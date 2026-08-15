from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from datamixxer.hub import HubCheck, check_hub_access, preflight_upload, repo_url, upload_folder
from datamixxer.io import read_json, read_yaml, write_json, write_jsonl, write_yaml
from datamixxer.standards import (
    artifact_version,
    dataset_card,
    now_iso,
    standard_repo_id,
)

DEFAULT_STORE_DIR = ".datamixxer/mixes"
Progress = Callable[[str], None]
MISSING = object()
PLACEHOLDER_DATASET_IDS = {"owner/dataset-name"}
PLACEHOLDER_REPO_IDS = {"owner/my_balanced_mix-v1"}
CONFIG_TEMPLATE = """# Stable id for the generated artifact and default repo naming.
id: my_balanced_mix
name: My Balanced Mix
version: v1

seed: 3407
buffer_size: 10000
dedupe: true

# Optional. Set to false or remove this block for train-only output.
split:
  test_size: 0.1

sources:
  - name: example
    # Replace with a real Hugging Face dataset id, for example HuggingFaceTB/smoltalk2.
    dataset_id: owner/dataset-name
    # Remove this line if the dataset has no config/subset.
    config: default
    split: train
    count: 1000
    metadata:
      domain: example

output:
  store_dir: .datamixxer/mixes
  train_file: train.jsonl
  test_file: test.jsonl
  push_to_hub: false
  hub:
    repo_id: owner/my_balanced_mix-v1
"""
EMPTY_CONFIG_TEMPLATE = """# Stable id for the generated artifact and default repo naming.
id: my_balanced_mix
name: My Balanced Mix
version: v1

seed: 3407
buffer_size: 10000
dedupe: true

# Optional. Set to false or remove this block for train-only output.
split:
  test_size: 0.1

sources: []

output:
  store_dir: .datamixxer/mixes
  train_file: train.jsonl
  test_file: test.jsonl
  push_to_hub: false
  hub: {}
"""


@dataclass(frozen=True)
class MixSource:
    index: int
    name: str
    dataset_id: str
    config: str | None
    split: str
    count: int
    train_count: int
    test_count: int
    raw: dict[str, Any]


@dataclass(frozen=True)
class MixArtifact:
    manifest: dict[str, Any]
    path: Path


@dataclass(frozen=True)
class MixPlan:
    config: dict[str, Any]
    hash_value: str
    path: Path
    sources: list[MixSource]
    artifact: MixArtifact | None


@dataclass(frozen=True)
class DatasetInspection:
    dataset_id: str
    configs: list[str]
    splits_by_config: dict[str | None, list[str]]


@dataclass(frozen=True)
class RowSample:
    source: MixSource
    rows: list[dict[str, Any]]
    scanned: int


def stream_rows(dataset_id: str, config: str | None, split: str, seed: int, buffer_size: int):
    from datasets import disable_progress_bar, load_dataset

    disable_progress_bar()
    dataset = load_dataset(dataset_id, config, split=split, streaming=True)
    shuffled = dataset.shuffle(seed=seed, buffer_size=buffer_size)
    yield from shuffled


def close_iterator(iterator: Any) -> None:
    close = getattr(iterator, "close", None)
    if callable(close):
        close()


def write_config_template(path: str | Path, *, overwrite: bool = False, empty: bool = False) -> Path:
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists; pass --force to overwrite it")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(EMPTY_CONFIG_TEMPLATE if empty else CONFIG_TEMPLATE, encoding="utf-8")
    return output


def write_new_config(
    path: str | Path,
    *,
    dataset_id: str,
    split: str,
    count: int,
    source_name: str | None = None,
    dataset_config: str | None = None,
    mix_id: str | None = None,
    name: str | None = None,
    test_size: Any = 0.1,
    repo_id: str | None = None,
    owner: str | None = None,
    overwrite: bool = False,
) -> Path:
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists; pass --force to overwrite it")
    artifact_id = mix_id or output.stem.replace("-", "_")
    config: dict[str, Any] = {
        "id": artifact_id,
        "name": name or _title_from_id(artifact_id),
        "version": "v1",
        "seed": 3407,
        "buffer_size": 10000,
        "dedupe": True,
        "split": {"test_size": test_size},
        "sources": [
            {
                "name": source_name or _source_name_from_dataset(dataset_id),
                "dataset_id": dataset_id,
                "split": split,
                "count": count,
            }
        ],
        "output": {
            "store_dir": DEFAULT_STORE_DIR,
            "train_file": "train.jsonl",
            "test_file": "test.jsonl",
            "push_to_hub": False,
            "hub": {},
        },
    }
    if dataset_config:
        config["sources"][0]["config"] = dataset_config
    if repo_id:
        config["output"]["hub"]["repo_id"] = repo_id
    elif owner:
        config["output"]["hub"]["owner"] = owner
    write_yaml(output, config)
    return output


def _format_config_issues(title: str, issues: list[str]) -> str:
    return title + ":\n" + "\n".join(f"- {issue}" for issue in issues)


def validate_config(config: dict[str, Any], *, require_hub: bool = False) -> None:
    if not _present(config.get("id")):
        raise ValueError("config is missing required field `id`")
    _as_int(config.get("seed", 3407), "seed")
    buffer_size = _as_int(config.get("buffer_size", 10_000), "buffer_size")
    if buffer_size <= 0:
        raise ValueError("buffer_size must be greater than 0")

    output = config.get("output") or {}
    if output is not None and not isinstance(output, dict):
        raise ValueError("`output` must be a mapping when provided")
    hub = output.get("hub", {})
    if hub is not None and not isinstance(hub, dict):
        raise ValueError("`output.hub` must be a mapping when provided")
    if require_hub and not standard_repo_id(
        hub or {},
        fallback_name=str(config["id"]),
        version=artifact_version(config),
        required=False,
    ):
        raise ValueError(
            "`--push-to-hub` requires `output.hub.repo_id`, `output.hub.owner`, or `--repo-id`"
        )

    dedupe_config = config.get("dedupe", True)
    if not (
        isinstance(dedupe_config, (bool, str, dict))
        or dedupe_config is None
    ):
        raise ValueError("dedupe must be a boolean, string, or mapping")
    normalize_tagging_rules(config)
    normalize_sources(config)
    warnings = config_warnings(config)
    if warnings:
        raise ValueError(_format_config_issues(f"Config has {len(warnings)} setup items to fix", warnings))


def config_warnings(config: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    try:
        sources = normalize_sources(config)
    except ValueError:
        return warnings

    for source in sources:
        if source.dataset_id in PLACEHOLDER_DATASET_IDS:
            warnings.append(
                f"{_source_path(source)}.dataset_id: replace placeholder dataset_id {source.dataset_id!r} "
                "with a real Hugging Face dataset id"
            )
        if source.dataset_id in PLACEHOLDER_DATASET_IDS and source.config == "default":
            warnings.append(
                f"{_source_path(source)}.config: remove 'default' if the dataset "
                "has no config or replace it with a real config name"
            )

    output = config.get("output") or {}
    hub = output.get("hub") or {}
    repo_id = hub.get("repo_id")
    if repo_id in PLACEHOLDER_REPO_IDS:
        warnings.append(
            f"output.hub.repo_id: replace placeholder {repo_id!r} before publishing"
        )
    return warnings


def validate_config_file(config_path: str | Path, *, require_hub: bool = False) -> dict[str, Any]:
    config = read_yaml(config_path)
    validate_config(config, require_hub=require_hub)
    return config


def check_source_access(config: dict[str, Any]) -> list[str]:
    """Validate Hugging Face dataset/config/split access without streaming rows."""
    from datasets import get_dataset_split_names

    errors: list[str] = []
    for source in normalize_sources(config):
        try:
            splits = get_dataset_split_names(source.dataset_id, config_name=source.config)
        except Exception as exc:
            errors.append(
                f"{source.name}: cannot access {source.dataset_id}"
                f"{f'/{source.config}' if source.config else ''}: {exc}"
            )
            continue
        if source.split not in splits:
            errors.append(
                f"{source.name}: split {source.split!r} was not found in "
                f"{source.dataset_id}{f'/{source.config}' if source.config else ''}; "
                f"available splits: {', '.join(splits) or 'none'}"
            )
    return errors


def validate_sample_rows(config: dict[str, Any], sample_rows: int) -> list[str]:
    """Read a small row sample to catch row-schema-dependent config problems."""
    if sample_rows <= 0:
        return []
    errors: list[str] = []
    seed = int(config.get("seed", 3407))
    buffer_size = int(config.get("buffer_size", 10_000))
    dedupe_config = config.get("dedupe", True)
    for source in normalize_sources(config):
        iterator = None
        try:
            iterator = stream_rows(
                source.dataset_id,
                source.config,
                source.split,
                seed + source.index,
                buffer_size,
            )
            checked = 0
            for row in iterator:
                if not isinstance(row, dict):
                    continue
                dedupe_key(row, dedupe_config)
                checked += 1
                if checked >= sample_rows:
                    break
            if checked == 0:
                errors.append(f"{source.name}: no dictionary rows found in the first sampled rows")
        except Exception as exc:
            errors.append(f"{source.name}: sampled row validation failed: {exc}")
        finally:
            if iterator is not None:
                close_iterator(iterator)
    return errors


def collect_row_samples(config: dict[str, Any], sample_rows: int) -> list[RowSample]:
    """Read a small row sample from each source for schema and content previews."""
    if sample_rows <= 0:
        raise ValueError("sample_rows must be greater than 0")
    validate_config(config)
    seed = int(config.get("seed", 3407))
    buffer_size = int(config.get("buffer_size", 10_000))
    dedupe_config = config.get("dedupe", True)
    samples: list[RowSample] = []
    for source in normalize_sources(config):
        rows: list[dict[str, Any]] = []
        scanned = 0
        iterator = stream_rows(
            source.dataset_id,
            source.config,
            source.split,
            seed + source.index,
            buffer_size,
        )
        try:
            for row in iterator:
                scanned += 1
                if not isinstance(row, dict):
                    continue
                dedupe_key(row, dedupe_config)
                rows.append(row)
                if len(rows) >= sample_rows:
                    break
        finally:
            close_iterator(iterator)
        if not rows:
            raise RuntimeError(f"{source.name}: no dictionary rows found in sampled source rows")
        samples.append(RowSample(source=source, rows=rows, scanned=scanned))
    return samples


def inspect_dataset(dataset_id: str, config: str | None = None) -> DatasetInspection:
    from datasets import get_dataset_config_names, get_dataset_split_names

    try:
        configs = get_dataset_config_names(dataset_id)
    except Exception:
        configs = []

    selected_configs: list[str | None]
    if config is not None:
        selected_configs = [config]
    elif configs:
        selected_configs = configs
    else:
        selected_configs = [None]

    splits_by_config: dict[str | None, list[str]] = {}
    for config_name in selected_configs:
        splits_by_config[config_name] = get_dataset_split_names(dataset_id, config_name=config_name)
    return DatasetInspection(
        dataset_id=dataset_id,
        configs=[str(item) for item in configs],
        splits_by_config=splits_by_config,
    )


def add_source_to_config(
    config_path: str | Path,
    *,
    name: str,
    dataset_id: str,
    split: str,
    count: int,
    config: str | None = None,
    metadata: dict[str, Any] | None = None,
    replace_placeholder: bool = True,
) -> dict[str, Any]:
    if not Path(config_path).exists():
        raise FileNotFoundError(f"{config_path} does not exist. Run `datamixxer init {config_path}` first.")
    mix_config = read_yaml(config_path)
    sources = mix_config.setdefault("sources", [])
    if not isinstance(sources, list):
        raise ValueError("`sources` must be a list before add-source can append to it")
    source: dict[str, Any] = {
        "name": name,
        "dataset_id": dataset_id,
        "split": split,
        "count": count,
    }
    if config:
        source["config"] = config
    if metadata:
        source["metadata"] = metadata
    placeholder_index = _placeholder_source_index(sources) if replace_placeholder else None
    if placeholder_index is None:
        sources.append(source)
    else:
        sources[placeholder_index] = source
        _clear_placeholder_hub_repo(mix_config)
    write_yaml(config_path, mix_config)
    return source


def mix_hash(config: dict[str, Any]) -> str:
    validate_config(config)
    payload = mix_hash_payload(config)
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def mix_hash_payload(config: dict[str, Any]) -> dict[str, Any]:
    sources = normalize_sources(config)
    tagging = normalize_tagging_rules(config)
    payload = {
        "schema_version": 1,
        "seed": int(config.get("seed", 3407)),
        "buffer_size": int(config.get("buffer_size", 10_000)),
        "dedupe": config.get("dedupe", True),
        "sources": [
            {
                "index": source.index,
                "name": source.name,
                "dataset_id": source.dataset_id,
                "config": source.config,
                "split": source.split,
                "count": source.count,
                "train_count": source.train_count,
                "test_count": source.test_count,
                "restyle": source.raw.get("restyle"),
                "metadata": source.raw.get("metadata") or {},
            }
            for source in sources
        ],
    }
    if tagging:
        payload["tagging"] = tagging
    return payload


def store_dir(config: dict[str, Any] | None = None, explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    if config is None:
        return Path(DEFAULT_STORE_DIR)
    output = config.get("output") or {}
    return Path(config.get("store_dir") or output.get("store_dir") or DEFAULT_STORE_DIR)


def output_dir_for(config: dict[str, Any], hash_value: str) -> Path:
    return store_dir(config) / hash_value


def list_mix_artifacts(store: str | Path | None = None) -> list[MixArtifact]:
    artifacts: list[MixArtifact] = []
    for manifest_path in Path(store or DEFAULT_STORE_DIR).glob("*/manifest.json"):
        manifest = read_json(manifest_path)
        if isinstance(manifest, dict):
            artifacts.append(MixArtifact(manifest=manifest, path=manifest_path.parent))
    return sorted(
        artifacts,
        key=lambda artifact: (
            str(artifact.manifest.get("artifact_id", "")),
            str(artifact.manifest.get("artifact_version", "")),
            str(artifact.manifest.get("mix_hash", "")),
        ),
    )


def resolve_mix_artifact(reference: str, store: str | Path | None = None) -> MixArtifact:
    ref_path = Path(reference)
    if ref_path.exists():
        manifest_path = ref_path / "manifest.json" if ref_path.is_dir() else ref_path
        manifest = read_json(manifest_path)
        if isinstance(manifest, dict):
            return MixArtifact(manifest=manifest, path=manifest_path.parent)

    artifacts = list_mix_artifacts(store)
    hash_matches = [
        artifact
        for artifact in artifacts
        if str(artifact.manifest.get("mix_hash", "")).startswith(reference)
    ]
    if len(hash_matches) == 1:
        return hash_matches[0]
    if len(hash_matches) > 1:
        raise ValueError(f"Ambiguous mix hash {reference!r}: {_format_candidates(hash_matches)}")

    id_matches = [
        artifact for artifact in artifacts if str(artifact.manifest.get("artifact_id", "")) == reference
    ]
    if len(id_matches) == 1:
        return id_matches[0]
    if len(id_matches) > 1:
        raise ValueError(f"Ambiguous mix id {reference!r}: {_format_candidates(id_matches)}")

    path_matches = [artifact for artifact in artifacts if str(artifact.path) == reference]
    if len(path_matches) == 1:
        return path_matches[0]
    store_label = str(store or DEFAULT_STORE_DIR)
    raise FileNotFoundError(
        f"No mix found for {reference!r} in {store_label}. Run `datamixxer list"
        f"{f' --store-dir {store_label}' if store else ''}` to see available mixes, "
        "or pass a mix hash prefix, artifact id, artifact directory, or manifest path."
    )


def _format_candidates(artifacts: list[MixArtifact]) -> str:
    return ", ".join(
        f"{artifact.manifest.get('mix_hash', '')[:12]} ({artifact.manifest.get('artifact_id')})"
        for artifact in artifacts
    )


def find_existing_mix(config: dict[str, Any], hash_value: str) -> MixArtifact | None:
    manifest_path = output_dir_for(config, hash_value) / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if isinstance(manifest, dict) and manifest.get("mix_hash") == hash_value:
            artifact = MixArtifact(manifest=manifest, path=manifest_path.parent)
            if mix_is_complete(artifact):
                return artifact
    return None


def plan_mix(config_path: str | Path) -> MixPlan:
    config = validate_config_file(config_path)
    hash_value = mix_hash(config)
    return MixPlan(
        config=config,
        hash_value=hash_value,
        path=output_dir_for(config, hash_value),
        sources=normalize_sources(config),
        artifact=find_existing_mix(config, hash_value),
    )


def mix_is_complete(artifact: MixArtifact) -> bool:
    splits = artifact.manifest.get("splits")
    if not isinstance(splits, dict) or not splits:
        return False
    for split_info in splits.values():
        if not isinstance(split_info, dict):
            return False
        filename = split_info.get("file")
        if not filename or not (artifact.path / str(filename)).exists():
            return False
    return (artifact.path / "README.md").exists()


def collect_mix(config: dict[str, Any], progress: Progress | None = None) -> dict[str, list[dict[str, Any]]]:
    validate_config(config)
    seed = int(config.get("seed", 3407))
    buffer_size = int(config.get("buffer_size", 10_000))
    sources = normalize_sources(config)
    dedupe_config = config.get("dedupe", True)
    seen_keys: set[str] = set()
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": []}
    if any(source.test_count for source in sources):
        rows_by_split["test"] = []

    for source in sources:
        needed = source.train_count + source.test_count
        collected: list[dict[str, Any]] = []
        scanned = 0
        duplicates = 0
        skipped_non_dict = 0
        if progress:
            progress(
                "Collecting "
                f"{source.name}: {needed} rows from {_source_label(source)} "
                f"(train={source.train_count}, test={source.test_count})"
            )
        iterator = stream_rows(
            source.dataset_id,
            source.config,
            source.split,
            seed + source.index,
            buffer_size,
        )
        start = perf_counter()

        try:
            for row in iterator:
                scanned += 1
                if not isinstance(row, dict):
                    skipped_non_dict += 1
                    continue
                key = dedupe_key(row, dedupe_config)
                if key is not None and key in seen_keys:
                    duplicates += 1
                    continue
                if key is not None:
                    seen_keys.add(key)
                collected.append(row)
                if len(collected) >= needed:
                    break
                if progress and scanned % _progress_interval(needed) == 0:
                    elapsed = max(perf_counter() - start, 0.001)
                    rate = scanned / elapsed
                    progress(
                        "Progress "
                        f"{source.name}: collected={len(collected)}/{needed} "
                        f"scanned={scanned} duplicates={duplicates} "
                        f"skipped_non_dict={skipped_non_dict} rate={rate:.1f} rows/s"
                    )
        finally:
            close_iterator(iterator)

        if len(collected) != needed:
            raise RuntimeError(
                f"Only collected {len(collected)} of {needed} rows for {source.name} "
                f"({source.dataset_id}, {source.split}); scanned={scanned}, "
                f"duplicates={duplicates}, skipped_non_dict={skipped_non_dict}"
            )

        test_rows = collected[: source.test_count]
        train_rows = collected[source.test_count :]
        rows_by_split["train"].extend(materialize_rows(train_rows, source, "train"))
        if source.test_count:
            rows_by_split.setdefault("test", []).extend(materialize_rows(test_rows, source, "test"))
        if progress:
            progress(
                "Collected "
                f"{source.name}: train={len(train_rows)} test={len(test_rows)} "
                f"scanned={scanned} duplicates={duplicates} skipped_non_dict={skipped_non_dict}"
            )

    return apply_tagging(
        {name: rows for name, rows in rows_by_split.items() if rows},
        config,
    )


def normalize_sources(config: dict[str, Any]) -> list[MixSource]:
    items_key = "sources" if config.get("sources") is not None else "plan"
    items = config.get(items_key)
    if not isinstance(items, list) or not items:
        raise ValueError("config must define a non-empty `sources` list or `plan` list")

    shared_source = config.get("source") or {}
    if shared_source is not None and not isinstance(shared_source, dict):
        raise ValueError("`source` must be a mapping when provided")

    split_config = config.get("split") or {}
    if split_config is not None and not isinstance(split_config, dict):
        raise ValueError("`split` must be a mapping when provided")
    default_test_size = config.get("test_size", split_config.get("test_size"))

    normalized: list[MixSource] = []
    for index, raw_item in enumerate(items):
        path = f"{items_key}[{index}]"
        if not isinstance(raw_item, dict):
            raise ValueError(f"{path} must be a mapping")
        item = dict(raw_item)
        dataset_id = item.get("dataset_id") or shared_source.get("dataset_id")
        if not dataset_id:
            raise ValueError(f"{path} is missing dataset_id")
        dataset_config = item.get("config", item.get("subset", shared_source.get("config")))
        source_split = item.get("split")
        if not source_split:
            raise ValueError(f"{path} is missing split")
        metadata = item.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError(f"{path}.metadata must be a mapping")
        name = str(item.get("name") or f"source_{index}")
        train_count, test_count, total_count = resolve_counts(
            item,
            default_test_size,
            path=path,
        )
        normalized.append(
            MixSource(
                index=index,
                name=name,
                dataset_id=str(dataset_id),
                config=None if dataset_config is None else str(dataset_config),
                split=str(source_split),
                count=total_count,
                train_count=train_count,
                test_count=test_count,
                raw=item,
            )
        )
    return normalized


def normalize_tagging_rules(config: dict[str, Any]) -> list[dict[str, Any]]:
    tagging = config.get("tagging", [])
    if tagging in (None, False):
        return []
    if isinstance(tagging, dict):
        raw_rules = [tagging]
    elif isinstance(tagging, list):
        raw_rules = tagging
    else:
        raise ValueError("tagging must be a mapping, list, false, or null")

    rules: list[dict[str, Any]] = []
    for index, raw_rule in enumerate(raw_rules):
        path = f"tagging[{index}]"
        if not isinstance(raw_rule, dict):
            raise ValueError(f"{path} must be a mapping")
        rate = _first_present(raw_rule, ("rate", "percentage", "percent"), path)
        split_test_count(100, rate, path=f"{path}.rate")
        tags = raw_rule.get("tags")
        if not isinstance(tags, dict) or not tags:
            raise ValueError(f"{path}.tags must be a non-empty mapping")
        try:
            json.dumps(tags, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}.tags must be JSON serializable") from exc
        balance_by = _normalize_string_list(
            raw_rule.get(
                "balance_by",
                ["source_dataset", "source_config", "source_split", "output_split"],
            ),
            path=f"{path}.balance_by",
        )
        output_splits = _normalize_optional_string_list(
            _first_present(
                raw_rule,
                ("output_splits", "output_split", "splits", "split"),
                path,
                default=None,
            ),
            path=f"{path}.output_splits",
        )
        rules.append(
            {
                "rate": rate,
                "tags": dict(tags),
                "balance_by": balance_by,
                "output_splits": output_splits,
            }
        )
    return rules


def _first_present(
    data: dict[str, Any],
    keys: tuple[str, ...],
    path: str,
    *,
    default: Any = MISSING,
) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    if default is not MISSING:
        return default
    raise ValueError(f"{path} is missing {keys[0]}")


def _normalize_optional_string_list(value: Any, *, path: str) -> list[str] | None:
    if value in (None, False):
        return None
    return _normalize_string_list(value, path=path)


def _normalize_string_list(value: Any, *, path: str) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        raise ValueError(f"{path} must be a string or list of strings")
    normalized = [str(item).strip() for item in values]
    if not normalized or any(not item for item in normalized):
        raise ValueError(f"{path} must not be empty")
    return normalized


def resolve_counts(
    item: dict[str, Any],
    default_test_size: Any,
    *,
    path: str = "source",
) -> tuple[int, int, int]:
    if "train_count" in item or "test_count" in item:
        train_count = _as_int(item.get("train_count", 0), f"{path}.train_count")
        test_count = _as_int(item.get("test_count", 0), f"{path}.test_count")
        if train_count < 0 or test_count < 0:
            raise ValueError(f"{path}.train_count and {path}.test_count must be non-negative")
        if train_count + test_count <= 0:
            raise ValueError(f"{path}.train_count and {path}.test_count must request at least one row")
        return train_count, test_count, train_count + test_count

    if "count" not in item:
        raise ValueError(f"{path} is missing count")
    total_count = _as_int(item["count"], f"{path}.count")
    if total_count <= 0:
        raise ValueError(f"{path}.count must be greater than 0")

    test_count = split_test_count(
        total_count,
        item.get("test_size", default_test_size),
        path=f"{path}.test_size",
    )
    return total_count - test_count, test_count, total_count


def split_test_count(total_count: int, test_size: Any, *, path: str = "test_size") -> int:
    if test_size in (None, False):
        return 0
    if isinstance(test_size, str) and test_size.strip().endswith("%"):
        try:
            fraction = float(test_size.strip()[:-1]) / 100
        except ValueError as exc:
            raise ValueError(f"{path} must be a percentage, fraction, or row count") from exc
        return round(total_count * fraction)
    if isinstance(test_size, float) and 0 <= test_size < 1:
        return round(total_count * test_size)
    if isinstance(test_size, str) and "." in test_size:
        try:
            fraction = float(test_size)
        except ValueError as exc:
            raise ValueError(f"{path} must be a percentage, fraction, or row count") from exc
        if 0 <= fraction < 1:
            return round(total_count * fraction)
    test_count = _as_int(test_size, path)
    if test_count < 0 or test_count > total_count:
        raise ValueError(f"{path} as a count must be between 0 and count")
    return test_count


def dedupe_key(row: dict[str, Any], dedupe_config: Any) -> str | None:
    if dedupe_config in (False, None):
        return None
    if dedupe_config is True:
        value = row.get("messages", row)
    elif isinstance(dedupe_config, str):
        value = select_key(row, dedupe_config)
        if value is MISSING:
            raise ValueError(f"dedupe field {dedupe_config!r} was not found in a source row")
    elif isinstance(dedupe_config, dict):
        if dedupe_config.get("enabled", True) is False:
            return None
        fields = dedupe_config.get("fields") or dedupe_config.get("field") or dedupe_config.get("key")
        if fields is None:
            value = row.get("messages", row)
        elif isinstance(fields, list):
            value = {}
            for field in fields:
                selected = select_key(row, str(field))
                if selected is MISSING:
                    raise ValueError(f"dedupe field {str(field)!r} was not found in a source row")
                value[field] = selected
        else:
            field = str(fields)
            value = select_key(row, field)
            if value is MISSING:
                raise ValueError(f"dedupe field {field!r} was not found in a source row")
    else:
        raise ValueError("dedupe must be a boolean, string, or mapping")
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def select_key(row: dict[str, Any], path: str) -> Any:
    if path in ("$row", "*"):
        return row
    value: Any = row
    for part in path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return MISSING
    return value


def _progress_interval(needed: int) -> int:
    return max(1_000, needed // 10)


def materialize_rows(
    rows: list[dict[str, Any]], source: MixSource, output_split: str
) -> list[dict[str, Any]]:
    materialized: list[dict[str, Any]] = []
    for row in rows:
        output = dict(row)
        metadata = source.raw.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"metadata for {source.name} must be a mapping")
        output.update(metadata)
        if "restyle" in source.raw:
            output["restyle"] = bool(source.raw["restyle"])
        output["bucket"] = source.name
        output["source_dataset"] = source.dataset_id
        output["source_config"] = source.config
        output["source_split"] = source.split
        output["output_split"] = output_split
        output.setdefault("source", row.get("source", source.split))
        materialized.append(output)
    return materialized


def apply_tagging(
    rows_by_split: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    rules = normalize_tagging_rules(config)
    if not rules:
        return rows_by_split

    seed = int(config.get("seed", 3407))
    for rule_index, rule in enumerate(rules):
        target_splits = set(rule["output_splits"] or rows_by_split)
        balance_by = rule["balance_by"]
        for split_name, rows in rows_by_split.items():
            if split_name not in target_splits:
                continue
            groups: dict[tuple[str, ...], list[int]] = {}
            for row_index, row in enumerate(rows):
                key = tuple(_tagging_group_value(row, field) for field in balance_by)
                groups.setdefault(key, []).append(row_index)
            for group_key, row_indexes in groups.items():
                tag_count = split_test_count(
                    len(row_indexes),
                    rule["rate"],
                    path=f"tagging[{rule_index}].rate",
                )
                if tag_count <= 0:
                    continue
                ranked = sorted(
                    row_indexes,
                    key=lambda row_index: _tagging_rank(
                        seed=seed,
                        rule_index=rule_index,
                        split_name=split_name,
                        group_key=group_key,
                        row_index=row_index,
                        row=rows[row_index],
                    ),
                )
                for row_index in ranked[:tag_count]:
                    rows[row_index].update(rule["tags"])
    return rows_by_split


def _tagging_group_value(row: dict[str, Any], field: str) -> str:
    value = select_key(row, field)
    if value is MISSING:
        return ""
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _tagging_rank(
    *,
    seed: int,
    rule_index: int,
    split_name: str,
    group_key: tuple[str, ...],
    row_index: int,
    row: dict[str, Any],
) -> str:
    payload = {
        "seed": seed,
        "rule_index": rule_index,
        "split_name": split_name,
        "group_key": group_key,
        "row_index": row_index,
        "row": row,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def counts_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return counts


def write_mix_artifacts(
    config: dict[str, Any],
    rows_by_split: dict[str, list[dict[str, Any]]],
    hash_value: str | None = None,
    config_path: str | Path | None = None,
    progress: Progress | None = None,
) -> MixArtifact:
    validate_config(config)
    output = config.get("output") or {}
    hash_value = hash_value or mix_hash(config)
    version = artifact_version(config)
    hub_repo_id = standard_repo_id(
        output.get("hub", {}),
        fallback_name=config["id"],
        version=version,
        required=False,
    )
    output_dir = output_dir_for(config, hash_value)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = output.get("train_file", "train.jsonl")
    test_file = output.get("test_file", "test.jsonl")
    split_files = {"train": train_file, "test": test_file}
    for split, rows in rows_by_split.items():
        if progress:
            progress(f"Writing {split_files.get(split, f'{split}.jsonl')}: {len(rows)} rows")
        write_jsonl(output_dir / split_files.get(split, f"{split}.jsonl"), rows)

    sources = normalize_sources(config)
    manifest_sources = [
        {
            "name": source.name,
            "dataset_id": source.dataset_id,
            "config": source.config,
            "split": source.split,
            "count": source.count,
            "train_count": source.train_count,
            "test_count": source.test_count,
        }
        for source in sources
    ]
    manifest = {
        "created_at": now_iso(),
        "artifact_type": "dataset_mix",
        "artifact_id": config["id"],
        "artifact_version": version,
        "mix_hash": hash_value,
        "mix_hash_payload": mix_hash_payload(config),
        "hub_repo_id": hub_repo_id,
        "config_path": str(config_path) if config_path is not None else None,
        "artifact_dir": str(output_dir),
        "store_dir": str(store_dir(config)),
        "name": config.get("name"),
        "seed": config.get("seed", 3407),
        "buffer_size": config.get("buffer_size", 10_000),
        "sources": manifest_sources,
        "splits": {
            split: {"rows": len(rows), "file": split_files.get(split, f"{split}.jsonl")}
            for split, rows in rows_by_split.items()
        },
        "buckets_by_split": {
            split: counts_by_key(rows, "bucket") for split, rows in rows_by_split.items()
        },
        "total_rows": sum(len(rows) for rows in rows_by_split.values()),
        "hub": {
            "pushed": False,
            "repo_id": hub_repo_id,
            "url": None,
            "pushed_at": None,
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    (output_dir / "README.md").write_text(
        dataset_card(config=config, manifest=manifest, repo_id=hub_repo_id),
        encoding="utf-8",
    )
    return MixArtifact(manifest=manifest, path=output_dir)


def push_mix_artifact(
    artifact: MixArtifact,
    *,
    repo_id: str | None = None,
    private: bool | None = None,
    commit_message: str | None = None,
) -> MixArtifact:
    manifest = artifact.manifest
    resolved_repo_id = repo_id or manifest.get("hub_repo_id") or manifest.get("hub", {}).get("repo_id")
    if not resolved_repo_id:
        raise KeyError("hub.repo_id or --repo-id is required for publishing")
    resolved_private = bool(private) if private is not None else False
    resolved_commit = commit_message or "Upload datamixxer dataset"
    upload_url = repo_url(str(resolved_repo_id), "dataset")
    previous_manifest = json.loads(json.dumps(manifest, ensure_ascii=False))

    preflight_upload(
        repo_id=str(resolved_repo_id),
        repo_type="dataset",
        private=resolved_private,
    )
    manifest["hub_repo_id"] = str(resolved_repo_id)
    manifest["hub"] = {
        "pushed": True,
        "repo_id": str(resolved_repo_id),
        "url": upload_url,
        "pushed_at": now_iso(),
        "private": resolved_private,
        "commit_message": resolved_commit,
    }
    write_json(artifact.path / "manifest.json", manifest)
    try:
        upload_folder(
            repo_id=str(resolved_repo_id),
            repo_type="dataset",
            folder_path=artifact.path,
            private=resolved_private,
            commit_message=resolved_commit,
        )
    except Exception:
        previous_manifest["hub"] = {
            **(previous_manifest.get("hub") or {}),
            "pushed": False,
            "repo_id": str(resolved_repo_id),
            "url": upload_url,
            "failed_at": now_iso(),
        }
        write_json(artifact.path / "manifest.json", previous_manifest)
        raise
    return MixArtifact(manifest=manifest, path=artifact.path)


def push_mix(
    reference: str,
    *,
    store: str | Path | None = None,
    repo_id: str | None = None,
    private: bool | None = None,
    commit_message: str | None = None,
) -> MixArtifact:
    return push_mix_artifact(
        resolve_mix_artifact(reference, store),
        repo_id=repo_id,
        private=private,
        commit_message=commit_message,
    )


def build_mix(config_path: str | Path, *, push: bool | None = None, force: bool = False) -> MixArtifact:
    config = validate_config_file(config_path)
    output = config.get("output") or {}
    should_push = output.get("push_to_hub", False) if push is None else push
    if should_push:
        validate_config(config, require_hub=True)
    hash_value = mix_hash(config)
    artifact = None if force else find_existing_mix(config, hash_value)
    sources = normalize_sources(config)
    print(f"Building mix {hash_value[:12]}")
    print(f"Store: {output_dir_for(config, hash_value)}")
    print(f"Plan: {sum(source.count for source in sources)} rows across {len(sources)} sources")
    if artifact is None:
        rows_by_split = collect_mix(config, progress=print)
        artifact = write_mix_artifacts(
            config,
            rows_by_split,
            hash_value=hash_value,
            config_path=config_path,
            progress=print,
        )
        print(f"Created mix {hash_value[:12]} at {artifact.path}")
    else:
        print(f"Reusing existing mix {hash_value[:12]} at {artifact.path}")
        print("Sampling inputs are unchanged. Use --force to rebuild anyway.")

    if should_push:
        hub = output["hub"]
        print(
            "Uploading "
            f"{standard_repo_id(hub, fallback_name=config['id'], version=artifact_version(config))} "
            f"({'private' if hub.get('private', False) else 'public'})"
        )
        artifact = push_mix_artifact(
            artifact,
            repo_id=standard_repo_id(
                hub,
                fallback_name=config["id"],
                version=artifact_version(config),
            ),
            private=bool(hub.get("private", False)),
            commit_message=hub.get("commit_message", "Upload datamixxer dataset"),
        )
        print(f"Uploaded dataset to {artifact.manifest['hub']['url']}")
    print_build_summary(artifact)
    return artifact


def explain_hash(config: dict[str, Any]) -> str:
    validate_config(config)
    return json.dumps(mix_hash_payload(config), indent=2, ensure_ascii=False, sort_keys=True)


def hub_check_for_config(config_path: str | Path, *, repo_id: str | None = None, private: bool = False) -> HubCheck:
    config = validate_config_file(config_path)
    output = config.get("output") or {}
    hub = output.get("hub") or {}
    resolved_repo_id = repo_id or standard_repo_id(
        hub,
        fallback_name=config["id"],
        version=artifact_version(config),
    )
    return check_hub_access(
        repo_id=str(resolved_repo_id),
        repo_type="dataset",
        private=bool(private or hub.get("private", False)),
    )


def _source_label(source: MixSource) -> str:
    if source.config:
        return f"{source.dataset_id}/{source.config}:{source.split}"
    return f"{source.dataset_id}:{source.split}"


def _source_path(source: MixSource) -> str:
    key = "sources" if "dataset_id" in source.raw else "plan"
    return f"{key}[{source.index}]"


def print_build_summary(artifact: MixArtifact) -> None:
    manifest = artifact.manifest
    print("\nOutputs:")
    print(f"  artifact: {artifact.path}")
    print(f"  manifest: {artifact.path / 'manifest.json'}")
    print(f"  dataset card: {artifact.path / 'README.md'}")
    for split, info in (manifest.get("splits") or {}).items():
        print(f"  {split}: {artifact.path / str(info.get('file'))} ({info.get('rows')} rows)")
    print("Next:")
    print(f"  datamixxer show {str(manifest.get('mix_hash', ''))[:12]}")
    hub = manifest.get("hub") or {}
    repo_id = hub.get("repo_id") or manifest.get("hub_repo_id")
    if repo_id and not hub.get("pushed"):
        print(f"  datamixxer publish {manifest.get('config_path') or manifest.get('mix_hash')} --repo-id {repo_id}")


def _placeholder_source_index(sources: list[Any]) -> int | None:
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            continue
        if source.get("dataset_id") in PLACEHOLDER_DATASET_IDS:
            return index
    return None


def _clear_placeholder_hub_repo(config: dict[str, Any]) -> None:
    output = config.get("output")
    if not isinstance(output, dict):
        return
    hub = output.get("hub")
    if not isinstance(hub, dict):
        return
    if hub.get("repo_id") in PLACEHOLDER_REPO_IDS:
        hub.pop("repo_id")


def _title_from_id(value: str) -> str:
    return str(value).replace("_", " ").replace("-", " ").title()


def _source_name_from_dataset(dataset_id: str) -> str:
    return slug_candidate(dataset_id.rsplit("/", 1)[-1])


def slug_candidate(value: str) -> str:
    candidate = str(value).strip().lower().replace("-", "_")
    candidate = "".join(char if char.isalnum() or char == "_" else "_" for char in candidate)
    candidate = "_".join(part for part in candidate.split("_") if part)
    return candidate or "source"


def _present(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _as_int(value: Any, path: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be an integer") from exc
