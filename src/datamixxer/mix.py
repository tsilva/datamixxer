from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable

from datamixxer.hub import HubCheck, check_hub_access, preflight_upload, repo_url, upload_folder
from datamixxer.io import read_json, read_yaml, write_json, write_jsonl
from datamixxer.standards import (
    artifact_version,
    dataset_card,
    now_iso,
    standard_repo_id,
)

DEFAULT_STORE_DIR = ".datamixxer/mixes"
Progress = Callable[[str], None]
CONFIG_TEMPLATE = """id: my_balanced_mix
name: My Balanced Mix
version: v1
seed: 3407
buffer_size: 10000
dedupe: true

split:
  test_size: 0.1

sources:
  - name: example
    dataset_id: owner/dataset-name
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


def stream_rows(dataset_id: str, config: str | None, split: str, seed: int, buffer_size: int):
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, config, split=split, streaming=True)
    shuffled = dataset.shuffle(seed=seed, buffer_size=buffer_size)
    yield from shuffled


def write_config_template(path: str | Path, *, overwrite: bool = False) -> Path:
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists; pass --force to overwrite it")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(CONFIG_TEMPLATE, encoding="utf-8")
    return output


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
    normalize_sources(config)


def validate_config_file(config_path: str | Path, *, require_hub: bool = False) -> dict[str, Any]:
    config = read_yaml(config_path)
    validate_config(config, require_hub=require_hub)
    return config


def mix_hash(config: dict[str, Any]) -> str:
    validate_config(config)
    payload = mix_hash_payload(config)
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def mix_hash_payload(config: dict[str, Any]) -> dict[str, Any]:
    sources = normalize_sources(config)
    return {
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
    raise FileNotFoundError(f"No mix found for {reference!r}")


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

    return {name: rows for name, rows in rows_by_split.items() if rows}


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
        return train_count, test_count, train_count + test_count

    if "count" not in item:
        raise ValueError(f"{path} is missing count")
    total_count = _as_int(item["count"], f"{path}.count")
    if total_count < 0:
        raise ValueError(f"{path}.count must be non-negative")

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
    elif isinstance(dedupe_config, dict):
        if dedupe_config.get("enabled", True) is False:
            return None
        fields = dedupe_config.get("fields") or dedupe_config.get("field") or dedupe_config.get("key")
        if fields is None:
            value = row.get("messages", row)
        elif isinstance(fields, list):
            value = {field: select_key(row, str(field)) for field in fields}
        else:
            value = select_key(row, str(fields))
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
            return None
    return value


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
    print(f"Building mix {hash_value[:12]}")
    print(f"Store: {output_dir_for(config, hash_value)}")
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


def _present(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _as_int(value: Any, path: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be an integer") from exc
