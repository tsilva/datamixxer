from __future__ import annotations

import json

import datamixxer.hub as hub
import datamixxer.mix as mix
import pytest
from datamixxer.cli import print_mix_plan
from datamixxer.io import read_yaml
from datamixxer.mix import (
    add_source_to_config,
    build_mix,
    check_source_access,
    collect_row_samples,
    dedupe_key,
    explain_hash,
    hub_check_for_config,
    list_mix_artifacts,
    materialize_rows,
    mix_hash,
    normalize_sources,
    plan_mix,
    push_mix,
    resolve_mix_artifact,
    split_test_count,
    validate_config,
    validate_config_file,
    validate_sample_rows,
    write_new_config,
    write_config_template,
)


def test_split_test_count_supports_fraction_percent_and_count() -> None:
    assert split_test_count(100, 0.1) == 10
    assert split_test_count(100, "10%") == 10
    assert split_test_count(100, "0.1") == 10
    assert split_test_count(100, 7) == 7


def test_validate_config_rejects_zero_row_sources() -> None:
    with pytest.raises(ValueError, match="sources\\[0\\].count must be greater than 0"):
        validate_config(
            {
                "id": "sample",
                "sources": [
                    {"name": "alpha", "dataset_id": "org/alpha", "split": "train", "count": 0}
                ],
            }
        )

    with pytest.raises(ValueError, match="must request at least one row"):
        validate_config(
            {
                "id": "sample",
                "sources": [
                    {
                        "name": "alpha",
                        "dataset_id": "org/alpha",
                        "split": "train",
                        "train_count": 0,
                        "test_count": 0,
                    }
                ],
            }
        )


def test_normalize_sources_supports_llmstyler_plan_shape() -> None:
    config = {
        "source": {"dataset_id": "org/source", "config": "SFT"},
        "split": {"test_size": 0.2},
        "plan": [{"name": "bucket", "split": "train", "count": 10, "restyle": True}],
    }

    sources = normalize_sources(config)

    assert len(sources) == 1
    assert sources[0].dataset_id == "org/source"
    assert sources[0].config == "SFT"
    assert sources[0].split == "train"
    assert sources[0].train_count == 8
    assert sources[0].test_count == 2


def test_normalize_sources_supports_multi_dataset_sources() -> None:
    config = {
        "sources": [
            {"name": "a", "dataset_id": "org/a", "split": "train", "count": 5},
            {
                "name": "b",
                "dataset_id": "org/b",
                "config": "default",
                "split": "validation",
                "train_count": 3,
                "test_count": 2,
            },
        ]
    }

    sources = normalize_sources(config)

    assert [(source.name, source.dataset_id, source.train_count, source.test_count) for source in sources] == [
        ("a", "org/a", 5, 0),
        ("b", "org/b", 3, 2),
    ]


def test_validate_config_reports_missing_id(tmp_path) -> None:
    config_path = tmp_path / "mix.yaml"
    config_path.write_text(
        "\n".join(
            [
                "sources:",
                "  - name: alpha",
                "    dataset_id: org/alpha",
                "    split: train",
                "    count: 1",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required field `id`"):
        validate_config_file(config_path)


def test_write_config_template_refuses_existing_file(tmp_path) -> None:
    config_path = tmp_path / "mix.yaml"
    write_config_template(config_path)

    with pytest.raises(FileExistsError, match="already exists"):
        write_config_template(config_path)

    write_config_template(config_path, overwrite=True)
    assert "sources:" in config_path.read_text(encoding="utf-8")


def test_write_empty_config_template_has_no_placeholder_sources(tmp_path) -> None:
    config_path = tmp_path / "mix.yaml"

    write_config_template(config_path, empty=True)

    config = read_yaml(config_path)
    assert config["sources"] == []


def test_write_new_config_creates_valid_first_source(tmp_path) -> None:
    config_path = tmp_path / "math-mix.yaml"

    write_new_config(
        config_path,
        dataset_id="org/math",
        dataset_config="default",
        split="train",
        count=25,
        repo_id="owner/math-mix",
    )

    config = validate_config_file(config_path)
    sources = normalize_sources(config)
    assert config["id"] == "math_mix"
    assert sources[0].name == "math"
    assert sources[0].dataset_id == "org/math"
    assert sources[0].config == "default"
    assert sources[0].count == 25
    assert config["output"]["hub"]["repo_id"] == "owner/math-mix"


def test_validate_config_rejects_unedited_starter_placeholders(tmp_path) -> None:
    config_path = tmp_path / "mix.yaml"
    write_config_template(config_path)

    with pytest.raises(ValueError, match="Config has 3 setup items to fix"):
        validate_config_file(config_path)


def test_dedupe_defaults_to_messages_when_present() -> None:
    row_a = {"messages": [{"role": "user", "content": "hi"}], "id": 1}
    row_b = {"messages": [{"role": "user", "content": "hi"}], "id": 2}

    assert dedupe_key(row_a, True) == dedupe_key(row_b, True)


def test_dedupe_explicit_missing_field_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="dedupe field 'messages' was not found"):
        dedupe_key({"text": "hello"}, "messages")


def test_dedupe_mapping_missing_field_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="dedupe field 'messages.content' was not found"):
        dedupe_key({"messages": []}, {"field": "messages.content"})


def test_materialize_rows_adds_mix_metadata_and_item_metadata() -> None:
    source = normalize_sources(
        {
            "sources": [
                {
                    "name": "bucket",
                    "dataset_id": "org/data",
                    "config": "cfg",
                    "split": "train",
                    "count": 1,
                    "restyle": True,
                    "metadata": {"domain": "chat"},
                }
            ]
        }
    )[0]

    rows = materialize_rows([{"messages": [], "source": "upstream"}], source, "test")

    assert rows == [
        {
            "messages": [],
            "source": "upstream",
            "domain": "chat",
            "restyle": True,
            "bucket": "bucket",
            "source_dataset": "org/data",
            "source_config": "cfg",
            "source_split": "train",
            "output_split": "test",
        }
    ]


def test_collect_mix_preserves_bucket_balance_in_test_split(monkeypatch) -> None:
    def fake_stream_rows(dataset_id, config, split, seed, buffer_size):
        for index in range(20):
            yield {
                "id": f"{dataset_id}-{split}-{index}",
                "messages": [{"role": "user", "content": f"{dataset_id}-{split}-{index}"}],
            }

    monkeypatch.setattr(mix, "stream_rows", fake_stream_rows)

    rows_by_split = mix.collect_mix(
        {
            "id": "sample",
            "seed": 1,
            "buffer_size": 5,
            "split": {"test_size": 0.25},
            "sources": [
                {"name": "alpha", "dataset_id": "org/alpha", "split": "train", "count": 8},
                {"name": "beta", "dataset_id": "org/beta", "split": "train", "count": 4},
            ],
        }
    )

    assert mix.counts_by_key(rows_by_split["train"], "bucket") == {"alpha": 6, "beta": 3}
    assert mix.counts_by_key(rows_by_split["test"], "bucket") == {"alpha": 2, "beta": 1}
    assert all(row["output_split"] == "train" for row in rows_by_split["train"])
    assert all(row["output_split"] == "test" for row in rows_by_split["test"])


def test_collect_mix_reports_progress(monkeypatch) -> None:
    def fake_stream_rows(dataset_id, config, split, seed, buffer_size):
        yield {"id": "one", "messages": [{"role": "user", "content": "hi"}]}
        yield {"id": "two", "messages": [{"role": "user", "content": "there"}]}

    monkeypatch.setattr(mix, "stream_rows", fake_stream_rows)
    messages: list[str] = []

    mix.collect_mix(
        {
            "id": "sample",
            "sources": [{"name": "alpha", "dataset_id": "org/alpha", "split": "train", "count": 1}],
        },
        progress=messages.append,
    )

    assert any("Collecting alpha" in message for message in messages)
    assert any("Collected alpha" in message and "scanned=1" in message for message in messages)


def test_collect_mix_closes_stream_iterator_after_collecting_needed_rows(monkeypatch) -> None:
    class ClosableRows:
        def __init__(self) -> None:
            self.closed = False
            self.rows = iter(
                [
                    {"id": "one", "messages": [{"role": "user", "content": "hi"}]},
                    {"id": "two", "messages": [{"role": "user", "content": "there"}]},
                ]
            )

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.rows)

        def close(self) -> None:
            self.closed = True

    stream = ClosableRows()

    def fake_stream_rows(dataset_id, config, split, seed, buffer_size):
        return stream

    monkeypatch.setattr(mix, "stream_rows", fake_stream_rows)

    rows_by_split = mix.collect_mix(
        {
            "id": "sample",
            "sources": [{"name": "alpha", "dataset_id": "org/alpha", "split": "train", "count": 1}],
        }
    )

    assert len(rows_by_split["train"]) == 1
    assert stream.closed is True


def test_collect_mix_tags_rate_by_provenance_for_target_output_split(monkeypatch) -> None:
    def fake_stream_rows(dataset_id, config, split, seed, buffer_size):
        for index in range(20):
            yield {
                "id": f"{dataset_id}-{index}",
                "messages": [{"role": "user", "content": f"{dataset_id}-{index}"}],
            }

    monkeypatch.setattr(mix, "stream_rows", fake_stream_rows)

    rows_by_split = mix.collect_mix(
        {
            "id": "sample",
            "seed": 1,
            "split": {"test_size": 0.2},
            "sources": [
                {"name": "alpha", "dataset_id": "org/alpha", "split": "train", "count": 10},
                {"name": "beta", "dataset_id": "org/beta", "split": "train", "count": 10},
            ],
            "tagging": [
                {
                    "rate": 0.25,
                    "output_splits": ["train"],
                    "tags": {"restyle": True, "target_style": "plain"},
                }
            ],
        }
    )

    tagged_train = [row for row in rows_by_split["train"] if row.get("restyle") is True]
    tagged_test = [row for row in rows_by_split["test"] if row.get("restyle") is True]

    assert mix.counts_by_key(tagged_train, "source_dataset") == {"org/alpha": 2, "org/beta": 2}
    assert all(row["target_style"] == "plain" for row in tagged_train)
    assert tagged_test == []


def test_validate_config_rejects_invalid_tagging_rules() -> None:
    with pytest.raises(ValueError, match="tagging\\[0\\].tags must be a non-empty mapping"):
        validate_config(
            {
                "id": "sample",
                "sources": [
                    {"name": "alpha", "dataset_id": "org/alpha", "split": "train", "count": 1}
                ],
                "tagging": [{"rate": 0.2, "tags": {}}],
            }
        )


def test_mix_hash_changes_for_tagging() -> None:
    base = {
        "id": "sample",
        "seed": 1,
        "sources": [{"name": "alpha", "dataset_id": "org/alpha", "split": "train", "count": 2}],
    }
    tagged = {
        **base,
        "tagging": [{"rate": 0.5, "output_splits": ["train"], "tags": {"restyle": True}}],
    }

    assert mix_hash(base) != mix_hash(tagged)


def test_empty_tagging_does_not_change_mix_hash() -> None:
    base = {
        "id": "sample",
        "seed": 1,
        "sources": [{"name": "alpha", "dataset_id": "org/alpha", "split": "train", "count": 2}],
    }

    assert mix_hash(base) == mix_hash({**base, "tagging": []})


def test_mix_hash_ignores_output_but_changes_for_seed() -> None:
    base = {
        "id": "sample",
        "seed": 1,
        "sources": [{"name": "alpha", "dataset_id": "org/alpha", "split": "train", "count": 2}],
        "output": {"dir": "one"},
    }
    moved = {**base, "output": {"dir": "two"}}
    changed_seed = {**base, "seed": 2}

    assert mix_hash(base) == mix_hash(moved)
    assert mix_hash(base) != mix_hash(changed_seed)


def test_explain_hash_shows_normalized_sampling_inputs() -> None:
    rendered = explain_hash(
        {
            "id": "sample",
            "seed": 1,
            "sources": [{"name": "alpha", "dataset_id": "org/alpha", "split": "train", "count": 2}],
        }
    )
    payload = json.loads(rendered)

    assert payload["seed"] == 1
    assert payload["sources"][0]["name"] == "alpha"
    assert payload["sources"][0]["train_count"] == 2


def test_check_source_access_reports_missing_split(monkeypatch) -> None:
    def fake_get_dataset_split_names(path, config_name=None):
        assert path == "org/alpha"
        assert config_name is None
        return ["train", "validation"]

    monkeypatch.setattr("datasets.get_dataset_split_names", fake_get_dataset_split_names)

    errors = check_source_access(
        {
            "id": "sample",
            "sources": [
                {
                    "name": "alpha",
                    "dataset_id": "org/alpha",
                    "split": "test",
                    "count": 1,
                }
            ],
        }
    )

    assert errors == [
        "alpha: split 'test' was not found in org/alpha; available splits: train, validation"
    ]


def test_build_mix_reuses_existing_hash(monkeypatch, tmp_path) -> None:
    calls = {"count": 0}

    def fake_collect(config, progress=None):
        calls["count"] += 1
        return {
            "train": [
                {
                    "id": "row-1",
                    "messages": [{"role": "user", "content": "hi"}],
                    "bucket": "alpha",
                    "source_dataset": "org/alpha",
                    "source_config": None,
                    "source_split": "train",
                    "output_split": "train",
                    "source": "train",
                }
            ]
        }

    monkeypatch.setattr(mix, "collect_mix", fake_collect)
    config_path = tmp_path / "mix.yaml"
    store_dir = tmp_path / "store"
    config_path.write_text(
        "\n".join(
            [
                "id: sample",
                "seed: 1",
                "sources:",
                "  - name: alpha",
                "    dataset_id: org/alpha",
                "    split: train",
                "    count: 1",
                "output:",
                f"  store_dir: {store_dir}",
                "  push_to_hub: false",
            ]
        ),
        encoding="utf-8",
    )

    first = build_mix(config_path)
    second = build_mix(config_path)

    assert first.manifest["mix_hash"] == second.manifest["mix_hash"]
    assert calls["count"] == 1
    assert first.path == store_dir / first.manifest["mix_hash"]
    assert (first.path / "manifest.json").exists()
    assert list_mix_artifacts(store_dir)[0].manifest["artifact_id"] == "sample"


def test_build_mix_reuse_message_explains_force(monkeypatch, tmp_path, capsys) -> None:
    def fake_collect(config, progress=None):
        return {
            "train": [
                {
                    "id": "row-1",
                    "messages": [{"role": "user", "content": "hi"}],
                    "bucket": "alpha",
                    "source_dataset": "org/alpha",
                    "source_config": None,
                    "source_split": "train",
                    "output_split": "train",
                    "source": "train",
                }
            ]
        }

    monkeypatch.setattr(mix, "collect_mix", fake_collect)
    config_path = tmp_path / "mix.yaml"
    store_dir = tmp_path / "store"
    config_path.write_text(
        "\n".join(
            [
                "id: sample",
                "sources:",
                "  - name: alpha",
                "    dataset_id: org/alpha",
                "    split: train",
                "    count: 1",
                "output:",
                f"  store_dir: {store_dir}",
            ]
        ),
        encoding="utf-8",
    )

    build_mix(config_path)
    build_mix(config_path)

    output = capsys.readouterr().out
    assert "Sampling inputs are unchanged. Use --force to rebuild anyway." in output


def test_plan_mix_resolves_counts_hash_path_and_status(tmp_path) -> None:
    config_path = tmp_path / "mix.yaml"
    store_dir = tmp_path / "store"
    config_path.write_text(
        "\n".join(
            [
                "id: sample",
                "version: v2",
                "seed: 1",
                "split:",
                "  test_size: 0.25",
                "sources:",
                "  - name: alpha",
                "    dataset_id: org/alpha",
                "    split: train",
                "    count: 8",
                "output:",
                f"  store_dir: {store_dir}",
            ]
        ),
        encoding="utf-8",
    )

    planned = plan_mix(config_path)

    assert planned.artifact is None
    assert planned.path == store_dir / planned.hash_value
    assert [(source.name, source.train_count, source.test_count) for source in planned.sources] == [
        ("alpha", 6, 2)
    ]


def test_print_mix_plan_includes_short_hash_and_next_steps(tmp_path, capsys) -> None:
    config_path = tmp_path / "mix.yaml"
    config_path.write_text(
        "\n".join(
            [
                "id: sample",
                "sources:",
                "  - name: alpha",
                "    dataset_id: org/alpha",
                "    split: train",
                "    count: 1",
            ]
        ),
        encoding="utf-8",
    )
    planned = plan_mix(config_path)

    print_mix_plan(planned, config_path=str(config_path))

    output = capsys.readouterr().out
    assert f"Short hash: {planned.hash_value[:12]}" in output
    assert "next:" in output
    assert f"datamixxer build {config_path}" in output


def test_validate_sample_rows_catches_dedupe_field_errors(monkeypatch) -> None:
    def fake_stream_rows(dataset_id, config, split, seed, buffer_size):
        yield {"text": "hello"}

    monkeypatch.setattr(mix, "stream_rows", fake_stream_rows)

    errors = validate_sample_rows(
        {
            "id": "sample",
            "dedupe": {"field": "messages.content"},
            "sources": [{"name": "alpha", "dataset_id": "org/alpha", "split": "train", "count": 1}],
        },
        1,
    )

    assert "alpha: sampled row validation failed" in errors[0]
    assert "dedupe field 'messages.content' was not found" in errors[0]


def test_collect_row_samples_returns_rows_and_scan_counts(monkeypatch) -> None:
    def fake_stream_rows(dataset_id, config, split, seed, buffer_size):
        yield "skip"
        yield {"id": "one", "messages": [{"role": "user", "content": "hi"}]}
        yield {"id": "two", "messages": [{"role": "user", "content": "there"}]}

    monkeypatch.setattr(mix, "stream_rows", fake_stream_rows)

    samples = collect_row_samples(
        {
            "id": "sample",
            "sources": [{"name": "alpha", "dataset_id": "org/alpha", "split": "train", "count": 2}],
        },
        2,
    )

    assert len(samples) == 1
    assert samples[0].source.name == "alpha"
    assert [row["id"] for row in samples[0].rows] == ["one", "two"]
    assert samples[0].scanned == 3


def test_add_source_to_config_appends_sources(tmp_path) -> None:
    config_path = tmp_path / "mix.yaml"
    config_path.write_text("id: sample\nsources: []\n", encoding="utf-8")

    source = add_source_to_config(
        config_path,
        name="alpha",
        dataset_id="org/alpha",
        config="default",
        split="train",
        count=5,
        metadata={"domain": "math"},
    )

    assert source["name"] == "alpha"
    sources = normalize_sources(validate_config_file(config_path))
    assert [(item.name, item.dataset_id, item.config, item.count) for item in sources] == [
        ("alpha", "org/alpha", "default", 5)
    ]


def test_add_source_replaces_starter_placeholder_by_default(tmp_path) -> None:
    config_path = tmp_path / "mix.yaml"
    write_config_template(config_path)

    add_source_to_config(
        config_path,
        name="alpha",
        dataset_id="org/alpha",
        split="train",
        count=5,
    )

    sources = normalize_sources(validate_config_file(config_path))
    assert [(item.name, item.dataset_id, item.count) for item in sources] == [
        ("alpha", "org/alpha", 5)
    ]


def test_hub_check_for_config_uses_config_repo(monkeypatch, tmp_path) -> None:
    def fake_check_hub_access(*, repo_id, repo_type, private):
        return mix.HubCheck(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            user="tester",
            url=f"https://huggingface.co/datasets/{repo_id}",
        )

    monkeypatch.setattr(mix, "check_hub_access", fake_check_hub_access)
    config_path = tmp_path / "mix.yaml"
    config_path.write_text(
        "\n".join(
            [
                "id: sample",
                "sources:",
                "  - name: alpha",
                "    dataset_id: org/alpha",
                "    split: train",
                "    count: 1",
                "output:",
                "  hub:",
                "    repo_id: owner/sample",
                "    private: true",
            ]
        ),
        encoding="utf-8",
    )

    check = hub_check_for_config(config_path)

    assert check.repo_id == "owner/sample"
    assert check.private is True


def test_hub_check_does_not_create_repo(monkeypatch) -> None:
    calls = []

    class FakeApi:
        def whoami(self):
            return {"name": "tester"}

        def repo_info(self, *, repo_id, repo_type):
            calls.append(("repo_info", repo_id, repo_type))

        def create_repo(self, **kwargs):
            calls.append(("create_repo", kwargs))

    monkeypatch.setattr(hub, "_api", FakeApi)

    check = hub.check_hub_access(repo_id="owner/sample", repo_type="dataset", private=True)

    assert check.user == "tester"
    assert calls == [("repo_info", "owner/sample", "dataset")]


def test_upload_preflight_can_create_repo(monkeypatch) -> None:
    calls = []

    class FakeApi:
        def whoami(self):
            return {"name": "tester"}

        def repo_info(self, *, repo_id, repo_type):
            calls.append(("repo_info", repo_id, repo_type))

        def create_repo(self, **kwargs):
            calls.append(("create_repo", kwargs))

    monkeypatch.setattr(hub, "_api", FakeApi)

    hub.preflight_upload(repo_id="owner/sample", repo_type="dataset", private=True)

    assert calls == [
        (
            "create_repo",
            {
                "repo_id": "owner/sample",
                "repo_type": "dataset",
                "private": True,
                "exist_ok": True,
            },
        )
    ]


def test_push_mix_updates_manifest_before_upload(monkeypatch, tmp_path) -> None:
    def fake_preflight_upload(*, repo_id, repo_type, private):
        assert repo_id == "owner/sample"
        assert repo_type == "dataset"
        assert private is True

    def fake_upload_folder(*, repo_id, repo_type, folder_path, private, commit_message):
        uploaded_manifest = json.loads((folder_path / "manifest.json").read_text(encoding="utf-8"))
        assert repo_id == "owner/sample"
        assert repo_type == "dataset"
        assert private is True
        assert commit_message == "sync sample"
        assert folder_path == output_dir
        assert uploaded_manifest["hub"]["pushed"] is True

    monkeypatch.setattr(mix, "preflight_upload", fake_preflight_upload)
    monkeypatch.setattr(mix, "upload_folder", fake_upload_folder)
    store_dir = tmp_path / "store"
    output_dir = store_dir / "abcdef123456"
    manifest = {
        "artifact_id": "sample",
        "artifact_version": "v1",
        "artifact_dir": str(output_dir),
        "store_dir": str(store_dir),
        "mix_hash": "abcdef123456",
        "total_rows": 1,
        "hub_repo_id": None,
        "hub": {"pushed": False, "repo_id": None, "url": None, "pushed_at": None},
        "splits": {"train": {"rows": 1, "file": "train.jsonl"}},
    }
    output_dir.mkdir(parents=True)
    (output_dir / "train.jsonl").write_text("{}\n", encoding="utf-8")
    (output_dir / "README.md").write_text("# sample\n", encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    pushed = push_mix(
        "abcdef123456",
        store=store_dir,
        repo_id="owner/sample",
        private=True,
        commit_message="sync sample",
    )

    assert pushed.manifest["hub"]["pushed"] is True
    assert pushed.manifest["hub"]["url"] == "https://huggingface.co/datasets/owner/sample"
    saved = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert saved["hub"]["pushed"] is True


def test_resolve_mix_artifact_rejects_ambiguous_ids(tmp_path) -> None:
    store_dir = tmp_path / "store"
    for hash_value in ("abcdef123456", "123456abcdef"):
        artifact_dir = store_dir / hash_value
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "artifact_id": "sample",
                    "artifact_version": "v1",
                    "mix_hash": hash_value,
                    "splits": {"train": {"rows": 1, "file": "train.jsonl"}},
                }
            ),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="Ambiguous mix id"):
        resolve_mix_artifact("sample", store_dir)


def test_resolve_mix_artifact_error_suggests_list(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="datamixxer list --store-dir"):
        resolve_mix_artifact("missing", tmp_path / "store")
