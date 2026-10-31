from __future__ import annotations

import json

import pytest

from datamixxer.cli import print_show
from datamixxer.io import write_yaml
from datamixxer.mix import (
    materialize_rows,
    mix_hash,
    normalize_sources,
    stream_source_rows,
    write_mix_artifacts,
)


def config(tmp_path):
    return {
        "id": "contract",
        "sources": [{"name": "sample", "dataset_id": "org/data", "split": "train", "count": 2}],
        "output": {"store_dir": str(tmp_path / "store with spaces")},
    }


def test_json_show_distinguishes_planned_and_built_artifacts(tmp_path, capsys):
    value = config(tmp_path)
    path = tmp_path / "mix.yaml"
    write_yaml(path, value)
    print_show(str(path), None, as_json=True)
    planned = json.loads(capsys.readouterr().out)
    assert planned["schema_version"] == 1
    assert planned["built"] is False
    artifact = write_mix_artifacts(value, {"train": [{"messages": []}, {"messages": []}]})
    print_show(str(path), None, as_json=True)
    built = json.loads(capsys.readouterr().out)
    assert built["built"] is True
    assert built["path"] == str(artifact.path.resolve()) == planned["path"]
    assert built["manifest"]["splits"]["train"]["rows"] == 2
    (artifact.path / "train.jsonl").unlink()
    print_show(str(path), None, as_json=True)
    assert json.loads(capsys.readouterr().out)["built"] is False


def test_revision_changes_hash_and_reaches_dataset_loader(tmp_path, monkeypatch):
    value = config(tmp_path)
    original_hash = mix_hash(value)
    value["sources"][0]["revision"] = "a" * 40
    assert mix_hash(value) != original_hash
    source = normalize_sources(value)[0]
    called = {}

    def fake_load(dataset_id, dataset_config, **kwargs):
        called.update(kwargs)

        class Dataset:
            def shuffle(self, **kwargs):
                return [{"messages": []}]

        return Dataset()

    monkeypatch.setattr("datasets.load_dataset", fake_load)
    assert list(stream_source_rows(source, 1, 10)) == [{"messages": []}]
    assert called["revision"] == "a" * 40
    assert materialize_rows([{"messages": []}], source, "train")[0]["source_revision"] == "a" * 40
    artifact = write_mix_artifacts(value, {"train": [{"messages": []}]})
    assert artifact.manifest["sources"][0]["revision"] == "a" * 40


@pytest.mark.parametrize("revision", ["", 42, [], {}])
def test_invalid_revisions_are_rejected(tmp_path, revision):
    value = config(tmp_path)
    value["sources"][0]["revision"] = revision
    with pytest.raises(ValueError, match="revision"):
        normalize_sources(value)
