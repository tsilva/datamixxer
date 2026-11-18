"""Explicit finite tokens must reach every Hub/dataset entrypoint."""

import sys
from types import SimpleNamespace

import pytest

from datamixxer import hub, mix


@pytest.mark.parametrize("token", [None, "finite_public_placeholder"])
def test_hub_client_receives_only_explicit_environment_token(monkeypatch, token):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    if token:
        monkeypatch.setenv("HF_TOKEN", token)
    calls = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=lambda **options: calls.append(options) or object()
    ))
    hub._api()
    assert calls == ([{"token": token}] if token else [{}])


def test_dataset_stream_and_metadata_receive_explicit_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "finite_public_placeholder")
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    calls = []

    def record(kind, result):
        def call(*arguments, **options):
            calls.append((kind, arguments, options))
            return result
        return call

    dataset = SimpleNamespace(shuffle=lambda **options: iter([{"text": "fixture"}]))
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(
        disable_progress_bar=lambda: None,
        load_dataset=record("stream", dataset),
        get_dataset_config_names=record("configs", ["fixture"]),
        get_dataset_split_names=record("splits", ["train"]),
    ))
    assert list(mix.stream_rows("owner/fixture", "fixture", "train", 1, 10,
                                revision="fixed-revision")) == [{"text": "fixture"}]
    assert mix.inspect_dataset("owner/fixture").splits_by_config == {"fixture": ["train"]}
    config = {"sources": [{"name": "fixture", "dataset_id": "owner/fixture",
                           "config": "fixture", "split": "train", "count": 1}]}
    assert mix.check_source_access(config) == []
    assert [kind for kind, _, _ in calls] == ["stream", "configs", "splits", "splits"]
    assert all(options["token"] == "finite_public_placeholder" for _, _, options in calls)
    assert calls[0][2]["revision"] == "fixed-revision"


@pytest.mark.parametrize("token", ["", " ", "fixture\ninvalid", "x" * 16385])
def test_invalid_explicit_token_stops_before_sdk_or_cached_login(monkeypatch, token):
    monkeypatch.setenv("HF_TOKEN", token)
    calls = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=lambda **options: calls.append(options)
    ))
    with pytest.raises(RuntimeError, match="explicitly supplied Hugging Face token is invalid"):
        hub._api()
    assert calls == []


def test_rejected_explicit_token_does_not_retry_with_cached_auth(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "rejected_public_placeholder")
    calls = []

    def load(*arguments, **options):
        calls.append(options)
        raise RuntimeError("Synthetic authentication rejection")

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(
        disable_progress_bar=lambda: None, load_dataset=load,
    ))
    with pytest.raises(RuntimeError, match="Synthetic authentication rejection"):
        list(mix.stream_rows("owner/fixture", None, "train", 1, 10))
    assert len(calls) == 1
    assert calls[0]["token"] == "rejected_public_placeholder"
