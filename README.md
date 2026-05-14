<div align="center">
  <img src="./logo.png" alt="datamixxer" width="420" />

  **🧪 Deterministic balanced dataset mixes 🧪**
</div>

`datamixxer` is a Python CLI for building balanced subsamples from Hugging Face
datasets. It reads a YAML plan, streams each source split, shuffles
deterministically, deduplicates rows, and writes a versioned dataset mix.

Each build produces JSONL split files, `manifest.json`, and a dataset card. When
a test split is configured, each source bucket is split independently so train
and test keep the same blend.

## Install

```bash
uv sync
```

Start a config, validate it, preview the plan, then build:

```bash
uv run datamixxer init my_mix.yaml
uv run datamixxer validate configs/smoltalk_style_mix.yaml
uv run datamixxer plan configs/smoltalk_style_mix.yaml --explain-hash
uv run datamixxer build configs/smoltalk_style_mix.yaml --no-push-to-hub
```

## Config

```yaml
id: my_balanced_mix
name: My Balanced Mix
version: v1
seed: 3407
buffer_size: 10000
dedupe: true

split:
  test_size: 0.1

sources:
  - name: math
    dataset_id: org/math-dataset
    config: default
    split: train
    count: 1000
  - name: code
    dataset_id: org/code-dataset
    split: train
    count: 1000
    metadata:
      domain: code

output:
  store_dir: .datamixxer/mixes
  train_file: train.jsonl
  test_file: test.jsonl
  push_to_hub: false
  hub:
    repo_id: owner/my_balanced_mix-v1
```

Required fields:

- `id`: stable artifact id used in manifests and default repo naming.
- `sources` or `plan`: non-empty list of source buckets.
- Source `dataset_id`: Hugging Face dataset id, unless inherited from `source`.
- Source `split`: Hugging Face split name to stream.
- Source `count`, or explicit `train_count`/`test_count`.

Common optional fields:

- `name`: human-readable dataset mix name.
- `version` or `artifact_version`: artifact version. Numeric values are normalized with `v`.
- `seed`: base shuffle seed. Defaults to `3407`.
- `buffer_size`: streaming shuffle buffer. Defaults to `10000`.
- `split.test_size`: fraction, percentage, or row count for each bucket's test split.
- `dedupe`: `true`, `false`, a field path such as `messages`, or a mapping with `field`/`fields`.
- Source `config` or `subset`: dataset config/subset name.
- Source `metadata`: mapping copied into every output row for that source.
- Source `restyle`: boolean copied into output rows for `llmstyler` compatibility.
- `output.store_dir`: local artifact store. Defaults to `.datamixxer/mixes`.
- `output.train_file` and `output.test_file`: output JSONL filenames.
- `output.push_to_hub`: whether `build` should upload by default.
- `output.hub.repo_id` or `output.hub.owner`: Hub target for publishing.
- `output.hub.private`: create or update the target Hub repo as private.
- `output.hub.commit_message`: Hub upload commit message.

The older shared-source shape used by `llmstyler` is also supported:

```yaml
source:
  dataset_id: HuggingFaceTB/smoltalk2
  config: SFT
plan:
  - name: capability_magpie
    split: smoltalk_smollm3_smol_magpie_ultra_no_think
    count: 300
    restyle: false
```

## Commands

```bash
uv run datamixxer init my_mix.yaml                                      # write starter config
uv run datamixxer validate configs/smoltalk_style_mix.yaml              # validate config
uv run datamixxer plan configs/smoltalk_style_mix.yaml --explain-hash   # preview rows and hash inputs
uv run datamixxer build configs/smoltalk_style_mix.yaml --no-push-to-hub  # build locally
uv run datamixxer build configs/smoltalk_style_mix.yaml --push-to-hub     # build and upload
uv run datamixxer list                                                    # list local mixes
uv run datamixxer show <mix-hash-or-artifact-id>                          # inspect a mix
uv run datamixxer hub-check configs/smoltalk_style_mix.yaml               # check Hub auth/repo access
uv run datamixxer push <mix-hash-or-artifact-id> --repo-id owner/name      # upload later
uv sync --extra dev                                                       # install test/lint tools
uv run pytest                                                             # run tests
uv run ruff check .                                                       # run lint checks
```

## Notes

- `mix_hash` is computed from sampling inputs: sources, counts, train/test
  balance, seed, buffer size, dedupe settings, and output-affecting source
  metadata.
- Output paths and Hub settings do not affect `mix_hash`. If a complete mix with
  the same hash already exists, `build` reuses it unless `--force` is passed.
  Use `plan --explain-hash` to inspect the normalized hash inputs.
- Local mixes are stored under `.datamixxer/mixes/<mix_hash>` by default.
  `output.store_dir` controls where `build` writes artifacts and where `list`,
  `show`, and `push` look for them.
- Each output row keeps the source row fields and adds `bucket`,
  `source_dataset`, `source_config`, `source_split`, `output_split`, and any
  per-source `metadata`.
- Hub uploads require `HF_TOKEN` or `huggingface-cli login` before using
  `--push-to-hub` or `datamixxer push`. Use `hub-check` before a long build to
  verify authentication and target repo access.

## Architecture

![datamixxer architecture diagram](./architecture.png)

## License

No license file is present in this repository yet.
