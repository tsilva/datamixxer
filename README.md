<p align="center">
  <img src="https://raw.githubusercontent.com/tsilva/datamixxer/main/logo.png" alt="datamixxer" width="420" />
  <br />
  <!-- repo-tagline:start -->
  <strong>🧪 Deterministic balanced dataset mixes 🧪</strong>
  <!-- repo-tagline:end -->
</p>

datamixxer builds balanced dataset mixes from Hugging Face sources. It
streams rows, samples deterministically from each source, and deduplicates
overlaps. Each build writes JSONL splits, a manifest, and a dataset card.

## Quickstart

From a checkout with Python 3.11 or newer and uv:

```bash
uv sync --config-file uv.toml --frozen
uv run datamixxer new my_mix.yaml --dataset HuggingFaceTB/smoltalk2 --dataset-config SFT --split train --count 1000
uv run datamixxer plan my_mix.yaml
uv run datamixxer build my_mix.yaml
```

The plan shows the requested mix. The build materializes its versioned output
locally. See the [configuration and workflow guide](https://github.com/tsilva/datamixxer/blob/main/docs/guide.md)
for split settings, inspection, validation, and publishing to the Hub.

For datasets that need style rewrites, use
[llmstyler](https://github.com/tsilva/llmstyler) with the tagged rows.

## License

[MIT License](https://github.com/tsilva/datamixxer/blob/main/LICENSE).
