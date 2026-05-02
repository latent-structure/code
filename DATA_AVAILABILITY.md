# Data Availability

This repository keeps the code, configs, and manuscript-facing summaries needed to reproduce the
paper. It does not rely on shipping every raw artifact through GitHub.

## Included In The Release Surface

- Source code under `scripts/`
- Manuscript text in `main.tex`
- Concept and manifest tables under `data/concepts/` and `data/manifests/`

## External Or Local Resources

The full reproduction path assumes locally available copies of:

- model checkpoints for Qwen, Mistral, and Llama-VL
- raw THINGS resources and behavioral resources
- image datasets and any release archives referenced by the manifests
- cached extracted embeddings and generations when running downstream-only reproduction

The scope-extension analyses for imSitu and MIT-States use release copies of those datasets from
`datasets/` when available locally. The intended layout is repo-relative:

```text
repo/
  data/
  datasets/
  .cache/hf
```

If the dataset archive lives elsewhere, symlink it into `repo/datasets/` so the paths resolve
without editing the code.

## Not Shipped In GitHub

- full raw generation dumps
- large embedding bundles
- model caches and Hugging Face downloads
- Slurm logs and scheduler artifacts

The reviewer-facing reproduction path is therefore: install the environment, ensure the local
dataset/model resources are present, then run `scripts/reproduce_neurips.py` in `smoke`,
`downstream`, or `full` mode.
