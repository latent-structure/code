# Prompting Is Not Grounding: The Geometry of Concept Space in Vision-Language Models

This repository contains the code and analyses for a study of how sensory prompting
and visual grounding reshape concept geometry in vision-language models, across three
model families (Qwen, Mistral, Llama) and 1,854 concrete object concepts from THINGS,  504 event labels from imSitu, and 2,207 attribute-object compositions from MIT-States.

## What the Paper Shows

- Matched visual grounding changes concept geometry substantially more than sensory
  prompting alone, and the change is reference-aligned rather than generic compression.
- Global geometry is image-dominant (~0.87–0.94 mixture weight across families), while
  local concept identity remains text-anchored even under conflicting visual input.
- The Llama family shows a clean dissociation: visual routing is present and
  geometrically meaningful, but external reference-space alignment is weaker than in
  Qwen and Mistral — showing that visual restructuring and benchmark alignment are
  separable properties.
- The geometry-to-behavior bridge is concentrated and strong
  in continuous analyses: description-similarity drift tracks explicit leakage at
  r = 0.43, and CLIP forced-choice shows a 40-point named-image choice gap between
  matched and mismatched grounding.
- The global-local profile extends to THINGSplus moderators, imSitu event concepts,
  and an abstract SimLex pilot.

## Setup

### Requirements

```bash
conda env create -f environment.yml
conda activate prompting-is-not-grounding
```

### Data and Model Checkpoints

Datasets and model checkpoints are not included in the repository. The code expects:

- A local Hugging Face cache at `HF_HOME` (default: `.cache/hf`)
- External datasets under `repo/datasets/` or a symlink pointing there
- See `DATA_AVAILABILITY.md` for the full release policy and download instructions

A clean local layout looks like this:

```
repo/
  data/         # repo-tracked concept lists, manifests, and controls
  datasets/     # local copies of external datasets
  .cache/hf     # Hugging Face model cache
```

## Reproduction

The main entrypoint is `scripts/reproduce_neurips.py`, which supports four modes:

```bash
# Fast integrity check (no checkpoints required)
python scripts/reproduce_neurips.py --mode smoke

# Rebuild manuscript outputs from existing artifacts
python scripts/reproduce_neurips.py --mode downstream

# Full pipeline (requires local checkpoints and datasets)
python scripts/reproduce_neurips.py --mode full

# Re-render paper figures from existing analysis outputs
python scripts/reproduce_neurips.py --mode figures
```

Start with `--mode smoke` to verify the environment. Use `--mode full` only when
all local resources are in place.

## Repository Layout

```
config/           model, prompt, seed, dataset, and analysis settings
data/concepts/    concept lists and analysis subsets
data/manifests/   image, mismatch, hierarchy, Lancaster, and archive manifests
outputs/metrics/  compact metrics used by the manuscript
outputs/tables/   compact tables
outputs/figures/  paper figures
scripts/          manuscript pipeline and reproduction entrypoint
tests/            integrity checks for the release surface
legacy/           archived exploratory scripts (not needed for reproduction)
```

### Pipeline Overview

| Script | Role |
|--------|------|
| `scripts/reproduce_neurips.py` | Reproduction entrypoint |
| `scripts/00_run_results_v5_pipeline.py` | Core extraction and RSA pipeline |
| `scripts/35_compute_behavior_bridge_extensions.py` – `scripts/68_compute_llama_attention_geometry_coupling.py` | Manuscript analyses beyond the core pipeline |
| `scripts/34_make_paper_figures.py` | Final figure rendering |

Exploratory scripts are archived under `legacy/scripts/` and are not part of
the reproduction path.

## Data

| Dataset | Role |
|---------|------|
| THINGS | Primary concept set and behavioral similarity reference |
| THINGSplus | Object-level moderator norms |
| Lancaster Sensorimotor Norms | Sensorimotor reference space |
| SimLex-999 | Abstract concept pilot |
| imSitu | Event concept generalization |

See `DATA_AVAILABILITY.md` for download instructions and licensing.
