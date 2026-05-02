# RESULTS: Corrected Results Ledger for Manuscript Transfer

Main findings:
1. Matched grounding > sensory prompting in Qwen/Mistral RSA; Llama weaker.
2. Matched grounding compresses PR across Qwen/Mistral/Llama.
3. Prompt+image geometry is image-dominant.
4. Mismatched images perturb geometry but preserve local text identity.
5. Grounded language states align with the VLM?s own visual tower.

## 0. Current Status and Extraction Audit

The old concept-span extraction bug was severe enough that old results should be treated as invalid
or exploratory. Exact token-subsequence matching failed for almost all concepts in the primary
conditions.

| Family | Text prompting failure | Matched-image VLM failure | Old matched-image median sequence length | Median concept-token length | Old dilution ratio |
|---|---:|---:|---:|---:|---:|
| Qwen | 1853 / 1854 (99.95%) | 1853 / 1854 (99.95%) | 213 | 2 | 106x |
| Mistral | 1852 / 1854 (99.89%) | 1852 / 1854 (99.89%) | 457 | 2 | 228x |
| Llama | 1853 / 1854 (99.95%) | 1853 / 1854 (99.95%) | 19 | 2 | 9x |

For `T_prompt_primary`, the old fallback pooled about 8.5x to 9x the intended concept span across
families. The corrected bundle is now the only valid basis for the paper.

| Corrected artifact | Status |
|---|---|
| Merged embedding bundle | `outputs/embeddings/pooled_embeddings_full.npz` |
| Metadata summary | `outputs/embeddings/embedding_metadata_full_summary.json` |
| Mode | `merged_real_model_extraction` |
| Precision | `bf16` |
| Source tags | `qwen`, `mistral`, `llama` |
| Embedding records | 1208 |
| Concept subset | `data/concepts/things_max_1854_concepts.csv` |
| Concept count | 1854 |
| Span pooling verified | `True` |

Refresh status at the time of the latest update: corrected RDM construction, neighbor
restructuring, Procrustes, partial RSA, variance partitioning, layerwise trajectories,
full-archive preparation, and 1000-permutation mixture validation have completed. No pending
analysis job is required to interpret the results below.

## RQ1. Do Prompting and Grounding Induce the Same Conceptual Geometry?

### Primary Qwen RSA: matched grounding beats sensory prompting across anchors

These are the corrected Qwen primary RSA values over the 1854-concept THINGS-max branch.

| Reference space | Prompt only | Matched image | Prompt + image | Matched - prompt | Prompt+image - matched |
|---|---:|---:|---:|---:|---:|
| THINGS behavioral similarity | 0.1913 | 0.2723 | 0.2777 | +0.0810 | +0.0055 |
| Controlled THINGS | 0.1843 | 0.2627 | 0.2675 | +0.0784 | +0.0047 |
| SigLIP2 | 0.1423 | 0.1934 | 0.1875 | +0.0511 | -0.0058 |
| Lancaster perceptual | 0.1151 | 0.2252 | 0.2498 | +0.1100 | +0.0247 |

Interpretation: in the corrected primary Qwen results, matched-image grounding is more aligned than
sensory prompting with human behavioral, controlled human, learned visual, and Lancaster perceptual
reference spaces. The earlier narrative that sensory prompting better preserves human-like structure
is not supported on the corrected primary branch.

### Cross-family RSA: grounding advantage is strong in Qwen and Mistral, weak in Llama

Matched-image advantage over `T_prompt_primary`:

| Family | THINGS | Controlled THINGS | SigLIP2 | CLIP ViT-L/14 | DINOv2 | Lancaster perceptual |
|---|---:|---:|---:|---:|---:|---:|
| Qwen | +0.0810 | +0.0784 | +0.0511 | +0.0822 | +0.0403 | +0.1100 |
| Mistral | +0.1138 | +0.1084 | +0.0821 | +0.0765 | +0.0563 | +0.0621 |
| Llama | +0.0033 | +0.0041 | +0.0108 | -0.0343 | +0.0063 | -0.0011 |

Interpretation: the corrected cross-family result supports grounding > prompting in Qwen and
Mistral, but the Llama effect is weak and partly anchor-dependent. The safe manuscript wording is
that the effect is directionally present but architecture-dependent, not uniformly large.

### Prompt + image geometry is image-dominant

The prompt+image geometry was modeled as a rank-RDM regression mixture of prompt-only and
matched-image RDMs.

| Quantity | Value |
|---|---:|
| Prompt weight | 0.0128 |
| Matched-image weight | 0.9444 |
| Mixture R2 | 0.9026 |
| Residual norm | 0.3122 |
| Permutations | 1000 |
| Permutation p-value | 0.0010 |
| Null mean R2 | 0.000001 |
| Null 95th-percentile R2 | 0.000003 |
| Label | `image_dominant` |

Interpretation: once matched visual evidence is present, the global concept geometry is explained
almost entirely by matched-image grounding, not by sensory-prompt geometry. The 1000-permutation
null confirms that this mixture fit is far above chance-level shuffled target geometry.

### Mismatched images perturb geometry but almost never hijack local text identity

The mismatched-image analysis asks whether a text target with an unrelated image source moves closer
to the target concept's matched representation or to the mismatched image-source concept's matched
representation.

| Quantity | Value |
|---|---:|
| Mismatched pairs | 1854 |
| Margin | 0.01 cosine distance |
| Text-retention rate | 0.9984 |
| Image-hijack rate | 0.0005 |
| Ambiguous rate | 0.0011 |
| Approx. text-retained count | 1851 / 1854 |
| Approx. image-hijack count | 1 / 1854 |
| Approx. ambiguous count | 2 / 1854 |

Margin robustness:

| Margin | Text retention | Image hijack | Ambiguous |
|---:|---:|---:|---:|
| 0.000 | 0.9995 | 0.0005 | 0.0000 |
| 0.005 | 0.9984 | 0.0005 | 0.0011 |
| 0.010 | 0.9984 | 0.0005 | 0.0011 |
| 0.020 | 0.9984 | 0.0000 | 0.0016 |

Rank-based validation:

| Quantity | Value |
|---|---:|
| Median target rank among matched anchors | 1 |
| Mean target rank | 1.02 |
| Median mismatched image-source rank | 331 |
| Mean mismatched image-source rank | 519.77 |

Interpretation: mismatched images deform the representation but almost never overwrite the text
concept identity. This supports the mechanism "image-shaped global manifold, text-anchored local
identity."

### Procrustes and nearest-neighbor restructuring: perturbations deform grounded geometry

These refreshed summaries use the corrected full embedding metadata and report mid-to-late layer
means. Neighbor overlap uses k=10 nearest-neighbor Jaccard; lower Jaccard means stronger local
neighborhood replacement.

| Family | Comparison | Mean Procrustes disparity | k=10 mean neighbor Jaccard | k=10 mean rank shift |
|---|---|---:|---:|---:|
| Qwen text | neutral vs prompt | 0.2382 | 0.3149 | 2.3360 |
| Qwen-VL | matched vs degraded image | 0.0288 | 0.4997 | 1.9733 |
| Qwen-VL | matched vs mismatched image | 0.2090 | 0.1799 | 3.0742 |
| Qwen-VL | matched vs blank image | 0.2093 | 0.2139 | 3.1980 |
| Mistral text | neutral vs prompt | 0.2345 | 0.3769 | 2.0420 |
| Mistral-VL | matched vs degraded image | 0.0539 | 0.5924 | 1.6032 |
| Mistral-VL | matched vs mismatched image | 0.2904 | 0.2285 | 2.6944 |
| Mistral-VL | matched vs blank image | 0.2928 | 0.2731 | 2.5391 |
| Llama text | neutral vs prompt | 0.1676 | 0.4506 | 1.8695 |
| Llama-VL | matched vs degraded image | 0.0553 | 0.5159 | 1.8172 |
| Llama-VL | matched vs mismatched image | 0.3014 | 0.1816 | 3.0343 |
| Llama-VL | matched vs blank image | 0.3302 | 0.2108 | 2.8840 |

Interpretation: degraded images preserve much more of the matched-image geometry than mismatched or
blank controls across all three VLM families. Mismatched and blank images produce larger global
shape changes and stronger local-neighborhood replacement. Prompting also substantially changes text
geometry, so the safe claim is not that image perturbations always exceed prompting, but that visual
perturbations strongly restructure the grounded VLM manifold when image-text correspondence is
removed.

## RQ2. Does Visual Input Function as Enrichment or Constraint?

### Cross-family intrinsic dimensionality: matched grounding compresses the global concept space

Mid-to-late mean participation ratio over the 1854 sensory concepts:

| Family | Text-only VLM | Matched image | Prompt + image | Degraded image | Mismatched image | Blank image | Matched reduction vs text-only |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen-VL | 147.05 | 113.28 | 107.17 | 118.29 | 151.46 | 143.64 | 22.96% lower |
| Mistral-VL | 193.35 | 165.04 | 134.68 | 176.14 | 216.31 | 197.36 | 14.64% lower |
| Llama-VL | 193.14 | 137.43 | 150.67 | 132.94 | 191.04 | 175.85 | 28.84% lower |

Core pattern across all three families:

| Contrast | Supported? |
|---|---|
| Matched image < text-only VLM | Yes |
| Matched image < mismatched image | Yes |
| Matched image < blank image | Yes |

Interpretation: semantically matched visual grounding compresses the mid-to-late concept manifold.
Because mismatched and blank images do not compress similarly, the effect is not just "image
present"; it depends on image-text correspondence. Degraded images are a useful caveat: in Llama,
degraded-image PR is even lower than matched-image PR, so PR alone is not sufficient to define
successful grounding.

### PR-RSA relationship: reference-aligned compression, not generic compression

Across eight Qwen conditions, participation ratio is negatively correlated with RSA:

| Reference space | Spearman rho between PR and RSA |
|---|---:|
| THINGS behavioral similarity | -0.571 |
| Controlled THINGS | -0.571 |
| SigLIP2 | -0.786 |
| Lancaster perceptual | -0.333 |

Selected condition examples:

| Condition | THINGS | Controlled THINGS | SigLIP2 | Lancaster perceptual | Mean PR |
|---|---:|---:|---:|---:|---:|
| `M_prompt_plus_matched_image` | 0.2777 | 0.2675 | 0.1875 | 0.2498 | 107.17 |
| `M_matched_image` | 0.2723 | 0.2627 | 0.1934 | 0.2252 | 113.28 |
| `T_prompt_primary` | 0.1913 | 0.1843 | 0.1423 | 0.1151 | 113.01 |
| `M_text_only` | 0.1611 | 0.1586 | 0.1220 | 0.1158 | 147.05 |
| `M_mismatched_image` | 0.1964 | 0.1916 | 0.1165 | 0.1624 | 151.46 |
| `M_blank_image` | 0.1708 | 0.1678 | 0.1216 | 0.1283 | 143.64 |

Interpretation: lower PR tends to accompany better reference alignment, especially for SigLIP2, but
compression alone is not enough. `T_prompt_primary` has PR similar to `M_matched_image` but much
weaker RSA. The mechanism is therefore reference-aligned compression rather than generic
low-dimensionality.

## RQ3. What Geometric Mechanism Explains the Divergence?

### Layerwise global-local dissociation

| Quantity | Value |
|---|---:|
| Global-local dissociation present | `True` |
| Concepts | 1854 |
| Mixture layers | 33 |
| Identity-retention layers | 37 |
| Mean matched-image mixture weight | 0.9391 |
| Mean prompt mixture weight | 0.0381 |
| Last-half matched-image mixture weight | 0.9191 |
| Last-half prompt mixture weight | 0.0586 |
| Mean text-retention rate | 0.9983 |
| Mean image-hijack rate | 0.0007 |
| Last-half text-retention rate | 0.9970 |
| Last-half image-hijack rate | 0.0011 |

Interpretation: visual input dominates global geometry across layers, while local text identity is
preserved. Image shapes the manifold, text anchors
concept identity.

### Layerwise reference-alignment trajectories

The refreshed layerwise trajectory analysis tracks where matched-image RSA first exceeds
prompt-only RSA and where the matched-minus-prompt gap peaks.

| Reference space | First matched > prompt layer | Peak matched > prompt layer | Mean matched - prompt gap | Last-half mean gap |
|---|---:|---:|---:|---:|
| THINGS behavioral similarity | 10 | 25 | +0.0482 | +0.0722 |
| Controlled THINGS | 10 | 25 | +0.0479 | +0.0723 |
| SigLIP2 | 3 | 19 | +0.0270 | +0.0498 |
| Lancaster perceptual | 2 | 25 | +0.0506 | +0.0811 |

Last-half condition means:

| Reference space | Prompt only | Matched image | Text-only VLM | Mismatched image | Blank image |
|---|---:|---:|---:|---:|---:|
| THINGS behavioral similarity | 0.1695 | 0.2248 | 0.1537 | 0.1720 | 0.1608 |
| Controlled THINGS | 0.1617 | 0.2162 | 0.1493 | 0.1657 | 0.1562 |
| SigLIP2 | 0.1246 | 0.1615 | 0.1151 | 0.1129 | 0.1172 |
| Lancaster perceptual | 0.1024 | 0.1812 | 0.1130 | 0.1348 | 0.1181 |

Interpretation: matched-image grounding overtakes sensory prompting in the measured layer stack and
has a stronger late-layer advantage across human, controlled-human, visual, and perceptual anchors.
The crossover occurs earliest for Lancaster and SigLIP2, and later for THINGS-style anchors.

### Human residual and visual residual analyses

Human residual RSA after visual/category/lexical controls:

| Condition | Human residual RSA | Raw THINGS RSA |
|---|---:|---:|
| `T_prompt_primary` | 0.1149 | 0.1913 |
| `M_text_only` | 0.1018 | 0.1611 |
| `M_blank_image` | 0.1100 | 0.1708 |
| `M_mismatched_image` | 0.1390 | 0.1964 |
| `M_matched_image` | 0.1739 | 0.2723 |
| `M_prompt_plus_matched_image` | 0.1815 | 0.2777 |

Key contrast:

| Contrast | Value |
|---|---:|
| Human residual prompt - matched | -0.0590 |
| Human residual matched - prompt | +0.0590 |

Visual residual matched-minus-prompt:

| Residual anchor | Matched - prompt |
|---|---:|
| SigLIP2 residual | +0.0157 |
| CLIP ViT-L/14 residual | +0.0584 |
| DINOv2 residual | +0.0222 |

Interpretation: the hoped-for result that prompting wins on uniquely human residual structure did
not appear. Matched grounding still beats prompting after residualizing visual/category/lexical
structure, and it also captures visual residual structure better than prompting.

### Human partial RSA

| Quantity | Prompt | Matched image | Degraded image | Prompt - matched |
|---|---:|---:|---:|---:|
| Raw THINGS partial-RSA package | 0.1913 | 0.2723 | 0.2614 | -0.0810 |
| Joint-controlled | 0.1852 | 0.2633 | 0.2532 | -0.0780 |

Interpretation: after coarse proxy controls, matched grounding remains more aligned with the
residual human behavioral structure than prompt-only. The control adjustment is small relative to
the matched-minus-prompt gap: the largest prompt reduction from controls is 0.0061, while the
joint-controlled matched advantage is 0.0780.

### Variance partitioning: prompting and grounding explain different components

| Condition | Total fit | Unique human family | Unique anchor family | Unique proxy family |
|---|---:|---:|---:|---:|
| `T_prompt_primary` | 0.0513 | 0.0180 | 0.0036 | 0.0111 |
| `M_matched_image` | 0.0883 | 0.0370 | 0.0063 | 0.0078 |
| `M_mismatched_image` | 0.0497 | 0.0236 | 0.0009 | 0.0102 |

Key contrasts:

| Quantity | Value |
|---|---:|
| Highest unique human-family condition | `M_matched_image` |
| Highest unique anchor-family condition | `M_matched_image` |
| Matched - mismatched unique anchor variance | +0.0054 |

Interpretation: the refreshed variance partitioning strengthens the corrected narrative. Matched
grounding has the highest total fit, highest unique human-family component, and highest unique
anchor-family component. The earlier nuance that prompting carried the largest unique human-family
variance is not supported by the refreshed analysis.

## RQ4. Is Grounding Aligned With Generic Visual Structure or Specific Reference Spaces?

### Internal visual-tower alignment: grounded language states align with the VLM's own visual tower

Anchor: Qwen internal visual tower, `pooler_output`, 1854 concepts.

| Condition | RSA to Qwen internal visual tower |
|---|---:|
| `M_matched_image` | 0.2078 |
| `M_degraded_image` | 0.2066 |
| `M_prompt_plus_matched_image` | 0.2057 |
| `T_neutral` | 0.1575 |
| `T_prompt_primary` | 0.1440 |
| `M_blank_image` | 0.1360 |
| `M_text_only` | 0.1307 |
| `M_mismatched_image` | 0.1241 |

Key contrasts:

| Contrast | Value |
|---|---:|
| Matched - prompt | +0.0638 |
| Matched - text-only VLM | +0.0771 |
| Prompt+image - matched | -0.0021 |
| Degraded - matched | -0.0012 |
| Mismatched - matched | -0.0837 |
| Blank - matched | -0.0718 |

Interpretation: matched grounding pulls language-side concept geometry toward the model's own
internal visual representation space. Prompt+image and degraded-image states are nearly as aligned
as matched images, whereas blank and mismatched images are much lower. This is stronger mechanistic
evidence than external visual anchors alone.

### Lancaster sensorimotor validation

Lancaster overlap: 1519 resolved concepts.

| Lancaster reference | Prompt RSA | Matched RSA | Mismatched RSA | Blank RSA | Matched - prompt | Matched - mismatched | Matched - blank |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full sensorimotor | 0.1168 | 0.2512 | 0.1567 | 0.1303 | +0.1345 | +0.0945 | +0.1209 |
| Perceptual | 0.1151 | 0.2252 | 0.1624 | 0.1283 | +0.1101 | +0.0628 | +0.0969 |
| Haptic/material | 0.0588 | 0.0852 | 0.0456 | 0.0493 | +0.0264 | +0.0396 | +0.0359 |

Bootstrap CIs for matched-minus-prompt:

| Lancaster reference | Mean gap | CI low | CI high |
|---|---:|---:|---:|
| Full sensorimotor | +0.1345 | +0.1323 | +0.1365 |
| Perceptual | +0.1101 | +0.1079 | +0.1121 |
| Haptic/material | +0.0264 | +0.0245 | +0.0285 |

Bootstrap CIs for matched-image controls:

| Lancaster reference | Contrast | Mean gap | CI low | CI high |
|---|---|---:|---:|---:|
| Full sensorimotor | Matched - mismatched | +0.0945 | +0.0927 | +0.0963 |
| Full sensorimotor | Matched - blank | +0.1209 | +0.1190 | +0.1226 |
| Perceptual | Matched - mismatched | +0.0628 | +0.0610 | +0.0646 |
| Perceptual | Matched - blank | +0.0969 | +0.0950 | +0.0986 |
| Haptic/material | Matched - mismatched | +0.0396 | +0.0379 | +0.0414 |
| Haptic/material | Matched - blank | +0.0359 | +0.0341 | +0.0377 |

Interpretation: matched-image grounding improves alignment over sensory prompting across Lancaster
sensorimotor subspaces. The effect is strongest for full sensorimotor and perceptual norms, smaller
but positive for haptic/material norms. Matched images also beat mismatched and blank controls,
supporting semantic image-text correspondence rather than generic image presence.

### Multi-image and full-archive robustness

The canonical primary branch uses a single matched image per concept. These branches test whether
the result is dependent on one idiosyncratic JPEG. They are smaller and should be presented as
robustness/exploratory analyses.

Multi-image diagnostic:

| Anchor | Multi-image prototype | Single-image grounding | Prompt only |
|---|---:|---:|---:|
| THINGS behavioral similarity | 0.2996 | 0.2971 | 0.2276 |
| Controlled THINGS | 0.2881 | 0.2871 | 0.2247 |
| SigLIP2 | 0.3518 | 0.3195 | 0.0499 |

Multi-image representation consistency:

| Quantity | Mean | Min | Max |
|---|---:|---:|---:|
| Within-concept image similarity | 0.9937 | 0.9823 | 0.9990 |
| Nearest between-concept similarity | 0.8575 | 0.8252 | 0.8734 |
| Image-to-prompt similarity | -0.0028 | -0.0168 | +0.0193 |

Concept-level stability summary:

| Quantity | Value |
|---|---:|
| Concepts | 15 |
| Images per concept | 5 |
| Stable concepts | 15 / 15 |
| Mean prototype-to-single similarity | 0.9978 |

Interpretation: on the 15-concept multi-image diagnostic, concept-level image prototypes preserve
or slightly improve over single-image grounding and remain above prompt-only for these anchors. The
high within-concept similarity and high prototype-to-single similarity support the use of one matched
image per concept in the primary branch, but this diagnostic remains small.

Full THINGS archive branch:

| Quantity | Value |
|---|---:|
| Full archive prep available concepts | 1653 |
| Full archive prep excluded concepts | 201 |
| Full archive prep total images | 23341 |
| Full archive prep mean images per concept | 14.12 |
| Full archive prep min / max images per concept | 3 / 35 |
| Prototype diagnostic concepts | 35 |
| Prototype diagnostic total images | 480 |
| Prototype diagnostic mean images per concept | 13.71 |

| Anchor | Single image | Prototype size 3 | Prototype size 5 | Prototype size 10 | All-image prototype | Prompt only |
|---|---:|---:|---:|---:|---:|---:|
| THINGS behavioral similarity | 0.3850 | 0.5382 | 0.5521 | 0.5346 | 0.5415 | 0.5141 |
| Controlled THINGS | 0.2561 | 0.3358 | 0.3616 | 0.3383 | 0.3389 | 0.1757 |
| SigLIP2 | 0.4721 | 0.3749 | 0.3966 | 0.3999 | 0.3897 | 0.3741 |

All-image prototype comparisons:

| Anchor | Comparison | Gap | CI low | CI high |
|---|---|---:|---:|---:|
| THINGS | prototype - prompt | +0.0340 | -0.1162 | +0.1760 |
| THINGS | prototype - single | +0.1371 | -0.0011 | +0.2776 |
| Controlled THINGS | prototype - prompt | +0.1502 | -0.0848 | +0.3653 |
| Controlled THINGS | prototype - single | +0.0636 | -0.1406 | +0.2325 |
| SigLIP2 | prototype - single | -0.0777 | -0.2832 | +0.1304 |

Full-archive image variance and prototype closeness:

| Quantity | Value |
|---|---:|
| Mean within-concept image similarity | 0.9908 |
| Mean nearest between-concept similarity | 0.9875 |
| Mean image-to-prototype similarity | 0.9957 |
| Mean all-image prototype-to-single similarity | 0.9934 |
| Mean all-image prototype-to-prompt similarity | 0.0086 |
| Mixed-stability concepts | 29 / 35 |
| Image-fragile concepts | 6 / 35 |

Interpretation: archive preparation now covers a much broader 1653-concept pool, but the completed
prototype diagnostic remains the 35-concept branch reported here. Prototype averaging helps over
single-image grounding for THINGS-style anchors in this small diagnostic, but the CIs are wide and
the SigLIP2 pattern differs. Prototype representations remain extremely close to the single-image
representation, while archive image sets are not always cleanly separated from nearest
between-concept images. This should not carry primary inferential weight.

## Secondary Findings and Caveats

### Hierarchy-depth analysis: no coarse/fine crossover

| Level | Leader | Leader RSA | Matched-image RSA | Prompt-only RSA |
|---|---|---:|---:|---:|
| Coarse category | `M_prompt_plus_matched_image` | 0.0927 | 0.0837 | 0.0837 |
| Subtype | `M_prompt_plus_matched_image` | 0.0660 | 0.0586 | 0.0456 |

Interpretation: there is no clean "prompting wins coarse, grounding wins fine" crossover. Instead,
prompt+image leads both hierarchy levels, matched image is close to prompt at the coarse level, and
prompt-only is weaker at subtype level.

### Linear probe category decoding: grounded states encode labels more linearly

Linear probes were fit to predict metadata labels from condition-level concept representations. This
is a secondary diagnostic of label separability, not a replacement for RSA.

Coarse category decoding:

| Quantity | Value |
|---|---:|
| Concepts | 1854 |
| Classes | 30 |
| Coverage | 1.0000 |
| Chance balanced accuracy | 0.0333 |

| Condition | Balanced accuracy | Macro-F1 |
|---|---:|---:|
| `M_degraded_image` | 0.5788 | 0.6088 |
| `M_matched_image` | 0.5756 | 0.6013 |
| `M_prompt_plus_matched_image` | 0.5697 | 0.5961 |
| `M_mismatched_image` | 0.4810 | 0.5085 |
| `T_neutral` | 0.4609 | 0.4788 |
| `M_blank_image` | 0.4583 | 0.4789 |
| `T_prompt_primary` | 0.4398 | 0.4605 |
| `M_text_only` | 0.4335 | 0.4503 |

Subtype decoding:

| Quantity | Value |
|---|---:|
| Concepts | 1818 |
| Classes | 49 |
| Coverage | 0.9806 |
| Chance balanced accuracy | 0.0204 |

| Condition | Balanced accuracy | Macro-F1 |
|---|---:|---:|
| `M_matched_image` | 0.4802 | 0.4959 |
| `M_degraded_image` | 0.4786 | 0.4960 |
| `M_prompt_plus_matched_image` | 0.4692 | 0.4832 |
| `M_mismatched_image` | 0.4391 | 0.4665 |
| `M_blank_image` | 0.3815 | 0.4026 |
| `T_prompt_primary` | 0.3799 | 0.4015 |
| `M_text_only` | 0.3707 | 0.3902 |
| `T_neutral` | 0.3702 | 0.3943 |

Interpretation: visually grounded states make coarse category and subtype labels more linearly
decodable than prompt-only or text-only conditions. The degraded-image result is close to matched
image, so this should be framed as category/subtype separability under visual input rather than a
specific proof of semantically correct grounding.

### Human local geometry / small human-anchor audit: exploratory caveat

The small human-local branch is based on a much smaller behavioral-overlap subset and should be
labeled exploratory.

| Quantity | Value |
|---|---:|
| All-concept prompt local alignment | 0.4542 |
| All-concept matched-image local alignment | 0.4218 |
| All-concept degraded-image local alignment | 0.4095 |
| Prompt - matched local gap | +0.0324 |
| Matched - degraded local gap | +0.0123 |

Subtype leaders:

| Subtype | Best condition |
|---|---|
| Appearance/color | `M_matched_image` |
| Smell/taste proxy | `M_matched_image` |
| Sound-linked | `T_prompt_primary` |
| Texture/material | `M_degraded_image` |

Interpretation: local human-neighbor structure gives a limited prompt-favoring caveat, especially
for sound-linked concepts. Because this branch is small, it should be used for nuance rather than as
a headline result.

### Reliability / ceiling context

| Reliability check | Value |
|---|---:|
| Full repeated-triplet consistency | 0.6842 +/- 0.0076 |
| Repeated triplets | 2000 |
| Triplet rows | 124855 |
| 48-concept split-half RDM Spearman rho | 0.8954 |
| 48-concept RDM pairs | 1128 |
| Full 1854-concept split-half RDM available | `False` |

Interpretation: THINGS behavioral structure is reliable, but full 1854-concept RSA values should be
interpreted comparatively rather than ceiling-normalized.

## Claims Supported by the Current Evidence

1. Prompting and grounding do not induce the same geometry.
2. Corrected Qwen and Mistral RSA show matched grounding > sensory prompting across the main anchors.
3. Llama shows a much weaker and partly anchor-dependent effect, so cross-family claims must be
   phrased as architecture-dependent.
4. Matched visual grounding compresses/constrains the mid-to-late concept manifold across all three
   VLM families relative to text-only, blank, and mismatched controls.
5. Prompt+image states are globally image-dominant rather than additive in a balanced way.
6. Mismatched images perturb global geometry but almost never hijack local text identity.
7. Matched grounding aligns language-side geometry with the Qwen VLM's own visual-tower geometry.
8. Variance partitioning supports the corrected grounding-favoring story: matched grounding has the
   highest unique human-family and unique anchor-family components in the refreshed analysis.

## Claims Not Supported / Wording to Avoid

1. Do not claim that sensory prompting substitutes for visual grounding.
2. Do not claim that prompting wins on uniquely human residual or unique human-family structure;
   corrected residual and refreshed variance-partitioning analyses show matched grounding still wins.
3. Do not claim a uniformly large cross-family RSA effect; Llama is weak.
4. Do not claim PR reduction alone proves successful grounding; PR must be interpreted with RSA and
   perturbation controls.
5. Do not treat small-N human-local, multi-image, or full-archive branches as primary inferential
   evidence.
6. Do not mix RSA numbers from different aggregation pipelines without labeling them.

## Source Map

Primary source artifacts used here:

| Result family | Source files |
|---|---|
| Corrected extraction | `outputs/embeddings/embedding_metadata_full_summary.json`, `outputs/embeddings/pooled_embeddings_full.npz` |
| Cross-family RSA | `outputs/metrics/cross_family_rsa_full_summary.json`, `outputs/metrics/cross_family_rsa_full.csv` |
| Qwen primary RSA / prompt+image | `outputs/metrics/modality_interference_summary.json`, `outputs/metrics/modality_interference_alignment.csv` |
| Intrinsic dimensionality | `outputs/metrics/intrinsic_dimensionality.csv` |
| PR-RSA relationship | `outputs/metrics/id_alignment_summary.json`, `outputs/metrics/id_alignment_correlation.csv` |
| Residual analyses | `outputs/metrics/residual_interaction_summary.json`, `outputs/metrics/residual_reference_alignment.csv` |
| Human partial/local RSA | `outputs/metrics/human_partial_rsa_summary.json`, `outputs/metrics/human_local_geometry_summary.json` |
| Variance partitioning | `outputs/metrics/variance_partitioning_summary.json`, `outputs/tables/variance_partitioning_table.csv` |
| Internal visual tower | `outputs/metrics/internal_visual_tower_summary.json` |
| Mismatched hijacking | `outputs/metrics/mismatched_hijacking_summary.json`, `outputs/metrics/mismatched_hijacking_validation_summary.json` |
| Global-local dissociation | `outputs/metrics/layerwise_global_local_summary.json` |
| Mixture validation | `outputs/metrics/mixture_decomposition_validation_summary.json` |
| Lancaster | `outputs/tables/lancaster_main_result_table.csv`, `outputs/metrics/lancaster_alignment.csv`, `outputs/metrics/lancaster_gap_bootstrap.csv` |
| Multi-image/full archive | `outputs/metrics/multi_image_prototype_summary.csv`, `outputs/metrics/multi_image_consistency.csv`, `outputs/tables/multi_image_reversal_table.csv`, `outputs/metrics/full_things_prototype_summary.csv`, `outputs/metrics/full_things_image_variance.csv`, `outputs/tables/full_things_prototype_anchor_rsa_table.csv`, `outputs/tables/full_things_prototype_comparison_table.csv` |
| Linear probes | `outputs/metrics/linear_probe_summary.json`, `outputs/metrics/linear_probe_results.csv` |
| Geometry restructuring | `outputs/metrics/procrustes_summary.csv`, `outputs/metrics/neighbor_restructuring.csv` |
| Hierarchy depth | `outputs/metrics/hierarchy_depth_summary.json` |
| Reliability | `outputs/metrics/things_reliability_ceiling.json` |
