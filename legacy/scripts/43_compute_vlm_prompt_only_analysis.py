from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from analysis_common import ordered_embedding_for_concepts
from common import (
    ROOT,
    append_run_log,
    canonical_condition_name,
    condensed_cosine_distance,
    embeddings_path,
    load_project_config,
    metrics_path,
    output_path,
    percentile_interval,
    spearman_corr,
    write_csv,
    write_json,
)
from hardening_common import (
    build_proxy_rdms,
    lancaster_matrix_for_concepts,
    load_active_concept_rows,
    load_siglip_reference,
    load_things_reference,
    residual_rsa,
    selected_layers,
    write_text,
)


CONDITIONS = ["M_text_only", "M_prompt_only", "M_matched_image", "M_prompt_plus_matched_image"]
ANCHORS = ["THINGS", "controlled_THINGS", "SigLIP2", "CLIP_ViT_L_14", "DINOv2", "lancaster_perceptual"]
BOOTSTRAP_ANCHORS = ["THINGS", "controlled_THINGS", "SigLIP2", "lancaster_perceptual"]


def family_specs(config: dict[str, Any]) -> list[dict[str, str]]:
    return [dict(row) for row in config["analysis"]["analysis"].get("cross_family_families", [])]


def tagged_paths(tag: str) -> tuple[Path, Path]:
    base = ROOT / "outputs" / "embeddings"
    return base / f"pooled_embeddings_{tag}.npz", base / f"embedding_metadata_{tag}.json"


def load_combined_bundle(extra_tags: list[str]) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[str, np.ndarray], dict[str, list[int]], dict[str, Any]]:
    bundles = [(embeddings_path("pooled_embeddings_full.npz"), embeddings_path("embedding_metadata_full.json"))]
    bundles.extend(tagged_paths(tag) for tag in extra_tags)

    pooled: dict[str, np.ndarray] = {}
    metadata_lookup: dict[tuple[str, str, int], dict[str, Any]] = {}
    layers_by_model: dict[str, set[int]] = {}
    merged_records: list[dict[str, Any]] = []
    merged_metadata: dict[str, Any] | None = None
    record_id = 0

    for npz_path, metadata_path in bundles:
        if not npz_path.exists() or not metadata_path.exists():
            raise RuntimeError(f"Missing bundle files: {npz_path} / {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        arrays = np.load(npz_path, mmap_mode="r")
        if merged_metadata is None:
            merged_metadata = dict(metadata)
            merged_metadata["records"] = []
        for record in metadata["records"]:
            if record["domain"] != "sensory":
                continue
            condition = canonical_condition_name(record["condition"])
            key = (record["model_id"], condition, int(record["layer"]))
            if key in metadata_lookup:
                continue
            new_record = {**record, "condition": condition, "record_id": record_id}
            pooled[f"record_{record_id}"] = np.asarray(arrays[f"record_{record['record_id']}"], dtype=np.float32)
            metadata_lookup[key] = new_record
            layers_by_model.setdefault(record["model_id"], set()).add(int(record["layer"]))
            merged_records.append(new_record)
            record_id += 1

    if merged_metadata is None:
        raise RuntimeError("No embedding bundle was loaded.")
    merged_metadata["records"] = merged_records
    return metadata_lookup, pooled, {model: sorted(layers) for model, layers in layers_by_model.items()}, merged_metadata


def mean_embedding(
    metadata_lookup: dict[tuple[str, str, int], dict[str, Any]],
    pooled: dict[str, np.ndarray],
    model_id: str,
    condition: str,
    layers: list[int],
) -> tuple[np.ndarray, list[str]]:
    matrices = []
    concepts: list[str] | None = None
    for layer in layers:
        record = metadata_lookup.get((model_id, condition, layer))
        if record is None:
            continue
        if concepts is None:
            concepts = [concept.lower() for concept in record["concepts"]]
        matrices.append(np.asarray(pooled[f"record_{record['record_id']}"], dtype=np.float32))
    if concepts is None or not matrices:
        raise RuntimeError(f"Missing embeddings for {model_id} {condition}")
    return np.mean(np.stack(matrices), axis=0, dtype=np.float32), concepts


def ordered_static_anchor(name: str, target_concepts: list[str]) -> np.ndarray:
    mapping = {
        "CLIP_ViT_L_14": ("clip_vitl14_embeddings.npy", "clip_vitl14_concepts.json"),
        "DINOv2": ("dinov2_embeddings.npy", "dinov2_concepts.json"),
    }
    emb_name, concept_name = mapping[name]
    matrix = np.load(ROOT / "data" / "anchors" / emb_name)
    concepts = [concept.lower() for concept in json.loads((ROOT / "data" / "anchors" / concept_name).read_text(encoding="utf-8"))]
    return ordered_embedding_for_concepts(matrix, concepts, target_concepts)


def square_from_condensed(condensed: np.ndarray, n: int) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=float)
    matrix[np.triu_indices(n, k=1)] = condensed
    return matrix + matrix.T


def condensed_from_square_sample(square: np.ndarray, sample_idx: np.ndarray) -> np.ndarray:
    sampled = square[np.ix_(sample_idx, sample_idx)]
    return np.asarray(sampled[np.triu_indices(len(sample_idx), k=1)], dtype=float)


def bootstrap_gap(
    left_matrix: np.ndarray,
    right_matrix: np.ndarray,
    anchor_matrix: np.ndarray,
    *,
    mode: str,
    control_matrices: list[np.ndarray] | None,
    n_resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = left_matrix.shape[0]
    gaps = []
    for _ in range(n_resamples):
        sample_idx = rng.integers(0, n, size=n)
        left_rdm = condensed_cosine_distance(left_matrix[sample_idx])
        right_rdm = condensed_cosine_distance(right_matrix[sample_idx])
        anchor_rdm = condensed_from_square_sample(anchor_matrix, sample_idx)
        if mode == "residual":
            controls = [condensed_from_square_sample(control, sample_idx) for control in control_matrices or []]
            gap = residual_rsa(right_rdm, anchor_rdm, controls) - residual_rsa(left_rdm, anchor_rdm, controls)
        else:
            gap = spearman_corr(right_rdm, anchor_rdm) - spearman_corr(left_rdm, anchor_rdm)
        gaps.append(gap)
    values = np.asarray(gaps, dtype=float)
    low, high = percentile_interval(values, 0.95)
    return float(values.mean()), low, high


def participation_ratio(matrix: np.ndarray) -> float:
    centered = np.asarray(matrix, dtype=float) - np.asarray(matrix, dtype=float).mean(axis=0, keepdims=True)
    gram = centered @ centered.T
    trace = float(np.trace(gram))
    frob_sq = float(np.square(gram).sum())
    if trace <= 1e-12 or frob_sq <= 1e-12:
        return 0.0
    return float((trace**2) / frob_sq)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the reviewer-critical within-VLM prompt-only analysis.")
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--extra-tags", default="", help="Comma-separated tagged extraction bundles containing M_prompt_only.")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args()

    config = load_project_config(args.config)
    mid_fraction = float(config["analysis"]["analysis"]["mid_to_late_fraction"])
    extra_tags = [tag.strip() for tag in args.extra_tags.split(",") if tag.strip()]
    metadata_lookup, pooled, layers_by_model, metadata = load_combined_bundle(extra_tags)

    things_behavior, things_concepts, _ = load_things_reference()
    target_concepts = [row["concept"].lower() for row in load_active_concept_rows(args.config, domain="sensory")]
    target_idx = [things_concepts.index(concept) for concept in target_concepts]
    things_subset = things_behavior[np.ix_(target_idx, target_idx)]
    things_rdm = np.asarray(1.0 - things_subset[np.triu_indices(len(target_concepts), k=1)], dtype=float)
    things_matrix = 1.0 - things_subset.astype(float)
    proxy_rdms = build_proxy_rdms(target_concepts)
    proxy_names = ["subtype_membership", "coarse_category_structure", "sound_linked_vs_other", "lexical_trigram_distance"]
    proxy_matrices = [square_from_condensed(proxy_rdms[name], len(target_concepts)) for name in proxy_names]

    siglip_matrix, siglip_concepts = load_siglip_reference(metadata_lookup, pooled, layers_by_model, metadata)
    siglip_ordered = ordered_embedding_for_concepts(siglip_matrix, siglip_concepts, target_concepts)
    anchor_rdms = {
        "THINGS": things_rdm,
        "controlled_THINGS": things_rdm,
        "SigLIP2": condensed_cosine_distance(siglip_ordered),
        "CLIP_ViT_L_14": condensed_cosine_distance(ordered_static_anchor("CLIP_ViT_L_14", target_concepts)),
        "DINOv2": condensed_cosine_distance(ordered_static_anchor("DINOv2", target_concepts)),
    }
    siglip_matrix_square = square_from_condensed(anchor_rdms["SigLIP2"], len(target_concepts))

    lancaster_all = [concept.lower() for concept in json.loads((ROOT / "data" / "anchors" / "lancaster_perceptual_concepts.json").read_text(encoding="utf-8"))]
    target_set = set(target_concepts)
    lancaster_concepts = [concept for concept in lancaster_all if concept in target_set]
    lancaster_reference = lancaster_matrix_for_concepts(
        lancaster_concepts,
        ["Auditory.mean", "Gustatory.mean", "Haptic.mean", "Interoceptive.mean", "Olfactory.mean", "Visual.mean"],
    )
    lancaster_rdm = condensed_cosine_distance(lancaster_reference)
    lancaster_matrix_square = square_from_condensed(lancaster_rdm, len(lancaster_concepts))

    rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"num_concepts": len(target_concepts), "extra_tags": extra_tags, "families": {}}

    for family in family_specs(config):
        family_name = family["family_name"]
        model_id = family["multimodal_model"]
        if model_id not in layers_by_model:
            summary["families"][family_name] = {"status": "missing_model", "model": model_id}
            continue
        layers = selected_layers(layers_by_model[model_id], mid_fraction)
        try:
            condition_matrices = {
                condition: mean_embedding(metadata_lookup, pooled, model_id, condition, layers)
                for condition in CONDITIONS
            }
        except RuntimeError as exc:
            summary["families"][family_name] = {"status": "missing_condition", "error": str(exc)}
            continue

        scores: dict[str, dict[str, float]] = {}
        ordered_by_condition: dict[str, np.ndarray] = {}
        lancaster_by_condition: dict[str, np.ndarray] = {}
        for condition, (matrix, concepts) in condition_matrices.items():
            ordered = ordered_embedding_for_concepts(matrix, concepts, target_concepts)
            ordered_by_condition[condition] = ordered
            model_rdm = condensed_cosine_distance(ordered)
            scores[condition] = {
                "THINGS": spearman_corr(model_rdm, things_rdm),
                "controlled_THINGS": residual_rsa(model_rdm, things_rdm, [proxy_rdms[name] for name in proxy_names]),
                "SigLIP2": spearman_corr(model_rdm, anchor_rdms["SigLIP2"]),
                "CLIP_ViT_L_14": spearman_corr(model_rdm, anchor_rdms["CLIP_ViT_L_14"]),
                "DINOv2": spearman_corr(model_rdm, anchor_rdms["DINOv2"]),
            }
            ordered_lancaster = ordered_embedding_for_concepts(matrix, concepts, lancaster_concepts)
            lancaster_by_condition[condition] = ordered_lancaster
            scores[condition]["lancaster_perceptual"] = spearman_corr(condensed_cosine_distance(ordered_lancaster), lancaster_rdm)
            scores[condition]["participation_ratio"] = participation_ratio(ordered)
            for anchor_name in [*ANCHORS, "participation_ratio"]:
                rows.append(
                    {
                        "family_name": family_name,
                        "model_id": model_id,
                        "condition": condition,
                        "metric": anchor_name,
                        "value": scores[condition][anchor_name],
                        "num_concepts": len(lancaster_concepts) if anchor_name == "lancaster_perceptual" else len(target_concepts),
                    }
                )

        contrasts = {}
        for anchor_name in [*ANCHORS, "participation_ratio"]:
            for left, right, contrast_name in [
                ("M_text_only", "M_prompt_only", "prompt_only_minus_text_only"),
                ("M_prompt_only", "M_matched_image", "matched_minus_prompt_only"),
                ("M_matched_image", "M_prompt_plus_matched_image", "prompt_plus_image_minus_matched"),
                ("M_prompt_only", "M_prompt_plus_matched_image", "prompt_plus_image_minus_prompt_only"),
            ]:
                delta = scores[right][anchor_name] - scores[left][anchor_name]
                contrasts[f"{anchor_name}:{contrast_name}"] = delta
                contrast_rows.append(
                    {
                        "family_name": family_name,
                        "model_id": model_id,
                        "metric": anchor_name,
                        "contrast_name": contrast_name,
                        "left_condition": left,
                        "right_condition": right,
                        "delta": delta,
                    }
                )

        bootstrap_specs = {
            "THINGS": (things_matrix, "spearman", None, ordered_by_condition["M_prompt_only"], ordered_by_condition["M_matched_image"], len(target_concepts)),
            "controlled_THINGS": (things_matrix, "residual", proxy_matrices, ordered_by_condition["M_prompt_only"], ordered_by_condition["M_matched_image"], len(target_concepts)),
            "SigLIP2": (siglip_matrix_square, "spearman", None, ordered_by_condition["M_prompt_only"], ordered_by_condition["M_matched_image"], len(target_concepts)),
            "lancaster_perceptual": (lancaster_matrix_square, "spearman", None, lancaster_by_condition["M_prompt_only"], lancaster_by_condition["M_matched_image"], len(lancaster_concepts)),
        }
        for idx, anchor_name in enumerate(BOOTSTRAP_ANCHORS):
            anchor_matrix, mode, controls, left_matrix, right_matrix, n_concepts = bootstrap_specs[anchor_name]
            mean_gap, ci_low, ci_high = bootstrap_gap(
                left_matrix,
                right_matrix,
                anchor_matrix,
                mode=mode,
                control_matrices=controls,
                n_resamples=args.bootstrap_resamples,
                seed=4310 + idx + 100 * len(bootstrap_rows),
            )
            bootstrap_rows.append(
                {
                    "family_name": family_name,
                    "anchor_name": anchor_name,
                    "contrast_name": "matched_minus_prompt_only",
                    "observed_gap": contrasts[f"{anchor_name}:matched_minus_prompt_only"],
                    "bootstrap_mean_gap": mean_gap,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "num_concepts": n_concepts,
                    "bootstrap_resamples": args.bootstrap_resamples,
                }
            )

        summary["families"][family_name] = {"status": "ok", "model_id": model_id, "contrasts": contrasts}

    write_csv(metrics_path("vlm_prompt_only_condition_scores.csv"), rows, ["family_name", "model_id", "condition", "metric", "value", "num_concepts"])
    write_csv(metrics_path("vlm_prompt_only_contrasts.csv"), contrast_rows, ["family_name", "model_id", "metric", "contrast_name", "left_condition", "right_condition", "delta"])
    write_csv(
        metrics_path("vlm_prompt_only_matched_bootstrap.csv"),
        bootstrap_rows,
        ["family_name", "anchor_name", "contrast_name", "observed_gap", "bootstrap_mean_gap", "ci95_low", "ci95_high", "num_concepts", "bootstrap_resamples"],
    )
    write_json(metrics_path("vlm_prompt_only_summary.json"), summary)

    lines = ["# VLM Prompt-Only Analysis", "", "This analysis compares a sensory prompt without an image against matched-image grounding within the same VLM."]
    for family_name, payload in summary["families"].items():
        lines.append("")
        lines.append(f"## {family_name}")
        lines.append(f"- status: `{payload['status']}`")
        if payload["status"] == "ok":
            for anchor_name in BOOTSTRAP_ANCHORS:
                key = f"{anchor_name}:matched_minus_prompt_only"
                lines.append(f"- `{anchor_name}` matched-minus-prompt-only: `{payload['contrasts'][key]:.4f}`")
    write_text(output_path("reports", "main_results", "vlm_prompt_only_report.md"), "\n".join(lines))
    append_run_log("VLM Prompt-Only Analysis", ["Wrote within-VLM prompt-only analysis outputs."])


if __name__ == "__main__":
    main()
