from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from analysis_common import ordered_embedding_for_concepts
from common import ROOT, append_run_log, condensed_cosine_distance, load_project_config, metrics_path, output_path, percentile_interval, spearman_corr, write_csv, write_json
from hardening_common import (
    build_proxy_rdms,
    lancaster_matrix_for_concepts,
    load_embedding_bundle,
    load_project_backbone,
    load_siglip_reference,
    load_things_reference,
    mean_embedding_for_condition,
    residual_rsa,
    selected_layers,
    write_text,
)


CONDITIONS = [
    "T_neutral",
    "T_prompt_primary",
    "M_text_only",
    "M_matched_image",
    "M_prompt_plus_matched_image",
    "M_degraded_image",
    "M_mismatched_image",
    "M_blank_image",
]
ANCHORS = ["THINGS", "controlled_THINGS", "SigLIP2", "CLIP_ViT_L_14", "DINOv2", "lancaster_perceptual"]
BOOTSTRAP_ANCHORS = ["THINGS", "controlled_THINGS", "SigLIP2", "lancaster_perceptual"]


def family_specs(config: dict[str, Any]) -> list[dict[str, str]]:
    return [dict(row) for row in config["analysis"]["analysis"].get("cross_family_families", [])]


def model_for_condition(spec: dict[str, str], condition: str) -> str:
    return spec["text_model"] if condition.startswith("T_") else spec["multimodal_model"]


def ordered_static_anchor(name: str, target_concepts: list[str]) -> np.ndarray:
    mapping = {
        "CLIP_ViT_L_14": ("clip_vitl14_embeddings.npy", "clip_vitl14_concepts.json"),
        "DINOv2": ("dinov2_embeddings.npy", "dinov2_concepts.json"),
    }
    emb_name, concept_name = mapping[name]
    matrix = np.load(ROOT / "data" / "anchors" / emb_name)
    concepts = [concept.lower() for concept in json.loads((ROOT / "data" / "anchors" / concept_name).read_text(encoding="utf-8"))]
    return ordered_embedding_for_concepts(matrix, concepts, target_concepts)


def add_anchor_row(
    rows: list[dict[str, Any]],
    family: dict[str, str],
    anchor_name: str,
    condition: str,
    score: float,
    num_concepts: int,
    num_pairs: int,
) -> None:
    rows.append(
        {
            "family_name": family["family_name"],
            "family_role": family.get("family_role", ""),
            "text_model": family["text_model"],
            "multimodal_model": family["multimodal_model"],
            "condition": condition,
            "anchor_name": anchor_name,
            "rsa_score": score,
            "num_concepts": num_concepts,
            "num_pairs": num_pairs,
            "row_type": "condition_score",
            "contrast_name": "",
            "contrast_delta": "",
        }
    )


def add_contrast_row(
    rows: list[dict[str, Any]],
    family: dict[str, str],
    anchor_name: str,
    contrast_name: str,
    delta: float,
) -> None:
    rows.append(
        {
            "family_name": family["family_name"],
            "family_role": family.get("family_role", ""),
            "text_model": family["text_model"],
            "multimodal_model": family["multimodal_model"],
            "condition": "",
            "anchor_name": anchor_name,
            "rsa_score": "",
            "num_concepts": "",
            "num_pairs": "",
            "row_type": "contrast",
            "contrast_name": contrast_name,
            "contrast_delta": delta,
        }
    )


def square_from_condensed(condensed: np.ndarray, n: int) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=float)
    matrix[np.triu_indices(n, k=1)] = condensed
    return matrix + matrix.T


def condensed_from_square_sample(square: np.ndarray, sample_idx: np.ndarray) -> np.ndarray:
    sampled = square[np.ix_(sample_idx, sample_idx)]
    return np.asarray(sampled[np.triu_indices(len(sample_idx), k=1)], dtype=float)


def bootstrap_matched_prompt_gap(
    *,
    prompt_matrix: np.ndarray,
    matched_matrix: np.ndarray,
    anchor_matrix: np.ndarray,
    mode: str,
    control_matrices: list[np.ndarray] | None,
    n_resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = prompt_matrix.shape[0]
    gaps = []
    for _ in range(n_resamples):
        sample_idx = rng.integers(0, n, size=n)
        prompt_rdm = condensed_cosine_distance(prompt_matrix[sample_idx])
        matched_rdm = condensed_cosine_distance(matched_matrix[sample_idx])
        anchor_rdm = condensed_from_square_sample(anchor_matrix, sample_idx)
        if mode == "residual":
            controls = [condensed_from_square_sample(control, sample_idx) for control in control_matrices or []]
            gap = residual_rsa(matched_rdm, anchor_rdm, controls) - residual_rsa(prompt_rdm, anchor_rdm, controls)
        else:
            gap = spearman_corr(matched_rdm, anchor_rdm) - spearman_corr(prompt_rdm, anchor_rdm)
        gaps.append(gap)
    values = np.asarray(gaps, dtype=float)
    ci_low, ci_high = percentile_interval(values, 0.95)
    return float(values.mean()), ci_low, ci_high


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test concept limit.")
    parser.add_argument("--bootstrap-resamples", type=int, default=0, help="Concept-level bootstrap resamples for matched-minus-prompt headline gaps.")
    args = parser.parse_args()

    config, _primary_text, _primary_multimodal, mid_fraction = load_project_backbone(args.config)
    if args.bootstrap_resamples <= 0:
        args.bootstrap_resamples = int(config["analysis"]["budgets"]["bootstrap_resamples"])
    metadata_lookup, pooled, layers_by_model, metadata = load_embedding_bundle()
    things_behavior, things_concepts, _ = load_things_reference()
    target_concepts = things_concepts[: args.limit] if args.limit else things_concepts
    target_idx = [things_concepts.index(concept) for concept in target_concepts]
    things_subset = things_behavior[np.ix_(target_idx, target_idx)]
    things_rdm = np.asarray(1.0 - things_subset[np.triu_indices(len(target_concepts), k=1)], dtype=float)
    things_distance_matrix = 1.0 - things_subset.astype(float)
    proxy_rdms = build_proxy_rdms(target_concepts)
    proxy_matrices = [square_from_condensed(proxy_rdms[name], len(target_concepts)) for name in [
        "subtype_membership",
        "coarse_category_structure",
        "sound_linked_vs_other",
        "lexical_trigram_distance",
    ]]

    siglip_matrix, siglip_concepts = load_siglip_reference(metadata_lookup, pooled, layers_by_model, metadata)
    siglip_ordered = ordered_embedding_for_concepts(siglip_matrix, siglip_concepts, target_concepts)
    anchor_rdms = {
        "SigLIP2": condensed_cosine_distance(siglip_ordered),
        "CLIP_ViT_L_14": condensed_cosine_distance(ordered_static_anchor("CLIP_ViT_L_14", target_concepts)),
        "DINOv2": condensed_cosine_distance(ordered_static_anchor("DINOv2", target_concepts)),
    }
    siglip_rdm_matrix = square_from_condensed(anchor_rdms["SigLIP2"], len(target_concepts))

    lancaster_all = [concept.lower() for concept in json.loads((ROOT / "data" / "anchors" / "lancaster_perceptual_concepts.json").read_text(encoding="utf-8"))]
    target_set = set(target_concepts)
    lancaster_concepts = [concept for concept in lancaster_all if concept in target_set]
    lancaster_reference = lancaster_matrix_for_concepts(
        lancaster_concepts,
        ["Auditory.mean", "Gustatory.mean", "Haptic.mean", "Interoceptive.mean", "Olfactory.mean", "Visual.mean"],
    )
    lancaster_rdm = condensed_cosine_distance(lancaster_reference)
    lancaster_rdm_matrix = square_from_condensed(lancaster_rdm, len(lancaster_concepts))

    rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"families": {}, "num_concepts": len(target_concepts), "num_lancaster_concepts": len(lancaster_concepts)}
    for family in family_specs(config):
        condition_scores: dict[str, dict[str, float]] = {}
        condition_matrices: dict[str, np.ndarray] = {}
        lancaster_condition_matrices: dict[str, np.ndarray] = {}
        family_name = family["family_name"]
        missing_models = [model for model in {family["text_model"], family["multimodal_model"]} if model not in layers_by_model]
        if missing_models:
            summary["families"][family_name] = {"status": "blocked", "missing_models": missing_models}
            continue

        selected_by_model = {model: selected_layers(layers_by_model[model], mid_fraction) for model in {family["text_model"], family["multimodal_model"]}}
        for condition in CONDITIONS:
            model_id = model_for_condition(family, condition)
            matrix, concepts = mean_embedding_for_condition(metadata_lookup, pooled, model_id, condition, selected_by_model[model_id])
            ordered_things = ordered_embedding_for_concepts(matrix, concepts, target_concepts)
            condition_matrices[condition] = ordered_things
            model_rdm = condensed_cosine_distance(ordered_things)
            condition_scores[condition] = {}
            score = spearman_corr(model_rdm, things_rdm)
            condition_scores[condition]["THINGS"] = score
            add_anchor_row(rows, family, "THINGS", condition, score, len(target_concepts), len(things_rdm))
            score = residual_rsa(
                model_rdm,
                things_rdm,
                [
                    proxy_rdms["subtype_membership"],
                    proxy_rdms["coarse_category_structure"],
                    proxy_rdms["sound_linked_vs_other"],
                    proxy_rdms["lexical_trigram_distance"],
                ],
            )
            condition_scores[condition]["controlled_THINGS"] = score
            add_anchor_row(rows, family, "controlled_THINGS", condition, score, len(target_concepts), len(things_rdm))
            for anchor_name, anchor_rdm in anchor_rdms.items():
                score = spearman_corr(model_rdm, anchor_rdm)
                condition_scores[condition][anchor_name] = score
                add_anchor_row(rows, family, anchor_name, condition, score, len(target_concepts), len(anchor_rdm))
            ordered_lancaster = ordered_embedding_for_concepts(matrix, concepts, lancaster_concepts)
            lancaster_condition_matrices[condition] = ordered_lancaster
            score = spearman_corr(condensed_cosine_distance(ordered_lancaster), lancaster_rdm)
            condition_scores[condition]["lancaster_perceptual"] = score
            add_anchor_row(rows, family, "lancaster_perceptual", condition, score, len(lancaster_concepts), len(lancaster_rdm))

        contrasts: dict[str, float] = {}
        for anchor_name in ANCHORS:
            matched_minus_prompt = condition_scores["M_matched_image"][anchor_name] - condition_scores["T_prompt_primary"][anchor_name]
            prompt_image_minus_matched = condition_scores["M_prompt_plus_matched_image"][anchor_name] - condition_scores["M_matched_image"][anchor_name]
            matched_minus_blank = condition_scores["M_matched_image"][anchor_name] - condition_scores["M_blank_image"][anchor_name]
            matched_minus_mismatch = condition_scores["M_matched_image"][anchor_name] - condition_scores["M_mismatched_image"][anchor_name]
            for contrast_name, delta in [
                ("matched_minus_prompt", matched_minus_prompt),
                ("prompt_image_minus_matched", prompt_image_minus_matched),
                ("matched_minus_blank", matched_minus_blank),
                ("matched_minus_mismatched", matched_minus_mismatch),
            ]:
                add_contrast_row(rows, family, anchor_name, contrast_name, delta)
                contrasts[f"{anchor_name}:{contrast_name}"] = delta
        bootstrap_specs = {
            "THINGS": (things_distance_matrix, "spearman", None, condition_matrices["T_prompt_primary"], condition_matrices["M_matched_image"], len(target_concepts)),
            "controlled_THINGS": (things_distance_matrix, "residual", proxy_matrices, condition_matrices["T_prompt_primary"], condition_matrices["M_matched_image"], len(target_concepts)),
            "SigLIP2": (siglip_rdm_matrix, "spearman", None, condition_matrices["T_prompt_primary"], condition_matrices["M_matched_image"], len(target_concepts)),
            "lancaster_perceptual": (
                lancaster_rdm_matrix,
                "spearman",
                None,
                lancaster_condition_matrices["T_prompt_primary"],
                lancaster_condition_matrices["M_matched_image"],
                len(lancaster_concepts),
            ),
        }
        bootstrap_summary: dict[str, dict[str, float]] = {}
        for anchor_index, anchor_name in enumerate(BOOTSTRAP_ANCHORS):
            anchor_matrix, mode, controls, prompt_matrix, matched_matrix, n_concepts = bootstrap_specs[anchor_name]
            mean_gap, ci_low, ci_high = bootstrap_matched_prompt_gap(
                prompt_matrix=prompt_matrix,
                matched_matrix=matched_matrix,
                anchor_matrix=anchor_matrix,
                mode=mode,
                control_matrices=controls,
                n_resamples=args.bootstrap_resamples,
                seed=1009 + anchor_index + 100 * len(bootstrap_rows),
            )
            observed_gap = contrasts[f"{anchor_name}:matched_minus_prompt"]
            bootstrap_rows.append(
                {
                    "family_name": family_name,
                    "anchor_name": anchor_name,
                    "contrast_name": "matched_minus_prompt",
                    "observed_gap": observed_gap,
                    "bootstrap_mean_gap": mean_gap,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "num_concepts": n_concepts,
                    "bootstrap_resamples": args.bootstrap_resamples,
                    "unit": "concept",
                }
            )
            bootstrap_summary[anchor_name] = {
                "observed_gap": observed_gap,
                "bootstrap_mean_gap": mean_gap,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
        summary["families"][family_name] = {"status": "ok", "contrasts": contrasts, "matched_minus_prompt_bootstrap": bootstrap_summary}

    suffix = "_smoke" if args.limit else ""
    write_csv(
        metrics_path(f"cross_family_rsa_full{suffix}.csv"),
        rows,
        [
            "family_name",
            "family_role",
            "text_model",
            "multimodal_model",
            "condition",
            "anchor_name",
            "rsa_score",
            "num_concepts",
            "num_pairs",
            "row_type",
            "contrast_name",
            "contrast_delta",
        ],
    )
    write_json(metrics_path(f"cross_family_rsa_full_summary{suffix}.json"), summary)
    write_csv(
        metrics_path(f"cross_family_rsa_matched_prompt_bootstrap{suffix}.csv"),
        bootstrap_rows,
        [
            "family_name",
            "anchor_name",
            "contrast_name",
            "observed_gap",
            "bootstrap_mean_gap",
            "ci95_low",
            "ci95_high",
            "num_concepts",
            "bootstrap_resamples",
            "unit",
        ],
    )
    lines = ["# Full Cross-Family RSA Report", "", "## Summary"]
    for family_name, payload in summary["families"].items():
        lines.append(f"- `{family_name}` status: `{payload['status']}`")
        if payload["status"] == "ok":
            lines.append(f"- `{family_name}` THINGS matched-minus-prompt: `{payload['contrasts']['THINGS:matched_minus_prompt']:.4f}`")
            lines.append(f"- `{family_name}` SigLIP2 matched-minus-prompt: `{payload['contrasts']['SigLIP2:matched_minus_prompt']:.4f}`")
    write_text(output_path("reports", "main_results", f"cross_family_rsa_full_report{suffix}.md"), "\n".join(lines))
    append_run_log("Full Cross-Family RSA", [f"Wrote full cross-family RSA outputs with suffix `{suffix}`."])


if __name__ == "__main__":
    main()
