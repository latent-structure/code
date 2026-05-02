from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from analysis_common import aggregate_condition_embedding, ordered_embedding_for_concepts
from common import ROOT, append_run_log, condensed_cosine_distance, load_project_config, metrics_path, output_path, read_csv, spearman_corr, write_csv, write_json
from hardening_common import (
    build_proxy_rdms,
    condition_model_id,
    lancaster_matrix_for_concepts,
    load_embedding_bundle,
    load_project_backbone,
    load_siglip_reference,
    load_things_reference,
    residual_rsa,
    selected_layers,
    write_text,
)


CONDITIONS = ["T_prompt_primary", "M_matched_image", "M_prompt_plus_matched_image", "M_text_only"]
ANCHORS = ["THINGS behavioral similarity", "controlled_THINGS", "SigLIP2", "lancaster_perceptual"]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def anchor_scores(
    anchor_name: str,
    things_model_rdm: np.ndarray,
    lancaster_model_rdm: np.ndarray,
    behavior_rdm: np.ndarray,
    proxy_rdms: dict[str, np.ndarray],
    siglip_rdm: np.ndarray,
    lancaster_rdm: np.ndarray,
) -> float:
    if anchor_name == "THINGS behavioral similarity":
        return spearman_corr(things_model_rdm, behavior_rdm)
    if anchor_name == "controlled_THINGS":
        return residual_rsa(
            things_model_rdm,
            behavior_rdm,
            [
                proxy_rdms["subtype_membership"],
                proxy_rdms["coarse_category_structure"],
                proxy_rdms["sound_linked_vs_other"],
                proxy_rdms["lexical_trigram_distance"],
            ],
        )
    if anchor_name == "SigLIP2":
        return spearman_corr(things_model_rdm, siglip_rdm)
    if anchor_name == "lancaster_perceptual":
        return spearman_corr(lancaster_model_rdm, lancaster_rdm)
    raise KeyError(anchor_name)


def integration_pattern(prompt_score: float, matched_score: float, combined_score: float) -> str:
    best_base = max(prompt_score, matched_score)
    if combined_score < min(prompt_score, matched_score):
        return "interference"
    if combined_score >= best_base + 0.005:
        return "additive"
    if abs(prompt_score - matched_score) <= 0.01:
        return "balanced"
    return "prompt_dominant" if prompt_score > matched_score else "image_dominant"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    args = parser.parse_args()

    config = load_project_config(args.config)
    _backbone_config, backbone_text, backbone_multimodal, mid_fraction = load_project_backbone(args.config)
    metadata_lookup, pooled, layers_by_model, metadata = load_embedding_bundle()
    things_behavior, things_concepts, _ = load_things_reference()
    proxy_rdms = build_proxy_rdms(things_concepts)
    siglip_reference, siglip_concepts = load_siglip_reference(metadata_lookup, pooled, layers_by_model, metadata)
    lancaster_concepts = [concept.lower() for concept in json.loads((ROOT / "data" / "anchors" / "lancaster_perceptual_concepts.json").read_text(encoding="utf-8"))]
    lancaster_reference = lancaster_matrix_for_concepts(lancaster_concepts, ["Auditory.mean", "Gustatory.mean", "Haptic.mean", "Interoceptive.mean", "Olfactory.mean", "Visual.mean"])
    behavior_rdm = 1.0 - things_behavior[np.triu_indices(len(things_concepts), k=1)]
    siglip_rdm = condensed_cosine_distance(siglip_reference)
    lancaster_rdm = condensed_cosine_distance(lancaster_reference)

    selected_text_layers = selected_layers(layers_by_model[backbone_text], mid_fraction)
    selected_multimodal_layers = selected_layers(layers_by_model[backbone_multimodal], mid_fraction)

    condition_scores: dict[str, dict[str, float]] = defaultdict(dict)
    rows = []
    for condition in CONDITIONS:
        model_id = condition_model_id(backbone_text, backbone_multimodal, condition)
        layers = selected_text_layers if condition.startswith("T_") else selected_multimodal_layers
        embedding, concepts = aggregate_condition_embedding(metadata_lookup, pooled, model_id, condition, layers)
        things_ordered = ordered_embedding_for_concepts(embedding, concepts, things_concepts)
        things_model_rdm = condensed_cosine_distance(things_ordered)
        lancaster_ordered = ordered_embedding_for_concepts(embedding, concepts, lancaster_concepts)
        lancaster_model_rdm = condensed_cosine_distance(lancaster_ordered)
        for anchor_name in ANCHORS:
            score = anchor_scores(
                anchor_name,
                things_model_rdm,
                lancaster_model_rdm,
                behavior_rdm,
                proxy_rdms,
                siglip_rdm,
                lancaster_rdm,
            )
            rows.append(
                {
                    "anchor_name": anchor_name,
                    "condition": condition,
                    "model_id": model_id,
                    "rsa_score": score,
                    "comparison_to_prompt": "",
                    "comparison_to_matched": "",
                    "best_base_condition": "",
                    "integration_pattern": "",
                }
            )
            condition_scores[anchor_name][condition] = score

    # fill pairwise comparison rows and summaries
    summary = {"anchors": {}, "overall_pattern": ""}
    for anchor_name in ANCHORS:
        prompt_score = condition_scores[anchor_name]["T_prompt_primary"]
        matched_score = condition_scores[anchor_name]["M_matched_image"]
        combined_score = condition_scores[anchor_name]["M_prompt_plus_matched_image"]
        best_base = "T_prompt_primary" if prompt_score >= matched_score else "M_matched_image"
        pattern = integration_pattern(prompt_score, matched_score, combined_score)
        summary["anchors"][anchor_name] = {
            "prompt_rsa": prompt_score,
            "matched_rsa": matched_score,
            "combined_rsa": combined_score,
            "text_only_rsa": condition_scores[anchor_name]["M_text_only"],
            "combined_minus_best_base": combined_score - max(prompt_score, matched_score),
            "combined_minus_prompt": combined_score - prompt_score,
            "combined_minus_matched": combined_score - matched_score,
            "best_base_condition": best_base,
            "integration_pattern": pattern,
        }

        for row in rows:
            if row["anchor_name"] != anchor_name:
                continue
            condition = row["condition"]
            row["comparison_to_prompt"] = condition_scores[anchor_name][condition] - prompt_score
            row["comparison_to_matched"] = condition_scores[anchor_name][condition] - matched_score
            row["best_base_condition"] = best_base
            row["integration_pattern"] = pattern if condition == "M_prompt_plus_matched_image" else ""

    overall_votes = [item["integration_pattern"] for item in summary["anchors"].values()]
    if overall_votes.count("interference") >= 2:
        summary["overall_pattern"] = "interference"
    elif overall_votes.count("additive") >= 2:
        summary["overall_pattern"] = "additive"
    elif overall_votes.count("image_dominant") >= overall_votes.count("prompt_dominant"):
        summary["overall_pattern"] = "image_dominant"
    else:
        summary["overall_pattern"] = "prompt_dominant"

    write_csv(
        metrics_path("modality_interference_alignment.csv"),
        rows,
        [
            "anchor_name",
            "condition",
            "model_id",
            "rsa_score",
            "comparison_to_prompt",
            "comparison_to_matched",
            "best_base_condition",
            "integration_pattern",
        ],
    )
    write_json(metrics_path("modality_interference_summary.json"), summary)
    report_lines = [
        "# Modality Interference Report",
        "",
        "## Summary",
        f"- Overall integration pattern: `{summary['overall_pattern']}`",
        "",
        "## Anchor Breakdown",
    ]
    for anchor_name in ANCHORS:
        item = summary["anchors"][anchor_name]
        report_lines.extend(
            [
                f"### {anchor_name}",
                f"- Prompt RSA: `{item['prompt_rsa']:.4f}`",
                f"- Matched RSA: `{item['matched_rsa']:.4f}`",
                f"- Combined RSA: `{item['combined_rsa']:.4f}`",
                f"- Combined-minus-best-base: `{item['combined_minus_best_base']:.4f}`",
                f"- Best base condition: `{item['best_base_condition']}`",
                f"- Integration pattern: `{item['integration_pattern']}`",
            ]
        )
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "- The combined prompt-plus-image condition tests whether prompt structure survives multimodal fusion or is overridden by the visual stream.",
            "- Negative combined-minus-best-base values indicate interference; positive values indicate additive integration.",
        ]
    )
    write_text(output_path("reports", "main_results", "modality_interference_report.md"), "\n".join(report_lines))
    append_run_log(
        "Modality Interference",
        [
            f"Wrote modality-interference alignment to {metrics_path('modality_interference_alignment.csv').relative_to(ROOT)}.",
            f"Wrote modality-interference report to {output_path('reports', 'main_results', 'modality_interference_report.md').relative_to(ROOT)}.",
        ],
    )


if __name__ == "__main__":
    main()
