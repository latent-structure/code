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
HUMAN_ANCHORS = ["THINGS behavioral similarity", "controlled_THINGS", "SigLIP2", "lancaster_perceptual"]
CONDITION_SUBSETS = {
    "all_conditions": CONDITIONS,
    "primary_comparison": [
        "T_prompt_primary",
        "M_text_only",
        "M_matched_image",
        "M_prompt_plus_matched_image",
    ],
    "multimodal_perturbations": [
        "M_text_only",
        "M_blank_image",
        "M_mismatched_image",
        "M_degraded_image",
        "M_matched_image",
        "M_prompt_plus_matched_image",
    ],
    "visual_input_only": [
        "M_blank_image",
        "M_mismatched_image",
        "M_degraded_image",
        "M_matched_image",
    ],
}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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
    things_behavior_rdm = np.asarray(1.0 - things_behavior[np.triu_indices(len(things_concepts), k=1)], dtype=float)
    siglip_rdm = condensed_cosine_distance(siglip_reference)
    lancaster_rdm = condensed_cosine_distance(lancaster_reference)
    intrinsic_rows = read_csv(metrics_path("intrinsic_dimensionality.csv"))
    if not intrinsic_rows and metrics_path("intrinsic_dimensionality_things38.csv").exists():
        intrinsic_rows = read_csv(metrics_path("intrinsic_dimensionality_things38.csv"))
    probe_rows = read_csv(metrics_path("linear_probe_results.csv")) if metrics_path("linear_probe_results.csv").exists() else []

    selected_text_layers = selected_layers(layers_by_model[backbone_text], mid_fraction)
    selected_multimodal_layers = selected_layers(layers_by_model[backbone_multimodal], mid_fraction)

    condition_pr: dict[str, list[float]] = defaultdict(list)
    selected_layers_by_model = {
        backbone_text: set(selected_text_layers),
        backbone_multimodal: set(selected_multimodal_layers),
    }
    for row in intrinsic_rows:
        if row["domain"] != "sensory":
            continue
        model_id = row["model"]
        if model_id not in selected_layers_by_model:
            continue
        if int(row["layer"]) not in selected_layers_by_model[model_id]:
            continue
        condition_pr[row["condition"]].append(float(row["participation_ratio"]))

    condition_mean_pr = {condition: mean(values) for condition, values in condition_pr.items() if values}
    condition_anchor_scores: dict[str, dict[str, float]] = {anchor: {} for anchor in HUMAN_ANCHORS}
    for condition in CONDITIONS:
        model_id = condition_model_id(backbone_text, backbone_multimodal, condition)
        layers = selected_text_layers if condition.startswith("T_") else selected_multimodal_layers
        embedding, concepts = aggregate_condition_embedding(metadata_lookup, pooled, model_id, condition, layers)
        ordered_things = ordered_embedding_for_concepts(embedding, concepts, things_concepts)
        things_model_rdm = condensed_cosine_distance(ordered_things)
        ordered_lancaster = ordered_embedding_for_concepts(embedding, concepts, lancaster_concepts)
        lancaster_model_rdm = condensed_cosine_distance(ordered_lancaster)

        condition_anchor_scores["THINGS behavioral similarity"][condition] = spearman_corr(things_model_rdm, things_behavior_rdm)
        condition_anchor_scores["controlled_THINGS"][condition] = residual_rsa(
            things_model_rdm,
            things_behavior_rdm,
            [
                proxy_rdms["subtype_membership"],
                proxy_rdms["coarse_category_structure"],
                proxy_rdms["sound_linked_vs_other"],
                proxy_rdms["lexical_trigram_distance"],
            ],
        )
        condition_anchor_scores["SigLIP2"][condition] = spearman_corr(things_model_rdm, siglip_rdm)
        condition_anchor_scores["lancaster_perceptual"][condition] = spearman_corr(lancaster_model_rdm, lancaster_rdm)

    rows = []
    summary = {"conditions": {}, "correlations": []}
    for condition in CONDITIONS:
        summary["conditions"][condition] = {
            "mean_participation_ratio": condition_mean_pr.get(condition, 0.0),
            "human_anchor_scores": {anchor: condition_anchor_scores[anchor].get(condition, 0.0) for anchor in HUMAN_ANCHORS},
        }

    for subset_name, subset_conditions in CONDITION_SUBSETS.items():
        for anchor_name in HUMAN_ANCHORS:
            x = []
            y = []
            included_conditions = []
            for condition in subset_conditions:
                if condition not in condition_mean_pr or condition not in condition_anchor_scores[anchor_name]:
                    continue
                x.append(condition_mean_pr[condition])
                y.append(condition_anchor_scores[anchor_name][condition])
                included_conditions.append(condition)
            if len(x) >= 3:
                rho = spearman_corr(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
                rows.append(
                    {
                        "comparison_type": "human_anchor",
                        "comparison_name": anchor_name,
                        "metric_name": "rsa_score",
                        "condition_subset": subset_name,
                        "spearman_rho": rho,
                        "num_conditions": len(x),
                        "condition_set": ",".join(included_conditions),
                    }
                )
                summary["correlations"].append(
                    {
                        "comparison_type": "human_anchor",
                        "comparison_name": anchor_name,
                        "metric_name": "rsa_score",
                        "condition_subset": subset_name,
                        "spearman_rho": rho,
                        "num_conditions": len(x),
                    }
                )

    probe_metric_rows = defaultdict(dict)
    for row in probe_rows:
        probe_metric_rows[(row["target_label"], row["metric_name"])][row["condition"]] = float(row["metric_value"])

    for subset_name, subset_conditions in CONDITION_SUBSETS.items():
        for (target_label, metric_name), condition_scores in sorted(probe_metric_rows.items()):
            x = []
            y = []
            included_conditions = []
            for condition in subset_conditions:
                if condition not in condition_mean_pr or condition not in condition_scores:
                    continue
                x.append(condition_mean_pr[condition])
                y.append(condition_scores[condition])
                included_conditions.append(condition)
            if len(x) >= 3:
                rho = spearman_corr(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
                rows.append(
                    {
                        "comparison_type": "linear_probe",
                        "comparison_name": target_label,
                        "metric_name": metric_name,
                        "condition_subset": subset_name,
                        "spearman_rho": rho,
                        "num_conditions": len(x),
                        "condition_set": ",".join(included_conditions),
                    }
                )
                summary["correlations"].append(
                    {
                        "comparison_type": "linear_probe",
                        "comparison_name": target_label,
                        "metric_name": metric_name,
                        "condition_subset": subset_name,
                        "spearman_rho": rho,
                        "num_conditions": len(x),
                    }
                )

    write_csv(
        metrics_path("id_alignment_correlation.csv"),
        rows,
        ["comparison_type", "comparison_name", "metric_name", "condition_subset", "spearman_rho", "num_conditions", "condition_set"],
    )
    write_json(metrics_path("id_alignment_summary.json"), summary)
    report_lines = [
        "# ID Alignment Report",
        "",
        "## Summary",
        "- The report correlates condition-level participation ratio with human-anchor RSA and linear-probe scores.",
    ]
    for row in rows:
        report_lines.append(
            f"- `{row['condition_subset']}` `{row['comparison_type']}` `{row['comparison_name']}` `{row['metric_name']}` rho=`{float(row['spearman_rho']):.4f}` over `{row['num_conditions']}` conditions"
        )
    report_path = output_path("reports", "main_results", "id_alignment_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    append_run_log(
        "ID Alignment Correlation",
        [
            f"Wrote ID-alignment correlation metrics to {metrics_path('id_alignment_correlation.csv').relative_to(ROOT)}.",
            f"Wrote ID-alignment report to {report_path.relative_to(ROOT)}.",
        ],
    )


if __name__ == "__main__":
    main()
