from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations

import numpy as np

from analysis_common import aggregate_condition_embedding, load_hierarchy_mapping, ordered_embedding_for_concepts
from common import ROOT, append_run_log, condensed_cosine_distance, load_project_config, metrics_path, output_path, spearman_corr, write_csv, write_json
from hardening_common import condition_model_id, load_embedding_bundle, load_project_backbone, selected_layers, write_text


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
LEVELS = ["coarse_category", "subtype"]


def build_binary_rdm(concepts: list[str], labels: dict[str, str]) -> dict[str, np.ndarray]:
    pairs = list(combinations(range(len(concepts)), 2))
    rdms: dict[str, list[float]] = {level: [] for level in LEVELS}
    for left_idx, right_idx in pairs:
        left = concepts[left_idx]
        right = concepts[right_idx]
        for level in LEVELS:
            rdms[level].append(0.0 if labels[level][left] == labels[level][right] else 1.0)
    return {level: np.asarray(values, dtype=float) for level, values in rdms.items()}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    args = parser.parse_args()

    config = load_project_config(args.config)
    _backbone_config, backbone_text, backbone_multimodal, mid_fraction = load_project_backbone(args.config)
    metadata_lookup, pooled, layers_by_model, _ = load_embedding_bundle()
    hierarchy_lookup, hierarchy_rows = load_hierarchy_mapping(args.config)
    selected_text_layers = selected_layers(layers_by_model[backbone_text], mid_fraction)
    selected_multimodal_layers = selected_layers(layers_by_model[backbone_multimodal], mid_fraction)
    target_concepts = sorted(hierarchy_lookup)
    labels = {
        "coarse_category": {concept: hierarchy_lookup[concept]["coarse_category"] for concept in target_concepts},
        "subtype": {concept: hierarchy_lookup[concept]["subtype"] for concept in target_concepts},
    }
    reference_rdms = build_binary_rdm(target_concepts, labels)

    rows = []
    for condition in CONDITIONS:
        model_id = condition_model_id(backbone_text, backbone_multimodal, condition)
        layers = selected_text_layers if condition.startswith("T_") else selected_multimodal_layers
        for layer in layers:
            embedding, concepts = aggregate_condition_embedding(metadata_lookup, pooled, model_id, condition, [layer])
            ordered = ordered_embedding_for_concepts(embedding, concepts, target_concepts)
            model_rdm = condensed_cosine_distance(ordered)
            for level in LEVELS:
                rows.append(
                    {
                        "condition": condition,
                        "layer": layer,
                        "level": level,
                        "rsa_score": spearman_corr(model_rdm, reference_rdms[level]),
                        "num_concepts": len(target_concepts),
                    }
                )

    summary = {
        "conditions": {},
        "coarse_level_leader": "",
        "subtype_level_leader": "",
        "cross_over_present": False,
        "cross_over_anchor": "",
    }
    for level in LEVELS:
        level_rows = [row for row in rows if row["level"] == level]
        by_condition = defaultdict(list)
        for row in level_rows:
            by_condition[row["condition"]].append(float(row["rsa_score"]))
        summary["conditions"][level] = {
            condition: {
                "mean_rsa": mean(scores),
                "first_half_mean_rsa": mean(scores[: len(scores) // 2]) if scores else 0.0,
                "last_half_mean_rsa": mean(scores[len(scores) // 2 :]) if scores else 0.0,
            }
            for condition, scores in by_condition.items()
        }

    coarse_means = summary["conditions"]["coarse_category"]
    subtype_means = summary["conditions"]["subtype"]
    summary["coarse_level_leader"] = max(coarse_means.items(), key=lambda item: float(item[1]["mean_rsa"]))[0]
    summary["subtype_level_leader"] = max(subtype_means.items(), key=lambda item: float(item[1]["mean_rsa"]))[0]
    summary["cross_over_present"] = summary["coarse_level_leader"] != summary["subtype_level_leader"]
    summary["cross_over_anchor"] = f"{summary['coarse_level_leader']}__vs__{summary['subtype_level_leader']}"

    write_csv(
        metrics_path("hierarchy_depth_alignment.csv"),
        rows,
        ["condition", "layer", "level", "rsa_score", "num_concepts"],
    )
    write_json(metrics_path("hierarchy_depth_summary.json"), summary)
    report_lines = [
        "# Hierarchy Depth Report",
        "",
        "## Summary",
        f"- Coarse-level leader: `{summary['coarse_level_leader']}`",
        f"- Subtype-level leader: `{summary['subtype_level_leader']}`",
        f"- Cross-over present: `{summary['cross_over_present']}`",
        f"- Cross-over anchor: `{summary['cross_over_anchor']}`",
        "",
        "## Level Breakdown",
    ]
    for level in LEVELS:
        report_lines.append(f"### {level}")
        for condition, item in sorted(summary["conditions"][level].items()):
            report_lines.append(f"- `{condition}` mean_rsa=`{float(item['mean_rsa']):.4f}`")
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "- The coarse level approximates the superordinate regime; subtype approximates the subordinate regime.",
            "- A cross-over is present when different conditions lead at the two levels.",
        ]
    )
    write_text(output_path("reports", "main_results", "hierarchy_depth_report.md"), "\n".join(report_lines))
    append_run_log(
        "Hierarchy Depth",
        [
            f"Wrote hierarchy-depth alignment to {metrics_path('hierarchy_depth_alignment.csv').relative_to(ROOT)}.",
            f"Wrote hierarchy-depth report to {output_path('reports', 'main_results', 'hierarchy_depth_report.md').relative_to(ROOT)}.",
        ],
    )


if __name__ == "__main__":
    main()
