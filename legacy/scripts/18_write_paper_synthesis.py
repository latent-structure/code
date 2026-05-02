from __future__ import annotations

import argparse
import json
from collections import defaultdict

from common import ROOT, append_run_log, metrics_path, output_path, read_csv


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_text(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.parse_args()

    anchor_rows = read_csv(metrics_path("anchor_robustness.csv"))
    subtype_rows = read_csv(output_path("outputs", "tables", "human_anchor_subtype_breakdown.csv"))
    audit_summary = read_csv(output_path("outputs", "tables", "human_anchor_coarseness_tests.csv"))
    decision_text = output_path("reports", "decision", "go_no_go_recommendation.md").read_text(encoding="utf-8")
    neighbor_rows = [row for row in read_csv(metrics_path("neighbor_restructuring.csv")) if row["bootstrap_id"] == "aggregate"]
    procrustes_rows = [row for row in read_csv(metrics_path("procrustes_summary.csv")) if row["bootstrap_id"] == "aggregate"]
    local_geometry_rows = read_csv(metrics_path("human_local_geometry.csv")) if metrics_path("human_local_geometry.csv").exists() else []
    partial_summary = json.loads(metrics_path("human_partial_rsa_summary.json").read_text(encoding="utf-8")) if metrics_path("human_partial_rsa_summary.json").exists() else {}
    variance_summary = json.loads(metrics_path("variance_partitioning_summary.json").read_text(encoding="utf-8")) if metrics_path("variance_partitioning_summary.json").exists() else {}
    multi_image_rows = read_csv(metrics_path("multi_image_prototype_summary.csv")) if metrics_path("multi_image_prototype_summary.csv").exists() else []
    full_things_rows = read_csv(metrics_path("full_things_prototype_anchor_rsa.csv")) if metrics_path("full_things_prototype_anchor_rsa.csv").exists() else []
    full_things_comparison_rows = read_csv(output_path("outputs", "tables", "full_things_prototype_comparison_table.csv")) if output_path("outputs", "tables", "full_things_prototype_comparison_table.csv").exists() else []
    lancaster_table_rows = read_csv(output_path("outputs", "tables", "lancaster_main_result_table.csv")) if output_path("outputs", "tables", "lancaster_main_result_table.csv").exists() else []
    layerwise_rows = read_csv(metrics_path("layerwise_trajectory_summary.csv")) if metrics_path("layerwise_trajectory_summary.csv").exists() else []

    anchor_summary_lines = []
    by_anchor: dict[str, dict[str, str]] = {}
    for row in anchor_rows:
        by_anchor.setdefault(row["anchor_name"], row)
    for anchor_name in sorted(by_anchor):
        row = by_anchor[anchor_name]
        anchor_summary_lines.append(
            f"- `{anchor_name}` support=`{row['supports_anchor_ordering']}` mean_rsa(M_matched_image)=`{float(row['mean_rsa']):.4f}` on representative row `{row['condition']}`"
        )

    subtype_lines = [
        f"- `{row['subtype']}` matched_above_prompt=`{row['matched_above_prompt']}` mismatch_collapse=`{row['mismatch_collapse_holds']}` blank_collapse=`{row['blank_collapse_holds']}`"
        for row in subtype_rows
    ]

    coarseness_lines = [
        f"- `{row['proxy_name']}` rho=`{float(row['spearman_with_human_rdm']):.4f}` coarse=`{row['supports_semantically_coarse_anchor']}`"
        for row in audit_summary
    ]

    neighbor_summary: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in neighbor_rows:
        neighbor_summary[(row["condition_a"], row["condition_b"])].append(float(row["mean_jaccard"]))
    neighbor_lines = [
        f"- `{condition_a}` vs `{condition_b}` mean_jaccard=`{mean(values):.4f}`"
        for (condition_a, condition_b), values in sorted(neighbor_summary.items())
    ]

    procrustes_summary: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in procrustes_rows:
        procrustes_summary[(row["condition_a"], row["condition_b"])].append(float(row["procrustes_disparity"]))
    procrustes_lines = [
        f"- `{condition_a}` vs `{condition_b}` disparity=`{mean(values):.4f}`"
        for (condition_a, condition_b), values in sorted(procrustes_summary.items())
    ]
    local_geometry_lines = [
        f"- `{row['group_name']}` `{row['condition']}` mean_local_alignment=`{float(row['mean_local_alignment']):.4f}`"
        for row in local_geometry_rows
        if row["granularity"] in {"all_concepts", "subtype"}
    ]
    partial_lines = (
        [
            f"- raw prompt=`{partial_summary['raw_scores']['T_prompt_primary']:.4f}` matched=`{partial_summary['raw_scores']['M_matched_image']:.4f}`",
            f"- joint-controlled prompt=`{partial_summary['joint_control_scores']['T_prompt_primary']:.4f}` matched=`{partial_summary['joint_control_scores']['M_matched_image']:.4f}`",
            f"- largest prompt reduction control=`{partial_summary['largest_prompt_reduction_control']}`",
        ]
        if partial_summary
        else []
    )
    variance_lines = (
        [
            f"- highest_unique_human_condition=`{variance_summary['highest_unique_human_condition']}`",
            f"- highest_unique_anchor_condition=`{variance_summary['highest_unique_anchor_condition']}`",
            f"- matched_minus_mismatched_anchor_unique=`{float(variance_summary['matched_minus_mismatched_anchor_unique']):.4f}`",
        ]
        if variance_summary
        else []
    )
    multi_image_lines = [
        f"- `{row['anchor_name']}` `{row['representation']}` rsa=`{float(row['rsa_score']):.4f}`"
        for row in multi_image_rows
    ]
    full_things_lines = [
        f"- `{row['anchor_name']}` `{row['representation']}` rsa=`{float(row['rsa_score']):.4f}`"
        for row in full_things_rows
        if row["representation"] in {"single_image_grounding", "prototype_all_images", "T_prompt_primary"}
    ] + [
        f"- `{row['anchor_name']}` `{row['comparison_name']}` mean_gap=`{float(row['mean_gap']):.4f}`"
        for row in full_things_comparison_rows
    ]
    lancaster_lines = [
        f"- `{row['anchor_name']}` `{row['condition']}` mean_rsa=`{float(row['mean_rsa']):.4f}`"
        for row in lancaster_table_rows
        if row["condition"] in {"T_prompt_primary", "M_matched_image"}
    ]
    layerwise_lines = [
        f"- `{row['anchor_name']}` `{row['summary_type']}` value=`{row['value'] or 'none'}`"
        for row in layerwise_rows
        if row["summary_type"] != "trajectory"
    ]

    report = "\n".join(
        [
            "# Dissociation Support Report",
            "",
            "## Claim Frame",
            "- The project no longer targets anchor-robust matched-image dominance as its primary claim.",
            "- The active branch treats prompting and grounding as distinct representational regimes.",
            "",
            "## Decision Record",
            decision_text.strip(),
            "",
            "## Anchor Evidence",
            *(anchor_summary_lines or ["- No anchor summary rows were available."]),
            "",
            "## Human Anchor Structure",
            *(coarseness_lines or ["- No human-anchor coarseness rows were available."]),
            "",
            "## Subtype Dissociation",
            *(subtype_lines or ["- No subtype breakdown rows were available."]),
            "",
            "## Geometry Restructuring",
            "### Neighbor Restructuring",
            *(neighbor_lines or ["- No neighbor restructuring rows were available."]),
            "",
            "### Procrustes",
            *(procrustes_lines or ["- No Procrustes rows were available."]),
            "",
            "### Human Local Geometry",
            *(local_geometry_lines or ["- No human local-geometry rows were available."]),
            "",
            "## Residual Human Structure",
            *(partial_lines or ["- No partial human-RSA summary was available."]),
            "",
            "## Variance Decomposition",
            *(variance_lines or ["- No variance-partitioning summary was available."]),
            "",
            "## Multi-Image Stability",
            *(multi_image_lines or ["- No multi-image summary was available."]),
            "",
            "## Full THINGS Archive",
            *(full_things_lines or ["- No full-THINGS summary was available."]),
            "",
            "## Lancaster Sensorimotor Anchor",
            *(lancaster_lines or ["- No Lancaster summary was available."]),
            "",
            "## Layerwise Trajectories",
            *(layerwise_lines or ["- No layerwise trajectory summary was available."]),
        ]
    )

    write_text(output_path("reports", "main_results", "dissociation_support_report.md"), report)
    append_run_log(
        "Paper Synthesis",
        [
            f"Wrote dissociation support report to {output_path('reports', 'main_results', 'dissociation_support_report.md').relative_to(ROOT)}.",
        ],
    )


if __name__ == "__main__":
    main()
