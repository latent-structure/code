from __future__ import annotations

import argparse
from collections import defaultdict

from common import ROOT, metrics_path, output_path, read_csv, write_csv, append_run_log


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.parse_args()

    alignment_rows = [row for row in read_csv(metrics_path("layerwise_alignment_full.csv")) if row["bootstrap_id"] == "aggregate"]
    stability_rows = read_csv(metrics_path("layerwise_stability_full.csv"))
    robustness_rows = read_csv(metrics_path("anchor_robustness.csv"))
    layer_summary_rows = read_csv(metrics_path("anchor_layer_band_summary.csv")) if metrics_path("anchor_layer_band_summary.csv").exists() else []
    reversal_rows = read_csv(metrics_path("degraded_vs_matched_reversals.csv")) if metrics_path("degraded_vs_matched_reversals.csv").exists() else []
    human_anchor_audit_rows = read_csv(metrics_path("human_anchor_audit.csv")) if metrics_path("human_anchor_audit.csv").exists() else []
    human_anchor_concept_rows = read_csv(metrics_path("human_anchor_concept_diagnostics.csv")) if metrics_path("human_anchor_concept_diagnostics.csv").exists() else []
    human_anchor_coarseness_rows = read_csv(metrics_path("human_anchor_coarseness_tests.csv")) if metrics_path("human_anchor_coarseness_tests.csv").exists() else []
    human_local_geometry_rows = read_csv(metrics_path("human_local_geometry.csv")) if metrics_path("human_local_geometry.csv").exists() else []
    human_partial_rows = read_csv(metrics_path("human_partial_rsa.csv")) if metrics_path("human_partial_rsa.csv").exists() else []
    human_blockwise_rows = read_csv(metrics_path("human_blockwise_rsa.csv")) if metrics_path("human_blockwise_rsa.csv").exists() else []
    variance_partition_rows = read_csv(metrics_path("variance_partitioning.csv")) if metrics_path("variance_partitioning.csv").exists() else []
    multi_image_rows = read_csv(metrics_path("multi_image_consistency.csv")) if metrics_path("multi_image_consistency.csv").exists() else []
    multi_image_summary_rows = read_csv(metrics_path("multi_image_prototype_summary.csv")) if metrics_path("multi_image_prototype_summary.csv").exists() else []
    full_things_variance_rows = read_csv(metrics_path("full_things_image_variance.csv")) if metrics_path("full_things_image_variance.csv").exists() else []
    full_things_summary_rows = read_csv(metrics_path("full_things_prototype_summary.csv")) if metrics_path("full_things_prototype_summary.csv").exists() else []
    full_things_anchor_rows = read_csv(metrics_path("full_things_prototype_anchor_rsa.csv")) if metrics_path("full_things_prototype_anchor_rsa.csv").exists() else []
    procrustes_rows = read_csv(metrics_path("procrustes_summary.csv")) if metrics_path("procrustes_summary.csv").exists() else []
    if not procrustes_rows and metrics_path("procrustes_summary_things38.csv").exists():
        procrustes_rows = read_csv(metrics_path("procrustes_summary_things38.csv"))
    intrinsic_dimension_rows = read_csv(metrics_path("intrinsic_dimensionality.csv")) if metrics_path("intrinsic_dimensionality.csv").exists() else []
    if not intrinsic_dimension_rows and metrics_path("intrinsic_dimensionality_things38.csv").exists():
        intrinsic_dimension_rows = read_csv(metrics_path("intrinsic_dimensionality_things38.csv"))
    modality_interference_rows = read_csv(metrics_path("modality_interference_alignment.csv")) if metrics_path("modality_interference_alignment.csv").exists() else []
    linear_probe_rows = read_csv(metrics_path("linear_probe_results.csv")) if metrics_path("linear_probe_results.csv").exists() else []
    id_alignment_rows = read_csv(metrics_path("id_alignment_correlation.csv")) if metrics_path("id_alignment_correlation.csv").exists() else []
    hierarchy_depth_rows = read_csv(metrics_path("hierarchy_depth_alignment.csv")) if metrics_path("hierarchy_depth_alignment.csv").exists() else []
    lancaster_table_rows = read_csv(output_path("outputs", "tables", "lancaster_main_result_table.csv")) if output_path("outputs", "tables", "lancaster_main_result_table.csv").exists() else []
    layerwise_rows = read_csv(metrics_path("layerwise_trajectory_summary.csv")) if metrics_path("layerwise_trajectory_summary.csv").exists() else []

    alignment_summary: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in alignment_rows:
        alignment_summary[(row.get("model", row.get("model_id", "")), row["condition"], row.get("anchor_name", row.get("anchor_model_id", "")))].append(float(row["rsa_score"]))
    alignment_table = [
        {"model": model, "condition": condition, "anchor_name": anchor_name, "mean_rsa": mean(scores)}
        for (model, condition, anchor_name), scores in sorted(alignment_summary.items())
    ]

    perturbation_summary: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in stability_rows:
        value = row.get("abs_drop")
        if value in ("", None):
            continue
        perturbation_summary[(row.get("model", row.get("model_id", "")), row["perturbation"], row.get("anchor_name", row.get("anchor_model_id", "")))].append(float(value))
    perturbation_table = [
        {"model": model, "perturbation": perturbation, "anchor_name": anchor_name, "mean_abs_drop": mean(scores)}
        for (model, perturbation, anchor_name), scores in sorted(perturbation_summary.items())
    ]

    output_path("outputs", "tables").mkdir(parents=True, exist_ok=True)
    write_csv(output_path("outputs", "tables", "alignment_summary_table.csv"), alignment_table, ["model", "condition", "anchor_name", "mean_rsa"])
    write_csv(output_path("outputs", "tables", "perturbation_summary_table.csv"), perturbation_table, ["model", "perturbation", "anchor_name", "mean_abs_drop"])
    write_csv(
        output_path("outputs", "tables", "anchor_robustness_table.csv"),
        robustness_rows,
        list(robustness_rows[0].keys()) if robustness_rows else ["pair_id", "text_model", "multimodal_model", "condition", "layer_band", "anchor_name", "anchor_type", "mean_rsa", "ordering_rank", "supports_anchor_ordering", "violation_notes"],
    )
    if layer_summary_rows:
        write_csv(output_path("outputs", "tables", "anchor_layer_support_table.csv"), layer_summary_rows, list(layer_summary_rows[0].keys()))
    if reversal_rows:
        write_csv(output_path("outputs", "tables", "degraded_vs_matched_reversals_table.csv"), reversal_rows, list(reversal_rows[0].keys()))
    if human_anchor_audit_rows:
        write_csv(output_path("outputs", "tables", "human_anchor_audit_table.csv"), human_anchor_audit_rows, list(human_anchor_audit_rows[0].keys()))
    if human_anchor_concept_rows:
        write_csv(
            output_path("outputs", "tables", "human_anchor_concept_diagnostics_table.csv"),
            human_anchor_concept_rows,
            list(human_anchor_concept_rows[0].keys()),
        )
    if human_anchor_coarseness_rows:
        write_csv(
            output_path("outputs", "tables", "human_anchor_coarseness_tests.csv"),
            human_anchor_coarseness_rows,
            list(human_anchor_coarseness_rows[0].keys()),
        )
    if human_local_geometry_rows:
        write_csv(
            output_path("outputs", "tables", "human_local_geometry_table.csv"),
            human_local_geometry_rows,
            list(human_local_geometry_rows[0].keys()),
        )
    if human_partial_rows:
        write_csv(
            output_path("outputs", "tables", "human_partial_rsa_table.csv"),
            human_partial_rows,
            list(human_partial_rows[0].keys()),
        )
    if human_blockwise_rows:
        write_csv(
            output_path("outputs", "tables", "human_blockwise_rsa_table.csv"),
            human_blockwise_rows,
            list(human_blockwise_rows[0].keys()),
        )
    if variance_partition_rows:
        write_csv(
            output_path("outputs", "tables", "variance_partitioning_table.csv"),
            variance_partition_rows,
            list(variance_partition_rows[0].keys()),
        )
    if multi_image_rows:
        write_csv(
            output_path("outputs", "tables", "multi_image_consistency_table.csv"),
            multi_image_rows,
            list(multi_image_rows[0].keys()),
        )
    if multi_image_summary_rows:
        write_csv(
            output_path("outputs", "tables", "multi_image_prototype_summary_table.csv"),
            multi_image_summary_rows,
            list(multi_image_summary_rows[0].keys()),
        )
    if full_things_variance_rows:
        write_csv(
            output_path("outputs", "tables", "full_things_concept_stability_table.csv"),
            full_things_variance_rows,
            list(full_things_variance_rows[0].keys()),
        )
    if full_things_summary_rows:
        write_csv(
            output_path("outputs", "tables", "full_things_prototype_summary_table.csv"),
            full_things_summary_rows,
            list(full_things_summary_rows[0].keys()),
        )
    if full_things_anchor_rows:
        write_csv(
            output_path("outputs", "tables", "full_things_prototype_anchor_rsa_table.csv"),
            full_things_anchor_rows,
            list(full_things_anchor_rows[0].keys()),
        )
    if procrustes_rows:
        write_csv(
            output_path("outputs", "tables", "procrustes_summary_table.csv"),
            procrustes_rows,
            list(procrustes_rows[0].keys()),
        )
    if intrinsic_dimension_rows:
        write_csv(
            output_path("outputs", "tables", "intrinsic_dimensionality_table.csv"),
            intrinsic_dimension_rows,
            list(intrinsic_dimension_rows[0].keys()),
        )
    if modality_interference_rows:
        write_csv(
            output_path("outputs", "tables", "modality_interference_table.csv"),
            modality_interference_rows,
            list(modality_interference_rows[0].keys()),
        )
    if linear_probe_rows:
        write_csv(
            output_path("outputs", "tables", "linear_probe_table.csv"),
            linear_probe_rows,
            list(linear_probe_rows[0].keys()),
        )
    if id_alignment_rows:
        write_csv(
            output_path("outputs", "tables", "id_alignment_correlation_table.csv"),
            id_alignment_rows,
            list(id_alignment_rows[0].keys()),
        )
    if hierarchy_depth_rows:
        write_csv(
            output_path("outputs", "tables", "hierarchy_depth_alignment_table.csv"),
            hierarchy_depth_rows,
            list(hierarchy_depth_rows[0].keys()),
        )
    if lancaster_table_rows:
        write_csv(
            output_path("outputs", "tables", "lancaster_main_result_table.csv"),
            lancaster_table_rows,
            list(lancaster_table_rows[0].keys()),
        )
    if layerwise_rows:
        write_csv(
            output_path("outputs", "tables", "layerwise_trajectory_summary_table.csv"),
            layerwise_rows,
            list(layerwise_rows[0].keys()),
        )
    append_run_log(
        "Tables",
        [
            "Wrote alignment, perturbation, anchor robustness, human-anchor diagnostic, local-geometry, partial-RSA, variance-partitioning, multi-image, full-THINGS, canonical Procrustes, intrinsic-dimensionality, modality-interference, linear-probe, ID-correlation, hierarchy-depth, Lancaster, and layerwise tables to outputs/tables.",
        ],
    )


if __name__ == "__main__":
    main()
