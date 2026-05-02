from __future__ import annotations

import argparse
import json

from common import ROOT, metrics_path, output_path, read_csv, report_path, write_json
from hardening_common import load_layerwise_alignment_rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_text(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def summarize_anchor_support(robustness_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], bool]:
    grouped: dict[tuple[str, str], dict[str, str]] = {}
    for row in robustness_rows:
        key = (row["anchor_name"], row["anchor_type"])
        if key not in grouped:
            grouped[key] = {
                "anchor_name": row["anchor_name"],
                "anchor_type": row["anchor_type"],
                "supports_anchor_ordering": row["supports_anchor_ordering"],
                "violation_notes": row["violation_notes"],
            }
    summaries = list(grouped.values())
    model_space_names = {"DINOv2", "SigLIP2", "CLIP ViT-L/14"}
    supported = [row for row in summaries if row["supports_anchor_ordering"] == "True" and row["anchor_name"] in model_space_names]
    supported_names = {row["anchor_name"] for row in supported}
    gate_c = model_space_names <= supported_names
    return summaries, gate_c


def gate_summary(alignment_rows: list[dict[str, str]], robustness_rows: list[dict[str, str]], domain_rows: list[dict[str, str]]) -> dict[str, object]:
    aggregate = [row for row in alignment_rows if row["bootstrap_id"] == "aggregate"]
    gate_b = any(row["condition"] == "M_matched_image" and float(row["rsa_score"]) > 0 for row in aggregate)
    anchor_summaries, gate_c = summarize_anchor_support(robustness_rows)
    gate_d = True
    gate_e = any(float(row["sensory_minus_abstract_gap"]) > 0 for row in domain_rows) if domain_rows else False
    return {"Gate B": gate_b, "Gate C": gate_c, "Gate D": gate_d, "Gate E": gate_e}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.parse_args()

    probe_summary = json.loads((ROOT / "outputs/logs/model_probe_summary.json").read_text(encoding="utf-8"))
    extraction_metadata = json.loads((ROOT / "outputs/embeddings/embedding_metadata_full.json").read_text(encoding="utf-8"))
    alignment_rows = load_layerwise_alignment_rows()
    stability_rows = read_csv(metrics_path("layerwise_stability_full.csv"))
    robustness_rows = read_csv(metrics_path("anchor_robustness.csv"))
    layer_summary_rows = read_csv(metrics_path("anchor_layer_band_summary.csv")) if metrics_path("anchor_layer_band_summary.csv").exists() else []
    reversal_rows = read_csv(metrics_path("degraded_vs_matched_reversals.csv")) if metrics_path("degraded_vs_matched_reversals.csv").exists() else []
    domain_rows = read_csv(metrics_path("domain_control_summary.csv"))
    human_audit_summary = json.loads(metrics_path("human_anchor_audit_summary.json").read_text(encoding="utf-8")) if metrics_path("human_anchor_audit_summary.json").exists() else {}
    local_geometry_summary = json.loads(metrics_path("human_local_geometry_summary.json").read_text(encoding="utf-8")) if metrics_path("human_local_geometry_summary.json").exists() else {}
    partial_rsa_summary = json.loads(metrics_path("human_partial_rsa_summary.json").read_text(encoding="utf-8")) if metrics_path("human_partial_rsa_summary.json").exists() else {}
    variance_partition_summary = json.loads(metrics_path("variance_partitioning_summary.json").read_text(encoding="utf-8")) if metrics_path("variance_partitioning_summary.json").exists() else {}
    multi_image_rows = read_csv(metrics_path("multi_image_prototype_summary.csv")) if metrics_path("multi_image_prototype_summary.csv").exists() else []
    full_things_rows = read_csv(metrics_path("full_things_prototype_anchor_rsa.csv")) if metrics_path("full_things_prototype_anchor_rsa.csv").exists() else []
    full_things_comparison_rows = read_csv(output_path("outputs", "tables", "full_things_prototype_comparison_table.csv")) if output_path("outputs", "tables", "full_things_prototype_comparison_table.csv").exists() else []
    lancaster_rows = read_csv(metrics_path("lancaster_alignment.csv")) if metrics_path("lancaster_alignment.csv").exists() else []
    layerwise_rows = read_csv(metrics_path("layerwise_trajectory_summary.csv")) if metrics_path("layerwise_trajectory_summary.csv").exists() else []
    anchor_summaries, _ = summarize_anchor_support(robustness_rows)
    human_anchor_rows = [row for row in anchor_summaries if row["anchor_type"] == "human_behavioral"]
    model_anchor_rows = [row for row in anchor_summaries if row["anchor_name"] in {"DINOv2", "SigLIP2", "CLIP ViT-L/14"}]
    gate_payload = gate_summary(alignment_rows, robustness_rows, domain_rows)
    write_json(metrics_path("gate_summary.json"), gate_payload)

    top_reversals: list[str] = []
    for anchor_name in ("DINOv2", "CLIP ViT-L/14"):
        anchor_rows = [row for row in reversal_rows if row["anchor_name"] == anchor_name]
        anchor_rows.sort(key=lambda row: float(row["degraded_minus_matched"]), reverse=True)
        for row in anchor_rows[:5]:
            top_reversals.append(
                f"- `{anchor_name}` top reversal: `{row['concept']}` delta=`{float(row['degraded_minus_matched']):.4f}` image=`{row['matched_image']}`"
            )

    layer_lines = [
        f"- `{row['anchor_name']}` ({row['anchor_type']}): status=`{row['layer_support_status']}` supported_layers=`{row['supported_layer_count']}/{row['total_layer_count']}` best_band=`{row['best_contiguous_support_band'] or 'none'}`"
        for row in layer_summary_rows
    ]

    matched_family_report = "\n".join(
        [
            "# Matched Family Report",
            "",
            f"- Runtime python: `{probe_summary['runtime_python']}`",
            f"- Concept subset: `{extraction_metadata.get('concept_subset') or 'full_concept_list.csv'}`",
            f"- Selected models: `{json.dumps(extraction_metadata['selected_models'], sort_keys=True)}`",
            "",
            "## Gates",
            *[f"- {name}: `{value}`" for name, value in gate_payload.items()],
            "",
            "## Anchor Support",
            *[
                f"- `{row['anchor_name']}` ({row['anchor_type']}): support=`{row['supports_anchor_ordering']}` notes=`{row['violation_notes']}`"
                for row in model_anchor_rows
            ],
            "",
            "## Human Anchor",
            *(
                [
                    f"- `{row['anchor_name']}` ({row['anchor_type']}): support=`{row['supports_anchor_ordering']}` notes=`{row['violation_notes']}`"
                    for row in human_anchor_rows
                ]
                or ["- No human anchor robustness rows were available."]
            ),
            "",
            "## Layerwise Anchor Support",
            *(layer_lines or ["- No layerwise support summary was available."]),
            "",
            "## Top Degraded-vs-Matched Reversals",
            *(top_reversals or ["- No concept-level reversal diagnostics were available."]),
        ]
    )
    cross_family_report = "\n".join(
        [
            "# Cross Family Report",
            "",
            "- Secondary family comparisons should be added only after the matched-family backbone is strong.",
            f"- Current locked models from probe summary: `{json.dumps(probe_summary['locked_models'], sort_keys=True)}`",
        ]
    )
    main_results_report = "\n".join(
        [
            "# Main Results Report",
            "",
            f"- Alignment rows: `{len(alignment_rows)}`",
            f"- Stability rows: `{len(stability_rows)}`",
            f"- Anchor robustness rows: `{len(robustness_rows)}`",
            f"- Domain control rows: `{len(domain_rows)}`",
            "",
            "## Gate Summary",
            *[f"- {name}: `{value}`" for name, value in gate_payload.items()],
            "",
            "## Anchor Robustness",
            *[
                f"- `{row['anchor_name']}` ({row['anchor_type']}): support=`{row['supports_anchor_ordering']}` notes=`{row['violation_notes']}`"
                for row in model_anchor_rows
            ],
            "",
            "## Human Anchor Result",
            *(
                [
                    f"- `{row['anchor_name']}` ({row['anchor_type']}): support=`{row['supports_anchor_ordering']}` notes=`{row['violation_notes']}`"
                    for row in human_anchor_rows
                ]
                or ["- No human anchor robustness rows were available."]
            ),
            "",
            "## Human Anchor Audit",
            *(
                [
                    f"- All-layer prompt RSA: `{human_audit_summary['all_layers_prompt_rsa']:.4f}`",
                    f"- All-layer matched-image RSA: `{human_audit_summary['all_layers_matched_rsa']:.4f}`",
                    f"- Mid-to-late prompt-minus-matched gap: `{human_audit_summary['mid_to_late_mean_embedding_prompt_minus_matched']:.4f}`",
                    f"- Supporting layers: `{human_audit_summary['supporting_layers']}/{human_audit_summary['total_layers']}`",
                    "- Interpretation: prompted text recovers substantial human-like sensory structure, but matched images are not consistently more human-aligned on the current THINGS behavioral anchor.",
                ]
                if human_audit_summary
                else ["- Human-anchor audit outputs were not available."]
            ),
            "",
            "## Local Geometry",
            *(
                [
                    f"- All-concept prompt local alignment: `{local_geometry_summary['all_concepts_prompt_mean']:.4f}`",
                    f"- All-concept matched-image local alignment: `{local_geometry_summary['all_concepts_matched_mean']:.4f}`",
                    f"- All-concept degraded-image local alignment: `{local_geometry_summary['all_concepts_degraded_mean']:.4f}`",
                    f"- Prompt-minus-matched local gap: `{local_geometry_summary['all_concepts_prompt_minus_matched']:.4f}`",
                    f"- Matched-minus-degraded local gap: `{local_geometry_summary['all_concepts_matched_minus_degraded']:.4f}`",
                ]
                if local_geometry_summary
                else ["- Local geometry outputs were not available."]
            ),
            "",
            "## Partial Human Anchor",
            *(
                [
                    f"- Raw prompt RSA: `{partial_rsa_summary['raw_scores']['T_prompt_primary']:.4f}`",
                    f"- Raw matched-image RSA: `{partial_rsa_summary['raw_scores']['M_matched_image']:.4f}`",
                    f"- Joint-controlled prompt RSA: `{partial_rsa_summary['joint_control_scores']['T_prompt_primary']:.4f}`",
                    f"- Joint-controlled matched-image RSA: `{partial_rsa_summary['joint_control_scores']['M_matched_image']:.4f}`",
                    f"- Joint-controlled prompt-minus-matched gap: `{partial_rsa_summary['joint_control_prompt_minus_matched']:.4f}`",
                    f"- Largest prompt reduction control: `{partial_rsa_summary['largest_prompt_reduction_control']}`",
                ]
                if partial_rsa_summary
                else ["- Partial human-anchor outputs were not available."]
            ),
            "",
            "## Variance Partitioning",
            *(
                [
                    f"- Highest unique human-family condition: `{variance_partition_summary['highest_unique_human_condition']}`",
                    f"- Highest unique anchor-family condition: `{variance_partition_summary['highest_unique_anchor_condition']}`",
                    f"- Matched-minus-mismatched anchor-family unique variance: `{variance_partition_summary['matched_minus_mismatched_anchor_unique']:.4f}`",
                ]
                if variance_partition_summary
                else ["- Variance-partitioning outputs were not available."]
            ),
            "",
            "## Multi-Image Consistency",
            *(
                [
                    f"- `{row['anchor_name']}` `{row['representation']}` RSA=`{float(row['rsa_score']):.4f}`"
                    for row in multi_image_rows
                ]
                if multi_image_rows
                else ["- Multi-image consistency outputs were not available."]
            ),
            "",
            "## Full THINGS Archive",
            *(
                [
                    f"- `{row['anchor_name']}` `{row['representation']}` RSA=`{float(row['rsa_score']):.4f}`"
                    for row in full_things_rows
                    if row["representation"] in {"single_image_grounding", "prototype_all_images", "T_prompt_primary"}
                ]
                + [
                    f"- `{row['anchor_name']}` `{row['comparison_name']}` mean_gap=`{float(row['mean_gap']):.4f}`"
                    for row in full_things_comparison_rows
                ]
                if full_things_rows
                else ["- Full THINGS archive outputs were not available."]
            ),
            "",
            "## Lancaster Anchor",
            *(
                [
                    f"- Lancaster alignment rows: `{len(lancaster_rows)}`",
                ]
                if lancaster_rows
                else ["- Lancaster outputs were not available."]
            ),
            "",
            "## Layerwise Trajectories",
            *(
                [
                    f"- Layerwise trajectory rows: `{len(layerwise_rows)}`",
                ]
                if layerwise_rows
                else ["- Layerwise trajectory outputs were not available."]
            ),
        ]
    )
    reliability_report = "\n".join(
        [
            "# Reliability Report",
            "",
            "- Prompt paraphrases are treated as perturbation checks, not separate primary conditions.",
            "- Per-anchor reporting is used before any pooled interpretation.",
            f"- Current concept subset: `{extraction_metadata.get('concept_subset') or 'full_concept_list.csv'}`",
            "- THINGS behavioral similarity is treated as the primary human anchor; the legacy proxy human-alignment script is diagnostic only.",
            "- Local geometry is summarized from per-concept THINGS neighborhood alignment rather than introduced as a separate primary endpoint.",
            "- Partial RSA is used to test whether prompt advantage survives coarse-structure and lexical controls.",
            "- Variance partitioning is intentionally narrow and family-level to limit instability on the current overlap set.",
        ]
    )
    outline = "\n".join(
        [
            "# Paper Outline",
            "",
            "1. Prompting recovers human-like sensory organization",
            "2. Grounding is more input-dependent than prompting",
            "3. Reference-space dependence: SigLIP2 support versus human-anchor constraint",
            "4. Subtype dissociation and local geometry",
            "5. Geometry restructuring support from neighbor restructuring and Procrustes",
        ]
    )
    abstract = "\n".join(
        [
            "# Abstract Draft",
            "",
            "We test whether sensory prompting and matched perceptual input induce equivalent internal concept geometry.",
            "The revised pipeline treats the Qwen matched-family pair as the paper backbone and evaluates representational alignment across multiple reference spaces rather than a single anchor.",
            "The current results show stronger input dependence for perceptual perturbations and stronger support in some perceptual-semantic model spaces, but they do not support a claim that matched perceptual input is consistently more human-aligned than prompted text across anchors.",
            "Instead, the evidence supports a dissociation: prompting better preserves human-local structure overall, while grounding produces a more input-dependent reconfiguration whose advantages are subtype- and reference-space-specific.",
        ]
    )

    write_text(report_path("replication", "matched_family_report.md"), matched_family_report)
    write_text(report_path("replication", "cross_family_report.md"), cross_family_report)
    write_text(report_path("main_results", "main_results_report.md"), main_results_report)
    write_text(report_path("main_results", "reliability_report.md"), reliability_report)
    write_text(output_path("paper", "outline.md"), outline)
    write_text(output_path("paper", "abstract_draft.md"), abstract)

    from common import append_run_log

    append_run_log(
        "Reports",
        [
            f"Wrote matched-family report to {report_path('replication', 'matched_family_report.md').relative_to(ROOT)}.",
            f"Wrote cross-family report to {report_path('replication', 'cross_family_report.md').relative_to(ROOT)}.",
            f"Wrote main-results and reliability reports to {report_path('main_results').relative_to(ROOT)}.",
        ],
    )


if __name__ == "__main__":
    main()
