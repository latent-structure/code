from __future__ import annotations

import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from common import ROOT, append_run_log, metrics_path, output_path, percentile_interval, write_csv
from hardening_common import (
    CONDITION_ORDER,
    LANCASTER_SPACES,
    condition_model_id,
    lancaster_matrix_for_concepts,
    load_embedding_bundle,
    load_project_backbone,
    mean_embedding_for_condition,
    selected_layers,
    write_text,
)


CONDITIONS = ["T_neutral", "T_prompt_primary", "M_text_only", "M_matched_image", "M_degraded_image", "M_mismatched_image", "M_blank_image"]


def build_bootstrap_rows(
    rdms_by_condition: dict[str, np.ndarray],
    reference_rdm: np.ndarray,
    resamples: int,
    seed: int,
    space_name: str,
) -> list[dict[str, object]]:
    from common import spearman_corr

    rng = np.random.default_rng(seed)
    n = len(reference_rdm)
    rows = []
    for sample_id in range(resamples):
        idx = rng.integers(0, n, size=n)
        prompt = spearman_corr(rdms_by_condition["T_prompt_primary"][idx], reference_rdm[idx])
        matched = spearman_corr(rdms_by_condition["M_matched_image"][idx], reference_rdm[idx])
        mismatched = spearman_corr(rdms_by_condition["M_mismatched_image"][idx], reference_rdm[idx])
        blank = spearman_corr(rdms_by_condition["M_blank_image"][idx], reference_rdm[idx])
        rows.append(
            {
                "anchor_name": space_name,
                "sample_id": sample_id,
                "prompt_rsa": prompt,
                "matched_rsa": matched,
                "mismatched_rsa": mismatched,
                "blank_rsa": blank,
                "matched_minus_prompt": matched - prompt,
                "matched_minus_mismatched": matched - mismatched,
                "matched_minus_blank": matched - blank,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    args = parser.parse_args()

    config, backbone_text, backbone_multimodal, mid_fraction = load_project_backbone(args.config)
    metadata_lookup, pooled, layers_by_model, _ = load_embedding_bundle()
    sensory_rows = [row for row in json.loads((ROOT / "outputs" / "embeddings" / "embedding_metadata_full.json").read_text(encoding="utf-8"))["records"] if row["domain"] == "sensory"]
    concept_rows = [row for row in json.loads((ROOT / "data" / "anchors" / "lancaster_full_sensorimotor_concepts.json").read_text(encoding="utf-8"))]
    concepts = concept_rows
    text_layers = selected_layers(layers_by_model[backbone_text], mid_fraction)
    multimodal_layers = selected_layers(layers_by_model[backbone_multimodal], mid_fraction)

    alignment_rows = []
    bootstrap_rows = []
    table_rows = []
    fig_spaces = list(LANCASTER_SPACES.keys())
    figure, axes = plt.subplots(1, len(fig_spaces), figsize=(6 * len(fig_spaces), 5), sharey=True)
    if len(fig_spaces) == 1:
        axes = [axes]

    for axis, (space_name, dimensions) in zip(axes, LANCASTER_SPACES.items()):
        ref_matrix = lancaster_matrix_for_concepts(concepts, dimensions)
        from common import condensed_cosine_distance, spearman_corr

        ref_rdm = condensed_cosine_distance(ref_matrix)
        aggregate_rdms: dict[str, np.ndarray] = {}
        layer_scores: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for condition in CONDITIONS:
            model_id = condition_model_id(backbone_text, backbone_multimodal, condition)
            layers = layers_by_model[model_id]
            for layer in layers:
                embedding, condition_concepts = mean_embedding_for_condition(metadata_lookup, pooled, model_id, condition, [layer])
                condition_index = {concept: idx for idx, concept in enumerate(condition_concepts)}
                ordered_embedding = np.asarray([embedding[condition_index[concept]] for concept in concepts], dtype=float)
                score = spearman_corr(condensed_cosine_distance(ordered_embedding), ref_rdm)
                alignment_rows.append(
                    {
                        "anchor_name": space_name,
                        "condition": condition,
                        "layer": layer,
                        "aggregation": "single_layer",
                        "rsa_score": score,
                        "num_concepts": len(concepts),
                    }
                )
                layer_scores[condition].append((layer, score))

            selected = text_layers if condition.startswith("T_") else multimodal_layers
            aggregate_embedding, aggregate_concepts = mean_embedding_for_condition(metadata_lookup, pooled, model_id, condition, selected)
            aggregate_index = {concept: idx for idx, concept in enumerate(aggregate_concepts)}
            ordered_aggregate = np.asarray([aggregate_embedding[aggregate_index[concept]] for concept in concepts], dtype=float)
            aggregate_rdm = condensed_cosine_distance(ordered_aggregate)
            aggregate_rdms[condition] = aggregate_rdm
            aggregate_score = spearman_corr(aggregate_rdm, ref_rdm)
            alignment_rows.append(
                {
                    "anchor_name": space_name,
                    "condition": condition,
                    "layer": "",
                    "aggregation": "mid_to_late_mean_embedding",
                    "rsa_score": aggregate_score,
                    "num_concepts": len(concepts),
                }
            )

        rows = build_bootstrap_rows(
            aggregate_rdms,
            ref_rdm,
            int(config["analysis"]["budgets"].get("bootstrap_resamples", 1000)),
            int(config["seeds"]["global"]),
            space_name,
        )
        bootstrap_rows.extend(rows)
        matched_minus_prompt = np.asarray([float(row["matched_minus_prompt"]) for row in rows], dtype=float)
        matched_minus_mismatched = np.asarray([float(row["matched_minus_mismatched"]) for row in rows], dtype=float)
        matched_minus_blank = np.asarray([float(row["matched_minus_blank"]) for row in rows], dtype=float)
        matched_ci = percentile_interval(matched_minus_prompt, float(config["analysis"]["analysis"]["ci_level"]))
        mismatch_ci = percentile_interval(matched_minus_mismatched, float(config["analysis"]["analysis"]["ci_level"]))
        blank_ci = percentile_interval(matched_minus_blank, float(config["analysis"]["analysis"]["ci_level"]))
        for condition in ["T_prompt_primary", "M_matched_image", "M_mismatched_image", "M_blank_image"]:
            scores = np.asarray([float(row[f"{condition.split('_', 1)[1].lower() if condition.startswith('T_') else condition.replace('M_', '').replace('_image', '')}_rsa"]) if False else 0.0 for row in []], dtype=float)
        table_rows.extend(
            [
                {
                    "anchor_name": space_name,
                    "condition": "T_prompt_primary",
                    "mean_rsa": next(row["rsa_score"] for row in alignment_rows[::-1] if row["anchor_name"] == space_name and row["condition"] == "T_prompt_primary" and row["aggregation"] == "mid_to_late_mean_embedding"),
                    "ci_low": "",
                    "ci_high": "",
                    "comparison_name": "",
                    "comparison_mean": "",
                    "comparison_ci_low": "",
                    "comparison_ci_high": "",
                },
                {
                    "anchor_name": space_name,
                    "condition": "M_matched_image",
                    "mean_rsa": next(row["rsa_score"] for row in alignment_rows[::-1] if row["anchor_name"] == space_name and row["condition"] == "M_matched_image" and row["aggregation"] == "mid_to_late_mean_embedding"),
                    "ci_low": "",
                    "ci_high": "",
                    "comparison_name": "matched_minus_prompt",
                    "comparison_mean": float(matched_minus_prompt.mean()),
                    "comparison_ci_low": matched_ci[0],
                    "comparison_ci_high": matched_ci[1],
                },
                {
                    "anchor_name": space_name,
                    "condition": "M_mismatched_image",
                    "mean_rsa": next(row["rsa_score"] for row in alignment_rows[::-1] if row["anchor_name"] == space_name and row["condition"] == "M_mismatched_image" and row["aggregation"] == "mid_to_late_mean_embedding"),
                    "ci_low": "",
                    "ci_high": "",
                    "comparison_name": "matched_minus_mismatched",
                    "comparison_mean": float(matched_minus_mismatched.mean()),
                    "comparison_ci_low": mismatch_ci[0],
                    "comparison_ci_high": mismatch_ci[1],
                },
                {
                    "anchor_name": space_name,
                    "condition": "M_blank_image",
                    "mean_rsa": next(row["rsa_score"] for row in alignment_rows[::-1] if row["anchor_name"] == space_name and row["condition"] == "M_blank_image" and row["aggregation"] == "mid_to_late_mean_embedding"),
                    "ci_low": "",
                    "ci_high": "",
                    "comparison_name": "matched_minus_blank",
                    "comparison_mean": float(matched_minus_blank.mean()),
                    "comparison_ci_low": blank_ci[0],
                    "comparison_ci_high": blank_ci[1],
                },
            ]
        )

        for condition in ["T_neutral", "T_prompt_primary", "M_text_only", "M_matched_image", "M_mismatched_image"]:
            pairs = sorted(layer_scores[condition])
            axis.plot([layer for layer, _ in pairs], [score for _, score in pairs], label=condition)
        axis.set_title(space_name.replace("_", " "))
        axis.set_xlabel("Layer")
        axis.set_ylabel("RSA")

    axes[0].legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path("outputs", "figures", "fig_lancaster_alignment.png"), dpi=180)
    plt.close(figure)

    write_csv(
        metrics_path("lancaster_alignment.csv"),
        alignment_rows,
        ["anchor_name", "condition", "layer", "aggregation", "rsa_score", "num_concepts"],
    )
    write_csv(
        metrics_path("lancaster_gap_bootstrap.csv"),
        bootstrap_rows,
        ["anchor_name", "sample_id", "prompt_rsa", "matched_rsa", "mismatched_rsa", "blank_rsa", "matched_minus_prompt", "matched_minus_mismatched", "matched_minus_blank"],
    )
    write_csv(
        output_path("outputs", "tables", "lancaster_main_result_table.csv"),
        table_rows,
        ["anchor_name", "condition", "mean_rsa", "ci_low", "ci_high", "comparison_name", "comparison_mean", "comparison_ci_low", "comparison_ci_high"],
    )

    report_lines = [
        "# Lancaster Report",
        "",
        "## Overlap",
        f"- Lancaster overlap with the sensory set: `{len(concepts)}` concepts",
        "- Feature spaces reported: `lancaster_full_sensorimotor`, `lancaster_perceptual`, `lancaster_haptic_material`",
        "",
        "## Main Results",
    ]
    for space_name in fig_spaces:
        prompt = next(float(row["mean_rsa"]) for row in table_rows if row["anchor_name"] == space_name and row["condition"] == "T_prompt_primary")
        matched = next(float(row["mean_rsa"]) for row in table_rows if row["anchor_name"] == space_name and row["condition"] == "M_matched_image")
        gap = next(float(row["comparison_mean"]) for row in table_rows if row["anchor_name"] == space_name and row["condition"] == "M_matched_image")
        mismatch = next(float(row["comparison_mean"]) for row in table_rows if row["anchor_name"] == space_name and row["condition"] == "M_mismatched_image")
        blank = next(float(row["comparison_mean"]) for row in table_rows if row["anchor_name"] == space_name and row["condition"] == "M_blank_image")
        behavior = "more_like_siglip2" if gap > 0 else "more_like_things"
        report_lines.extend(
            [
                f"### {space_name}",
                f"- prompt RSA: `{prompt:.4f}`",
                f"- matched RSA: `{matched:.4f}`",
                f"- matched-minus-prompt: `{gap:.4f}`",
                f"- matched-minus-mismatched: `{mismatch:.4f}`",
                f"- matched-minus-blank: `{blank:.4f}`",
                f"- qualitative behavior: `{behavior}`",
            ]
        )
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "- Lancaster is intended to test sensorimotor and perceptual human structure rather than coarse behavioral similarity alone.",
            "- The key comparison is whether matched grounding improves relative to prompting more cleanly here than it does on raw THINGS.",
        ]
    )
    write_text(output_path("reports", "main_results", "lancaster_report.md"), "\n".join(report_lines))
    append_run_log(
        "Lancaster RSA",
        [
            f"Wrote Lancaster alignment metrics to {metrics_path('lancaster_alignment.csv').relative_to(ROOT)}.",
            f"Wrote Lancaster report to {output_path('reports', 'main_results', 'lancaster_report.md').relative_to(ROOT)}.",
        ],
    )


if __name__ == "__main__":
    main()
