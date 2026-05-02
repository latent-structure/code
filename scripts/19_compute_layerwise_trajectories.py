from __future__ import annotations

import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from common import ROOT, append_run_log, metrics_path, output_path, write_csv
from hardening_common import (
    condition_model_id,
    lancaster_matrix_for_concepts,
    load_embedding_bundle,
    load_project_backbone,
    load_siglip_reference,
    load_things_reference,
    mean_embedding_for_condition,
    residual_rsa,
    build_proxy_rdms,
    write_text,
)


ANCHORS = [
    "THINGS behavioral similarity",
    "controlled_THINGS",
    "SigLIP2",
    "lancaster_perceptual",
]
CONDITIONS = ["T_neutral", "T_prompt_primary", "M_text_only", "M_matched_image", "M_mismatched_image", "M_blank_image", "M_degraded_image"]


def first_positive_gap(layers: list[int], values: list[float], threshold: float = 0.01) -> int | None:
    for layer, value in zip(layers, values):
        if value > threshold:
            return layer
    return None


def save_anchor_plot(path, anchor_name: str, by_condition: dict[str, list[tuple[int, float]]], include_conditions: list[str]) -> None:
    plt.figure(figsize=(8, 5))
    for condition in include_conditions:
        pairs = by_condition.get(condition, [])
        if not pairs:
            continue
        plt.plot([layer for layer, _ in pairs], [score for _, score in pairs], label=condition)
    plt.xlabel("Layer")
    plt.ylabel("RSA")
    plt.title(anchor_name)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    args = parser.parse_args()

    _config, backbone_text, backbone_multimodal, _ = load_project_backbone(args.config)
    metadata_lookup, pooled, layers_by_model, metadata = load_embedding_bundle()
    things_behavior, things_concepts, things_index = load_things_reference()
    proxy_rdms = build_proxy_rdms(things_concepts)
    siglip_embedding, siglip_concepts = load_siglip_reference(metadata_lookup, pooled, layers_by_model, metadata)
    siglip_index = {concept: idx for idx, concept in enumerate(siglip_concepts)}
    from common import condensed_cosine_distance, spearman_corr

    siglip_reference_rdm = condensed_cosine_distance(np.asarray([siglip_embedding[siglip_index[concept]] for concept in things_concepts], dtype=float))
    lancaster_concepts = [concept.lower() for concept in json.loads((ROOT / "data" / "anchors" / "lancaster_perceptual_concepts.json").read_text(encoding="utf-8"))]
    lancaster_reference_rdm = condensed_cosine_distance(lancaster_matrix_for_concepts(lancaster_concepts, ["Auditory.mean", "Gustatory.mean", "Haptic.mean", "Interoceptive.mean", "Olfactory.mean", "Visual.mean"]))
    behavior_rdm = np.asarray((1.0 - things_behavior)[np.triu_indices(len(things_concepts), k=1)], dtype=float)

    rows = []
    by_anchor: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    for condition in CONDITIONS:
        model_id = condition_model_id(backbone_text, backbone_multimodal, condition)
        for layer in layers_by_model[model_id]:
            embedding, concepts = mean_embedding_for_condition(metadata_lookup, pooled, model_id, condition, [layer])
            concept_index = {concept: idx for idx, concept in enumerate(concepts)}
            ordered_embedding = np.asarray([embedding[concept_index[concept]] for concept in things_concepts], dtype=float)
            model_rdm = condensed_cosine_distance(ordered_embedding)
            ordered_lancaster_embedding = np.asarray([embedding[concept_index[concept]] for concept in lancaster_concepts], dtype=float)
            lancaster_model_rdm = condensed_cosine_distance(ordered_lancaster_embedding)
            if len(lancaster_model_rdm) != len(lancaster_reference_rdm):
                raise RuntimeError(
                    f"Lancaster RDM length mismatch for {condition} layer {layer}: "
                    f"model={len(lancaster_model_rdm)} reference={len(lancaster_reference_rdm)}"
                )
            raw_things = spearman_corr(model_rdm, behavior_rdm)
            controlled = residual_rsa(
                model_rdm,
                behavior_rdm,
                [
                    proxy_rdms["subtype_membership"],
                    proxy_rdms["coarse_category_structure"],
                    proxy_rdms["sound_linked_vs_other"],
                    proxy_rdms["lexical_trigram_distance"],
                ],
            )
            siglip = spearman_corr(model_rdm, siglip_reference_rdm)
            lancaster = spearman_corr(lancaster_model_rdm, lancaster_reference_rdm)
            for anchor_name, score in [
                ("THINGS behavioral similarity", raw_things),
                ("controlled_THINGS", controlled),
                ("SigLIP2", siglip),
                ("lancaster_perceptual", lancaster),
            ]:
                rows.append(
                    {
                        "anchor_name": anchor_name,
                        "condition": condition,
                        "layer": layer,
                        "summary_type": "trajectory",
                        "rsa_score": score,
                        "value": "",
                    }
                )
                by_anchor[anchor_name][condition].append((layer, score))

    summary_rows = []
    for anchor_name, condition_map in by_anchor.items():
        prompt_pairs = sorted(condition_map.get("T_prompt_primary", []))
        neutral_pairs = sorted(condition_map.get("T_neutral", []))
        matched_pairs = sorted(condition_map.get("M_matched_image", []))
        prompt_layers = [layer for layer, _ in prompt_pairs]
        prompt_minus_neutral = [prompt - neutral for (_, prompt), (_, neutral) in zip(prompt_pairs, neutral_pairs)]
        matched_minus_prompt = [matched - prompt for (_, matched), (_, prompt) in zip(matched_pairs, prompt_pairs)]
        prompt_div = first_positive_gap(prompt_layers, prompt_minus_neutral)
        matched_div = first_positive_gap(prompt_layers, matched_minus_prompt)
        peak_layer = prompt_layers[int(np.argmax(np.asarray(matched_minus_prompt, dtype=float)))] if matched_minus_prompt else None
        for key, value in [
            ("first_prompt_vs_neutral_layer", prompt_div),
            ("first_matched_vs_prompt_layer", matched_div),
            ("peak_matched_vs_prompt_layer", peak_layer),
        ]:
            summary_rows.append(
                {
                    "anchor_name": anchor_name,
                    "condition": "",
                    "layer": "",
                    "summary_type": key,
                    "rsa_score": "",
                    "value": "" if value is None else value,
                }
            )

    all_rows = rows + summary_rows
    write_csv(
        metrics_path("layerwise_trajectory_summary.csv"),
        all_rows,
        ["anchor_name", "condition", "layer", "summary_type", "rsa_score", "value"],
    )

    save_anchor_plot(output_path("outputs", "figures", "fig_layerwise_things.png"), "THINGS behavioral similarity", by_anchor["THINGS behavioral similarity"], ["T_neutral", "T_prompt_primary", "M_text_only", "M_matched_image", "M_mismatched_image"])
    save_anchor_plot(output_path("outputs", "figures", "fig_layerwise_things_controlled.png"), "controlled_THINGS", by_anchor["controlled_THINGS"], ["T_neutral", "T_prompt_primary", "M_text_only", "M_matched_image", "M_mismatched_image"])
    save_anchor_plot(output_path("outputs", "figures", "fig_layerwise_siglip2.png"), "SigLIP2", by_anchor["SigLIP2"], ["T_neutral", "T_prompt_primary", "M_text_only", "M_matched_image", "M_mismatched_image"])
    save_anchor_plot(output_path("outputs", "figures", "fig_layerwise_lancaster.png"), "lancaster_perceptual", by_anchor["lancaster_perceptual"], ["T_neutral", "T_prompt_primary", "M_text_only", "M_matched_image", "M_mismatched_image"])

    report_lines = [
        "# Layerwise Trajectory Report",
        "",
        "## Divergence Summary",
    ]
    for anchor_name in ANCHORS:
        anchor_summary = [row for row in summary_rows if row["anchor_name"] == anchor_name]
        report_lines.append(f"### {anchor_name}")
        for row in anchor_summary:
            report_lines.append(f"- {row['summary_type']}: `{row['value'] or 'none'}`")
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "- Layerwise trajectories are intended to show where prompting and grounding separate inside the backbone.",
            "- Controlled THINGS uses the same coarse-structure controls as the partial-RSA analysis, but evaluated layer by layer.",
        ]
    )
    write_text(output_path("reports", "main_results", "layerwise_trajectory_report.md"), "\n".join(report_lines))
    append_run_log(
        "Layerwise Trajectories",
        [
            f"Wrote layerwise trajectory summary to {metrics_path('layerwise_trajectory_summary.csv').relative_to(ROOT)}.",
            f"Wrote layerwise trajectory report to {output_path('reports', 'main_results', 'layerwise_trajectory_report.md').relative_to(ROOT)}.",
        ],
    )


if __name__ == "__main__":
    main()
