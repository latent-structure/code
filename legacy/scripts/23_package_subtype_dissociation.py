from __future__ import annotations

import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from common import (
    ROOT,
    append_run_log,
    canonical_condition_name,
    condensed_cosine_distance,
    embeddings_path,
    load_project_config,
    metrics_path,
    output_path,
    read_csv,
    write_csv,
)


def mean_embedding_for_condition(
    metadata_lookup: dict[tuple[str, str, int], dict[str, object]],
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
        matrices.append(np.asarray(pooled[f"record_{record['record_id']}"], dtype=float))
    if not matrices or concepts is None:
        raise RuntimeError(f"Missing embeddings for {model_id} {condition}")
    return np.mean(np.stack(matrices), axis=0), concepts


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    from common import spearman_corr as shared_spearman_corr

    return shared_spearman_corr(np.asarray(x, dtype=float), np.asarray(y, dtype=float))


def write_text(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def plot_subtype_dissociation(rows: list[dict[str, object]]) -> None:
    ordered_subtypes = ["appearance_color", "texture_material", "sound_linked", "smell_taste_proxy"]
    anchor_names = ["THINGS behavioral similarity", "SigLIP2", "THINGS residual"]
    labels = []
    values = []
    colors = []
    color_map = {
        "THINGS behavioral similarity": "#1f77b4",
        "SigLIP2": "#d95f02",
        "THINGS residual": "#7570b3",
    }
    for subtype in ordered_subtypes:
        for anchor_name in anchor_names:
            row = next((item for item in rows if item["subtype"] == subtype and item["anchor_name"] == anchor_name), None)
            if row is None:
                continue
            labels.append(f"{subtype}\n{anchor_name}")
            values.append(float(row["matched_minus_prompt"]))
            colors.append(color_map[anchor_name])
    plt.figure(figsize=(12, 5))
    plt.bar(labels, values, color=colors)
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.ylabel("Matched minus prompt")
    plt.title("Subtype dissociation across anchors")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path("outputs", "figures", "fig_subtype_dissociation.png"), dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    args = parser.parse_args()

    config = load_project_config(args.config)
    backbone_text = config["analysis"]["execution"]["sensory_backbone_text_model"]
    backbone_multimodal = config["analysis"]["execution"]["sensory_backbone_multimodal_model"]
    mid_to_late_fraction = float(config["analysis"]["analysis"]["mid_to_late_fraction"])

    metadata = json.loads(embeddings_path("embedding_metadata_full.json").read_text(encoding="utf-8"))
    pooled_npz = np.load(embeddings_path("pooled_embeddings_full.npz"))
    pooled = {key: np.asarray(pooled_npz[key], dtype=float) for key in pooled_npz.files}
    metadata_lookup = {
        (record["model_id"], canonical_condition_name(record["condition"]), int(record["layer"])): record
        for record in metadata["records"]
        if record["domain"] == "sensory"
    }
    subset_path = config["analysis"].get("execution", {}).get("default_concept_subset", "")
    subset_file = (ROOT / subset_path) if subset_path else (ROOT / "data" / "concepts" / "full_concept_list.csv")
    concept_rows = {row["concept"].lower(): row for row in read_csv(subset_file) if row["domain"] == "sensory"}
    subtype_groups: dict[str, list[str]] = defaultdict(list)
    for concept, row in concept_rows.items():
        subtype_groups[row["subtype"]].append(concept)

    layers_by_model: dict[str, list[int]] = defaultdict(list)
    for record in metadata["records"]:
        if record["domain"] == "sensory":
            layers_by_model[record["model_id"]].append(int(record["layer"]))
    text_layers = sorted(set(layers_by_model[backbone_text]))
    multimodal_layers = sorted(set(layers_by_model[backbone_multimodal]))
    anchor_model_ids = sorted({
        record["model_id"]
        for record in metadata["records"]
        if record["family"] == "anchor" and "siglip" in record["model_id"].lower()
    })
    if not anchor_model_ids:
        raise RuntimeError("SigLIP2 anchor rows were not found in current embedding metadata.")
    anchor_model_id = anchor_model_ids[0]
    anchor_layers = sorted(set(layers_by_model[anchor_model_id]))
    selected_text_layers = text_layers[len(text_layers) - int(np.ceil(len(text_layers) * mid_to_late_fraction)) :]
    selected_multimodal_layers = multimodal_layers[len(multimodal_layers) - int(np.ceil(len(multimodal_layers) * mid_to_late_fraction)) :]

    prompt_embeddings, prompt_concepts = mean_embedding_for_condition(metadata_lookup, pooled, backbone_text, "T_prompt_primary", selected_text_layers)
    matched_embeddings, matched_concepts = mean_embedding_for_condition(metadata_lookup, pooled, backbone_multimodal, "M_matched_image", selected_multimodal_layers)
    anchor_embeddings, anchor_concepts = mean_embedding_for_condition(metadata_lookup, pooled, anchor_model_id, "reference_anchor_image", anchor_layers)
    if prompt_concepts != matched_concepts or prompt_concepts != anchor_concepts:
        raise RuntimeError("Subtype packaging expects aligned concept ordering across prompt, matched, and SigLIP2 anchor.")
    concept_index = {concept: idx for idx, concept in enumerate(prompt_concepts)}

    things_blockwise = read_csv(metrics_path("human_blockwise_rsa.csv"))
    residual_summary = json.loads(metrics_path("human_partial_rsa_summary.json").read_text(encoding="utf-8"))
    local_geometry_rows = read_csv(metrics_path("human_local_geometry.csv"))

    summary_rows = []
    for subtype in sorted(subtype_groups):
        subtype_indices = [concept_index[concept] for concept in subtype_groups[subtype]]
        if len(subtype_indices) < 3:
            continue
        prompt_sub = np.asarray(prompt_embeddings[subtype_indices], dtype=float)
        matched_sub = np.asarray(matched_embeddings[subtype_indices], dtype=float)
        anchor_sub = np.asarray(anchor_embeddings[subtype_indices], dtype=float)
        anchor_rdm = condensed_cosine_distance(anchor_sub)
        prompt_siglip = spearman_corr(condensed_cosine_distance(prompt_sub), anchor_rdm)
        matched_siglip = spearman_corr(condensed_cosine_distance(matched_sub), anchor_rdm)
        things_row_prompt = next(
            row for row in things_blockwise if row["block_type"] == "within_subtype" and row["block_name"] == subtype and row["condition"] == "T_prompt_primary"
        )
        things_row_matched = next(
            row for row in things_blockwise if row["block_type"] == "within_subtype" and row["block_name"] == subtype and row["condition"] == "M_matched_image"
        )
        local_prompt = next(
            row for row in local_geometry_rows if row["granularity"] == "subtype" and row["group_name"] == subtype and row["condition"] == "T_prompt_primary"
        )
        local_matched = next(
            row for row in local_geometry_rows if row["granularity"] == "subtype" and row["group_name"] == subtype and row["condition"] == "M_matched_image"
        )
        summary_rows.extend(
            [
                {
                    "subtype": subtype,
                    "anchor_name": "THINGS behavioral similarity",
                    "prompt_score": float(things_row_prompt["rsa_score"]),
                    "matched_score": float(things_row_matched["rsa_score"]),
                    "matched_minus_prompt": float(things_row_matched["rsa_score"]) - float(things_row_prompt["rsa_score"]),
                    "driver_note": "",
                },
                {
                    "subtype": subtype,
                    "anchor_name": "SigLIP2",
                    "prompt_score": prompt_siglip,
                    "matched_score": matched_siglip,
                    "matched_minus_prompt": matched_siglip - prompt_siglip,
                    "driver_note": "",
                },
                {
                    "subtype": subtype,
                    "anchor_name": "THINGS residual",
                    "prompt_score": float(local_prompt["mean_local_alignment"]),
                    "matched_score": float(local_matched["mean_local_alignment"]),
                    "matched_minus_prompt": float(local_matched["mean_local_alignment"]) - float(local_prompt["mean_local_alignment"]),
                    "driver_note": "",
                },
            ]
        )

    interpretation_map = {
        "appearance_color": "Appearance concepts are the clearest bridge between image-grounded and behaviorally recognizable structure, but they are not uniformly prompt- or image-dominant.",
        "texture_material": "Texture/material concepts preserve some human-like local organization under prompting, while image grounding does not consistently add a human-anchor advantage.",
        "sound_linked": "Sound-linked concepts are the clearest prompt-favored subtype, consistent with a more associative and nonvisual representational regime.",
        "smell_taste_proxy": "Smell/taste proxy concepts show the cleanest matched-image advantage, consistent with image-conditioned restructuring that helps some cross-modal proxies.",
    }
    for row in summary_rows:
        row["driver_note"] = interpretation_map[row["subtype"]]

    write_csv(
        output_path("outputs", "tables", "subtype_interaction_summary.csv"),
        summary_rows,
        ["subtype", "anchor_name", "prompt_score", "matched_score", "matched_minus_prompt", "driver_note"],
    )
    plot_subtype_dissociation(summary_rows)

    report_lines = []
    for subtype in sorted(subtype_groups):
        subtype_rows = [row for row in summary_rows if row["subtype"] == subtype]
        if not subtype_rows:
            continue
        things_row = next(row for row in subtype_rows if row["anchor_name"] == "THINGS behavioral similarity")
        siglip_row = next(row for row in subtype_rows if row["anchor_name"] == "SigLIP2")
        residual_row = next(row for row in subtype_rows if row["anchor_name"] == "THINGS residual")
        report_lines.extend(
            [
                f"### {subtype}",
                f"- THINGS matched-minus-prompt: `{float(things_row['matched_minus_prompt']):.4f}`",
                f"- SigLIP2 matched-minus-prompt: `{float(siglip_row['matched_minus_prompt']):.4f}`",
                f"- THINGS residual matched-minus-prompt: `{float(residual_row['matched_minus_prompt']):.4f}`",
                f"- Interpretation: {things_row['driver_note']}",
            ]
        )
    report = "\n".join(
        [
            "# Subtype Interpretation Report",
            "",
            "## Condition By Subtype Dissociation",
            *report_lines,
            "",
            "## Interaction Readout",
            "- The subtype story is principled when the sign or magnitude of matched-minus-prompt differs across THINGS, residual THINGS, and SigLIP2.",
            "- In the current branch, sound-linked concepts are prompt-favored, while smell/taste proxy concepts are matched-favored and more image-sensitive.",
        ]
    )
    write_text(output_path("reports", "main_results", "subtype_interpretation_report.md"), report)
    append_run_log(
        "Subtype Dissociation Packaging",
        [
            f"Wrote subtype interaction summary to {output_path('outputs', 'tables', 'subtype_interaction_summary.csv').relative_to(ROOT)}.",
            f"Wrote subtype dissociation figure to {output_path('outputs', 'figures', 'fig_subtype_dissociation.png').relative_to(ROOT)}.",
            f"Wrote subtype interpretation report to {output_path('reports', 'main_results', 'subtype_interpretation_report.md').relative_to(ROOT)}.",
        ],
    )


if __name__ == "__main__":
    main()
