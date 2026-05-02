from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from common import (
    ROOT,
    append_run_log,
    canonical_condition_name,
    condensed_cosine_distance,
    load_project_config,
    metrics_path,
    read_csv,
    spearman_corr,
    write_csv,
)
from hardening_common import load_layerwise_alignment_rows


BACKBONE_CONDITIONS = [
    "T_neutral",
    "T_prompt_primary",
    "M_text_only",
    "M_matched_image",
    "M_degraded_image",
    "M_mismatched_image",
    "M_blank_image",
]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_anchor(scores: dict[str, float]) -> tuple[bool, str]:
    required = ["T_neutral", "T_prompt_primary", "M_text_only", "M_matched_image", "M_degraded_image", "M_mismatched_image"]
    missing = [condition for condition in required if condition not in scores]
    if missing:
        return False, f"missing_conditions:{','.join(missing)}"

    violations: list[str] = []
    if not scores["T_neutral"] < scores["T_prompt_primary"]:
        violations.append("text_prompt_not_above_neutral")
    if not scores["T_prompt_primary"] < scores["M_matched_image"]:
        violations.append("matched_not_above_prompt")
    if not scores["M_text_only"] < scores["M_matched_image"]:
        violations.append("matched_not_above_multimodal_text_only")
    if not scores["M_matched_image"] > scores["M_degraded_image"]:
        violations.append("degraded_not_below_matched")
    if not scores["M_degraded_image"] > scores["M_mismatched_image"]:
        violations.append("mismatched_not_below_degraded")
    if "M_blank_image" in scores and not scores["M_matched_image"] > scores["M_blank_image"]:
        violations.append("blank_not_below_matched")
    return (not violations), ";".join(violations) if violations else "ok"


def squareform(condensed: np.ndarray) -> np.ndarray:
    condensed = np.asarray(condensed, dtype=float)
    n = int((1 + np.sqrt(1 + 8 * len(condensed))) / 2)
    matrix = np.zeros((n, n), dtype=float)
    tri = np.triu_indices(n, k=1)
    matrix[tri] = condensed
    matrix[(tri[1], tri[0])] = condensed
    return matrix


def load_metadata_and_embeddings() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    metadata = json.loads((ROOT / "outputs" / "embeddings" / "embedding_metadata_full.json").read_text(encoding="utf-8"))
    payload = np.load(ROOT / "outputs" / "embeddings" / "pooled_embeddings_full.npz")
    embeddings = {key: np.asarray(payload[key], dtype=float) for key in payload.files}
    return metadata, embeddings


def load_anchor_embeddings(anchor_name: str) -> tuple[np.ndarray | None, list[str]]:
    mapping = {
        "DINOv2": ("data/anchors/dinov2_embeddings.npy", "data/anchors/dinov2_concepts.json"),
        "CLIP ViT-L/14": ("data/anchors/clip_vitl14_embeddings.npy", "data/anchors/clip_vitl14_concepts.json"),
    }
    if anchor_name not in mapping:
        return None, []
    emb_path, concept_path = mapping[anchor_name]
    embeddings = np.load(ROOT / emb_path)
    concepts = json.loads((ROOT / concept_path).read_text(encoding="utf-8"))
    return np.asarray(embeddings, dtype=float), concepts


def record_lookup(metadata: dict[str, object]) -> dict[tuple[str, str, int], dict[str, object]]:
    lookup = {}
    for record in metadata["records"]:
        lookup[(record["model_id"], canonical_condition_name(record["condition"]), int(record["layer"]))] = record
    return lookup


def compute_reversal_rows(config: dict[str, object], anchor_names: list[str]) -> list[dict[str, object]]:
    metadata, embeddings = load_metadata_and_embeddings()
    lookup = record_lookup(metadata)
    backbone_multimodal = config["analysis"]["execution"]["sensory_backbone_multimodal_model"]
    layers = sorted(
        {
            int(record["layer"])
            for record in metadata["records"]
            if record["model_id"] == backbone_multimodal and record["domain"] == "sensory"
        }
    )
    selected_layers = layers[len(layers) // 2 :]
    image_manifest = {row["concept"]: row for row in read_csv(ROOT / "data" / "manifests" / "image_manifest.csv")}

    out_rows: list[dict[str, object]] = []
    for anchor_name in anchor_names:
        anchor_embeddings, anchor_concepts = load_anchor_embeddings(anchor_name)
        if anchor_embeddings is None:
            continue
        anchor_index = {concept: idx for idx, concept in enumerate(anchor_concepts)}
        matched_mats = []
        degraded_mats = []
        concept_order: list[str] | None = None
        for layer in selected_layers:
            matched_record = lookup.get((backbone_multimodal, "M_matched_image", layer))
            degraded_record = lookup.get((backbone_multimodal, "M_degraded_image", layer))
            if matched_record is None or degraded_record is None:
                continue
            concepts = matched_record["concepts"]
            if any(concept not in anchor_index for concept in concepts):
                continue
            concept_order = list(concepts)
            matched_mats.append(np.asarray(embeddings[f"record_{matched_record['record_id']}"], dtype=float))
            degraded_mats.append(np.asarray(embeddings[f"record_{degraded_record['record_id']}"], dtype=float))
        if not matched_mats or concept_order is None:
            continue

        matched_avg = np.mean(np.stack(matched_mats), axis=0)
        degraded_avg = np.mean(np.stack(degraded_mats), axis=0)
        anchor_avg = anchor_embeddings[[anchor_index[concept] for concept in concept_order]]

        matched_dist = squareform(condensed_cosine_distance(matched_avg))
        degraded_dist = squareform(condensed_cosine_distance(degraded_avg))
        anchor_dist = squareform(condensed_cosine_distance(anchor_avg))

        for idx, concept in enumerate(concept_order):
            mask = np.ones(len(concept_order), dtype=bool)
            mask[idx] = False
            matched_score = spearman_corr(matched_dist[idx, mask], anchor_dist[idx, mask])
            degraded_score = spearman_corr(degraded_dist[idx, mask], anchor_dist[idx, mask])
            out_rows.append(
                {
                    "anchor_name": anchor_name,
                    "concept": concept,
                    "matched_score": matched_score,
                    "degraded_score": degraded_score,
                    "degraded_minus_matched": degraded_score - matched_score,
                    "matched_image": image_manifest.get(concept, {}).get("matched_image", ""),
                }
            )

    out_rows.sort(key=lambda row: (row["anchor_name"], -float(row["degraded_minus_matched"])))
    return out_rows


def compute_layer_support_rows(
    rows: list[dict[str, str]],
    backbone_text: str,
    backbone_multimodal: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        model = row.get("model", row.get("model_id", ""))
        if model not in {backbone_text, backbone_multimodal}:
            continue
        anchor_name = row.get("anchor_name", row.get("anchor_model_id", ""))
        anchor_type = row.get("anchor_type", "vision_language")
        grouped[(anchor_name, anchor_type, int(row["layer"]))][canonical_condition_name(row["condition"])].append(float(row["rsa_score"]))

    layer_rows: list[dict[str, object]] = []
    by_anchor: dict[tuple[str, str], list[tuple[int, bool]]] = defaultdict(list)
    for (anchor_name, anchor_type, layer), condition_scores in sorted(grouped.items()):
        means = {condition: mean(scores) for condition, scores in condition_scores.items()}
        support, notes = evaluate_anchor(means)
        layer_rows.append(
            {
                "anchor_name": anchor_name,
                "anchor_type": anchor_type,
                "layer": layer,
                "supports_anchor_ordering": support,
                "violation_notes": notes,
                "T_neutral": means.get("T_neutral", ""),
                "T_prompt_primary": means.get("T_prompt_primary", ""),
                "M_text_only": means.get("M_text_only", ""),
                "M_matched_image": means.get("M_matched_image", ""),
                "M_degraded_image": means.get("M_degraded_image", ""),
                "M_mismatched_image": means.get("M_mismatched_image", ""),
                "M_blank_image": means.get("M_blank_image", ""),
            }
        )
        by_anchor[(anchor_name, anchor_type)].append((layer, support))

    summary_rows: list[dict[str, object]] = []
    for (anchor_name, anchor_type), support_pairs in sorted(by_anchor.items()):
        support_pairs.sort()
        best_run = 0
        current_run = 0
        best_end = None
        supported_layer_count = 0
        for layer, supported in support_pairs:
            if supported:
                supported_layer_count += 1
                current_run += 1
                if current_run > best_run:
                    best_run = current_run
                    best_end = layer
            else:
                current_run = 0
        if best_end is None:
            band = ""
            status = "fails_all_layers"
        else:
            band = f"{best_end - best_run + 1}-{best_end}"
            status = "passes_aggregate" if supported_layer_count == len(support_pairs) else "passes_some_layers_but_not_aggregate"
        summary_rows.append(
            {
                "anchor_name": anchor_name,
                "anchor_type": anchor_type,
                "supported_layer_count": supported_layer_count,
                "total_layer_count": len(support_pairs),
                "best_contiguous_support_band": band,
                "best_contiguous_support_length": best_run,
                "layer_support_status": status,
            }
        )
    return layer_rows, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    args = parser.parse_args()

    config = load_project_config(args.config)
    backbone_text = config["analysis"]["execution"]["sensory_backbone_text_model"]
    backbone_multimodal = config["analysis"]["execution"]["sensory_backbone_multimodal_model"]
    rows = [
        row
        for row in load_layerwise_alignment_rows(args.config)
        if row["bootstrap_id"] == "aggregate" and row["domain"] == "sensory"
    ]

    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        model = row.get("model", row.get("model_id", ""))
        if model not in {backbone_text, backbone_multimodal}:
            continue
        anchor_name = row.get("anchor_name", row.get("anchor_model_id", ""))
        anchor_type = row.get("anchor_type", "vision_language")
        grouped[(anchor_name, anchor_type)][canonical_condition_name(row["condition"])].append(float(row["rsa_score"]))

    output_rows = []
    anchor_support_rows = []
    pair_id = f"{backbone_text}__vs__{backbone_multimodal}"
    for (anchor_name, anchor_type), condition_scores in sorted(grouped.items()):
        means = {condition: mean(scores) for condition, scores in condition_scores.items()}
        ranked = sorted(means.items(), key=lambda item: item[1], reverse=True)
        rank_map = {condition: rank + 1 for rank, (condition, _) in enumerate(ranked)}
        support, violations = evaluate_anchor(means)
        anchor_support_rows.append(
            {
                "anchor_name": anchor_name,
                "anchor_type": anchor_type,
                "supports_anchor_ordering": support,
                "violation_notes": violations,
            }
        )
        for condition in BACKBONE_CONDITIONS:
            if condition not in means:
                continue
            output_rows.append(
                {
                    "pair_id": pair_id,
                    "text_model": backbone_text,
                    "multimodal_model": backbone_multimodal,
                    "condition": condition,
                    "layer_band": config["analysis"]["execution"]["anchor_layer_band"],
                    "anchor_name": anchor_name,
                    "anchor_type": anchor_type,
                    "mean_rsa": means[condition],
                    "ordering_rank": rank_map[condition],
                    "supports_anchor_ordering": support,
                    "violation_notes": violations,
                }
            )

    write_csv(
        metrics_path("anchor_robustness.csv"),
        output_rows,
        [
            "pair_id",
            "text_model",
            "multimodal_model",
            "condition",
            "layer_band",
            "anchor_name",
            "anchor_type",
            "mean_rsa",
            "ordering_rank",
            "supports_anchor_ordering",
            "violation_notes",
        ],
    )
    write_csv(
        metrics_path("anchor_support_summary.csv"),
        anchor_support_rows,
        ["anchor_name", "anchor_type", "supports_anchor_ordering", "violation_notes"],
    )

    layer_rows, layer_summary_rows = compute_layer_support_rows(rows, backbone_text, backbone_multimodal)
    write_csv(
        metrics_path("anchor_layer_support.csv"),
        layer_rows,
        [
            "anchor_name",
            "anchor_type",
            "layer",
            "supports_anchor_ordering",
            "violation_notes",
            "T_neutral",
            "T_prompt_primary",
            "M_text_only",
            "M_matched_image",
            "M_degraded_image",
            "M_mismatched_image",
            "M_blank_image",
        ],
    )
    write_csv(
        metrics_path("anchor_layer_band_summary.csv"),
        layer_summary_rows,
        [
            "anchor_name",
            "anchor_type",
            "supported_layer_count",
            "total_layer_count",
            "best_contiguous_support_band",
            "best_contiguous_support_length",
            "layer_support_status",
        ],
    )

    reversal_rows = compute_reversal_rows(config, ["DINOv2", "CLIP ViT-L/14"])
    write_csv(
        metrics_path("degraded_vs_matched_reversals.csv"),
        reversal_rows,
        ["anchor_name", "concept", "matched_score", "degraded_score", "degraded_minus_matched", "matched_image"],
    )

    append_run_log(
        "Anchor Robustness",
        [
            f"Wrote matched-family anchor robustness summary to {metrics_path('anchor_robustness.csv').relative_to(config['_resolved_root'])}.",
            f"Anchors supporting the main ordering: {sum(1 for row in anchor_support_rows if row['supports_anchor_ordering'])}/{len(anchor_support_rows)}.",
            f"Wrote layerwise anchor support diagnostics to {metrics_path('anchor_layer_support.csv').relative_to(config['_resolved_root'])} and {metrics_path('anchor_layer_band_summary.csv').relative_to(config['_resolved_root'])}.",
            f"Wrote concept-level degraded-vs-matched reversals to {metrics_path('degraded_vs_matched_reversals.csv').relative_to(config['_resolved_root'])}.",
        ],
    )


if __name__ == "__main__":
    main()
