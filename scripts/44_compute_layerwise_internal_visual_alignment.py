from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from analysis_common import ordered_embedding_for_concepts
from common import ROOT, append_run_log, condensed_cosine_distance, embeddings_path, metrics_path, rankdata, write_csv, write_json
from hardening_common import load_embedding_bundle, mean_embedding_for_condition, write_text


FAMILY_SPECS: dict[str, dict[str, str]] = {
    "qwen": {
        "multimodal_model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "visual_cache_suffix": "_qwen",
    },
    "mistral": {
        "multimodal_model_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "visual_cache_suffix": "_mistral",
    },
    "llama": {
        "multimodal_model_id": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "visual_cache_suffix": "_llama",
    },
}

CONDITIONS = [
    "M_text_only",
    "M_matched_image",
    "M_prompt_plus_matched_image",
    "M_degraded_image",
    "M_mismatched_image",
    "M_blank_image",
]


def available_condition_layers(
    metadata_lookup: dict[tuple[str, str, int], dict[str, Any]],
    model_id: str,
    condition: str,
) -> list[int]:
    return sorted(
        layer
        for candidate_model, candidate_condition, layer in metadata_lookup
        if candidate_model == model_id and candidate_condition == condition
    )


def load_internal_visual_cache(family: str) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    suffix = FAMILY_SPECS[family]["visual_cache_suffix"]
    npz_path = embeddings_path(f"internal_visual_tower{suffix}.npz")
    json_path = embeddings_path(f"internal_visual_tower{suffix}.json")
    if not npz_path.exists() or not json_path.exists():
        raise RuntimeError(f"Missing internal visual-tower cache for {family}: {npz_path} / {json_path}")
    payload = np.load(npz_path, allow_pickle=False)
    concepts = [str(concept).lower() for concept in payload["concepts"].tolist()]
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    return embeddings, concepts, metadata


def pearson_centered(left: np.ndarray, right_centered: np.ndarray, right_norm: float) -> float:
    left_centered = left - left.mean()
    denom = float(np.linalg.norm(left_centered) * right_norm)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left_centered, right_centered) / denom)


def summarize_family(rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    fam_rows = [row for row in rows if row["family"] == family]
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in fam_rows:
        by_condition.setdefault(str(row["condition"]), []).append(row)
    for condition in by_condition:
        by_condition[condition] = sorted(by_condition[condition], key=lambda row: int(row["layer"]))

    text_rows = by_condition.get("M_text_only", [])
    matched_rows = by_condition.get("M_matched_image", [])
    summary: dict[str, Any] = {
        "num_layers": len(matched_rows),
        "conditions": {},
        "matched_minus_text_by_layer": [],
    }
    for condition, condition_rows in sorted(by_condition.items()):
        scores = np.asarray([float(row["rsa_score"]) for row in condition_rows], dtype=float)
        if scores.size == 0:
            continue
        half_start = scores.size // 2
        summary["conditions"][condition] = {
            "mean_rsa": float(scores.mean()),
            "first_layer_rsa": float(scores[0]),
            "last_layer_rsa": float(scores[-1]),
            "last_half_mean_rsa": float(scores[half_start:].mean()),
            "max_rsa": float(scores.max()),
            "max_layer": int(condition_rows[int(scores.argmax())]["layer"]),
        }

    if text_rows and matched_rows:
        text_by_layer = {int(row["layer"]): float(row["rsa_score"]) for row in text_rows}
        matched_by_layer = {int(row["layer"]): float(row["rsa_score"]) for row in matched_rows}
        common_layers = sorted(set(text_by_layer) & set(matched_by_layer))
        gaps = []
        for layer in common_layers:
            gap = matched_by_layer[layer] - text_by_layer[layer]
            gaps.append(gap)
            summary["matched_minus_text_by_layer"].append({"layer": layer, "gap": float(gap)})
        if gaps:
            gap_array = np.asarray(gaps, dtype=float)
            half_start = len(gap_array) // 2
            first_positive = next((layer for layer, gap in zip(common_layers, gaps) if gap > 0), None)
            summary["matched_minus_text"] = {
                "mean_gap": float(gap_array.mean()),
                "last_half_mean_gap": float(gap_array[half_start:].mean()),
                "last_layer_gap": float(gap_array[-1]),
                "max_gap": float(gap_array.max()),
                "max_gap_layer": int(common_layers[int(gap_array.argmax())]),
                "first_positive_layer": first_positive,
                "positive_layer_fraction": float(np.mean(gap_array > 0)),
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute layerwise language-side alignment to each VLM's internal visual-tower RDM.")
    parser.add_argument("--families", default="qwen,mistral,llama")
    args = parser.parse_args()

    requested = [item.strip() for item in args.families.split(",") if item.strip()]
    metadata_lookup, pooled, layers_by_model, _metadata = load_embedding_bundle()
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    for family in requested:
        if family not in FAMILY_SPECS:
            raise RuntimeError(f"Unknown family: {family}")
        internal_embeddings, internal_concepts, internal_metadata = load_internal_visual_cache(family)
        internal_ranked = rankdata(condensed_cosine_distance(internal_embeddings))
        internal_ranked_centered = internal_ranked - internal_ranked.mean()
        internal_ranked_norm = float(np.linalg.norm(internal_ranked_centered))
        model_id = FAMILY_SPECS[family]["multimodal_model_id"]
        model_layers = layers_by_model.get(model_id, [])
        if not model_layers:
            raise RuntimeError(f"No language-side layers found for {family}: {model_id}")
        for condition in CONDITIONS:
            layers = available_condition_layers(metadata_lookup, model_id, condition)
            if not layers:
                continue
            for layer in layers:
                embedding, concepts = mean_embedding_for_condition(metadata_lookup, pooled, model_id, condition, [layer])
                ordered = ordered_embedding_for_concepts(embedding, concepts, internal_concepts)
                score = pearson_centered(rankdata(condensed_cosine_distance(ordered)), internal_ranked_centered, internal_ranked_norm)
                rows.append(
                    {
                        "family": family,
                        "condition": condition,
                        "model_id": model_id,
                        "layer": int(layer),
                        "relative_layer": int(layer) / max(layers),
                        "rsa_score": score,
                        "num_concepts": len(internal_concepts),
                        "vision_model_id": internal_metadata.get("model_id", model_id),
                        "vision_source": internal_metadata.get("vision_source", ""),
                    }
                )
        summaries[family] = summarize_family(rows, family)

    write_csv(
        metrics_path("layerwise_internal_visual_alignment.csv"),
        rows,
        ["family", "condition", "model_id", "layer", "relative_layer", "rsa_score", "num_concepts", "vision_model_id", "vision_source"],
    )
    write_json(metrics_path("layerwise_internal_visual_alignment_summary.json"), summaries)
    lines = ["# Layerwise Internal Visual Alignment", ""]
    for family in requested:
        summary = summaries[family]
        gap = summary.get("matched_minus_text", {})
        matched = summary.get("conditions", {}).get("M_matched_image", {})
        text = summary.get("conditions", {}).get("M_text_only", {})
        lines.extend(
            [
                f"## {family}",
                f"- layers: `{summary.get('num_layers')}`",
                f"- matched mean RSA: `{matched.get('mean_rsa', float('nan')):.4f}`",
                f"- text-only mean RSA: `{text.get('mean_rsa', float('nan')):.4f}`",
                f"- matched-minus-text mean gap: `{gap.get('mean_gap', float('nan')):.4f}`",
                f"- matched-minus-text last-half gap: `{gap.get('last_half_mean_gap', float('nan')):.4f}`",
                f"- first positive matched-minus-text layer: `{gap.get('first_positive_layer')}`",
                f"- positive layer fraction: `{gap.get('positive_layer_fraction', float('nan')):.4f}`",
                "",
            ]
        )
    write_text(ROOT / "reports" / "main_results" / "layerwise_internal_visual_alignment_report.md", "\n".join(lines))
    append_run_log("Layerwise Internal Visual Alignment", [f"Computed layerwise internal visual alignment for {', '.join(requested)}."])


if __name__ == "__main__":
    main()
