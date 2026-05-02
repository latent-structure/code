from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

import numpy as np

from common import (
    ROOT,
    append_run_log,
    canonical_condition_name,
    embeddings_path,
    metrics_path,
    output_path,
    read_csv,
    write_csv,
    write_json,
)
from hardening_common import load_project_backbone, selected_layers, write_text


def suffix(limit: int) -> str:
    return "_smoke" if limit else ""


def build_metadata_lookup(metadata: dict[str, Any]) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[str, list[int]]]:
    lookup = {
        (record["model_id"], canonical_condition_name(record["condition"]), int(record["layer"])): record
        for record in metadata["records"]
        if record["domain"] == "sensory"
    }
    layers_by_model: dict[str, list[int]] = {}
    for record in metadata["records"]:
        if record["domain"] != "sensory":
            continue
        layers_by_model.setdefault(record["model_id"], []).append(int(record["layer"]))
    for model_id in list(layers_by_model):
        layers_by_model[model_id] = sorted(set(layers_by_model[model_id]))
    return lookup, layers_by_model


def mean_embedding_for_condition(
    arrays: Any,
    lookup: dict[tuple[str, str, int], dict[str, Any]],
    model_id: str,
    condition: str,
    layers: list[int],
) -> tuple[np.ndarray, list[str]]:
    matrices = []
    concepts: list[str] | None = None
    for layer in layers:
        record = lookup.get((model_id, condition, int(layer)))
        if record is None:
            continue
        if concepts is None:
            concepts = [concept.lower() for concept in record["concepts"]]
        matrices.append(np.asarray(arrays[f"record_{record['record_id']}"], dtype=np.float32))
    if not matrices or concepts is None:
        raise RuntimeError(f"Missing embeddings for {model_id} {condition}")
    return np.mean(np.stack(matrices), axis=0, dtype=np.float32), concepts


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(1.0 - np.clip(np.dot(left, right), -1.0, 1.0))


def classify_shift(target_distance: float, source_distance: float, margin: float) -> str:
    if source_distance + margin < target_distance:
        return "image_hijack"
    if target_distance + margin < source_distance:
        return "text_retention"
    return "ambiguous"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test row limit. Writes *_smoke outputs.")
    parser.add_argument("--margin", type=float, default=0.01)
    args = parser.parse_args()

    _config, _backbone_text, backbone_multimodal, mid_fraction = load_project_backbone(args.config)
    metadata = json.loads(embeddings_path("embedding_metadata_full.json").read_text(encoding="utf-8"))
    arrays = np.load(embeddings_path("pooled_embeddings_full.npz"))
    lookup, layers_by_model = build_metadata_lookup(metadata)
    layers = selected_layers(layers_by_model[backbone_multimodal], mid_fraction)

    matched_matrix, matched_concepts = mean_embedding_for_condition(
        arrays,
        lookup,
        backbone_multimodal,
        "M_matched_image",
        layers,
    )
    mismatched_matrix, mismatched_concepts = mean_embedding_for_condition(
        arrays,
        lookup,
        backbone_multimodal,
        "M_mismatched_image",
        layers,
    )
    matched_matrix = normalize_rows(matched_matrix)
    mismatched_matrix = normalize_rows(mismatched_matrix)
    matched_index = {concept.lower(): idx for idx, concept in enumerate(matched_concepts)}
    mismatched_index = {concept.lower(): idx for idx, concept in enumerate(mismatched_concepts)}

    mismatch_rows = read_csv(ROOT / "data" / "manifests" / "mismatch_map.csv")
    if args.limit:
        mismatch_rows = mismatch_rows[: args.limit]

    rows = []
    counts_by_mode: dict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    margins_by_mode: dict[str, list[float]] = defaultdict(list)
    for item in mismatch_rows:
        target = item["concept"].lower()
        source = item["mismatch_concept"].lower()
        mode = item.get("mismatch_mode", "")
        if target not in matched_index or target not in mismatched_index or source not in matched_index:
            continue
        mismatched_vec = mismatched_matrix[mismatched_index[target]]
        target_vec = matched_matrix[matched_index[target]]
        source_vec = matched_matrix[matched_index[source]]
        target_distance = cosine_distance(mismatched_vec, target_vec)
        source_distance = cosine_distance(mismatched_vec, source_vec)
        source_minus_target = source_distance - target_distance
        label = classify_shift(target_distance, source_distance, args.margin)
        counts_by_mode[mode][label] += 1
        counts_by_mode["all"][label] += 1
        margins_by_mode[mode].append(source_minus_target)
        margins_by_mode["all"].append(source_minus_target)
        rows.append(
            {
                "concept": target,
                "mismatch_concept": source,
                "mismatch_mode": mode,
                "distance_to_text_target": target_distance,
                "distance_to_image_source": source_distance,
                "source_minus_target_distance": source_minus_target,
                "shift_label": label,
            }
        )

    summary: dict[str, Any] = {
        "model_id": backbone_multimodal,
        "num_pairs": len(rows),
        "margin": args.margin,
        "modes": {},
    }
    for mode, counts in counts_by_mode.items():
        total = sum(counts.values())
        margins = np.asarray(margins_by_mode[mode], dtype=float)
        summary["modes"][mode] = {
            "num_pairs": total,
            "image_hijack_rate": 0.0 if total == 0 else counts["image_hijack"] / total,
            "text_retention_rate": 0.0 if total == 0 else counts["text_retention"] / total,
            "ambiguous_rate": 0.0 if total == 0 else counts["ambiguous"] / total,
            "mean_source_minus_target_distance": 0.0 if len(margins) == 0 else float(margins.mean()),
            "median_source_minus_target_distance": 0.0 if len(margins) == 0 else float(np.median(margins)),
        }

    out_suffix = suffix(args.limit)
    write_csv(
        metrics_path(f"mismatched_hijacking{out_suffix}.csv"),
        rows,
        [
            "concept",
            "mismatch_concept",
            "mismatch_mode",
            "distance_to_text_target",
            "distance_to_image_source",
            "source_minus_target_distance",
            "shift_label",
        ],
    )
    write_json(metrics_path(f"mismatched_hijacking_summary{out_suffix}.json"), summary)

    all_summary = summary["modes"].get("all", {})
    lines = [
        "# Mismatched-Image Hijacking Report",
        "",
        "## Summary",
        f"- Model: `{summary['model_id']}`",
        f"- Pairs analyzed: `{summary['num_pairs']}`",
        f"- Image-hijack rate: `{float(all_summary.get('image_hijack_rate', 0.0)):.4f}`",
        f"- Text-retention rate: `{float(all_summary.get('text_retention_rate', 0.0)):.4f}`",
        f"- Ambiguous rate: `{float(all_summary.get('ambiguous_rate', 0.0)):.4f}`",
        f"- Mean source-minus-target distance: `{float(all_summary.get('mean_source_minus_target_distance', 0.0)):.4f}`",
        "",
        "## By Mismatch Mode",
    ]
    for mode, payload in sorted(summary["modes"].items()):
        if mode == "all":
            continue
        lines.append(
            f"- `{mode}` image_hijack=`{payload['image_hijack_rate']:.4f}` "
            f"text_retention=`{payload['text_retention_rate']:.4f}` ambiguous=`{payload['ambiguous_rate']:.4f}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- Negative source-minus-target distance means the mismatched representation is closer to the image-source concept than to the text-target concept.",
            "- This analysis tests whether mismatched visual input hijacks the language-side concept state or whether text identity remains dominant.",
        ]
    )
    write_text(output_path("reports", "main_results", f"mismatched_hijacking_report{out_suffix}.md"), "\n".join(lines))
    append_run_log(
        "Mismatched-Image Hijacking",
        [
            f"Wrote mismatched-image hijacking rows to {metrics_path(f'mismatched_hijacking{out_suffix}.csv').relative_to(ROOT)}.",
            f"Wrote mismatched-image hijacking report to {output_path('reports', 'main_results', f'mismatched_hijacking_report{out_suffix}.md').relative_to(ROOT)}.",
        ],
    )


if __name__ == "__main__":
    main()
