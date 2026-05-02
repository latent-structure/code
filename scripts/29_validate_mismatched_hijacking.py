from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

import numpy as np

from common import ROOT, append_run_log, embeddings_path, metrics_path, output_path, read_csv, write_csv, write_json
from hardening_common import load_active_concept_rows, load_project_backbone, mean_embedding_for_condition, selected_layers, write_text


CONDITIONS = ["M_text_only", "M_matched_image", "M_mismatched_image", "M_blank_image"]
MARGINS = [0.0, 0.005, 0.01, 0.02]


def build_lookup(metadata: dict[str, Any]) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[str, list[int]]]:
    from common import canonical_condition_name

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
    return lookup, {model: sorted(set(layers)) for model, layers in layers_by_model.items()}


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def classify(target_distance: float, source_distance: float, margin: float) -> str:
    if source_distance + margin < target_distance:
        return "image_hijack"
    if target_distance + margin < source_distance:
        return "text_retention"
    return "ambiguous"


def concept_index(concepts: list[str]) -> dict[str, int]:
    return {concept.lower(): idx for idx, concept in enumerate(concepts)}


def rank_of_distance(distances: np.ndarray, index: int) -> int:
    return int(np.argsort(distances, kind="mergesort").tolist().index(index) + 1)


def valid_random_sources(
    target: str,
    mode: str,
    concepts: list[str],
    subtype: dict[str, str],
) -> list[str]:
    target_subtype = subtype[target]
    if mode == "within_subtype":
        return [concept for concept in concepts if concept != target and subtype[concept] == target_subtype]
    if mode == "cross_subtype":
        return [concept for concept in concepts if concept != target and subtype[concept] != target_subtype]
    return [concept for concept in concepts if concept != target]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test pair limit.")
    parser.add_argument("--seed", type=int, default=20260424)
    args = parser.parse_args()

    _config, _backbone_text, backbone_multimodal, mid_fraction = load_project_backbone(args.config)
    metadata = json.loads(embeddings_path("embedding_metadata_full.json").read_text(encoding="utf-8"))
    arrays = np.load(embeddings_path("pooled_embeddings_full.npz"))
    lookup, layers_by_model = build_lookup(metadata)
    layers = selected_layers(layers_by_model[backbone_multimodal], mid_fraction)

    matrices: dict[str, np.ndarray] = {}
    concepts_by_condition: dict[str, list[str]] = {}
    for condition in CONDITIONS:
        matrix, concepts = mean_embedding_for_condition(lookup, arrays, backbone_multimodal, condition, layers)
        matrices[condition] = normalize_rows(matrix)
        concepts_by_condition[condition] = [concept.lower() for concept in concepts]

    matched_index = concept_index(concepts_by_condition["M_matched_image"])
    mismatched_index = concept_index(concepts_by_condition["M_mismatched_image"])
    text_only_index = concept_index(concepts_by_condition["M_text_only"])
    blank_index = concept_index(concepts_by_condition["M_blank_image"])
    all_concepts = concepts_by_condition["M_matched_image"]
    subtype = {row["concept"].lower(): row["subtype"] for row in load_active_concept_rows(args.config, domain="sensory")}
    mismatch_rows = read_csv(ROOT / "data" / "manifests" / "mismatch_map.csv")
    if args.limit:
        mismatch_rows = mismatch_rows[: args.limit]
    rng = np.random.default_rng(args.seed)

    rows: list[dict[str, Any]] = []
    counts: dict[str, dict[float, defaultdict[str, int]]] = {"observed": {margin: defaultdict(int) for margin in MARGINS}, "random_null": {margin: defaultdict(int) for margin in MARGINS}}
    target_ranks = []
    source_ranks = []
    random_source_ranks = []
    for item in mismatch_rows:
        target = item["concept"].lower()
        source = item["mismatch_concept"].lower()
        mode = item.get("mismatch_mode", "")
        if target not in matched_index or target not in mismatched_index or source not in matched_index:
            continue
        mismatched_vec = matrices["M_mismatched_image"][mismatched_index[target]]
        target_vec = matrices["M_matched_image"][matched_index[target]]
        source_vec = matrices["M_matched_image"][matched_index[source]]
        distances_to_matched = 1.0 - np.clip(matrices["M_matched_image"] @ mismatched_vec, -1.0, 1.0)
        target_distance = float(distances_to_matched[matched_index[target]])
        source_distance = float(distances_to_matched[matched_index[source]])
        target_rank = rank_of_distance(distances_to_matched, matched_index[target])
        source_rank = rank_of_distance(distances_to_matched, matched_index[source])
        target_ranks.append(target_rank)
        source_ranks.append(source_rank)

        candidates = valid_random_sources(target, mode, all_concepts, subtype)
        if not candidates:
            candidates = [concept for concept in all_concepts if concept != target]
        random_source = str(rng.choice(candidates))
        random_source_distance = float(distances_to_matched[matched_index[random_source]])
        random_source_rank = rank_of_distance(distances_to_matched, matched_index[random_source])
        random_source_ranks.append(random_source_rank)

        text_only_vec = matrices["M_text_only"][text_only_index[target]]
        blank_vec = matrices["M_blank_image"][blank_index[target]]
        distance_to_text_only_target = float(1.0 - np.clip(np.dot(mismatched_vec, text_only_vec), -1.0, 1.0))
        distance_to_blank_target = float(1.0 - np.clip(np.dot(mismatched_vec, blank_vec), -1.0, 1.0))

        for margin in MARGINS:
            counts["observed"][margin][classify(target_distance, source_distance, margin)] += 1
            counts["random_null"][margin][classify(target_distance, random_source_distance, margin)] += 1
        rows.append(
            {
                "concept": target,
                "mismatch_concept": source,
                "random_mismatch_concept": random_source,
                "mismatch_mode": mode,
                "distance_to_text_target": target_distance,
                "distance_to_image_source": source_distance,
                "distance_to_random_source": random_source_distance,
                "target_rank_among_matched_anchors": target_rank,
                "source_rank_among_matched_anchors": source_rank,
                "random_source_rank_among_matched_anchors": random_source_rank,
                "mismatch_minus_text_only_target_distance": target_distance - distance_to_text_only_target,
                "mismatch_minus_blank_target_distance": target_distance - distance_to_blank_target,
            }
        )

    summary: dict[str, Any] = {
        "model_id": backbone_multimodal,
        "num_pairs": len(rows),
        "seed": args.seed,
        "mean_target_rank": float(np.mean(target_ranks)) if target_ranks else 0.0,
        "median_target_rank": float(np.median(target_ranks)) if target_ranks else 0.0,
        "mean_source_rank": float(np.mean(source_ranks)) if source_ranks else 0.0,
        "median_source_rank": float(np.median(source_ranks)) if source_ranks else 0.0,
        "mean_random_source_rank": float(np.mean(random_source_ranks)) if random_source_ranks else 0.0,
        "margin_sensitivity": {},
    }
    for label, by_margin in counts.items():
        summary["margin_sensitivity"][label] = {}
        for margin, margin_counts in by_margin.items():
            total = sum(margin_counts.values())
            summary["margin_sensitivity"][label][str(margin)] = {
                "image_hijack_rate": 0.0 if total == 0 else margin_counts["image_hijack"] / total,
                "text_retention_rate": 0.0 if total == 0 else margin_counts["text_retention"] / total,
                "ambiguous_rate": 0.0 if total == 0 else margin_counts["ambiguous"] / total,
            }

    suffix = "_smoke" if args.limit else ""
    write_csv(
        metrics_path(f"mismatched_hijacking_validation{suffix}.csv"),
        rows,
        [
            "concept",
            "mismatch_concept",
            "random_mismatch_concept",
            "mismatch_mode",
            "distance_to_text_target",
            "distance_to_image_source",
            "distance_to_random_source",
            "target_rank_among_matched_anchors",
            "source_rank_among_matched_anchors",
            "random_source_rank_among_matched_anchors",
            "mismatch_minus_text_only_target_distance",
            "mismatch_minus_blank_target_distance",
        ],
    )
    write_json(metrics_path(f"mismatched_hijacking_validation_summary{suffix}.json"), summary)
    observed_001 = summary["margin_sensitivity"]["observed"]["0.01"]
    lines = [
        "# Mismatched-Image Hijacking Validation Report",
        "",
        f"- Pairs analyzed: `{summary['num_pairs']}`",
        f"- Median target rank among matched anchors: `{summary['median_target_rank']:.2f}`",
        f"- Median source rank among matched anchors: `{summary['median_source_rank']:.2f}`",
        f"- Observed text-retention rate at margin 0.01: `{observed_001['text_retention_rate']:.4f}`",
        f"- Observed image-hijack rate at margin 0.01: `{observed_001['image_hijack_rate']:.4f}`",
        "- Robustness checks include rank among all matched anchors, random source nulls, and margin sensitivity.",
    ]
    write_text(output_path("reports", "main_results", f"mismatched_hijacking_validation_report{suffix}.md"), "\n".join(lines))
    append_run_log("Mismatched Hijacking Validation", [f"Wrote hijacking validation outputs with suffix `{suffix}`."])


if __name__ == "__main__":
    main()
