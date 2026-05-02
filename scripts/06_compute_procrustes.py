from __future__ import annotations

import argparse
import json

import numpy as np

from common import ROOT, append_run_log, canonical_condition_name, embeddings_path, load_project_config, metrics_path, read_csv, write_csv


COMPARISONS = [
    ("T_prompt_primary", "M_matched_image"),
    ("T_neutral", "T_prompt_primary"),
    ("M_matched_image", "M_degraded_image"),
    ("M_matched_image", "M_mismatched_image"),
    ("M_matched_image", "M_blank_image"),
]


def centered(matrix: np.ndarray) -> np.ndarray:
    return matrix - matrix.mean(axis=0, keepdims=True)


def procrustes_disparity(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = centered(left)
    right_centered = centered(right)
    norm_left = np.linalg.norm(left_centered)
    norm_right = np.linalg.norm(right_centered)
    if norm_left == 0 or norm_right == 0:
        return 0.0
    left_scaled = left_centered / norm_left
    right_scaled = right_centered / norm_right
    # The non-zero singular values of A^T B equal those of A B^T.
    # Compute them in concept space (n x n) rather than feature space (d x d).
    singular_values = np.linalg.svd(left_scaled @ right_scaled.T, compute_uv=False, full_matrices=False)
    return float(max(0.0, 2.0 - 2.0 * np.asarray(singular_values, dtype=float).sum()))


def load_subset_concepts(path: str) -> list[str]:
    subset_rows = read_csv((ROOT / path).resolve())
    return [row["concept"].lower() for row in subset_rows if row["domain"] == "sensory"]


def default_concept_subset(config: dict) -> str:
    return str(config["analysis"]["execution"]["default_concept_subset"])


def reorder_to_subset(matrix: np.ndarray, concepts: list[str], subset: list[str]) -> np.ndarray:
    index = {concept.lower(): idx for idx, concept in enumerate(concepts)}
    missing = [concept for concept in subset if concept not in index]
    if missing:
        raise RuntimeError(f"Requested subset concepts missing from embedding record: {', '.join(missing)}")
    ordered_idx = [index[concept] for concept in subset]
    return np.asarray(matrix[ordered_idx], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--concept-subset", default=None)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()

    config = load_project_config(args.config)
    arrays = np.load(embeddings_path("pooled_embeddings_full.npz"), mmap_mode="r")
    metadata = json.loads(embeddings_path("embedding_metadata_full.json").read_text(encoding="utf-8"))
    active_subset = args.concept_subset or default_concept_subset(config)
    subset_concepts = load_subset_concepts(active_subset)
    output_name = args.output_name or ("procrustes_summary_things38.csv" if args.concept_subset else "procrustes_summary.csv")
    grouped = {
        (record["model_id"], record["domain"], int(record["layer"]), canonical_condition_name(record["condition"])): {
            "matrix": arrays[f"record_{record['record_id']}"],
            "concepts": record["concepts"],
        }
        for record in metadata["records"]
    }

    rows = []
    for (model, domain, layer, _), payload in grouped.items():
        if domain != "sensory":
            continue
        for condition_a, condition_b in COMPARISONS:
            key_a = (model, domain, layer, condition_a)
            key_b = (model, domain, layer, condition_b)
            if key_a not in grouped or key_b not in grouped:
                continue
            left = np.asarray(grouped[key_a]["matrix"], dtype=float)
            right = np.asarray(grouped[key_b]["matrix"], dtype=float)
            if subset_concepts:
                left = reorder_to_subset(left, grouped[key_a]["concepts"], subset_concepts)
                right = reorder_to_subset(right, grouped[key_b]["concepts"], subset_concepts)
            rows.append(
                {
                    "model": model,
                    "layer": layer,
                    "domain": domain,
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "procrustes_disparity": procrustes_disparity(left, right),
                    "concept_subset": active_subset,
                    "bootstrap_id": "aggregate",
                }
            )

    write_csv(
        metrics_path(output_name),
        rows,
        ["model", "layer", "domain", "condition_a", "condition_b", "procrustes_disparity", "concept_subset", "bootstrap_id"],
    )
    append_run_log(
        "Procrustes",
        [
            f"Wrote Procrustes summary to {metrics_path(output_name).relative_to(config['_resolved_root'])}.",
            "Global alignment residuals are reported as support metrics rather than primary evidence.",
        ],
    )


if __name__ == "__main__":
    main()
