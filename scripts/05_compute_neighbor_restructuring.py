from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from common import append_run_log, canonical_condition_name, embeddings_path, load_project_config, metrics_path, write_csv
from hardening_common import load_layerwise_alignment_rows


COMPARISONS = [
    ("T_neutral", "T_prompt_primary"),
    ("T_prompt_primary", "M_matched_image"),
    ("M_matched_image", "M_degraded_image"),
    ("M_matched_image", "M_mismatched_image"),
    ("M_matched_image", "M_blank_image"),
]


def top_k_neighbors(matrix: np.ndarray, k: int) -> list[list[int]]:
    sims = matrix @ matrix.T
    np.fill_diagonal(sims, -np.inf)
    return [list(np.argsort(row)[-k:][::-1]) for row in sims]


def jaccard(a: list[int], b: list[int]) -> float:
    left = set(a)
    right = set(b)
    union = left | right
    return 0.0 if not union else len(left & right) / len(union)


def mean_rank_shift(a: list[int], b: list[int]) -> float:
    rank_a = {value: idx for idx, value in enumerate(a)}
    rank_b = {value: idx for idx, value in enumerate(b)}
    shared = rank_a.keys() & rank_b.keys()
    if not shared:
        return float(len(a))
    return float(sum(abs(rank_a[item] - rank_b[item]) for item in shared) / len(shared))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    args = parser.parse_args()

    config = load_project_config(args.config)
    _ = load_layerwise_alignment_rows(args.config)
    arrays = np.load(embeddings_path("pooled_embeddings_full.npz"))
    metadata = json.loads(embeddings_path("embedding_metadata_full.json").read_text(encoding="utf-8"))
    grouped = {
        (record["model_id"], record["domain"], int(record["layer"]), canonical_condition_name(record["condition"])): arrays[f"record_{record['record_id']}"]
        for record in metadata["records"]
    }

    rows = []
    for (model, domain, layer, _), _matrix in grouped.items():
        for condition_a, condition_b in COMPARISONS:
            key_a = (model, domain, layer, condition_a)
            key_b = (model, domain, layer, condition_b)
            if key_a not in grouped or key_b not in grouped:
                continue
            for k in (5, 10):
                neighbors_a = top_k_neighbors(grouped[key_a], min(k, grouped[key_a].shape[0] - 1))
                neighbors_b = top_k_neighbors(grouped[key_b], min(k, grouped[key_b].shape[0] - 1))
                jac = [jaccard(left, right) for left, right in zip(neighbors_a, neighbors_b)]
                shifts = [mean_rank_shift(left, right) for left, right in zip(neighbors_a, neighbors_b)]
                rows.append(
                    {
                        "model": model,
                        "layer": layer,
                        "domain": domain,
                        "condition_a": condition_a,
                        "condition_b": condition_b,
                        "k": k,
                        "mean_jaccard": float(np.mean(jac)),
                        "median_jaccard": float(np.median(jac)),
                        "mean_rank_shift": float(np.mean(shifts)),
                        "bootstrap_id": "aggregate",
                    }
                )

    write_csv(
        metrics_path("neighbor_restructuring.csv"),
        rows,
        ["model", "layer", "domain", "condition_a", "condition_b", "k", "mean_jaccard", "median_jaccard", "mean_rank_shift", "bootstrap_id"],
    )
    append_run_log(
        "Neighbor Restructuring",
        [
            f"Wrote neighbor restructuring metrics to {metrics_path('neighbor_restructuring.csv').relative_to(config['_resolved_root'])}.",
            "Default neighborhood sizes k=5 and k=10 were used.",
        ],
    )


if __name__ == "__main__":
    main()
