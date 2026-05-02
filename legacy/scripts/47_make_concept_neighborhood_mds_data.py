#!/usr/bin/env python3
"""Build compact concept-neighborhood MDS inputs for paper figures.

The output is intentionally small: for a few selected target concepts, it stores
condition-specific 2D classical-MDS coordinates for the target, its mismatched
source, and nearest neighbors under each condition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


QWEN_TEXT = "Qwen/Qwen3.5-9B"
QWEN_VLM = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_CONDITIONS = (
    "T_prompt_primary",
    "M_text_only",
    "M_matched_image",
    "M_mismatched_image",
)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def cosine_distance_matrix(x: np.ndarray) -> np.ndarray:
    z = l2_normalize(x.astype(np.float64, copy=False))
    d = 1.0 - z @ z.T
    np.fill_diagonal(d, 0.0)
    return np.clip(d, 0.0, 2.0)


def classical_mds(dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = dist.shape[0]
    d2 = dist**2
    h = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * h @ d2 @ h
    vals, vecs = np.linalg.eigh(b)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    pos = np.maximum(vals[:2], 0.0)
    coords = vecs[:, :2] * np.sqrt(pos)
    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
    return coords, vals


def load_qwen_records(metadata_path: Path, npz_path: Path) -> tuple[list[dict], np.lib.npyio.NpzFile]:
    meta = json.loads(metadata_path.read_text())
    all_records = meta["records"]
    qwen_records = [
        r
        for r in all_records
        if r.get("model_id") in {QWEN_TEXT, QWEN_VLM}
        or str(r.get("model_id", "")).startswith("google/siglip2")
    ]
    z = np.load(npz_path, allow_pickle=False)
    if len(qwen_records) != len(z.files):
        raise ValueError(
            f"Qwen record count mismatch: metadata={len(qwen_records)} npz={len(z.files)}"
        )
    return qwen_records, z


def pooled_condition_embeddings(
    records: list[dict],
    z: np.lib.npyio.NpzFile,
    condition: str,
    model_id: str,
) -> tuple[np.ndarray, list[str], list[int]]:
    matches = [
        (i, r)
        for i, r in enumerate(records)
        if r.get("condition") == condition and r.get("model_id") == model_id
    ]
    if not matches:
        raise ValueError(f"No records for {model_id} {condition}")
    layers = sorted({int(r["layer"]) for _, r in matches})
    start = layers[len(layers) // 2]
    selected = [(i, r) for i, r in matches if int(r["layer"]) >= start]
    arrs = [z[f"record_{i}"] for i, _ in selected]
    emb = np.mean(np.stack(arrs, axis=0), axis=0)
    concepts = selected[0][1]["concepts"]
    return emb.astype(np.float32), concepts, [int(r["layer"]) for _, r in selected]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="outputs/embeddings/embedding_metadata_full.json")
    ap.add_argument("--embeddings", default="outputs/embeddings/pooled_embeddings_qwen.npz")
    ap.add_argument("--concepts", default="data/concepts/things_max_1854_concepts.csv")
    ap.add_argument("--image-manifest", default="data/manifests/image_manifest.csv")
    ap.add_argument("--mismatch-map", default="data/manifests/mismatch_map.csv")
    ap.add_argument("--behavior", default="outputs/metrics/behavior_bridge_extensions_full_implicit_leakage.csv")
    ap.add_argument("--output", default="figures_data/derived/concept_neighborhood_mds.csv")
    ap.add_argument("--summary-output", default="figures_data/derived/concept_neighborhood_mds_summary.json")
    ap.add_argument("--targets", nargs="*", default=["amplifier", "goose", "apple"])
    ap.add_argument("--top-behavior-examples", type=int, default=2)
    ap.add_argument("--neighbors", type=int, default=18)
    args = ap.parse_args()

    records, z = load_qwen_records(Path(args.metadata), Path(args.embeddings))
    concept_df = pd.read_csv(args.concepts)
    image_df = pd.read_csv(args.image_manifest)
    mismatch_df = pd.read_csv(args.mismatch_map)
    behavior_df = pd.read_csv(args.behavior) if Path(args.behavior).exists() else pd.DataFrame()

    subtype = concept_df.set_index("concept")["subtype"].to_dict()
    image_path = image_df.set_index("concept")["matched_image"].to_dict()
    mismatch_source = mismatch_df.set_index("concept")["mismatch_concept"].to_dict()

    condition_models = {
        "T_prompt_primary": QWEN_TEXT,
        "M_text_only": QWEN_VLM,
        "M_matched_image": QWEN_VLM,
        "M_mismatched_image": QWEN_VLM,
    }
    embeddings: dict[str, np.ndarray] = {}
    layers_used: dict[str, list[int]] = {}
    concepts_ref: list[str] | None = None
    for cond, model in condition_models.items():
        emb, concepts, layers = pooled_condition_embeddings(records, z, cond, model)
        if concepts_ref is None:
            concepts_ref = concepts
        elif concepts != concepts_ref:
            raise ValueError(f"Concept order mismatch for {cond}")
        embeddings[cond] = l2_normalize(emb)
        layers_used[cond] = layers
    assert concepts_ref is not None
    concept_to_idx = {c: i for i, c in enumerate(concepts_ref)}

    targets: list[str] = [t for t in args.targets if t in concept_to_idx]
    if not behavior_df.empty and args.top_behavior_examples > 0:
        # Add examples where mismatch output is strongly source-like, but avoid duplicates.
        b = behavior_df.sort_values(
            ["source_minus_target_description_similarity", "source_attraction"],
            ascending=False,
        )
        for c in b["concept"].tolist():
            if c in concept_to_idx and c not in targets:
                targets.append(c)
            if len(targets) >= len(args.targets) + args.top_behavior_examples:
                break

    rows = []
    summaries = []
    for target in targets:
        tidx = concept_to_idx[target]
        source = mismatch_source.get(target)
        source_idx = concept_to_idx.get(source) if isinstance(source, str) else None

        union: set[int] = {tidx}
        if source_idx is not None:
            union.add(source_idx)
        neighbor_by_condition: dict[str, set[int]] = {}

        for cond, emb in embeddings.items():
            d = 1.0 - emb @ emb[tidx]
            order = np.argsort(d)
            neigh = [int(i) for i in order if int(i) != tidx][: args.neighbors]
            neighbor_by_condition[cond] = set(neigh)
            union.update(neigh)

        union_idx = sorted(union)
        for cond, emb in embeddings.items():
            sub = emb[union_idx]
            dist = cosine_distance_matrix(sub)
            coords, eigvals = classical_mds(dist)
            target_dist = 1.0 - emb @ emb[tidx]
            ranks = np.empty_like(target_dist, dtype=np.int64)
            ranks[np.argsort(target_dist)] = np.arange(1, len(target_dist) + 1)

            for local_i, global_i in enumerate(union_idx):
                concept = concepts_ref[global_i]
                rows.append(
                    {
                        "example_target": target,
                        "condition": cond,
                        "concept": concept,
                        "subtype": subtype.get(concept, ""),
                        "x": float(coords[local_i, 0]),
                        "y": float(coords[local_i, 1]),
                        "distance_to_target": float(target_dist[global_i]),
                        "rank_to_target": int(ranks[global_i]),
                        "is_target": concept == target,
                        "is_mismatch_source": concept == source,
                        "selected_as_neighbor_in_condition": global_i in neighbor_by_condition[cond],
                        "mismatch_source": source,
                        "matched_image": image_path.get(concept, ""),
                        "target_image": image_path.get(target, ""),
                        "mismatch_source_image": image_path.get(source, "") if isinstance(source, str) else "",
                    }
                )

            summaries.append(
                {
                    "example_target": target,
                    "condition": cond,
                    "num_points": len(union_idx),
                    "num_condition_neighbors": len(neighbor_by_condition[cond]),
                    "mismatch_source": source,
                    "mismatch_source_rank": int(ranks[source_idx]) if source_idx is not None else None,
                    "mismatch_source_distance": float(target_dist[source_idx]) if source_idx is not None else None,
                    "mds_eigenvalue_1": float(eigvals[0]) if len(eigvals) else 0.0,
                    "mds_eigenvalue_2": float(eigvals[1]) if len(eigvals) > 1 else 0.0,
                    "layers_used": layers_used[cond],
                }
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    Path(args.summary_output).write_text(json.dumps({"targets": targets, "rows": summaries}, indent=2))
    print(f"Wrote {out} with {len(rows)} rows for {len(targets)} target concepts.")
    print(f"Wrote {args.summary_output}.")


if __name__ == "__main__":
    main()
