from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import ROOT, append_run_log, read_csv, write_csv, write_json
from hardening_common import THINGS_BEHAVIOR_CONCEPTS, THINGS_BEHAVIOR_MATRIX


def normalize_concept(name: str) -> str:
    return str(name).strip().lower()


def things_lookup() -> tuple[np.ndarray, list[str], dict[str, int]]:
    matrix = np.load(THINGS_BEHAVIOR_MATRIX)
    concepts = [normalize_concept(c) for c in json.loads(THINGS_BEHAVIOR_CONCEPTS.read_text(encoding="utf-8"))]
    return matrix, concepts, {concept: idx for idx, concept in enumerate(concepts)}


def choose_neighbor(
    anchor: str,
    opposite: str,
    concepts: list[str],
    index: dict[str, int],
    similarity: np.ndarray,
    excluded: set[str],
) -> tuple[str, float, float, float]:
    """Pick a neighbor close to anchor but relatively far from opposite."""

    anchor_idx = index[anchor]
    opposite_idx = index[opposite]
    best: tuple[float, str, float, float] | None = None
    for candidate in concepts:
        if candidate in excluded or candidate not in index:
            continue
        cand_idx = index[candidate]
        sim_anchor = float(similarity[anchor_idx, cand_idx])
        sim_opposite = float(similarity[opposite_idx, cand_idx])
        score = sim_anchor - sim_opposite
        item = (score, candidate, sim_anchor, sim_opposite)
        if best is None or item > best:
            best = item
    if best is None:
        raise RuntimeError(f"No valid neighbor candidate for anchor={anchor} opposite={opposite}")
    score, candidate, sim_anchor, sim_opposite = best
    return candidate, sim_anchor, sim_opposite, score


def load_pair_image_similarity() -> dict[tuple[str, str], tuple[float, str]]:
    path = ROOT / "outputs" / "metrics" / "clip_forced_choice_behavior.csv"
    if not path.exists():
        return {}
    rows = pd.read_csv(path)
    rows = rows[rows["condition"].eq("M_mismatched_image")]
    out: dict[tuple[str, str], tuple[float, str]] = {}
    for _, row in rows.iterrows():
        key = (normalize_concept(row["concept"]), normalize_concept(row["mismatch_source"]))
        out[key] = (float(row["pair_image_similarity"]), str(row["pair_difficulty"]))
    return out


def load_source_attraction() -> dict[tuple[str, str], tuple[float, float]]:
    path = ROOT / "outputs" / "metrics" / "behavior_bridge_extensions_full_implicit_leakage.csv"
    if not path.exists():
        return {}
    out: dict[tuple[str, str], tuple[float, float]] = {}
    for row in read_csv(path):
        key = (normalize_concept(row["concept"]), normalize_concept(row["mismatch_source"]))
        out[key] = (float(row["source_attraction"]), float(row["source_minus_target_margin"]))
    return out


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    similarity, things_concepts, things_index = things_lookup()
    concept_rows = [
        row for row in read_csv(ROOT / args.concepts)
        if normalize_concept(row.get("domain", "sensory")) == "sensory"
    ]
    active = {normalize_concept(row["concept"]): row for row in concept_rows}
    active_names = sorted(active)

    mismatch_rows = read_csv(ROOT / args.mismatch_map)
    mismatch = {
        normalize_concept(row["concept"]): normalize_concept(row["mismatch_concept"])
        for row in mismatch_rows
    }
    pair_image = load_pair_image_similarity()
    source_attraction = load_source_attraction()

    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    for target in active_names:
        source = mismatch.get(target)
        if not source or source not in active or target not in things_index or source not in things_index:
            continue
        target_neighbor, tn_target_sim, tn_source_sim, tn_select_margin = choose_neighbor(
            target,
            source,
            active_names,
            things_index,
            similarity,
            {target, source},
        )
        source_neighbor, sn_source_sim, sn_target_sim, sn_select_margin = choose_neighbor(
            source,
            target,
            active_names,
            things_index,
            similarity,
            {target, source, target_neighbor},
        )
        if rng.random() < 0.5:
            option_a_role = "target_neighbor"
            option_a = target_neighbor
            option_b_role = "source_neighbor"
            option_b = source_neighbor
        else:
            option_a_role = "source_neighbor"
            option_a = source_neighbor
            option_b_role = "target_neighbor"
            option_b = target_neighbor
        target_source_similarity = float(similarity[things_index[target], things_index[source]])
        image_similarity, pair_difficulty = pair_image.get((target, source), (float("nan"), ""))
        attraction, margin = source_attraction.get((target, source), (float("nan"), float("nan")))
        rows.append(
            {
                "concept": target,
                "subtype": active[target]["subtype"],
                "mismatch_source": source,
                "mismatch_source_subtype": active[source]["subtype"],
                "target_neighbor": target_neighbor,
                "source_neighbor": source_neighbor,
                "option_a": option_a,
                "option_a_role": option_a_role,
                "option_b": option_b,
                "option_b_role": option_b_role,
                "target_source_things_similarity": target_source_similarity,
                "target_neighbor_to_target_similarity": tn_target_sim,
                "target_neighbor_to_source_similarity": tn_source_sim,
                "target_neighbor_selection_margin": tn_select_margin,
                "source_neighbor_to_source_similarity": sn_source_sim,
                "source_neighbor_to_target_similarity": sn_target_sim,
                "source_neighbor_selection_margin": sn_select_margin,
                "pair_image_similarity": image_similarity,
                "pair_difficulty": pair_difficulty,
                "source_attraction": attraction,
                "source_minus_target_margin": margin,
                "ab_seed": args.seed,
            }
        )
    if args.limit and args.limit < len(rows):
        rng = random.Random(args.seed)
        rows = sorted(rng.sample(rows, args.limit), key=lambda row: row["concept"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build item set for identity/similarity behavioral dissociation probe.")
    parser.add_argument("--concepts", default="data/concepts/things_max_1854_concepts.csv")
    parser.add_argument("--mismatch-map", default="data/manifests/mismatch_map.csv")
    parser.add_argument("--output", default="outputs/metrics/identity_similarity_probe_items.csv")
    parser.add_argument("--summary-output", default="outputs/metrics/identity_similarity_probe_items_summary.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260501)
    args = parser.parse_args()

    rows = build_rows(args)
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(ROOT / args.output, rows, fieldnames)
    summary = {
        "num_items": len(rows),
        "seed": args.seed,
        "mean_target_source_things_similarity": float(np.nanmean([float(r["target_source_things_similarity"]) for r in rows])) if rows else 0.0,
        "mean_pair_image_similarity": float(np.nanmean([float(r["pair_image_similarity"]) for r in rows])) if rows else 0.0,
        "mean_source_attraction": float(np.nanmean([float(r["source_attraction"]) for r in rows])) if rows else 0.0,
    }
    write_json(ROOT / args.summary_output, summary)
    append_run_log("Identity-Similarity Probe Items", [f"Wrote {len(rows)} probe items to {args.output}."])
    print(f"Wrote {args.output} with {len(rows)} rows.")


if __name__ == "__main__":
    main()
