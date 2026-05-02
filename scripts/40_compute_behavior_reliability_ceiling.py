from __future__ import annotations

import argparse
import importlib.util
from collections import defaultdict
from typing import Any, Callable

import numpy as np

from common import ROOT, append_run_log, percentile_interval, rankdata, read_csv, write_csv, write_json
from hardening_common import write_text


METRICS = ["visual_word_rate_per_100", "lancaster_visual_mean", "exemplar_specific_rate_per_100"]


def load_stage31_module() -> Any:
    script_path = ROOT / "scripts" / "31_score_behavior_probe.py"
    spec = importlib.util.spec_from_file_location("stage31_score_behavior_probe", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load helper module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return 0.0
    return float(np.dot(x, y) / denom)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return pearson_corr(rankdata(np.asarray(x, dtype=float)), rankdata(np.asarray(y, dtype=float)))


def split_half_reliability(values_by_concept: dict[str, list[float]], n_splits: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    concepts = sorted(values_by_concept)
    repeats = len(next(iter(values_by_concept.values())))
    if repeats < 2:
        raise RuntimeError("At least two repeats per concept are required for split-half reliability.")
    estimates = []
    for _ in range(n_splits):
        left_values = []
        right_values = []
        for concept in concepts:
            indices = rng.permutation(repeats)
            left_idx = indices[: repeats // 2]
            right_idx = indices[repeats // 2 :]
            left_values.append(float(np.mean([values_by_concept[concept][idx] for idx in left_idx])))
            right_values.append(float(np.mean([values_by_concept[concept][idx] for idx in right_idx])))
        estimates.append(spearman_corr(np.asarray(left_values), np.asarray(right_values)))
    arr = np.asarray(estimates, dtype=float)
    low, high = percentile_interval(arr, 0.95)
    return float(np.mean(arr)), low, high


def bootstrap_observed_corr(
    bridge_rows: list[dict[str, str]],
    predictor: str,
    endpoint: str,
    n_bootstrap: int,
    seed: int,
    fn: Callable[[np.ndarray, np.ndarray], float] = spearman_corr,
) -> tuple[float, float, float]:
    x = np.asarray([float(row[predictor]) for row in bridge_rows], dtype=float)
    y = np.asarray([float(row[endpoint]) for row in bridge_rows], dtype=float)
    observed = fn(x, y)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(x), size=len(x))
        values.append(fn(x[idx], y[idx]))
    low, high = percentile_interval(np.asarray(values, dtype=float), 0.95)
    return observed, low, high


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute same-concept behavior reliability ceilings from repeated stochastic generations.")
    parser.add_argument("--generations", default="outputs/generations/behavior_repeats_mismatched_200x5.csv")
    parser.add_argument("--bridge", default="outputs/metrics/behavior_geometry_bridge_full.csv")
    parser.add_argument("--output-stem", default="behavior_reliability_ceiling")
    parser.add_argument("--splits", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260429)
    args = parser.parse_args()

    stage31 = load_stage31_module()
    rows = read_csv(ROOT / args.generations)
    lancaster_norms = stage31.load_lancaster_norms()
    scored = [stage31.score_row(row, lancaster_norms) for row in rows]
    by_concept: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_concept[row["concept"].lower()].append(row)
    repeat_counts = {concept: len(items) for concept, items in by_concept.items()}
    if len(set(repeat_counts.values())) != 1:
        raise RuntimeError(f"Unequal repeat counts: {repeat_counts}")
    n_repeats = next(iter(repeat_counts.values()))
    bridge_all = {row["concept"].lower(): row for row in read_csv(ROOT / args.bridge)}
    bridge_rows = [bridge_all[concept] for concept in sorted(by_concept) if concept in bridge_all]

    output_rows = []
    for metric_idx, metric in enumerate(METRICS):
        values_by_concept = {
            concept: [float(row[metric]) for row in sorted(items, key=lambda row: int(row["repeat_id"]))]
            for concept, items in by_concept.items()
        }
        reliability, rel_low, rel_high = split_half_reliability(values_by_concept, args.splits, args.seed + metric_idx)
        observed, obs_low, obs_high = bootstrap_observed_corr(
            bridge_rows,
            "source_attraction",
            metric,
            args.bootstrap,
            args.seed + 100 + metric_idx,
        )
        normalized = observed / reliability if reliability != 0 else 0.0
        variance_fraction = normalized**2
        output_rows.append(
            {
                "metric": metric,
                "n_concepts": len(by_concept),
                "n_repeats": n_repeats,
                "split_half_reliability": reliability,
                "split_half_ci95_low": rel_low,
                "split_half_ci95_high": rel_high,
                "source_attraction_rho_full": observed,
                "source_attraction_ci95_low": obs_low,
                "source_attraction_ci95_high": obs_high,
                "ceiling_normalized_rho": normalized,
                "ceiling_normalized_variance_fraction": variance_fraction,
            }
        )

    summary = {
        "generation_file": args.generations,
        "n_concepts": len(by_concept),
        "n_repeats": n_repeats,
        "n_splits": args.splits,
        "n_bootstrap": args.bootstrap,
        "method_note": "Reliability is same-concept split-half Spearman reliability across stochastic repeated mismatched-image generations.",
        "rows": output_rows,
    }
    write_csv(
        ROOT / "outputs" / "metrics" / f"{args.output_stem}.csv",
        output_rows,
        [
            "metric",
            "n_concepts",
            "n_repeats",
            "split_half_reliability",
            "split_half_ci95_low",
            "split_half_ci95_high",
            "source_attraction_rho_full",
            "source_attraction_ci95_low",
            "source_attraction_ci95_high",
            "ceiling_normalized_rho",
            "ceiling_normalized_variance_fraction",
        ],
    )
    write_json(ROOT / "outputs" / "metrics" / f"{args.output_stem}.json", summary)
    lines = [
        "# Behavior Reliability Ceiling",
        "",
        f"- Concepts: `{len(by_concept)}`",
        f"- Repeats per concept: `{n_repeats}`",
        f"- Split-half resamples: `{args.splits}`",
        "",
        "| Metric | Reliability | 95% CI | Source-attraction rho | Ceiling-normalized rho | Normalized variance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in output_rows:
        lines.append(
            f"| `{row['metric']}` | {row['split_half_reliability']:.4f} | "
            f"[{row['split_half_ci95_low']:.4f}, {row['split_half_ci95_high']:.4f}] | "
            f"{row['source_attraction_rho_full']:.4f} | {row['ceiling_normalized_rho']:.4f} | "
            f"{row['ceiling_normalized_variance_fraction']:.4f} |"
        )
    write_text(ROOT / "reports" / "main_results" / f"{args.output_stem}_report.md", "\n".join(lines))
    append_run_log(
        "Behavior Reliability Ceiling",
        [
            f"Computed reliability ceilings for {len(by_concept)} concepts and {n_repeats} repeats.",
            f"Output stem: {args.output_stem}.",
        ],
    )


if __name__ == "__main__":
    main()
