from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from common import ROOT, append_run_log, percentile_interval, rankdata, write_csv, write_json
from hardening_common import write_text


PREDICTORS = [
    "source_attraction",
    "source_minus_target_margin",
    "rdm_disruption",
]

ENDPOINTS = [
    "description_drift",
    "source_description_similarity",
    "clip_source_minus_target_similarity",
    "clip_source_choice",
    "similarity_source_neighbor_choice",
    "mismatched_source_leakage",
    "visual_word_rate_per_100",
    "lancaster_visual_mean",
]

BINARY_ENDPOINTS = {
    "clip_source_choice",
    "similarity_source_neighbor_choice",
    "mismatched_source_leakage",
}

PRIMARY_ENDPOINTS = [
    "description_drift",
    "clip_source_minus_target_similarity",
    "similarity_source_neighbor_choice",
]


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0.0:
        return 0.0
    return float(np.dot(x, y) / denom)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return 0.0
    return pearson_corr(rankdata(x), rankdata(y))


def bootstrap_ci(x: np.ndarray, y: np.ndarray, fn: Callable[[np.ndarray, np.ndarray], float], n_bootstrap: int, seed: int) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(x), size=len(x))
        values.append(fn(x[idx], y[idx]))
    return percentile_interval(np.asarray(values, dtype=float), 0.95)


def permutation_p(x: np.ndarray, y: np.ndarray, observed: float, fn: Callable[[np.ndarray, np.ndarray], float], n_permutations: int, seed: int) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return 1.0
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_permutations):
        if abs(fn(x, rng.permutation(y))) >= abs(observed):
            count += 1
    return float((count + 1.0) / (n_permutations + 1.0))


def read_required(path: str) -> pd.DataFrame:
    full_path = ROOT / path
    if not full_path.exists():
        raise RuntimeError(f"Missing required input: {full_path}")
    return pd.read_csv(full_path)


def build_analysis_table(args: argparse.Namespace) -> pd.DataFrame:
    leakage = read_required(args.implicit_leakage)
    geometry = read_required(args.geometry_bridge)
    clip = read_required(args.clip_forced_choice)
    relational = read_required(args.relational_scores)

    leakage = leakage.copy()
    leakage["concept_key"] = leakage["concept"].str.lower()
    leakage["description_drift"] = leakage["source_description_similarity"] - leakage["target_description_similarity"]

    geometry = geometry.copy()
    geometry["concept_key"] = geometry["concept"].str.lower()
    geometry = geometry[["concept_key", "rdm_disruption", "target_perturbation"]]

    clip = clip[clip["condition"].eq("M_mismatched_image")].copy()
    clip["concept_key"] = clip["concept"].str.lower()
    clip["clip_source_minus_target_similarity"] = clip["source_similarity"] - clip["target_similarity"]
    clip = clip[
        [
            "concept_key",
            "target_similarity",
            "source_similarity",
            "target_margin",
            "target_choice",
            "source_choice",
            "clip_source_minus_target_similarity",
            "pair_image_similarity",
            "pair_difficulty",
        ]
    ].rename(columns={"source_choice": "clip_source_choice", "target_choice": "clip_target_choice"})

    relational = relational[
        relational["condition"].eq("M_mismatched_image") & relational["task"].eq("similarity")
    ].copy()
    relational["concept_key"] = relational["concept"].str.lower()
    relational = relational[
        [
            "concept_key",
            "similarity_valid",
            "similarity_source_neighbor_choice",
            "similarity_target_neighbor_choice",
            "target_source_things_similarity",
            "source_neighbor_selection_margin",
            "target_neighbor_selection_margin",
        ]
    ]

    table = leakage.merge(geometry, on="concept_key", how="left")
    table = table.merge(clip, on="concept_key", how="left")
    table = table.merge(relational, on="concept_key", how="left")
    table = table.sort_values("concept_key").reset_index(drop=True)
    return table


def correlation_rows(table: pd.DataFrame, n_bootstrap: int, n_permutations: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for predictor_idx, predictor in enumerate(PREDICTORS):
        if predictor not in table:
            continue
        x = table[predictor].to_numpy(dtype=float)
        for endpoint_idx, endpoint in enumerate(ENDPOINTS):
            if endpoint not in table:
                continue
            y = table[endpoint].to_numpy(dtype=float)
            fn = pearson_corr if endpoint in BINARY_ENDPOINTS else spearman_corr
            stat_name = "point_biserial_r" if endpoint in BINARY_ENDPOINTS else "spearman_rho"
            observed = fn(x, y)
            ci_low, ci_high = bootstrap_ci(x, y, fn, n_bootstrap, seed + 101 * predictor_idx + endpoint_idx)
            p_value = permutation_p(x, y, observed, fn, n_permutations, seed + 10000 + 101 * predictor_idx + endpoint_idx)
            rows.append(
                {
                    "predictor": predictor,
                    "endpoint": endpoint,
                    "endpoint_family": "primary" if endpoint in PRIMARY_ENDPOINTS else "supporting",
                    "statistic": stat_name,
                    "n": int(np.sum(np.isfinite(x) & np.isfinite(y))),
                    "estimate": observed,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "permutation_p": p_value,
                }
            )
    return rows


def quartile_rows(table: pd.DataFrame, predictor: str) -> list[dict[str, Any]]:
    if predictor not in table:
        return []
    q1 = table[predictor].quantile(0.25)
    q3 = table[predictor].quantile(0.75)
    groups = [
        ("bottom_quartile_source_attraction", table[table[predictor] <= q1]),
        ("top_quartile_source_attraction", table[table[predictor] >= q3]),
    ]
    rows: list[dict[str, Any]] = []
    for name, group in groups:
        row: dict[str, Any] = {
            "group": name,
            "n": int(len(group)),
            "mean_source_attraction": float(group["source_attraction"].mean()),
            "mean_source_minus_target_margin": float(group["source_minus_target_margin"].mean()),
        }
        for endpoint in ENDPOINTS:
            if endpoint in group:
                row[f"mean_{endpoint}"] = float(group[endpoint].mean())
        rows.append(row)
    if len(rows) == 2:
        delta: dict[str, Any] = {
            "group": "top_minus_bottom",
            "n": min(int(rows[0]["n"]), int(rows[1]["n"])),
            "mean_source_attraction": rows[1]["mean_source_attraction"] - rows[0]["mean_source_attraction"],
            "mean_source_minus_target_margin": rows[1]["mean_source_minus_target_margin"] - rows[0]["mean_source_minus_target_margin"],
        }
        for endpoint in ENDPOINTS:
            key = f"mean_{endpoint}"
            if key in rows[0] and key in rows[1]:
                delta[key] = rows[1][key] - rows[0][key]
        rows.append(delta)
    return rows


def report_lines(summary: dict[str, Any], correlations: list[dict[str, Any]], quartiles: list[dict[str, Any]]) -> list[str]:
    lines = [
        "# Behavior Drift Endpoint Reanalysis",
        "",
        f"- Concepts: `{summary['n_concepts']}`",
        f"- Bootstrap resamples: `{summary['n_bootstrap']}`",
        f"- Permutations: `{summary['n_permutations']}`",
        "- Primary question: do hidden-state source-attraction measures predict source-relative output drift?",
        "",
        "## Primary Correlations",
        "",
        "| Predictor | Endpoint | Statistic | Estimate | 95% CI | p |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in correlations:
        if row["endpoint_family"] != "primary":
            continue
        lines.append(
            f"| `{row['predictor']}` | `{row['endpoint']}` | `{row['statistic']}` | "
            f"{row['estimate']:+.4f} | [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | {row['permutation_p']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Supporting Correlations",
            "",
            "| Predictor | Endpoint | Statistic | Estimate | 95% CI | p |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in correlations:
        if row["endpoint_family"] != "supporting":
            continue
        lines.append(
            f"| `{row['predictor']}` | `{row['endpoint']}` | `{row['statistic']}` | "
            f"{row['estimate']:+.4f} | [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | {row['permutation_p']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Source-Attraction Quartiles",
            "",
            "| Group | n | Description drift | CLIP source-target | Source-neighbor choice | Explicit leakage | Lancaster visual |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in quartiles:
        lines.append(
            f"| `{row['group']}` | {int(row['n'])} | "
            f"{row.get('mean_description_drift', 0.0):+.4f} | "
            f"{row.get('mean_clip_source_minus_target_similarity', 0.0):+.4f} | "
            f"{row.get('mean_similarity_source_neighbor_choice', 0.0):+.4f} | "
            f"{row.get('mean_mismatched_source_leakage', 0.0):+.4f} | "
            f"{row.get('mean_lancaster_visual_mean', 0.0):+.4f} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute source-relative behavior drift endpoints.")
    parser.add_argument("--implicit-leakage", default="outputs/metrics/behavior_bridge_extensions_full_implicit_leakage.csv")
    parser.add_argument("--geometry-bridge", default="outputs/metrics/behavior_geometry_bridge_full.csv")
    parser.add_argument("--clip-forced-choice", default="outputs/metrics/clip_forced_choice_behavior.csv")
    parser.add_argument("--relational-scores", default="outputs/metrics/identity_similarity_probe_scores.csv")
    parser.add_argument("--output-stem", default="behavior_drift_endpoints")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    table = build_analysis_table(args)
    if args.limit:
        table = table.head(args.limit).copy()
    correlations = correlation_rows(table, args.bootstrap, args.permutations, args.seed)
    quartiles = quartile_rows(table, "source_attraction")
    summary = {
        "n_concepts": int(len(table)),
        "n_bootstrap": args.bootstrap,
        "n_permutations": args.permutations,
        "primary_endpoints": PRIMARY_ENDPOINTS,
        "supporting_endpoints": [endpoint for endpoint in ENDPOINTS if endpoint not in PRIMARY_ENDPOINTS],
        "inputs": {
            "implicit_leakage": args.implicit_leakage,
            "geometry_bridge": args.geometry_bridge,
            "clip_forced_choice": args.clip_forced_choice,
            "relational_scores": args.relational_scores,
        },
        "correlations": correlations,
        "quartiles": quartiles,
    }

    output_dir = ROOT / "outputs" / "metrics"
    report_dir = ROOT / "reports" / "main_results"
    table_path = output_dir / f"{args.output_stem}.csv"
    corr_path = output_dir / f"{args.output_stem}_correlations.csv"
    quartile_path = output_dir / f"{args.output_stem}_quartiles.csv"
    json_path = output_dir / f"{args.output_stem}_summary.json"
    report_path = report_dir / f"{args.output_stem}_report.md"
    table.to_csv(table_path, index=False)
    write_csv(corr_path, correlations, ["predictor", "endpoint", "endpoint_family", "statistic", "n", "estimate", "ci95_low", "ci95_high", "permutation_p"])
    quartile_fields = sorted({key for row in quartiles for key in row})
    write_csv(quartile_path, quartiles, quartile_fields)
    write_json(json_path, summary)
    write_text(report_path, "\n".join(report_lines(summary, correlations, quartiles)))
    append_run_log(
        "Behavior Drift Endpoint Reanalysis",
        [
            f"Wrote {table_path.relative_to(ROOT)} for {len(table)} concepts.",
            f"Wrote {report_path.relative_to(ROOT)}.",
        ],
    )
    print(f"Wrote {table_path.relative_to(ROOT)}")
    print(f"Wrote {corr_path.relative_to(ROOT)}")
    print(f"Wrote {quartile_path.relative_to(ROOT)}")
    print(f"Wrote {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
