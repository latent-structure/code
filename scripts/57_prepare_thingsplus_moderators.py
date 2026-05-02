from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from common import ROOT, append_run_log, percentile_interval, rankdata, write_csv, write_json
from hardening_common import write_text


THINGSPLUS_DIR = ROOT / "THINGS-database" / "osfstorage" / "02_object-level" / "trial-wise_tables"
PROPERTY_FILE = THINGSPLUS_DIR / "object-properties_trial-wise_thingsplus.tsv"
SIZE_FILE = THINGSPLUS_DIR / "size_trial-wise_thingsplus.tsv"
AROUSAL_FILE = THINGSPLUS_DIR / "arousal_trial-wise_thingsplus.tsv"

OUTCOMES = [
    "source_attraction",
    "source_minus_target_margin",
    "description_drift",
    "clip_source_minus_target_similarity",
    "mismatched_source_leakage",
    "rdm_disruption",
    "target_perturbation",
    "lancaster_visual_mean",
]


def normalize_key(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_")


def read_quoted_tab_file(path: Path) -> pd.DataFrame:
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            rows.append(row[0].split("\t"))
    header, body = rows[0], rows[1:]
    return pd.DataFrame(body, columns=header)


def aggregate_numeric(df: pd.DataFrame, value_cols: list[str], prefix: str) -> pd.DataFrame:
    df = df.copy()
    df["concept_key"] = df["uniqueID"].map(normalize_key)
    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    grouped = df.groupby("concept_key", as_index=False)[value_cols].mean()
    return grouped.rename(columns={col: f"{prefix}{col}" for col in value_cols})


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return 0.0 if denom == 0 else float(np.dot(x, y) / denom)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return pearson_corr(rankdata(np.asarray(x, dtype=float)), rankdata(np.asarray(y, dtype=float)))


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
    return percentile_interval(np.asarray(values), 0.95)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate THINGSplus moderators and link them to behavior/geometry outcomes.")
    parser.add_argument("--behavior-drift", default="outputs/metrics/behavior_drift_endpoints.csv")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260501)
    args = parser.parse_args()

    prop = read_quoted_tab_file(PROPERTY_FILE)
    prop_cols = ["manmade", "precious", "lives", "heavy", "natural", "moves", "grasp", "hold", "be.moved", "pleasant", "arousing"]
    prop_agg = aggregate_numeric(prop, prop_cols, "thingsplus_")

    size = pd.read_csv(SIZE_FILE, sep="\t", low_memory=False)
    size_agg = aggregate_numeric(size, ["Size", "RangeLength", "ratingLevel1"], "thingsplus_size_")

    arousal = pd.read_csv(AROUSAL_FILE, sep="\t", low_memory=False)
    arousal_agg = aggregate_numeric(arousal, ["arousing"], "thingsplus_arousal_")

    moderators = prop_agg.merge(size_agg, on="concept_key", how="outer").merge(arousal_agg, on="concept_key", how="outer")
    behavior = pd.read_csv(ROOT / args.behavior_drift)
    if "concept_key" not in behavior:
        behavior["concept_key"] = behavior["concept"].map(normalize_key)
    joined = behavior.merge(moderators, on="concept_key", how="left")

    moderator_cols = [col for col in moderators.columns if col != "concept_key"]
    rows: list[dict[str, Any]] = []
    for mod_idx, moderator in enumerate(moderator_cols):
        x = joined[moderator].to_numpy(dtype=float)
        for outcome_idx, outcome in enumerate(OUTCOMES):
            if outcome not in joined:
                continue
            y = joined[outcome].to_numpy(dtype=float)
            fn = pearson_corr if outcome == "mismatched_source_leakage" else spearman_corr
            estimate = fn(x, y)
            ci_low, ci_high = bootstrap_ci(x, y, fn, args.bootstrap, args.seed + 100 * mod_idx + outcome_idx)
            rows.append(
                {
                    "moderator": moderator,
                    "outcome": outcome,
                    "statistic": "point_biserial_r" if outcome == "mismatched_source_leakage" else "spearman_rho",
                    "n": int(np.sum(np.isfinite(x) & np.isfinite(y))),
                    "estimate": estimate,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                }
            )

    out_dir = ROOT / "outputs" / "scope_extensions"
    report_dir = ROOT / "reports" / "main_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joined.to_csv(out_dir / "thingsplus_concept_moderators.csv", index=False)
    write_csv(out_dir / "thingsplus_moderator_correlations.csv", rows, ["moderator", "outcome", "statistic", "n", "estimate", "ci95_low", "ci95_high"])
    top = sorted(rows, key=lambda row: abs(float(row["estimate"])), reverse=True)[:30]
    write_json(
        out_dir / "thingsplus_moderator_summary.json",
        {
            "n_joined": int(len(joined)),
            "n_with_any_thingsplus": int(joined[moderator_cols].notna().any(axis=1).sum()),
            "moderator_count": len(moderator_cols),
            "top_absolute_correlations": top,
        },
    )
    lines = [
        "# THINGSplus Object-Property Moderators",
        "",
        f"- Concepts joined: `{len(joined)}`",
        f"- Concepts with any THINGSplus moderator: `{int(joined[moderator_cols].notna().any(axis=1).sum())}`",
        "",
        "| Moderator | Outcome | Statistic | Estimate | 95% CI | n |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in top:
        lines.append(
            f"| `{row['moderator']}` | `{row['outcome']}` | `{row['statistic']}` | "
            f"{row['estimate']:+.4f} | [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | {row['n']} |"
        )
    write_text(report_dir / "thingsplus_moderator_report.md", "\n".join(lines))
    append_run_log("THINGSplus Moderators", [f"Wrote THINGSplus moderator analysis for {len(joined)} concepts."])


if __name__ == "__main__":
    main()
