from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from common import ROOT, append_run_log, percentile_interval, write_csv, write_json
from hardening_common import write_text


def normalize_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", " ", str(text), flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</?think>", " ", text, flags=re.IGNORECASE)
    text = text.lower().replace("_", " ").replace("-", " ")
    return re.sub(r"[^a-z0-9 ]+", " ", text)


def phrase_pattern(phrase: str) -> re.Pattern[str]:
    tokens = [re.escape(part) for part in normalize_text(phrase).split() if part]
    if not tokens:
        return re.compile(r"a^")
    return re.compile(r"\b" + r"\s+".join(tokens) + r"\b")


def first_match_position(text: str, phrase: str) -> int | None:
    match = phrase_pattern(phrase).search(text)
    return None if match is None else match.start()


def classify_identity(generated_text: str, target: str, source: str) -> tuple[int, int, int, str]:
    text = normalize_text(generated_text)
    target_pos = first_match_position(text, target)
    source_pos = first_match_position(text, source)
    if target_pos is None and source_pos is None:
        return 0, 0, 1, "invalid"
    if target_pos is not None and source_pos is None:
        return 1, 0, 0, "target"
    if source_pos is not None and target_pos is None:
        return 0, 1, 0, "source"
    if target_pos is not None and source_pos is not None:
        if target_pos < source_pos:
            return 1, 0, 0, "target_first"
        if source_pos < target_pos:
            return 0, 1, 0, "source_first"
    return 0, 0, 1, "ambiguous"


def bootstrap_ci(values: np.ndarray, fn: Callable[[np.ndarray], float], n_bootstrap: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(values), size=len(values))
        estimates.append(fn(values[idx]))
    return percentile_interval(np.asarray(estimates, dtype=float), 0.95)


def summarize_group(rows: pd.DataFrame, group: str, n_bootstrap: int, seed: int) -> dict[str, Any]:
    if rows.empty:
        return {
            "group": group,
            "n": 0,
            "target_identity_rate": "",
            "source_identity_rate": "",
            "invalid_rate": "",
            "clip_source_drift_rate": "",
            "description_source_drift_rate": "",
            "decoupled_clip_rate": "",
            "decoupled_description_rate": "",
            "target_identity_ci95_low": "",
            "target_identity_ci95_high": "",
            "decoupled_clip_ci95_low": "",
            "decoupled_clip_ci95_high": "",
        }
    target = rows["identity_target_choice"].to_numpy(dtype=float)
    source = rows["identity_source_choice"].to_numpy(dtype=float)
    invalid = rows["identity_invalid"].to_numpy(dtype=float)
    clip_drift = rows["clip_source_drift"].to_numpy(dtype=float)
    desc_drift = rows["description_source_drift"].to_numpy(dtype=float)
    decoupled_clip = rows["decoupled_clip_drift"].to_numpy(dtype=float)
    decoupled_desc = rows["decoupled_description_drift"].to_numpy(dtype=float)
    target_ci = bootstrap_ci(target, lambda values: float(np.mean(values)), n_bootstrap, seed)
    decoupled_ci = bootstrap_ci(decoupled_clip, lambda values: float(np.mean(values)), n_bootstrap, seed + 1000)
    return {
        "group": group,
        "n": len(rows),
        "target_identity_rate": float(np.mean(target)),
        "source_identity_rate": float(np.mean(source)),
        "invalid_rate": float(np.mean(invalid)),
        "clip_source_drift_rate": float(np.mean(clip_drift)),
        "description_source_drift_rate": float(np.mean(desc_drift)),
        "decoupled_clip_rate": float(np.mean(decoupled_clip)),
        "decoupled_description_rate": float(np.mean(decoupled_desc)),
        "target_identity_ci95_low": target_ci[0],
        "target_identity_ci95_high": target_ci[1],
        "decoupled_clip_ci95_low": decoupled_ci[0],
        "decoupled_clip_ci95_high": decoupled_ci[1],
    }


def report_lines(summary_rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "# Identity-Drift Decoupling Probe",
        "",
        "This probe tests whether mismatched visual input can perturb global descriptive behavior while preserving text-defined identity.",
        "",
        "| Group | n | Target identity | Source identity | Invalid | CLIP source drift | Description source drift | Target + CLIP drift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        if not row["n"]:
            continue
        lines.append(
            f"| {row['group']} | {row['n']} | {row['target_identity_rate']:.4f} | "
            f"{row['source_identity_rate']:.4f} | {row['invalid_rate']:.4f} | "
            f"{row['clip_source_drift_rate']:.4f} | {row['description_source_drift_rate']:.4f} | "
            f"{row['decoupled_clip_rate']:.4f} |"
        )
    return lines


def copy_to_figures_data(paths: list[Path]) -> None:
    target = ROOT / "figures_data" / "derived"
    target.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists():
            shutil.copy2(path, target / path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score identity forced-choice responses and join with drift metrics.")
    parser.add_argument("--generations", default="outputs/generations/identity_drift_probe_generations.csv")
    parser.add_argument("--clip", default="outputs/metrics/clip_forced_choice_behavior.csv")
    parser.add_argument("--bridge", default="outputs/metrics/behavior_bridge_extensions_full_implicit_leakage.csv")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--no-copy-figures-data", action="store_true")
    args = parser.parse_args()

    generations = pd.read_csv(ROOT / args.generations)
    clip = pd.read_csv(ROOT / args.clip)
    bridge = pd.read_csv(ROOT / args.bridge)

    rows = []
    for _, row in generations.iterrows():
        target = str(row["concept"]).lower()
        source = str(row["mismatch_source"]).lower()
        target_choice, source_choice, invalid, label = classify_identity(str(row["generated_text"]), target, source)
        rows.append(
            {
                **row.to_dict(),
                "concept": target,
                "mismatch_source": source,
                "identity_target_choice": target_choice,
                "identity_source_choice": source_choice,
                "identity_invalid": invalid,
                "identity_label": label,
            }
        )
    scored = pd.DataFrame(rows)

    clip_mismatch = clip[clip["condition"].eq("M_mismatched_image")][
        ["concept", "target_choice", "source_choice", "target_margin", "pair_image_similarity", "pair_difficulty"]
    ].rename(
        columns={
            "target_choice": "clip_target_choice",
            "source_choice": "clip_source_choice",
            "target_margin": "clip_target_margin",
        }
    )
    bridge_cols = [
        "concept",
        "source_attraction",
        "source_minus_target_margin",
        "source_description_similarity",
        "source_minus_target_description_similarity",
        "mismatched_source_leakage",
        "visual_word_rate_per_100",
        "lancaster_visual_mean",
    ]
    merged = scored.merge(clip_mismatch, on="concept", how="left").merge(bridge[bridge_cols], on="concept", how="left")
    merged["clip_source_drift"] = merged["clip_source_choice"].fillna(0).astype(int)
    merged["description_source_drift"] = (merged["source_minus_target_description_similarity"].fillna(-np.inf) > 0).astype(int)
    merged["decoupled_clip_drift"] = ((merged["identity_target_choice"] == 1) & (merged["clip_source_drift"] == 1)).astype(int)
    merged["decoupled_description_drift"] = ((merged["identity_target_choice"] == 1) & (merged["description_source_drift"] == 1)).astype(int)

    q1 = merged["source_attraction"].quantile(0.25)
    q3 = merged["source_attraction"].quantile(0.75)
    groups = [
        ("all", merged),
        ("bottom_quartile_source_attraction", merged[merged["source_attraction"] <= q1]),
        ("top_quartile_source_attraction", merged[merged["source_attraction"] >= q3]),
        ("bottom_half_source_attraction", merged[merged["source_attraction"] <= merged["source_attraction"].median()]),
        ("top_half_source_attraction", merged[merged["source_attraction"] > merged["source_attraction"].median()]),
        ("clip_source_drift_negative", merged[merged["clip_source_drift"] == 0]),
        ("clip_source_drift_positive", merged[merged["clip_source_drift"] == 1]),
    ]
    summary_rows = [summarize_group(df, name, args.bootstrap, args.seed + idx * 17) for idx, (name, df) in enumerate(groups)]

    suffix = args.suffix
    score_path = ROOT / f"outputs/metrics/identity_drift_probe_scores{suffix}.csv"
    summary_path = ROOT / f"outputs/metrics/identity_drift_decoupling_summary{suffix}.csv"
    quartile_path = ROOT / f"outputs/metrics/identity_drift_decoupling_quartiles{suffix}.csv"
    json_path = ROOT / f"outputs/metrics/identity_drift_decoupling_summary{suffix}.json"
    report_path = ROOT / f"reports/main_results/identity_drift_decoupling_report{suffix}.md"

    merged.to_csv(score_path, index=False)
    write_csv(summary_path, summary_rows, list(summary_rows[0].keys()))
    quartile_rows = [row for row in summary_rows if "quartile" in row["group"]]
    write_csv(quartile_path, quartile_rows, list(summary_rows[0].keys()))
    write_json(
        json_path,
        {
            "n": int(len(merged)),
            "source_attraction_q1": float(q1),
            "source_attraction_q3": float(q3),
            "summary_rows": summary_rows,
        },
    )
    write_text(report_path, "\n".join(report_lines(summary_rows)))
    if not args.no_copy_figures_data and not suffix:
        copy_to_figures_data([score_path, quartile_path])
    append_run_log(
        "Identity-Drift Decoupling Probe",
        [
            f"Scored {len(merged)} identity probe rows.",
            f"Wrote {score_path.relative_to(ROOT)}.",
        ],
    )
    print(f"Wrote {score_path.relative_to(ROOT)}")
    print(f"Wrote {summary_path.relative_to(ROOT)}")
    print(f"Wrote {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
