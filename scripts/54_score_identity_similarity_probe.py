from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from common import ROOT, append_run_log, percentile_interval, write_json
from hardening_common import write_text


CONDITIONS = ["M_text_only", "M_matched_image", "M_mismatched_image", "M_blank_image"]


def normalize_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", " ", str(text), flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</?think>", " ", text, flags=re.IGNORECASE)
    text = text.lower().replace("_", " ").replace("-", " ")
    return re.sub(r"[^a-z0-9 ]+", " ", text).strip()


def phrase_pattern(phrase: str) -> re.Pattern[str]:
    tokens = [re.escape(part) for part in normalize_text(phrase).split() if part]
    if not tokens:
        return re.compile(r"a^")
    return re.compile(r"\b" + r"\s+".join(tokens) + r"\b")


def first_pos(text: str, phrase: str) -> int | None:
    match = phrase_pattern(phrase).search(text)
    return None if match is None else match.start()


def classify_identity(text: str, target: str, source: str) -> str:
    norm = normalize_text(text)
    target_pos = first_pos(norm, target)
    source_pos = first_pos(norm, source)
    if target_pos is None and source_pos is None:
        return "invalid"
    if target_pos is not None and source_pos is None:
        return "target"
    if source_pos is not None and target_pos is None:
        return "source"
    if target_pos is not None and source_pos is not None:
        return "target" if target_pos < source_pos else "source"
    return "invalid"


def classify_similarity(text: str, option_a_role: str, option_b_role: str) -> str:
    norm = normalize_text(text)
    tokens = norm.split()
    first = tokens[0] if tokens else ""
    if first == "a":
        return option_a_role
    if first == "b":
        return option_b_role
    if re.search(r"\boption a\b|\ba\)", norm):
        return option_a_role
    if re.search(r"\boption b\b|\bb\)", norm):
        return option_b_role
    return "invalid"


def bootstrap_ci(values: np.ndarray, fn: Callable[[np.ndarray], float], n_bootstrap: int, seed: int) -> tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(values), size=len(values))
        estimates.append(fn(values[idx]))
    return percentile_interval(np.asarray(estimates, dtype=float), 0.95)


def rate(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else 0.0


def summarize_condition(df: pd.DataFrame, condition: str, n_bootstrap: int, seed: int) -> dict[str, Any]:
    sub = df[df["condition"].eq(condition)]
    identity = sub[sub["task"].eq("identity")]
    similarity = sub[sub["task"].eq("similarity")]
    merged = identity[["concept", "identity_target_correct"]].merge(
        similarity[["concept", "similarity_source_neighbor_choice"]],
        on="concept",
        how="inner",
    )
    source_values = similarity["similarity_source_neighbor_choice"].to_numpy(dtype=float)
    identity_values = identity["identity_target_correct"].to_numpy(dtype=float)
    co_values = (
        (merged["identity_target_correct"].to_numpy(dtype=int) == 1)
        & (merged["similarity_source_neighbor_choice"].to_numpy(dtype=int) == 1)
    ).astype(float)
    source_ci = bootstrap_ci(source_values, lambda x: float(np.mean(x)), n_bootstrap, seed)
    identity_ci = bootstrap_ci(identity_values, lambda x: float(np.mean(x)), n_bootstrap, seed + 101)
    co_ci = bootstrap_ci(co_values, lambda x: float(np.mean(x)), n_bootstrap, seed + 202)
    return {
        "condition": condition,
        "identity_n": int(len(identity)),
        "similarity_n": int(len(similarity)),
        "identity_valid_rate": rate(identity["identity_valid"]),
        "identity_target_correct_rate": rate(identity["identity_target_correct"]),
        "identity_source_choice_rate": rate(identity["identity_source_choice"]),
        "similarity_valid_rate": rate(similarity["similarity_valid"]),
        "similarity_source_neighbor_choice_rate": rate(similarity["similarity_source_neighbor_choice"]),
        "similarity_target_neighbor_choice_rate": rate(similarity["similarity_target_neighbor_choice"]),
        "target_identity_plus_source_neighbor_rate": float(np.mean(co_values)) if len(co_values) else 0.0,
        "source_neighbor_ci95_low": source_ci[0],
        "source_neighbor_ci95_high": source_ci[1],
        "identity_target_ci95_low": identity_ci[0],
        "identity_target_ci95_high": identity_ci[1],
        "cooccurrence_ci95_low": co_ci[0],
        "cooccurrence_ci95_high": co_ci[1],
    }


def contrast_rows(summary: pd.DataFrame) -> list[dict[str, Any]]:
    lookup = summary.set_index("condition").to_dict(orient="index")
    rows = []
    for base in ["M_text_only", "M_matched_image", "M_blank_image"]:
        if base not in lookup or "M_mismatched_image" not in lookup:
            continue
        rows.append(
            {
                "contrast": f"M_mismatched_image_minus_{base}",
                "source_neighbor_choice_delta": lookup["M_mismatched_image"]["similarity_source_neighbor_choice_rate"]
                - lookup[base]["similarity_source_neighbor_choice_rate"],
                "identity_target_correct_delta": lookup["M_mismatched_image"]["identity_target_correct_rate"]
                - lookup[base]["identity_target_correct_rate"],
                "cooccurrence_delta": lookup["M_mismatched_image"]["target_identity_plus_source_neighbor_rate"]
                - lookup[base]["target_identity_plus_source_neighbor_rate"],
            }
        )
    return rows


def quartile_summary(scored: pd.DataFrame) -> list[dict[str, Any]]:
    similarity = scored[(scored["condition"].eq("M_mismatched_image")) & (scored["task"].eq("similarity"))].copy()
    if similarity.empty or "source_attraction" not in similarity:
        return []
    q1 = similarity["source_attraction"].quantile(0.25)
    q3 = similarity["source_attraction"].quantile(0.75)
    groups = [
        ("bottom_quartile_source_attraction", similarity[similarity["source_attraction"] <= q1]),
        ("top_quartile_source_attraction", similarity[similarity["source_attraction"] >= q3]),
        ("semantically_distant_pairs", similarity[similarity["target_source_things_similarity"] <= similarity["target_source_things_similarity"].median()]),
        ("semantically_close_pairs", similarity[similarity["target_source_things_similarity"] > similarity["target_source_things_similarity"].median()]),
    ]
    rows = []
    for name, group in groups:
        rows.append(
            {
                "group": name,
                "n": int(len(group)),
                "source_neighbor_choice_rate": rate(group["similarity_source_neighbor_choice"]),
                "target_neighbor_choice_rate": rate(group["similarity_target_neighbor_choice"]),
                "valid_rate": rate(group["similarity_valid"]),
                "mean_source_attraction": float(group["source_attraction"].mean()) if len(group) else 0.0,
                "mean_target_source_things_similarity": float(group["target_source_things_similarity"].mean()) if len(group) else 0.0,
                "mean_pair_image_similarity": float(group["pair_image_similarity"].mean()) if len(group) else 0.0,
            }
        )
    return rows


def report_lines(summary: pd.DataFrame, contrasts: pd.DataFrame, quartiles: pd.DataFrame) -> list[str]:
    lines = [
        "# Similarity-Judgment Behavioral Probe",
        "",
        "This probe tests whether mismatched images bias relational similarity choices toward neighbors of the mismatched image source.",
        "",
        "## Condition Summary",
        "",
    ]
    has_identity = bool(summary["identity_n"].sum() > 0)
    if has_identity:
        lines.extend(
            [
                "| Condition | Identity target | Similarity source-neighbor | Target identity + source-neighbor | Valid identity | Valid similarity |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
    else:
        lines.extend(
            [
                "| Condition | Similarity source-neighbor | Similarity target-neighbor | Valid similarity |",
                "|---|---:|---:|---:|",
            ]
        )
    for _, row in summary.iterrows():
        if has_identity:
            lines.append(
                f"| `{row['condition']}` | {row['identity_target_correct_rate']:.4f} | "
                f"{row['similarity_source_neighbor_choice_rate']:.4f} | "
                f"{row['target_identity_plus_source_neighbor_rate']:.4f} | "
                f"{row['identity_valid_rate']:.4f} | {row['similarity_valid_rate']:.4f} |"
            )
        else:
            lines.append(
                f"| `{row['condition']}` | {row['similarity_source_neighbor_choice_rate']:.4f} | "
                f"{row['similarity_target_neighbor_choice_rate']:.4f} | {row['similarity_valid_rate']:.4f} |"
            )
    if not contrasts.empty:
        if has_identity:
            lines.extend(["", "## Mismatched Contrasts", "", "| Contrast | Source-neighbor delta | Identity-target delta | Co-occurrence delta |", "|---|---:|---:|---:|"])
        else:
            lines.extend(["", "## Mismatched Contrasts", "", "| Contrast | Source-neighbor delta |", "|---|---:|"])
        for _, row in contrasts.iterrows():
            if has_identity:
                lines.append(
                    f"| `{row['contrast']}` | {row['source_neighbor_choice_delta']:+.4f} | "
                    f"{row['identity_target_correct_delta']:+.4f} | {row['cooccurrence_delta']:+.4f} |"
                )
            else:
                lines.append(f"| `{row['contrast']}` | {row['source_neighbor_choice_delta']:+.4f} |")
    if not quartiles.empty:
        lines.extend(["", "## Mismatched Source-Attraction Groups", "", "| Group | n | Source-neighbor choice | Mean source attraction | Mean THINGS target-source similarity |", "|---|---:|---:|---:|---:|"])
        for _, row in quartiles.iterrows():
            lines.append(
                f"| `{row['group']}` | {int(row['n'])} | {row['source_neighbor_choice_rate']:.4f} | "
                f"{row['mean_source_attraction']:.4f} | {row['mean_target_source_things_similarity']:.4f} |"
            )
    return lines


def copy_to_figures_data(paths: list[Path]) -> None:
    target = ROOT / "figures_data" / "derived"
    target.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists():
            shutil.copy2(path, target / path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score identity/similarity behavioral dissociation probe.")
    parser.add_argument("--generations", default="outputs/generations/identity_similarity_probe_generations.csv")
    parser.add_argument("--items", default="outputs/metrics/identity_similarity_probe_items.csv")
    parser.add_argument("--score-output", default="outputs/metrics/identity_similarity_probe_scores.csv")
    parser.add_argument("--summary-output", default="outputs/metrics/identity_similarity_probe_summary.csv")
    parser.add_argument("--contrast-output", default="outputs/metrics/identity_similarity_probe_contrasts.csv")
    parser.add_argument("--quartile-output", default="outputs/metrics/identity_similarity_probe_quartiles.csv")
    parser.add_argument("--json-output", default="outputs/metrics/identity_similarity_probe_summary.json")
    parser.add_argument("--report-output", default="reports/main_results/identity_similarity_probe_report.md")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--no-copy-figures-data", action="store_true")
    args = parser.parse_args()

    generations = pd.read_csv(ROOT / args.generations)
    items = pd.read_csv(ROOT / args.items)
    scored_rows = []
    for _, row in generations.iterrows():
        item = row.to_dict()
        text = str(item["generated_text"])
        identity_label = ""
        similarity_label = ""
        if item["task"] == "identity":
            identity_label = classify_identity(text, str(item["concept"]), str(item["mismatch_source"]))
        elif item["task"] == "similarity":
            similarity_label = classify_similarity(text, str(item["option_a_role"]), str(item["option_b_role"]))
        scored_rows.append(
            {
                **item,
                "identity_label": identity_label,
                "identity_valid": int(identity_label in {"target", "source"}),
                "identity_target_correct": int(identity_label == "target"),
                "identity_source_choice": int(identity_label == "source"),
                "similarity_label": similarity_label,
                "similarity_valid": int(similarity_label in {"target_neighbor", "source_neighbor"}),
                "similarity_target_neighbor_choice": int(similarity_label == "target_neighbor"),
                "similarity_source_neighbor_choice": int(similarity_label == "source_neighbor"),
            }
        )
    scored = pd.DataFrame(scored_rows)
    scored = scored.merge(
        items[
            [
                "concept",
                "target_source_things_similarity",
                "pair_image_similarity",
                "pair_difficulty",
                "source_attraction",
                "source_minus_target_margin",
                "target_neighbor_selection_margin",
                "source_neighbor_selection_margin",
            ]
        ],
        on="concept",
        how="left",
    )

    summary_rows = [summarize_condition(scored, condition, args.bootstrap, args.seed + idx * 13) for idx, condition in enumerate(CONDITIONS)]
    summary = pd.DataFrame(summary_rows)
    contrasts = pd.DataFrame(contrast_rows(summary))
    quartiles = pd.DataFrame(quartile_summary(scored))

    score_path = ROOT / args.score_output
    summary_path = ROOT / args.summary_output
    contrast_path = ROOT / args.contrast_output
    quartile_path = ROOT / args.quartile_output
    json_path = ROOT / args.json_output
    report_path = ROOT / args.report_output
    score_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(score_path, index=False)
    summary.to_csv(summary_path, index=False)
    contrasts.to_csv(contrast_path, index=False)
    quartiles.to_csv(quartile_path, index=False)
    write_json(
        json_path,
        {
            "n_generation_rows": int(len(scored)),
            "n_concepts": int(scored["concept"].nunique()),
            "condition_summary": summary.to_dict(orient="records"),
            "contrasts": contrasts.to_dict(orient="records"),
            "quartiles": quartiles.to_dict(orient="records"),
        },
    )
    write_text(report_path, "\n".join(report_lines(summary, contrasts, quartiles)))
    if not args.no_copy_figures_data:
        copy_to_figures_data([score_path, summary_path, contrast_path, quartile_path])
    append_run_log("Identity-Similarity Behavioral Probe", [f"Scored {len(scored)} rows and wrote {report_path.relative_to(ROOT)}."])
    print(f"Wrote {score_path.relative_to(ROOT)}")
    print(f"Wrote {summary_path.relative_to(ROOT)}")
    print(f"Wrote {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
