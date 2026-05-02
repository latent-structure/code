from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import ROOT, append_run_log, read_csv, write_csv, write_json


VISUAL_WORDS = {
    "bright",
    "dark",
    "shiny",
    "glossy",
    "matte",
    "transparent",
    "opaque",
    "reflective",
    "color",
    "colored",
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "black",
    "white",
    "gray",
    "brown",
    "gold",
    "silver",
    "round",
    "square",
    "flat",
    "curved",
    "long",
    "short",
    "wide",
    "narrow",
    "large",
    "small",
    "pattern",
    "striped",
    "spotted",
    "smooth",
    "rough",
    "textured",
    "surface",
    "edge",
    "shape",
    "shadow",
    "glow",
    "visual",
    "visible",
}

TACTILE_WORDS = {
    "soft",
    "hard",
    "rough",
    "smooth",
    "warm",
    "cold",
    "cool",
    "wet",
    "dry",
    "sticky",
    "slippery",
    "grainy",
    "fuzzy",
    "fluffy",
    "sharp",
    "heavy",
    "light",
    "tactile",
    "touch",
    "feel",
}

AUDITORY_WORDS = {
    "sound",
    "sounds",
    "loud",
    "quiet",
    "ringing",
    "buzzing",
    "humming",
    "crackling",
    "clicking",
    "noise",
    "noisy",
    "silent",
    "auditory",
    "hear",
}

SMELL_TASTE_WORDS = {
    "smell",
    "scent",
    "aroma",
    "fragrant",
    "odor",
    "taste",
    "flavor",
    "sweet",
    "sour",
    "bitter",
    "salty",
    "savory",
    "spicy",
    "pungent",
}

GENERIC_MARKERS = {
    "typically",
    "usually",
    "commonly",
    "often",
    "generally",
    "used",
    "use",
    "purpose",
    "function",
    "designed",
    "type",
    "kind",
    "category",
    "example",
}

EXEMPLAR_SPECIFIC_WORDS = VISUAL_WORDS | TACTILE_WORDS | {
    "background",
    "foreground",
    "beside",
    "behind",
    "under",
    "above",
    "near",
    "on",
    "wooden",
    "metal",
    "metallic",
    "plastic",
    "fabric",
    "glass",
    "ceramic",
    "leather",
    "painted",
    "worn",
    "cracked",
    "polished",
    "dusty",
    "wet",
    "dry",
}

SENSORY_WORDS = VISUAL_WORDS | TACTILE_WORDS | AUDITORY_WORDS | SMELL_TASTE_WORDS
LANCASTER_COLUMNS = [
    "Auditory.mean",
    "Gustatory.mean",
    "Haptic.mean",
    "Interoceptive.mean",
    "Olfactory.mean",
    "Visual.mean",
    "Foot_leg.mean",
    "Hand_arm.mean",
    "Head.mean",
    "Mouth.mean",
    "Torso.mean",
]
PERCEPTUAL_COLUMNS = [
    "Auditory.mean",
    "Gustatory.mean",
    "Haptic.mean",
    "Interoceptive.mean",
    "Olfactory.mean",
    "Visual.mean",
]


def strip_think_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</?think>", " ", text, flags=re.IGNORECASE)
    return " ".join(text.replace("\n", " ").split())


def normalize_text(text: str) -> str:
    text = strip_think_blocks(text).lower().replace("_", " ").replace("-", " ")
    return re.sub(r"[^a-z0-9 ]+", " ", text)


def tokens(text: str) -> list[str]:
    return [part for part in normalize_text(text).split() if part]


def contains_concept(text: str, concept: str) -> bool:
    text_norm = f" {normalize_text(text)} "
    concept_norm = " ".join(tokens(concept))
    if not concept_norm:
        return False
    return f" {concept_norm} " in text_norm


def count_lexicon_words(text: str, lexicon: set[str]) -> int:
    return sum(1 for token in tokens(text) if token in lexicon)


def safe_rate(count: float, denominator: float, scale: float = 100.0) -> float:
    return (count / denominator) * scale if denominator > 0 else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    weight = pos - lo
    return float((1.0 - weight) * ordered[lo] + weight * ordered[hi])


def bootstrap_ci(values: list[float], n_bootstrap: int, seed: int) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    boot = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(len(values))] for _ in values]
        boot.append(mean(sample))
    return percentile(boot, 0.025), percentile(boot, 0.975)


def paired_permutation_p(differences: list[float], n_permutations: int, seed: int) -> float:
    if not differences:
        return 1.0
    observed = abs(mean(differences))
    if observed == 0:
        return 1.0
    rng = random.Random(seed)
    count = 0
    for _ in range(n_permutations):
        perm_mean = mean([value if rng.random() < 0.5 else -value for value in differences])
        if abs(perm_mean) >= observed:
            count += 1
    return (count + 1.0) / (n_permutations + 1.0)


def bh_fdr(p_values: list[float]) -> list[float]:
    n = len(p_values)
    order = sorted(range(n), key=lambda idx: p_values[idx])
    adjusted = [1.0] * n
    running = 1.0
    for rank_from_end, idx in enumerate(reversed(order), start=1):
        rank = n - rank_from_end + 1
        running = min(running, p_values[idx] * n / rank)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def load_lancaster_norms(path: Path = ROOT / "Lancaster_sensorimotor.csv") -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    norms = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            word = normalize_text(row["Word"]).strip()
            if not word:
                continue
            norms[word] = {column: float(row.get(column) or 0.0) for column in LANCASTER_COLUMNS}
    return norms


def lancaster_scores(text: str, norms: dict[str, dict[str, float]]) -> dict[str, float]:
    matched = [norms[token] for token in tokens(text) if token in norms]
    if not matched:
        return {
            "lancaster_token_overlap": 0,
            "lancaster_visual_mean": 0.0,
            "lancaster_perceptual_mean": 0.0,
            "lancaster_sensorimotor_mean": 0.0,
        }
    visual = mean([row["Visual.mean"] for row in matched])
    perceptual = mean([mean([row[column] for column in PERCEPTUAL_COLUMNS]) for row in matched])
    sensorimotor = mean([mean([row[column] for column in LANCASTER_COLUMNS]) for row in matched])
    return {
        "lancaster_token_overlap": len(matched),
        "lancaster_visual_mean": visual,
        "lancaster_perceptual_mean": perceptual,
        "lancaster_sensorimotor_mean": sensorimotor,
    }


def score_row(row: dict[str, str], lancaster_norms: dict[str, dict[str, float]]) -> dict[str, Any]:
    generated_raw = row["generated_text"]
    generated = strip_think_blocks(generated_raw)
    word_count = len(tokens(generated))
    visual_count = count_lexicon_words(generated, VISUAL_WORDS)
    sensory_count = count_lexicon_words(generated, SENSORY_WORDS)
    generic_count = count_lexicon_words(generated, GENERIC_MARKERS)
    exemplar_count = count_lexicon_words(generated, EXEMPLAR_SPECIFIC_WORDS)
    target_retention = contains_concept(generated, row["concept"])
    source = row.get("mismatch_source", "")
    source_leakage = row["condition"] == "M_mismatched_image" and bool(source) and contains_concept(generated, source)
    lancaster = lancaster_scores(generated, lancaster_norms)
    return {
        **row,
        "generated_text_stripped": generated,
        "think_block_removed": int(generated != generated_raw),
        "target_retention": int(target_retention),
        "mismatched_source_leakage": int(source_leakage),
        "visual_word_count": visual_count,
        "sensory_word_count": sensory_count,
        "generic_marker_count": generic_count,
        "exemplar_specific_count": exemplar_count,
        "output_word_count": word_count,
        "visual_word_rate_per_100": safe_rate(visual_count, word_count),
        "sensory_word_rate_per_100": safe_rate(sensory_count, word_count),
        "generic_marker_rate_per_100": safe_rate(generic_count, word_count),
        "exemplar_specific_rate_per_100": safe_rate(exemplar_count, word_count),
        **lancaster,
        "lancaster_token_overlap_rate_per_100": safe_rate(float(lancaster["lancaster_token_overlap"]), word_count),
        "lancaster_visual_per_100": safe_rate(float(lancaster["lancaster_visual_mean"]), 1.0),
        "lancaster_perceptual_per_100": safe_rate(float(lancaster["lancaster_perceptual_mean"]), 1.0),
    }


CONDITION_METRICS = [
    "target_retention",
    "mismatched_source_leakage",
    "visual_word_count",
    "sensory_word_count",
    "generic_marker_count",
    "exemplar_specific_count",
    "output_word_count",
    "visual_word_rate_per_100",
    "sensory_word_rate_per_100",
    "generic_marker_rate_per_100",
    "exemplar_specific_rate_per_100",
    "lancaster_token_overlap",
    "lancaster_token_overlap_rate_per_100",
    "lancaster_visual_mean",
    "lancaster_perceptual_mean",
    "lancaster_sensorimotor_mean",
]


PRIMARY_CONTRASTS = [
    ("M_matched_image", "M_text_only", "visual_word_rate_per_100"),
    ("M_matched_image", "M_blank_image", "visual_word_rate_per_100"),
    ("M_matched_image", "T_prompt_primary", "visual_word_rate_per_100"),
    ("M_prompt_plus_matched_image", "M_matched_image", "visual_word_rate_per_100"),
    ("M_matched_image", "M_text_only", "sensory_word_rate_per_100"),
    ("M_matched_image", "M_blank_image", "sensory_word_rate_per_100"),
    ("M_matched_image", "M_text_only", "lancaster_visual_mean"),
    ("M_matched_image", "M_blank_image", "lancaster_visual_mean"),
]


def summarize(scored_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        by_condition[row["condition"]].append(row)
    condition_summary = {}
    for condition, rows in sorted(by_condition.items()):
        condition_summary[condition] = {"n": len(rows)}
        for metric in CONDITION_METRICS:
            condition_summary[condition][f"mean_{metric}"] = mean([float(row.get(metric, 0.0)) for row in rows])

    def contrast(metric: str, left: str, right: str) -> float:
        return condition_summary.get(left, {}).get(f"mean_{metric}", 0.0) - condition_summary.get(right, {}).get(f"mean_{metric}", 0.0)

    return {
        "conditions": condition_summary,
        "contrasts": {
            "matched_minus_prompt_visual_rate_per_100": contrast("visual_word_rate_per_100", "M_matched_image", "T_prompt_primary"),
            "matched_minus_text_only_visual_rate_per_100": contrast("visual_word_rate_per_100", "M_matched_image", "M_text_only"),
            "prompt_plus_image_minus_matched_visual_rate_per_100": contrast("visual_word_rate_per_100", "M_prompt_plus_matched_image", "M_matched_image"),
            "matched_minus_blank_visual_rate_per_100": contrast("visual_word_rate_per_100", "M_matched_image", "M_blank_image"),
            "mismatched_source_leakage_rate": condition_summary.get("M_mismatched_image", {}).get("mean_mismatched_source_leakage", 0.0),
            "mismatched_target_retention_rate": condition_summary.get("M_mismatched_image", {}).get("mean_target_retention", 0.0),
        },
        "method_note": (
            "Deterministic output-level scoring of generated descriptions with concept-paired uncertainty; "
            "this is a secondary behavior probe, not a task-accuracy benchmark."
        ),
    }


def paired_differences(rows: list[dict[str, Any]], left: str, right: str, metric: str) -> list[float]:
    by_concept_condition = {(row["concept"], row["condition"]): row for row in rows}
    concepts = sorted({row["concept"] for row in rows})
    differences = []
    for concept in concepts:
        left_row = by_concept_condition.get((concept, left))
        right_row = by_concept_condition.get((concept, right))
        if left_row is None or right_row is None:
            continue
        differences.append(float(left_row[metric]) - float(right_row[metric]))
    return differences


def contrast_rows(
    scored_rows: list[dict[str, Any]],
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    p_values = []
    for idx, (left, right, metric) in enumerate(PRIMARY_CONTRASTS):
        differences = paired_differences(scored_rows, left, right, metric)
        ci_low, ci_high = bootstrap_ci(differences, n_bootstrap, seed + idx)
        p_value = paired_permutation_p(differences, n_permutations, seed + 1000 + idx)
        p_values.append(p_value)
        rows.append(
            {
                "left_condition": left,
                "right_condition": right,
                "metric": metric,
                "n_pairs": len(differences),
                "mean_difference": mean(differences),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "permutation_p": p_value,
                "fdr_q": 1.0,
            }
        )
    q_values = bh_fdr(p_values)
    for row, q_value in zip(rows, q_values):
        row["fdr_q"] = q_value
    return rows


def report_lines(summary: dict[str, Any], contrasts: list[dict[str, Any]]) -> list[str]:
    lines = [
        "# Behavior Probe Report",
        "",
        "This is a secondary output-level probe. It tests whether geometry-level differences are visible in deterministic generated descriptions; it is not a downstream task benchmark.",
        "",
        "## Condition Means",
        "",
        "| Condition | n | Target retention | Source leakage | Visual /100 | Sensory /100 | Lancaster visual | Generic /100 | Exemplar /100 | Words |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, row in summary["conditions"].items():
        lines.append(
            f"| `{condition}` | {row['n']} | {row['mean_target_retention']:.4f} | "
            f"{row['mean_mismatched_source_leakage']:.4f} | {row['mean_visual_word_rate_per_100']:.4f} | "
            f"{row['mean_sensory_word_rate_per_100']:.4f} | {row['mean_lancaster_visual_mean']:.4f} | "
            f"{row['mean_generic_marker_rate_per_100']:.4f} | {row['mean_exemplar_specific_rate_per_100']:.4f} | "
            f"{row['mean_output_word_count']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Prespecified Paired Contrasts",
            "",
            "| Left | Right | Metric | n | Mean diff | 95% CI | p | FDR q |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in contrasts:
        lines.append(
            f"| `{row['left_condition']}` | `{row['right_condition']}` | `{row['metric']}` | "
            f"{row['n_pairs']} | {row['mean_difference']:.4f} | "
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | {row['permutation_p']:.4f} | {row['fdr_q']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Key Descriptive Contrasts",
            "",
            f"- Matched - prompt visual words per 100: `{summary['contrasts']['matched_minus_prompt_visual_rate_per_100']:.4f}`",
            f"- Matched - text-only visual words per 100: `{summary['contrasts']['matched_minus_text_only_visual_rate_per_100']:.4f}`",
            f"- Prompt+image - matched visual words per 100: `{summary['contrasts']['prompt_plus_image_minus_matched_visual_rate_per_100']:.4f}`",
            f"- Matched - blank visual words per 100: `{summary['contrasts']['matched_minus_blank_visual_rate_per_100']:.4f}`",
            f"- Mismatched target-retention rate: `{summary['contrasts']['mismatched_target_retention_rate']:.4f}`",
            f"- Mismatched source-leakage rate: `{summary['contrasts']['mismatched_source_leakage_rate']:.4f}`",
        ]
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Score behavior-probe generations with deterministic lexical metrics.")
    parser.add_argument("--input", default="outputs/generations/behavior_probe_v2_generations.csv")
    parser.add_argument("--output-stem", default="behavior_probe_v2")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    input_path = ROOT / args.input
    suffix = "_smoke" if args.smoke else ""
    if args.smoke and input_path.name == "behavior_probe_v2_generations.csv":
        input_path = ROOT / "outputs" / "generations" / "behavior_probe_v2_generations_smoke.csv"
    rows = read_csv(input_path)
    lancaster_norms = load_lancaster_norms()
    scored_rows = [score_row(row, lancaster_norms) for row in rows]
    summary = summarize(scored_rows)
    contrasts = contrast_rows(scored_rows, args.bootstrap, args.permutations, args.seed)
    summary["prespecified_contrasts"] = contrasts
    summary["generation_file"] = str(input_path.relative_to(ROOT))
    summary["n_bootstrap"] = args.bootstrap
    summary["n_permutations"] = args.permutations
    summary["lancaster_norm_rows"] = len(lancaster_norms)

    fieldnames = [
        "concept",
        "subtype",
        "condition",
        "model_id",
        "prompt",
        "image_path",
        "mismatch_source",
        "prompt_template_version",
        "generated_text",
        "generated_text_stripped",
        "think_block_removed",
        "word_count",
        "seed",
        "max_new_tokens",
        "target_retention",
        "mismatched_source_leakage",
        "visual_word_count",
        "sensory_word_count",
        "generic_marker_count",
        "exemplar_specific_count",
        "output_word_count",
        "visual_word_rate_per_100",
        "sensory_word_rate_per_100",
        "generic_marker_rate_per_100",
        "exemplar_specific_rate_per_100",
        "lancaster_token_overlap",
        "lancaster_token_overlap_rate_per_100",
        "lancaster_visual_mean",
        "lancaster_perceptual_mean",
        "lancaster_sensorimotor_mean",
        "lancaster_visual_per_100",
        "lancaster_perceptual_per_100",
    ]
    write_csv(ROOT / "outputs" / "metrics" / f"{args.output_stem}_scores{suffix}.csv", scored_rows, fieldnames)
    write_csv(
        ROOT / "outputs" / "metrics" / f"{args.output_stem}_contrasts{suffix}.csv",
        contrasts,
        ["left_condition", "right_condition", "metric", "n_pairs", "mean_difference", "ci95_low", "ci95_high", "permutation_p", "fdr_q"],
    )
    write_json(ROOT / "outputs" / "metrics" / f"{args.output_stem}_summary{suffix}.json", summary)
    report_path = ROOT / "outputs" / "reports" / f"{args.output_stem}_report{suffix}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines(summary, contrasts)) + "\n", encoding="utf-8")
    append_run_log(
        "Behavior Probe Scoring",
        [
            f"Scored {len(scored_rows)} generation rows from {input_path.relative_to(ROOT)}.",
            f"Lancaster norm rows available: {len(lancaster_norms)}.",
            f"Bootstrap resamples: {args.bootstrap}; permutations: {args.permutations}.",
        ],
    )


if __name__ == "__main__":
    main()
