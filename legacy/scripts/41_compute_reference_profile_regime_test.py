from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from common import ROOT, append_run_log, metrics_path, rankdata, read_csv, spearman_corr, write_csv, write_json
from hardening_common import write_text


EXTERNAL_ANCHORS = ["THINGS", "controlled_THINGS", "SigLIP2", "CLIP_ViT_L_14", "DINOv2", "lancaster_perceptual"]
FAMILIES = ["qwen", "mistral", "llama"]
EPSILON = 1e-6


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0.0:
        return 0.0
    return float(np.dot(x, y) / denom)


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0.0:
        return 0.0
    return float(np.dot(x, y) / denom)


def regression_slope_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ beta
    residual = y - fitted
    return float(beta[1]), float(beta[0]), float(np.linalg.norm(residual))


def score_lookup() -> dict[tuple[str, str, str], float]:
    rows = read_csv(metrics_path("cross_family_rsa_full.csv"))
    lookup = {}
    for row in rows:
        if row["row_type"] != "condition_score":
            continue
        lookup[(row["family_name"], row["anchor_name"], row["condition"])] = float(row["rsa_score"])
    return lookup


def internal_visual_scores(family: str) -> dict[str, float]:
    path = metrics_path(f"internal_visual_tower_summary_{family}.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {condition: float(value) for condition, value in payload["scores"].items()}


def build_anchor_rows() -> list[dict[str, Any]]:
    lookup = score_lookup()
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        for anchor in EXTERNAL_ANCHORS:
            text_only = lookup[(family, anchor, "M_text_only")]
            prompt = lookup[(family, anchor, "T_prompt_primary")]
            matched = lookup[(family, anchor, "M_matched_image")]
            rows.append(anchor_row(family, anchor, "external", text_only, prompt, matched))
        internal = internal_visual_scores(family)
        rows.append(
            anchor_row(
                family,
                "internal_visual_tower",
                "internal",
                internal["M_text_only"],
                internal["T_prompt_primary"],
                internal["M_matched_image"],
            )
        )
    return rows


def anchor_row(family: str, anchor: str, anchor_type: str, text_only: float, prompt: float, matched: float) -> dict[str, Any]:
    prompt_gain = prompt - text_only
    grounding_gain = matched - text_only
    differential = grounding_gain - prompt_gain
    return {
        "family": family,
        "anchor": anchor,
        "anchor_type": anchor_type,
        "text_only_rsa": text_only,
        "prompt_rsa": prompt,
        "matched_rsa": matched,
        "prompt_gain_vs_text": prompt_gain,
        "grounding_gain_vs_text": grounding_gain,
        "differential_grounding_minus_prompt_gain": differential,
        "gain_ratio_grounding_over_prompt_abs": grounding_gain / max(abs(prompt_gain), EPSILON),
        "opposite_sign_gains": int(np.sign(prompt_gain) != 0 and np.sign(grounding_gain) != 0 and np.sign(prompt_gain) != np.sign(grounding_gain)),
        "grounding_exceeds_prompt_by_0p02": int(differential >= 0.02),
    }


def summarize_family(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for family in FAMILIES:
        subset = [row for row in rows if row["family"] == family]
        prompt = np.asarray([row["prompt_gain_vs_text"] for row in subset], dtype=float)
        grounding = np.asarray([row["grounding_gain_vs_text"] for row in subset], dtype=float)
        slope, intercept, residual_norm = regression_slope_intercept(prompt, grounding)
        summaries.append(
            {
                "family": family,
                "num_anchors": len(subset),
                "prompt_gain_l2": float(np.linalg.norm(prompt)),
                "grounding_gain_l2": float(np.linalg.norm(grounding)),
                "grounding_prompt_l2_ratio": float(np.linalg.norm(grounding) / max(np.linalg.norm(prompt), EPSILON)),
                "gain_profile_cosine": cosine_similarity(prompt, grounding),
                "gain_profile_pearson": pearson_corr(prompt, grounding),
                "gain_profile_spearman": spearman_corr(rankdata(prompt), rankdata(grounding)),
                "regression_slope_grounding_on_prompt": slope,
                "regression_intercept_grounding_on_prompt": intercept,
                "regression_residual_norm": residual_norm,
                "mean_prompt_gain": float(np.mean(prompt)),
                "mean_grounding_gain": float(np.mean(grounding)),
                "mean_differential_gain": float(np.mean(grounding - prompt)),
                "opposite_sign_anchor_count": int(sum(row["opposite_sign_gains"] for row in subset)),
                "grounding_exceeds_prompt_by_0p02_count": int(sum(row["grounding_exceeds_prompt_by_0p02"] for row in subset)),
            }
        )
    return summaries


def load_local_human_crossover() -> dict[str, Any] | None:
    path = metrics_path("human_local_geometry_summary.json")
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "prompt_local_human": payload["all_concepts_prompt_mean"],
        "matched_local_human": payload["all_concepts_matched_mean"],
        "degraded_local_human": payload["all_concepts_degraded_mean"],
        "prompt_minus_matched": payload["all_concepts_prompt_minus_matched"],
    }


def interpretation(summary_rows: list[dict[str, Any]], anchor_rows: list[dict[str, Any]]) -> str:
    qwen = next(row for row in summary_rows if row["family"] == "qwen")
    if qwen["gain_profile_cosine"] < 0.5 or qwen["opposite_sign_anchor_count"] >= 2:
        return "distinct_direction"
    if qwen["grounding_prompt_l2_ratio"] >= 2.0 and qwen["grounding_exceeds_prompt_by_0p02_count"] >= 4:
        return "non_scalar_amplification"
    if qwen["gain_profile_cosine"] >= 0.5 and qwen["grounding_prompt_l2_ratio"] >= 2.0:
        return "same_direction_larger_grounding"
    return "weak_or_mixed"


def report_lines(summary: dict[str, Any]) -> list[str]:
    lines = [
        "# Reference-Profile Regime Test",
        "",
        f"- Interpretation label: `{summary['interpretation_label']}`",
        "- Baseline: `M_text_only` in the same VLM family.",
        "- Prompt gain: `T_prompt_primary - M_text_only`.",
        "- Grounding gain: `M_matched_image - M_text_only`.",
        "",
        "## Family-Level Profile Diagnostics",
        "",
        "| Family | Anchors | Cosine | Pearson | L2 ratio | Mean prompt gain | Mean grounding gain | Opposite signs | Grounding > prompt by .02 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["family_summaries"]:
        lines.append(
            f"| {row['family']} | {row['num_anchors']} | {row['gain_profile_cosine']:.4f} | "
            f"{row['gain_profile_pearson']:.4f} | {row['grounding_prompt_l2_ratio']:.2f} | "
            f"{row['mean_prompt_gain']:+.4f} | {row['mean_grounding_gain']:+.4f} | "
            f"{row['opposite_sign_anchor_count']} | {row['grounding_exceeds_prompt_by_0p02_count']} |"
        )
    lines.extend(
        [
            "",
            "## Anchor-Level Gains",
            "",
            "| Family | Anchor | Prompt gain | Grounding gain | Grounding - prompt gain | Ratio |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["anchor_rows"]:
        lines.append(
            f"| {row['family']} | {row['anchor']} | {row['prompt_gain_vs_text']:+.4f} | "
            f"{row['grounding_gain_vs_text']:+.4f} | {row['differential_grounding_minus_prompt_gain']:+.4f} | "
            f"{row['gain_ratio_grounding_over_prompt_abs']:.2f} |"
        )
    if summary.get("local_human_crossover"):
        local = summary["local_human_crossover"]
        lines.extend(
            [
                "",
                "## Local-Human Crossover",
                "",
                f"- Prompt local-human alignment: `{local['prompt_local_human']:.4f}`",
                f"- Matched local-human alignment: `{local['matched_local_human']:.4f}`",
                f"- Prompt - matched local-human gap: `{local['prompt_minus_matched']:+.4f}`",
            ]
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare prompting and grounding reference-alignment gain profiles.")
    parser.parse_args()

    anchor_rows = build_anchor_rows()
    family_summaries = summarize_family(anchor_rows)
    summary = {
        "anchors": EXTERNAL_ANCHORS + ["internal_visual_tower"],
        "families": FAMILIES,
        "baseline_condition": "M_text_only",
        "prompt_condition": "T_prompt_primary",
        "grounding_condition": "M_matched_image",
        "anchor_rows": anchor_rows,
        "family_summaries": family_summaries,
        "local_human_crossover": load_local_human_crossover(),
    }
    summary["interpretation_label"] = interpretation(family_summaries, anchor_rows)
    write_csv(
        metrics_path("reference_profile_regime_test.csv"),
        anchor_rows,
        [
            "family",
            "anchor",
            "anchor_type",
            "text_only_rsa",
            "prompt_rsa",
            "matched_rsa",
            "prompt_gain_vs_text",
            "grounding_gain_vs_text",
            "differential_grounding_minus_prompt_gain",
            "gain_ratio_grounding_over_prompt_abs",
            "opposite_sign_gains",
            "grounding_exceeds_prompt_by_0p02",
        ],
    )
    write_csv(
        metrics_path("reference_profile_regime_test_summary.csv"),
        family_summaries,
        [
            "family",
            "num_anchors",
            "prompt_gain_l2",
            "grounding_gain_l2",
            "grounding_prompt_l2_ratio",
            "gain_profile_cosine",
            "gain_profile_pearson",
            "gain_profile_spearman",
            "regression_slope_grounding_on_prompt",
            "regression_intercept_grounding_on_prompt",
            "regression_residual_norm",
            "mean_prompt_gain",
            "mean_grounding_gain",
            "mean_differential_gain",
            "opposite_sign_anchor_count",
            "grounding_exceeds_prompt_by_0p02_count",
        ],
    )
    write_json(metrics_path("reference_profile_regime_test_summary.json"), summary)
    write_text(ROOT / "reports" / "main_results" / "reference_profile_regime_test_report.md", "\n".join(report_lines(summary)))
    append_run_log(
        "Reference-Profile Regime Test",
        [
            f"Computed gain-profile diagnostics for {len(FAMILIES)} families and {len(EXTERNAL_ANCHORS) + 1} anchors.",
            f"Interpretation label: {summary['interpretation_label']}.",
        ],
    )


if __name__ == "__main__":
    main()
