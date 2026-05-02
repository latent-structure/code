from __future__ import annotations

import argparse
from typing import Any, Callable

import numpy as np
import pandas as pd

from common import ROOT, append_run_log, metrics_path, percentile_interval, rankdata, write_csv, write_json
from hardening_common import write_text


ATTENTION_CONDITIONS = ["M_blank_image", "M_matched_image", "M_mismatched_image"]
LLAMA_CROSS_ATTENTION_LAYERS = [3, 8, 13, 18, 23, 28, 33, 38]
CONCEPT_OUTCOMES = [
    "source_attraction",
    "source_minus_target_margin",
    "rdm_disruption",
    "target_retention",
    "clip_target_margin",
    "clip_target_choice",
    "clip_source_choice",
]


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denom) if denom else float("nan")


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return float("nan")
    return pearson_corr(rankdata(x), rankdata(y))


def bootstrap_ci(x: np.ndarray, y: np.ndarray, fn: Callable[[np.ndarray, np.ndarray], float], *, seed: int, n_bootstrap: int) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, x.size, size=x.size)
        values.append(fn(x[idx], y[idx]))
    return percentile_interval(np.asarray(values, dtype=float), 0.95)


def correlation_rows(
    table: pd.DataFrame,
    *,
    predictors: list[str],
    outcomes: list[str],
    binary_outcomes: set[str],
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for predictor_idx, predictor in enumerate(predictors):
        x = table[predictor].to_numpy(dtype=float)
        for outcome_idx, outcome in enumerate(outcomes):
            y = table[outcome].to_numpy(dtype=float)
            fn = pearson_corr if outcome in binary_outcomes else spearman_corr
            statistic = "point_biserial_r" if outcome in binary_outcomes else "spearman_rho"
            estimate = fn(x, y)
            low, high = bootstrap_ci(
                x,
                y,
                fn,
                seed=seed + 1000 * predictor_idx + outcome_idx,
                n_bootstrap=n_bootstrap,
            )
            rows.append(
                {
                    "predictor": predictor,
                    "outcome": outcome,
                    "statistic": statistic,
                    "n": int(np.isfinite(x).sum()),
                    "estimate": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return rows


def load_attention(limit_concepts: int) -> pd.DataFrame:
    df = pd.read_csv(metrics_path("llama_cross_attention_probe.csv"))
    df = df[df["key_mode"].eq("cross_attention_visual_keys")].copy()
    df = df[df["condition"].isin(ATTENTION_CONDITIONS)].copy()
    df["concept"] = df["concept"].astype(str).str.lower()
    if limit_concepts:
        concepts = sorted(df["concept"].unique())[:limit_concepts]
        df = df[df["concept"].isin(concepts)].copy()
    df["attention_selectivity"] = df["concept_to_image_top1_attention"] - df["concept_to_image_normalized_entropy"]
    return df


def concept_attention_table(attention: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        attention.groupby(["concept", "condition"], as_index=False)
        .agg(
            top1_attention=("concept_to_image_top1_attention", "mean"),
            normalized_entropy=("concept_to_image_normalized_entropy", "mean"),
            attention_selectivity=("attention_selectivity", "mean"),
        )
        .pivot(index="concept", columns="condition")
    )
    grouped.columns = [f"{metric}_{condition}" for metric, condition in grouped.columns]
    grouped = grouped.reset_index()
    for metric in ["top1_attention", "normalized_entropy", "attention_selectivity"]:
        grouped[f"matched_minus_blank_{metric}"] = grouped[f"{metric}_M_matched_image"] - grouped[f"{metric}_M_blank_image"]
        grouped[f"mismatched_minus_blank_{metric}"] = grouped[f"{metric}_M_mismatched_image"] - grouped[f"{metric}_M_blank_image"]
        grouped[f"matched_minus_mismatched_{metric}"] = grouped[f"{metric}_M_matched_image"] - grouped[f"{metric}_M_mismatched_image"]
    return grouped


def load_concept_outcomes(limit_concepts: int) -> pd.DataFrame:
    bridge = pd.read_csv(metrics_path("behavior_geometry_bridge_llama_full.csv"))
    bridge = bridge.rename(columns={"target_margin": "bridge_target_margin"}).copy()
    bridge["concept"] = bridge["concept"].astype(str).str.lower()
    bridge = bridge[["concept", "source_attraction", "source_minus_target_margin", "rdm_disruption", "target_retention"]]

    clip = pd.read_csv(metrics_path("clip_forced_choice_behavior_llama_full.csv"))
    clip = clip[clip["condition"].eq("M_mismatched_image")].copy()
    clip["concept"] = clip["concept"].astype(str).str.lower()
    clip = clip.rename(
        columns={
            "target_margin": "clip_target_margin",
            "target_choice": "clip_target_choice",
            "source_choice": "clip_source_choice",
        }
    )[["concept", "clip_target_margin", "clip_target_choice", "clip_source_choice"]]
    out = bridge.merge(clip, on="concept", how="inner")
    if limit_concepts:
        concepts = sorted(out["concept"].unique())[:limit_concepts]
        out = out[out["concept"].isin(concepts)].copy()
    return out


def layer_attention_table(attention: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        attention.groupby(["attention_layer", "condition"], as_index=False)
        .agg(
            top1_attention=("concept_to_image_top1_attention", "mean"),
            normalized_entropy=("concept_to_image_normalized_entropy", "mean"),
            attention_selectivity=("attention_selectivity", "mean"),
        )
        .pivot(index="attention_layer", columns="condition")
    )
    grouped.columns = [f"{metric}_{condition}" for metric, condition in grouped.columns]
    grouped = grouped.reset_index().rename(columns={"attention_layer": "layer"})
    for metric in ["top1_attention", "normalized_entropy", "attention_selectivity"]:
        grouped[f"matched_minus_blank_{metric}"] = grouped[f"{metric}_M_matched_image"] - grouped[f"{metric}_M_blank_image"]
        grouped[f"mismatched_minus_blank_{metric}"] = grouped[f"{metric}_M_mismatched_image"] - grouped[f"{metric}_M_blank_image"]
        grouped[f"matched_minus_mismatched_{metric}"] = grouped[f"{metric}_M_matched_image"] - grouped[f"{metric}_M_mismatched_image"]
    return grouped


def load_layer_outcomes() -> pd.DataFrame:
    mixture = pd.read_csv(metrics_path("layerwise_prompt_image_mixture_llama.csv"))
    mixture = mixture[["layer", "prompt_weight", "matched_image_weight", "mixture_r2"]].copy()
    mixture["image_minus_prompt_weight"] = mixture["matched_image_weight"] - mixture["prompt_weight"]

    retention = pd.read_csv(metrics_path("layerwise_mismatched_identity_retention_llama.csv"))
    retention = retention[retention["mismatch_mode"].eq("all")][["layer", "text_retention_rate", "image_hijack_rate", "mean_source_minus_target_distance"]]

    internal = pd.read_csv(metrics_path("layerwise_internal_visual_alignment.csv"))
    internal = internal[internal["family"].eq("llama")].copy()
    internal_pivot = internal.pivot_table(index="layer", columns="condition", values="rsa_score", aggfunc="mean").reset_index()
    internal_rows = pd.DataFrame({"layer": internal_pivot["layer"]})
    if {"M_matched_image", "M_text_only"}.issubset(internal_pivot.columns):
        internal_rows["internal_visual_matched_minus_text"] = internal_pivot["M_matched_image"] - internal_pivot["M_text_only"]
    if {"M_prompt_plus_matched_image", "M_text_only"}.issubset(internal_pivot.columns):
        internal_rows["internal_visual_prompt_plus_minus_text"] = internal_pivot["M_prompt_plus_matched_image"] - internal_pivot["M_text_only"]

    external = pd.read_csv(metrics_path("cross_family_layerwise_rsa_trajectories.csv"))
    external = external[external["family_name"].eq("llama")].copy()
    external_pieces = []
    for anchor, sub in external.groupby("anchor_name"):
        pivot = sub.pivot_table(index="layer", columns="condition", values="rsa_score", aggfunc="mean").reset_index()
        if {"M_matched_image", "T_prompt_primary"}.issubset(pivot.columns):
            external_pieces.append(
                pd.DataFrame(
                    {
                        "layer": pivot["layer"],
                        f"external_rsa_matched_minus_prompt_{anchor}": pivot["M_matched_image"] - pivot["T_prompt_primary"],
                    }
                )
            )
    layer_index = pd.DataFrame({"layer": sorted(set(mixture["layer"]) | set(retention["layer"]) | set(internal_rows["layer"]))})
    out = layer_index.merge(mixture, on="layer", how="left").merge(retention, on="layer", how="left").merge(internal_rows, on="layer", how="left")
    for piece in external_pieces:
        out = out.merge(piece, on="layer", how="left")
    return out


def layer_correlations(layer_table: pd.DataFrame, predictors: list[str], outcomes: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for predictor in predictors:
        for outcome in outcomes:
            x = layer_table[predictor].to_numpy(dtype=float)
            y = layer_table[outcome].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            rows.append(
                {
                    "predictor": predictor,
                    "outcome": outcome,
                    "statistic": "spearman_rho",
                    "n": int(mask.sum()),
                    "estimate": spearman_corr(x, y),
                }
            )
    return rows


def write_report(
    *,
    concept_rows: list[dict[str, Any]],
    layer_rows: list[dict[str, Any]],
    concept_n: int,
    layer_n: int,
) -> None:
    top_concept = sorted(concept_rows, key=lambda row: abs(float(row["estimate"])), reverse=True)[:12]
    top_layer = sorted(layer_rows, key=lambda row: abs(float(row["estimate"])), reverse=True)[:16]
    lines = [
        "# Llama Attention-Geometry Coupling",
        "",
        f"- Concepts with complete attention/outcome joins: `{concept_n}`",
        f"- Cross-attention layers analyzed: `{layer_n}`",
        "- Concept-level CIs use bootstrap resampling over concepts.",
        "- Layer-level correlations are descriptive because only eight cross-attention layers are available.",
        "",
        "## Strongest Concept-Level Couplings",
        "",
        "| Predictor | Outcome | Statistic | Estimate | 95% CI | n |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in top_concept:
        lines.append(
            f"| `{row['predictor']}` | `{row['outcome']}` | `{row['statistic']}` | {row['estimate']:+.4f} | "
            f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | {row['n']} |"
        )
    lines.extend(
        [
            "",
            "## Strongest Layer-Level Couplings",
            "",
            "| Predictor | Outcome | Spearman rho | n |",
            "|---|---|---:|---:|",
        ]
    )
    for row in top_layer:
        lines.append(f"| `{row['predictor']}` | `{row['outcome']}` | {row['estimate']:+.4f} | {row['n']} |")
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "If attention predictors couple weakly to external RSA but real-image conditions still increase attention selectivity, Llama should be framed as visual routing without strong reference-space convergence.",
        ]
    )
    write_text(ROOT / "reports" / "main_results" / "llama_attention_geometry_coupling_report.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Couple Llama visual attention metrics to geometry and behavior outcomes.")
    parser.add_argument("--limit-concepts", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260501)
    args = parser.parse_args()

    attention = load_attention(args.limit_concepts)
    concept_attention = concept_attention_table(attention)
    concept_outcomes = load_concept_outcomes(args.limit_concepts)
    concept_table = concept_attention.merge(concept_outcomes, on="concept", how="inner")
    required_n = len(set(concept_attention["concept"]) & set(concept_outcomes["concept"]))
    if len(concept_table) < max(10, int(0.95 * required_n)):
        raise RuntimeError(f"Concept join retained {len(concept_table)} / {required_n} concepts.")

    concept_predictors = [
        "top1_attention_M_matched_image",
        "top1_attention_M_mismatched_image",
        "normalized_entropy_M_matched_image",
        "normalized_entropy_M_mismatched_image",
        "attention_selectivity_M_matched_image",
        "attention_selectivity_M_mismatched_image",
        "matched_minus_blank_top1_attention",
        "mismatched_minus_blank_top1_attention",
        "matched_minus_mismatched_top1_attention",
        "matched_minus_blank_normalized_entropy",
        "mismatched_minus_blank_normalized_entropy",
        "matched_minus_mismatched_normalized_entropy",
    ]
    concept_corrs = correlation_rows(
        concept_table,
        predictors=concept_predictors,
        outcomes=CONCEPT_OUTCOMES,
        binary_outcomes={"target_retention", "clip_target_choice", "clip_source_choice"},
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )

    layer_attention = layer_attention_table(attention)
    layer_outcomes = load_layer_outcomes()
    layer_table = layer_attention.merge(layer_outcomes, on="layer", how="left")
    if len(layer_table) != len(LLAMA_CROSS_ATTENTION_LAYERS):
        raise RuntimeError(f"Expected 8 cross-attention layers, got {len(layer_table)}.")
    layer_predictors = [
        "top1_attention_M_matched_image",
        "top1_attention_M_mismatched_image",
        "normalized_entropy_M_matched_image",
        "normalized_entropy_M_mismatched_image",
        "matched_minus_blank_top1_attention",
        "mismatched_minus_blank_top1_attention",
        "matched_minus_mismatched_top1_attention",
    ]
    layer_outcome_cols = [
        col
        for col in layer_table.columns
        if col
        in {
            "prompt_weight",
            "matched_image_weight",
            "image_minus_prompt_weight",
            "mixture_r2",
            "text_retention_rate",
            "image_hijack_rate",
            "mean_source_minus_target_distance",
            "internal_visual_matched_minus_text",
            "internal_visual_prompt_plus_minus_text",
            "external_rsa_matched_minus_prompt_THINGS",
            "external_rsa_matched_minus_prompt_controlled_THINGS",
            "external_rsa_matched_minus_prompt_SigLIP2",
            "external_rsa_matched_minus_prompt_lancaster_perceptual",
        }
    ]
    layer_corrs = layer_correlations(layer_table, layer_predictors, layer_outcome_cols)

    write_csv(metrics_path("llama_attention_geometry_coupling_concept.csv"), concept_table.to_dict("records"), list(concept_table.columns))
    write_csv(
        metrics_path("llama_attention_geometry_coupling_concept_correlations.csv"),
        concept_corrs,
        ["predictor", "outcome", "statistic", "n", "estimate", "ci95_low", "ci95_high"],
    )
    write_csv(metrics_path("llama_attention_geometry_coupling_layer.csv"), layer_table.to_dict("records"), list(layer_table.columns))
    write_csv(
        metrics_path("llama_attention_geometry_coupling_layer_correlations.csv"),
        layer_corrs,
        ["predictor", "outcome", "statistic", "n", "estimate"],
    )
    write_json(
        metrics_path("llama_attention_geometry_coupling_summary.json"),
        {
            "n_concepts": int(len(concept_table)),
            "n_layers": int(len(layer_table)),
            "concept_predictors": concept_predictors,
            "concept_outcomes": CONCEPT_OUTCOMES,
            "layer_predictors": layer_predictors,
            "layer_outcomes": layer_outcome_cols,
        },
    )
    write_report(concept_rows=concept_corrs, layer_rows=layer_corrs, concept_n=len(concept_table), layer_n=len(layer_table))
    append_run_log(
        "Llama Attention-Geometry Coupling",
        [
            "Computed concept-level and layer-level attention coupling diagnostics.",
            "Wrote reports/main_results/llama_attention_geometry_coupling_report.md.",
        ],
    )


if __name__ == "__main__":
    main()
