from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from common import ROOT, append_run_log, metrics_path, write_csv, write_json
from hardening_common import write_text


LLAMA_CROSS_ATTENTION_LAYERS = [3, 8, 13, 18, 23, 28, 33, 38]


def safe_mean(values: list[float]) -> float:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def add_pre_post_rows(rows: list[dict[str, Any]], *, trajectory: str, layer_values: dict[int, float], source: str) -> None:
    for cross_layer in LLAMA_CROSS_ATTENTION_LAYERS:
        pre_layer = cross_layer - 1
        post_layer = cross_layer
        if pre_layer not in layer_values or post_layer not in layer_values:
            continue
        pre = float(layer_values[pre_layer])
        post = float(layer_values[post_layer])
        rows.append(
            {
                "source": source,
                "trajectory": trajectory,
                "cross_attention_layer": cross_layer,
                "pre_layer": pre_layer,
                "post_layer": post_layer,
                "pre_value": pre,
                "post_value": post,
                "post_minus_pre": post - pre,
            }
        )


def add_layer_rows(rows: list[dict[str, Any]], *, trajectory: str, layer_values: dict[int, float], source: str) -> None:
    cross_set = set(LLAMA_CROSS_ATTENTION_LAYERS)
    for layer, value in sorted(layer_values.items()):
        rows.append(
            {
                "source": source,
                "trajectory": trajectory,
                "layer": layer,
                "value": float(value),
                "is_cross_attention_layer": layer in cross_set,
                "nearest_cross_attention_layer": min(LLAMA_CROSS_ATTENTION_LAYERS, key=lambda item: abs(item - layer)),
            }
        )


def load_mixture() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    df = pd.read_csv(metrics_path("layerwise_prompt_image_mixture_llama.csv"))
    layer_rows: list[dict[str, Any]] = []
    pre_post_rows: list[dict[str, Any]] = []
    for column in ["prompt_weight", "matched_image_weight", "mixture_r2"]:
        values = {int(row.layer): float(getattr(row, column)) for row in df.itertuples()}
        add_layer_rows(layer_rows, trajectory=column, layer_values=values, source="prompt_image_mixture")
        add_pre_post_rows(pre_post_rows, trajectory=column, layer_values=values, source="prompt_image_mixture")
    gap = {int(row.layer): float(row.matched_image_weight - row.prompt_weight) for row in df.itertuples()}
    add_layer_rows(layer_rows, trajectory="image_minus_prompt_weight", layer_values=gap, source="prompt_image_mixture")
    add_pre_post_rows(pre_post_rows, trajectory="image_minus_prompt_weight", layer_values=gap, source="prompt_image_mixture")
    return layer_rows, pre_post_rows


def load_retention() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    df = pd.read_csv(metrics_path("layerwise_mismatched_identity_retention_llama.csv"))
    df = df[df["mismatch_mode"].eq("all")].copy()
    layer_rows: list[dict[str, Any]] = []
    pre_post_rows: list[dict[str, Any]] = []
    for column in ["text_retention_rate", "image_hijack_rate", "mean_source_minus_target_distance"]:
        values = {int(row.layer): float(getattr(row, column)) for row in df.itertuples()}
        add_layer_rows(layer_rows, trajectory=column, layer_values=values, source="mismatched_identity")
        add_pre_post_rows(pre_post_rows, trajectory=column, layer_values=values, source="mismatched_identity")
    return layer_rows, pre_post_rows


def load_internal_visual() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    df = pd.read_csv(metrics_path("layerwise_internal_visual_alignment.csv"))
    df = df[df["family"].eq("llama")].copy()
    pivot = df.pivot_table(index="layer", columns="condition", values="rsa_score", aggfunc="mean")
    trajectories: dict[str, dict[int, float]] = {}
    if {"M_matched_image", "M_text_only"}.issubset(pivot.columns):
        trajectories["internal_visual_matched_minus_text"] = {
            int(layer): float(row["M_matched_image"] - row["M_text_only"]) for layer, row in pivot.iterrows()
        }
    if {"M_prompt_plus_matched_image", "M_text_only"}.issubset(pivot.columns):
        trajectories["internal_visual_prompt_plus_minus_text"] = {
            int(layer): float(row["M_prompt_plus_matched_image"] - row["M_text_only"]) for layer, row in pivot.iterrows()
        }
    if {"M_mismatched_image", "M_text_only"}.issubset(pivot.columns):
        trajectories["internal_visual_mismatch_minus_text"] = {
            int(layer): float(row["M_mismatched_image"] - row["M_text_only"]) for layer, row in pivot.iterrows()
        }
    layer_rows: list[dict[str, Any]] = []
    pre_post_rows: list[dict[str, Any]] = []
    for trajectory, values in trajectories.items():
        add_layer_rows(layer_rows, trajectory=trajectory, layer_values=values, source="internal_visual_alignment")
        add_pre_post_rows(pre_post_rows, trajectory=trajectory, layer_values=values, source="internal_visual_alignment")
    return layer_rows, pre_post_rows


def load_external_rsa() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    df = pd.read_csv(metrics_path("cross_family_layerwise_rsa_trajectories.csv"))
    df = df[df["family_name"].eq("llama")].copy()
    layer_rows: list[dict[str, Any]] = []
    pre_post_rows: list[dict[str, Any]] = []
    for anchor in sorted(df["anchor_name"].unique()):
        sub = df[df["anchor_name"].eq(anchor)]
        pivot = sub.pivot_table(index="layer", columns="condition", values="rsa_score", aggfunc="mean")
        if {"M_matched_image", "T_prompt_primary"}.issubset(pivot.columns):
            values = {int(layer): float(row["M_matched_image"] - row["T_prompt_primary"]) for layer, row in pivot.iterrows()}
            trajectory = f"external_rsa_matched_minus_prompt_{anchor}"
            add_layer_rows(layer_rows, trajectory=trajectory, layer_values=values, source="external_rsa")
            add_pre_post_rows(pre_post_rows, trajectory=trajectory, layer_values=values, source="external_rsa")
    return layer_rows, pre_post_rows


def summarize(layer_rows: list[dict[str, Any]], pre_post_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(layer_rows)
    pp = pd.DataFrame(pre_post_rows)
    rows = []
    for trajectory, sub in df.groupby("trajectory"):
        cross = sub[sub["is_cross_attention_layer"].astype(bool)]["value"].astype(float).tolist()
        non_cross = sub[~sub["is_cross_attention_layer"].astype(bool)]["value"].astype(float).tolist()
        pp_sub = pp[pp["trajectory"].eq(trajectory)]
        rows.append(
            {
                "trajectory": trajectory,
                "source": str(sub["source"].iloc[0]),
                "num_layers": int(len(sub)),
                "cross_layer_mean": safe_mean(cross),
                "non_cross_layer_mean": safe_mean(non_cross),
                "cross_minus_non_cross": safe_mean(cross) - safe_mean(non_cross),
                "mean_post_minus_pre_cross_attention": safe_mean(pp_sub["post_minus_pre"].astype(float).tolist()) if not pp_sub.empty else float("nan"),
                "max_abs_post_minus_pre_cross_attention": float(pp_sub["post_minus_pre"].abs().max()) if not pp_sub.empty else float("nan"),
                "strongest_cross_attention_layer": int(pp_sub.loc[pp_sub["post_minus_pre"].abs().idxmax(), "cross_attention_layer"]) if not pp_sub.empty else "",
            }
        )
    return rows


def write_report(summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Llama Cross-Attention Layer Diagnostics",
        "",
        f"- Llama cross-attention layers from config: `{LLAMA_CROSS_ATTENTION_LAYERS}`",
        "- This diagnostic aligns existing layerwise geometry trajectories to those layers; it does not re-run the model.",
        "",
        "| Source | Trajectory | Cross mean | Non-cross mean | Cross - non-cross | Mean post-pre | Strongest layer |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['source']}` | `{row['trajectory']}` | {row['cross_layer_mean']:+.4f} | "
            f"{row['non_cross_layer_mean']:+.4f} | {row['cross_minus_non_cross']:+.4f} | "
            f"{row['mean_post_minus_pre_cross_attention']:+.4f} | {row['strongest_cross_attention_layer']} |"
        )
    lines.extend(
        [
            "",
            "## Manuscript Interpretation",
            "- If image-dominant mixture is high while external/internal alignment gains remain comparatively weak, Llama is best framed as visual restructuring without strong reference-space convergence.",
            "- These diagnostics are architecture-consistent, not causal proof that sparse cross-attention causes the dissociation.",
        ]
    )
    write_text(ROOT / "reports" / "main_results" / "llama_cross_attention_layer_diagnostics_report.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Align Llama layerwise trajectories to configured cross-attention layers.")
    parser.parse_args()

    layer_rows: list[dict[str, Any]] = []
    pre_post_rows: list[dict[str, Any]] = []
    for loader in [load_mixture, load_retention, load_internal_visual, load_external_rsa]:
        loaded_layer, loaded_pre_post = loader()
        layer_rows.extend(loaded_layer)
        pre_post_rows.extend(loaded_pre_post)
    summary_rows = summarize(layer_rows, pre_post_rows)
    write_csv(
        metrics_path("llama_cross_attention_layer_diagnostics.csv"),
        layer_rows,
        ["source", "trajectory", "layer", "value", "is_cross_attention_layer", "nearest_cross_attention_layer"],
    )
    write_csv(
        metrics_path("llama_cross_attention_layer_prepost.csv"),
        pre_post_rows,
        ["source", "trajectory", "cross_attention_layer", "pre_layer", "post_layer", "pre_value", "post_value", "post_minus_pre"],
    )
    write_csv(
        metrics_path("llama_cross_attention_layer_diagnostics_summary.csv"),
        summary_rows,
        [
            "trajectory",
            "source",
            "num_layers",
            "cross_layer_mean",
            "non_cross_layer_mean",
            "cross_minus_non_cross",
            "mean_post_minus_pre_cross_attention",
            "max_abs_post_minus_pre_cross_attention",
            "strongest_cross_attention_layer",
        ],
    )
    write_json(
        metrics_path("llama_cross_attention_layer_diagnostics_summary.json"),
        {"cross_attention_layers": LLAMA_CROSS_ATTENTION_LAYERS, "summary": summary_rows},
    )
    write_report(summary_rows)
    append_run_log("Llama Cross-Attention Layer Diagnostics", ["Computed Llama cross-attention-aligned layer diagnostics."])


if __name__ == "__main__":
    main()
