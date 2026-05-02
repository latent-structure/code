from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from common import ROOT, output_path
from figure_style import apply_neurips_style, condition_color, condition_label, condition_marker, panel_label, save_figure


FIGURE_DIR = output_path("outputs", "figures", "paper")
FAMILY_LABELS = {"qwen": "Qwen", "mistral": "Mistral", "llama": "Llama"}
FAMILY_ORDER = ["qwen", "mistral", "llama"]
ANCHOR_LABELS = {
    "THINGS": "THINGS",
    "THINGS behavioral similarity": "THINGS",
    "controlled_THINGS": "Controlled\nTHINGS",
    "SigLIP2": "SigLIP2",
    "lancaster_perceptual": "Lancaster\nperceptual",
}
ANCHOR_COLORS = {
    "THINGS": "#0072B2",
    "controlled_THINGS": "#56B4E9",
    "SigLIP2": "#009E73",
    "lancaster_perceptual": "#D55E00",
}

# --- Utility functions (read_csv_rows, read_json, etc.) remain unchanged ---
def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def require_file(path: Path, allow_missing: bool) -> bool:
    if path.exists():
        return True
    if allow_missing:
        print(f"Skipping missing input: {path.relative_to(ROOT)}")
        return False
    raise FileNotFoundError(path)

def selected_names(only: str) -> set[str] | None:
    if not only:
        return None
    return {part.strip() for part in only.split(",") if part.strip()}

def save(fig: Any, stem: str) -> None:
    save_figure(fig, FIGURE_DIR / stem)
    plt.close(fig)

# --- Plotting Functions ---

def plot_cross_family_rsa(allow_missing: bool) -> None:
    path = output_path("outputs", "metrics", "cross_family_rsa_full.csv")
    if not require_file(path, allow_missing):
        return
    anchors = ["THINGS", "controlled_THINGS", "SigLIP2", "lancaster_perceptual"]
    rows = [
        row for row in read_csv_rows(path)
        if row["row_type"] == "contrast" and row["contrast_name"] == "matched_minus_prompt" and row["anchor_name"] in anchors
    ]
    values = {(row["family_name"], row["anchor_name"]): float(row["contrast_delta"]) for row in rows}

    apply_neurips_style(plt, nrows=1, ncols=1)
    fig, ax = plt.subplots(figsize=(6.0, 2.8)) # FIX: Slightly wider to accommodate legend
    x = np.arange(len(FAMILY_ORDER))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(anchors))
    
    for offset, anchor in zip(offsets, anchors):
        ax.bar(
            x + offset,
            [values.get((family, anchor), np.nan) for family in FAMILY_ORDER],
            width=width,
            label=ANCHOR_LABELS[anchor].replace("\n", " "),
            color=ANCHOR_COLORS[anchor],
            edgecolor="white", # FIX: Added slight edge for crispness
            linewidth=0.5
        )
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(x, [FAMILY_LABELS[family] for family in FAMILY_ORDER])
    ax.set_ylabel("Matched - prompt RSA")
    
    # FIX: Moved title to the axis, removed Suptitle. Moved legend entirely outside the plot to prevent data overlap.
    ax.set_title("Reference-aligned grounding is strong in Qwen/Mistral", pad=10)
    ax.legend(frameon=False, ncols=1, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    save(fig, "fig_cross_family_rsa_gaps")


def plot_global_local(allow_missing: bool) -> None:
    path = output_path("outputs", "metrics", "cross_family_global_local_summary.csv")
    if not require_file(path, allow_missing):
        return
    rows = {row["family"]: row for row in read_csv_rows(path)}

    apply_neurips_style(plt, nrows=1, ncols=2)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharex=False)
    y = np.arange(len(FAMILY_ORDER))

    axes[0].scatter([float(rows[f]["mean_prompt_weight"]) for f in FAMILY_ORDER], y, color=condition_color("T_prompt_primary"), label="Prompt weight", marker="o", s=60)
    axes[0].scatter([float(rows[f]["mean_matched_image_weight"]) for f in FAMILY_ORDER], y, color=condition_color("M_matched_image"), label="Image weight", marker="D", s=60)
    axes[0].set_yticks(y, [FAMILY_LABELS[f] for f in FAMILY_ORDER])
    axes[0].set_xlabel("Mean mixture weight")
    axes[0].set_xlim(-0.05, 1.05)
    axes[0].legend(frameon=False, loc="center right")
    panel_label(axes[0], "A")

    # FIX: Added jitter to the Y-axis so 1.0 and 0.0 values don't perfectly overlap, making all data points visible.
    jitter = 0.08
    axes[1].scatter([float(rows[f]["mean_text_retention_rate"]) for f in FAMILY_ORDER], y + jitter, color="#0072B2", label="Text retained", marker="o", s=60, alpha=0.8)
    axes[1].scatter([float(rows[f]["mean_image_hijack_rate"]) for f in FAMILY_ORDER], y - jitter, color="#D55E00", label="Image assigned", marker="X", s=60, alpha=0.8)
    axes[1].set_yticks(y, [FAMILY_LABELS[f] for f in FAMILY_ORDER])
    axes[1].set_xlabel("Identity assignment rate")
    axes[1].set_xlim(-0.05, 1.05)
    axes[1].legend(frameon=False, loc="center right")
    panel_label(axes[1], "B")

    # FIX: Removed Suptitle. In academic papers, this text belongs in the figure caption below the image.
    fig.tight_layout()
    save(fig, "fig_global_local_dissociation")


def mid_to_late_mean(values_by_layer: list[tuple[int, float]]) -> float:
    ordered = [value for _, value in sorted(values_by_layer)]
    count = int(math.ceil(len(ordered) * 0.5))
    return float(np.mean(ordered[-count:]))


def plot_pr_compression(allow_missing: bool) -> None:
    # ... (data loading remains unchanged) ...
    path = output_path("outputs", "metrics", "intrinsic_dimensionality.csv")
    family_path = output_path("outputs", "metrics", "cross_family_global_local_summary.csv")
    if not require_file(path, allow_missing) or not require_file(family_path, allow_missing):
        return
    family_rows = {row["family"]: row for row in read_csv_rows(family_path)}
    vlm_by_family = {family: row["multimodal_model_id"] for family, row in family_rows.items()}
    conditions = ["M_text_only", "M_blank_image", "M_mismatched_image", "M_degraded_image", "M_matched_image", "M_prompt_plus_matched_image"]
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in read_csv_rows(path):
        if row["domain"] != "sensory":
            continue
        for family, model_id in vlm_by_family.items():
            if row["model"] == model_id and row["condition"] in conditions:
                grouped[(family, row["condition"])].append((int(row["layer"]), float(row["participation_ratio"])))
    means = {key: mid_to_late_mean(values) for key, values in grouped.items()}

    apply_neurips_style(plt, nrows=1, ncols=1)
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    x = np.arange(len(conditions))
    
    # FIX: Categorical data should not be connected by lines. Changed to grouped dot-plot style.
    width = 0.2
    offsets = np.linspace(-width, width, len(FAMILY_ORDER))
    
    for offset, family in zip(offsets, FAMILY_ORDER):
        ax.plot(
            x + offset, # Jitter horizontally by family
            [means.get((family, condition), np.nan) for condition in conditions],
            marker="o",
            linestyle="none", # FIX: Removed spaghetti connecting lines
            markersize=7,
            label=FAMILY_LABELS[family],
        )
    ax.set_xticks(x, [condition_label(condition).replace(" ", "\n") for condition in conditions])
    ax.set_ylabel("Mid-to-late PR")
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    save(fig, "fig_pr_compression")


def plot_layerwise_rsa(allow_missing: bool) -> None:
    # ... (data loading remains unchanged) ...
    path = output_path("outputs", "metrics", "layerwise_trajectory_summary.csv")
    if not require_file(path, allow_missing):
        return
    anchors = ["THINGS behavioral similarity", "controlled_THINGS", "SigLIP2", "lancaster_perceptual"]
    conditions = ["T_prompt_primary", "M_text_only", "M_matched_image"]
    data: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in read_csv_rows(path):
        if row["summary_type"] != "trajectory" or row["anchor_name"] not in anchors or row["condition"] not in conditions:
            continue
        data[(row["anchor_name"], row["condition"])].append((int(row["layer"]), float(row["rsa_score"])))

    apply_neurips_style(plt, nrows=2, ncols=2)
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.5), sharex=False)
    for idx, (axis, anchor) in enumerate(zip(axes.ravel(), anchors)):
        for condition in conditions:
            pairs = sorted(data.get((anchor, condition), []))
            if not pairs:
                continue
            axis.plot(
                [layer for layer, _ in pairs],
                [score for _, score in pairs],
                color=condition_color(condition),
                marker=condition_marker(condition),
                markevery=max(1, len(pairs) // 6),
                linewidth=1.2, # FIX: Thinned lines slightly to reduce visual spaghetti clutter
                alpha=0.9,     # FIX: Added slight transparency so overlapping lines are readable
                label=condition_label(condition),
            )
        axis.set_title(ANCHOR_LABELS.get(anchor, anchor).replace("\n", " "), fontsize=10)
        axis.set_xlabel("Layer")
        axis.set_ylabel("RSA")
        panel_label(axis, chr(ord("A") + idx))
    
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig_layerwise_rsa_trajectories")


def plot_mismatched_identity(allow_missing: bool) -> None:
    summary_path = output_path("outputs", "metrics", "mismatched_hijacking_summary.json")
    validation_path = output_path("outputs", "metrics", "mismatched_hijacking_validation_summary.json")
    if not require_file(summary_path, allow_missing) or not require_file(validation_path, allow_missing):
        return
    validation = read_json(validation_path)
    margins = ["0.0", "0.005", "0.01", "0.02"]
    observed = validation["margin_sensitivity"]["observed"]

    apply_neurips_style(plt, nrows=1, ncols=2)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
    x = np.arange(len(margins))
    retention = np.asarray([observed[m]["text_retention_rate"] for m in margins])
    hijack = np.asarray([observed[m]["image_hijack_rate"] for m in margins])
    ambiguous = np.asarray([observed[m]["ambiguous_rate"] for m in margins])
    
    axes[0].bar(x, retention, color="#0072B2", label="Text retained")
    axes[0].bar(x, hijack, bottom=retention, color="#D55E00", label="Image assigned")
    axes[0].bar(x, ambiguous, bottom=retention + hijack, color="#999999", label="Ambiguous")
    axes[0].set_xticks(x, margins)
    axes[0].set_xlabel("Margin")
    axes[0].set_ylabel("Assignment rate")
    
    # FIX: Bar charts MUST start at 0. Starting at 0.98 is highly misleading.
    axes[0].set_ylim(0, 1.05) 
    axes[0].legend(frameon=False, loc="lower right")
    panel_label(axes[0], "A")

    rank_labels = ["Target", "Mismatched\nsource"]
    medians = [validation["median_target_rank"], validation["median_source_rank"]]
    means = [validation["mean_target_rank"], validation["mean_source_rank"]]
    
    # FIX: Replaced massive, misleading bars with a clean dot plot for mean/median.
    colors = ["#0072B2", "#D55E00"]
    for i in [0, 1]:
        axes[1].scatter(i, means[i], color=colors[i], marker="o", s=80, label="Mean" if i==0 else "")
        axes[1].scatter(i, medians[i], color=colors[i], marker="D", s=80, label="Median" if i==0 else "")
        # Draw a subtle line connecting mean and median to show spread
        axes[1].plot([i, i], [means[i], medians[i]], color='gray', linestyle='--', zorder=0)

    axes[1].set_xticks([0, 1], rank_labels)
    axes[1].set_xlim(-0.5, 1.5)
    axes[1].set_ylabel("Rank among matched anchors")
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False, loc="upper left")
    panel_label(axes[1], "B")
    
    # FIX: Removed Suptitle.
    fig.tight_layout()
    save(fig, "fig_mismatched_identity_retention")

def plot_variance_partitioning(allow_missing: bool) -> None:
    # ... (data loading remains unchanged) ...
    summary_path = output_path("outputs", "metrics", "variance_partitioning_summary.json")
    if not require_file(summary_path, allow_missing):
        return
    summary = read_json(summary_path)["conditions"]
    conditions = ["T_prompt_primary", "M_matched_image", "M_mismatched_image"]
    labels = ["Prompt", "Matched", "Mismatched"]
    components = [
        ("unique_human_family", "Unique human", "#0072B2"),
        ("unique_anchor_family", "Unique visual anchors", "#009E73"),
        ("unique_proxy_family", "Unique proxy", "#E69F00"),
        ("shared", "Shared explained", "#999999"),
    ]

    values: dict[str, list[float]] = {}
    residuals = []
    for condition in conditions:
        row = summary[condition]
        unique_sum = row["unique_human_family"] + row["unique_anchor_family"] + row["unique_proxy_family"]
        shared = max(0.0, row["total_model_fit"] - unique_sum)
        values.setdefault("unique_human_family", []).append(row["unique_human_family"])
        values.setdefault("unique_anchor_family", []).append(row["unique_anchor_family"])
        values.setdefault("unique_proxy_family", []).append(row["unique_proxy_family"])
        values.setdefault("shared", []).append(shared)
        residuals.append(max(0.0, 1.0 - row["total_model_fit"]))

    apply_neurips_style(plt, nrows=1, ncols=2)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharex=True) # FIX: Widened to fit legend outside
    x = np.arange(len(conditions))
    bottom = np.zeros(len(conditions))
    
    for key, label, color in components:
        component_values = np.asarray(values[key])
        axes[0].bar(x, component_values, bottom=bottom, label=label, color=color)
        axes[1].bar(x, component_values, bottom=bottom, label=label, color=color)
        bottom += component_values
        
    axes[0].bar(x, residuals, bottom=bottom, color="#E5E7EB", label="Residual")
    axes[0].set_title("Full scale", pad=10)
    axes[0].set_ylabel("Variance fraction")
    axes[0].set_xticks(x, labels)
    panel_label(axes[0], "A")

    axes[1].set_title("Explained components", pad=10)
    axes[1].set_ylabel("Variance fraction")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, max(bottom) * 1.2)
    panel_label(axes[1], "B")
    
    # FIX: Neatly aligned legend outside the plotting area
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.05, 1.0))
    # FIX: Removed Suptitle.
    fig.tight_layout()
    save(fig, "fig_variance_partitioning")

# ... plot_behavior_bridge remains largely unchanged ...
def plot_behavior_bridge(allow_missing: bool) -> None:
    data_path = output_path("outputs", "metrics", "behavior_geometry_bridge_full.csv")
    corr_path = output_path("outputs", "metrics", "behavior_geometry_bridge_full_correlations.csv")
    if not require_file(data_path, allow_missing) or not require_file(corr_path, allow_missing):
        return
    rows = read_csv_rows(data_path)
    corr_rows = read_csv_rows(corr_path)
    corr_lookup = {(row["predictor"], row["endpoint"]): row for row in corr_rows}
    panels = [
        ("rdm_disruption", "visual_word_rate_per_100", "Visual words /100"),
        ("rdm_disruption", "lancaster_visual_mean", "Lancaster visual"),
        ("source_minus_target_margin", "exemplar_specific_rate_per_100", "Exemplar-specific /100"),
        ("source_minus_target_margin", "mismatched_source_leakage", "Source leakage"),
    ]

    apply_neurips_style(plt, nrows=2, ncols=2)
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 4.1))
    for idx, (axis, (predictor, endpoint, ylabel)) in enumerate(zip(axes.ravel(), panels)):
        x = np.asarray([float(row[predictor]) for row in rows])
        y = np.asarray([float(row[endpoint]) for row in rows])
        if endpoint == "mismatched_source_leakage":
            groups = [x[y == 0], x[y == 1]]
            axis.boxplot(groups, labels=["No leak", "Leak"], widths=0.55, patch_artist=True)
            axis.set_ylabel(predictor.replace("_", " "))
        else:
            axis.scatter(x, y, s=7, alpha=0.35, color="#0072B2", edgecolors="none")
            if len(x) > 1:
                slope, intercept = np.polyfit(x, y, deg=1)
                grid = np.linspace(float(np.min(x)), float(np.max(x)), 100)
                axis.plot(grid, slope * grid + intercept, color="#D55E00", linewidth=1.2)
            axis.set_xlabel(predictor.replace("_", " "))
            axis.set_ylabel(ylabel)
        corr = corr_lookup.get((predictor, endpoint))
        if corr:
            axis.text(0.02, 0.98, f"{corr['statistic']}={float(corr['estimate']):.2f}", transform=axis.transAxes, va="top", ha="left")
        panel_label(axis, chr(ord("A") + idx))
    fig.tight_layout()
    save(fig, "fig_behavior_geometry_bridge")


FIGURES = {
    "cross_family_rsa": plot_cross_family_rsa,
    "global_local": plot_global_local,
    "pr_compression": plot_pr_compression,
    "layerwise_rsa": plot_layerwise_rsa,
    "mismatched_identity": plot_mismatched_identity,
    "variance_partitioning": plot_variance_partitioning,
    "behavior_bridge": plot_behavior_bridge,
}

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NeurIPS-style paper figures from completed result artifacts.")
    parser.add_argument("--config", default="config/analysis.yaml", help="Reserved for consistency with other scripts.")
    parser.add_argument("--only", default="", help=f"Comma-separated subset from: {', '.join(FIGURES)}")
    parser.add_argument("--allow-missing", action="store_true", help="Skip figures whose pending input files do not exist.")
    args = parser.parse_args()

    del args.config
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    requested = selected_names(args.only)
    unknown = requested - set(FIGURES) if requested else set()
    if unknown:
        raise ValueError(f"Unknown figure name(s): {', '.join(sorted(unknown))}")
    for name, function in FIGURES.items():
        if requested is not None and name not in requested:
            continue
        print(f"Generating {name}")
        function(args.allow_missing)

if __name__ == "__main__":
    main()
