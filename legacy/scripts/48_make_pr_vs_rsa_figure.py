from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_style import (
    apply_neurips_style,
    condition_color,
    condition_label,
    condition_marker,
    panel_label,
    save_figure,
)


FAMILY_ORDER = ["qwen", "mistral", "llama"]
FAMILY_LABELS = {"qwen": "Qwen", "mistral": "Mistral", "llama": "Llama"}
CONDITION_ORDER = [
    "T_neutral",
    "T_prompt_primary",
    "M_text_only",
    "M_blank_image",
    "M_mismatched_image",
    "M_degraded_image",
    "M_matched_image",
    "M_prompt_plus_matched_image",
]
ANNOTATIONS = {
    ("qwen", "T_prompt_primary"): "Prompt",
    ("qwen", "M_matched_image"): "Matched",
    ("llama", "M_matched_image"): "Matched",
    ("llama", "M_degraded_image"): "Degraded",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create PR-vs-RSA figure showing compression is not alignment."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("figures_data/derived/fig_cross_family_pr_vs_rsa_condition_points.csv"),
        help="Condition-level table with mid_to_late_pr and RSA columns.",
    )
    parser.add_argument(
        "--anchor",
        default="THINGS",
        choices=["THINGS", "controlled_THINGS", "SigLIP2"],
        help="RSA anchor to plot on the y-axis.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figures/paper/fig_pr_vs_rsa_alignment"),
        help="Output path stem. .png and .pdf are written.",
    )
    return parser.parse_args()


def annotate_point(ax: plt.Axes, row: pd.Series, text: str, *, dx: float = 2.0, dy: float = 0.004) -> None:
    ax.annotate(
        text,
        xy=(row["mid_to_late_pr"], row["_rsa"]),
        xytext=(row["mid_to_late_pr"] + dx, row["_rsa"] + dy),
        textcoords="data",
        fontsize=7,
        arrowprops={"arrowstyle": "-", "color": "0.45", "linewidth": 0.7},
        color="0.15",
    )


def plot(args: argparse.Namespace) -> None:
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    df = pd.read_csv(args.input)
    required = {"family_name", "condition", "mid_to_late_pr", args.anchor}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {args.input}: {sorted(missing)}")

    df = df.copy()
    df["_rsa"] = df[args.anchor]
    df = df[df["condition"].isin(CONDITION_ORDER)]
    df["condition"] = pd.Categorical(df["condition"], CONDITION_ORDER, ordered=True)
    df = df.sort_values(["family_name", "condition"])

    apply_neurips_style(plt, nrows=1, ncols=3)
    fig, axes = plt.subplots(1, 3, figsize=(6.7, 2.35), sharey=True)

    for idx, (ax, family) in enumerate(zip(axes, FAMILY_ORDER)):
        fam = df[df["family_name"] == family].copy()
        for _, row in fam.iterrows():
            cond = str(row["condition"])
            ax.scatter(
                row["mid_to_late_pr"],
                row["_rsa"],
                s=42,
                color=condition_color(cond),
                marker=condition_marker(cond),
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )

        if family == "qwen":
            prompt = fam[fam["condition"].astype(str) == "T_prompt_primary"]
            matched = fam[fam["condition"].astype(str) == "M_matched_image"]
            if len(prompt) and len(matched):
                p = prompt.iloc[0]
                m = matched.iloc[0]
                ax.plot(
                    [p["mid_to_late_pr"], m["mid_to_late_pr"]],
                    [p["_rsa"], m["_rsa"]],
                    color="0.55",
                    linewidth=0.9,
                    linestyle="--",
                    zorder=1,
                )
                ax.text(
                    0.05,
                    0.95,
                    "same PR,\ndifferent RSA",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=7,
                    color="0.25",
                )

        if family == "llama":
            degraded = fam[fam["condition"].astype(str) == "M_degraded_image"]
            matched = fam[fam["condition"].astype(str) == "M_matched_image"]
            if len(degraded) and len(matched):
                d = degraded.iloc[0]
                m = matched.iloc[0]
                ax.annotate(
                    "more compression,\nweaker RSA",
                    xy=(d["mid_to_late_pr"], d["_rsa"]),
                    xytext=(d["mid_to_late_pr"] + 8, d["_rsa"] - 0.018),
                    fontsize=7,
                    color="0.25",
                    arrowprops={"arrowstyle": "->", "color": "0.5", "linewidth": 0.7},
                )
                ax.plot(
                    [d["mid_to_late_pr"], m["mid_to_late_pr"]],
                    [d["_rsa"], m["_rsa"]],
                    color="0.55",
                    linewidth=0.9,
                    linestyle="--",
                    zorder=1,
                )

        for (fam_name, cond), label in ANNOTATIONS.items():
            if fam_name != family:
                continue
            rows = fam[fam["condition"].astype(str) == cond]
            if rows.empty:
                continue
            row = rows.iloc[0]
            dx = -20.0 if family == "qwen" and cond == "T_prompt_primary" else 2.0
            dy = -0.009 if family == "llama" and cond == "M_degraded_image" else 0.004
            annotate_point(ax, row, label, dx=dx, dy=dy)

        ax.set_title(FAMILY_LABELS[family])
        ax.set_xlabel("Mid-to-late participation ratio")
        ax.grid(True, color="0.9", linewidth=0.6)
        panel_label(ax, chr(ord("A") + idx), x=-0.13, y=1.07)

    axes[0].set_ylabel(f"RSA to {args.anchor.replace('_', ' ')}")

    handles = []
    labels = []
    for cond in CONDITION_ORDER:
        if cond not in set(df["condition"].astype(str)):
            continue
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=condition_marker(cond),
                linestyle="none",
                markerfacecolor=condition_color(cond),
                markeredgecolor="white",
                markeredgewidth=0.45,
                markersize=5.5,
            )
        )
        labels.append(condition_label(cond))

    fig.legend(
        handles,
        labels,
        frameon=False,
        ncols=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        columnspacing=0.9,
        handletextpad=0.35,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save_figure(fig, args.output)
    plt.close(fig)
    print(f"Wrote {args.output.with_suffix('.png')}")
    print(f"Wrote {args.output.with_suffix('.pdf')}")


def main() -> None:
    plot(parse_args())


if __name__ == "__main__":
    main()
