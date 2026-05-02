from __future__ import annotations

from pathlib import Path
from typing import Any


CONDITION_LABELS = {
    "T_neutral": "Neutral text",
    "T_prompt_primary": "Sensory prompt",
    "M_text_only": "VLM text-only",
    "M_matched_image": "Matched image",
    "M_prompt_plus_matched_image": "Prompt + image",
    "M_degraded_image": "Degraded image",
    "M_mismatched_image": "Mismatched image",
    "M_blank_image": "Blank image",
}

CONDITION_COLORS = {
    "T_neutral": "#6B7280",
    "T_prompt_primary": "#0072B2",
    "M_text_only": "#4D4D4D",
    "M_matched_image": "#009E73",
    "M_prompt_plus_matched_image": "#D55E00",
    "M_degraded_image": "#CC79A7",
    "M_mismatched_image": "#E69F00",
    "M_blank_image": "#999999",
}

CONDITION_MARKERS = {
    "T_neutral": "o",
    "T_prompt_primary": "o",
    "M_text_only": "s",
    "M_matched_image": "D",
    "M_prompt_plus_matched_image": "^",
    "M_degraded_image": "v",
    "M_mismatched_image": "X",
    "M_blank_image": "P",
}


def apply_neurips_style(plt: Any, *, nrows: int = 1, ncols: int = 1, usetex: bool = False) -> None:
    try:
        from tueplots import bundles

        plt.rcParams.update(bundles.neurips2021(nrows=nrows, ncols=ncols, usetex=usetex))
    except Exception:
        plt.rcParams.update(
            {
                "figure.figsize": (5.5, max(1.8, 1.8 * nrows)),
                "figure.dpi": 120,
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.02,
                "font.size": 8,
                "axes.labelsize": 8,
                "axes.titlesize": 8,
                "legend.fontsize": 7,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.linewidth": 0.8,
                "lines.linewidth": 1.5,
                "lines.markersize": 4,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )


def condition_label(condition: str) -> str:
    return CONDITION_LABELS.get(condition, condition.replace("_", " "))


def condition_color(condition: str) -> str:
    return CONDITION_COLORS.get(condition, "#333333")


def condition_marker(condition: str) -> str:
    return CONDITION_MARKERS.get(condition, "o")


def panel_label(axis: Any, label: str, *, x: float = -0.12, y: float = 1.05) -> None:
    axis.text(x, y, label, transform=axis.transAxes, fontsize=9, fontweight="bold", va="top", ha="left")


def save_figure(figure: Any, path: Path | str, *, dpi: int = 300) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
