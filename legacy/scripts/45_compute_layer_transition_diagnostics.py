from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from common import ROOT, append_run_log, metrics_path, read_csv, write_csv, write_json
from hardening_common import write_text


FAMILIES = ["qwen", "mistral", "llama"]


def read_family_csv(stem: str, family: str) -> list[dict[str, str]]:
    path = metrics_path(f"{stem}_{family}.csv")
    if not path.exists():
        raise RuntimeError(f"Missing {path}")
    return read_csv(path)


def safe_float(value: str) -> float:
    return float(value) if value not in {"", "nan", "None"} else float("nan")


def split_thirds(values: np.ndarray) -> dict[str, float]:
    n = len(values)
    early = values[: max(1, n // 3)]
    middle = values[max(1, n // 3) : max(2, 2 * n // 3)]
    late = values[max(2, 2 * n // 3) :]
    return {
        "early_mean": float(np.mean(early)),
        "middle_mean": float(np.mean(middle)),
        "late_mean": float(np.mean(late)),
        "late_minus_early": float(np.mean(late) - np.mean(early)),
    }


def fit_linear_sse(x: np.ndarray, y: np.ndarray) -> float:
    design = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    return float(np.square(resid).sum())


def best_one_break(x: np.ndarray, y: np.ndarray, min_segment: int = 4) -> dict[str, Any]:
    if len(y) < min_segment * 2 + 1:
        return {"break_layer": "", "piecewise_sse": float("nan"), "linear_sse": fit_linear_sse(x, y), "sse_reduction": float("nan")}
    linear = fit_linear_sse(x, y)
    best = {"break_idx": None, "piecewise_sse": float("inf")}
    for break_idx in range(min_segment, len(y) - min_segment):
        sse = fit_linear_sse(x[:break_idx], y[:break_idx]) + fit_linear_sse(x[break_idx:], y[break_idx:])
        if sse < best["piecewise_sse"]:
            best = {"break_idx": break_idx, "piecewise_sse": sse}
    reduction = 0.0 if linear <= 1e-12 else (linear - float(best["piecewise_sse"])) / linear
    return {
        "break_layer": int(x[int(best["break_idx"])]) if best["break_idx"] is not None else "",
        "piecewise_sse": float(best["piecewise_sse"]),
        "linear_sse": linear,
        "sse_reduction": float(reduction),
    }


def transition_metrics(layers: np.ndarray, values: np.ndarray, threshold: float | None = None) -> dict[str, Any]:
    diffs = np.diff(values)
    abs_diffs = np.abs(diffs)
    steepest_idx = int(abs_diffs.argmax()) if len(abs_diffs) else 0
    thirds = split_thirds(values)
    metrics: dict[str, Any] = {
        "first_layer": int(layers[0]),
        "last_layer": int(layers[-1]),
        "first_value": float(values[0]),
        "last_value": float(values[-1]),
        "mean_value": float(values.mean()),
        "max_value": float(values.max()),
        "max_layer": int(layers[int(values.argmax())]),
        "steepest_change_from_layer": int(layers[steepest_idx]) if len(abs_diffs) else "",
        "steepest_change_to_layer": int(layers[steepest_idx + 1]) if len(abs_diffs) else "",
        "steepest_abs_delta": float(abs_diffs[steepest_idx]) if len(abs_diffs) else 0.0,
        **thirds,
        **best_one_break(layers.astype(float), values),
    }
    if threshold is not None:
        first_cross = next((int(layer) for layer, value in zip(layers, values) if value >= threshold), "")
        metrics["first_threshold_layer"] = first_cross
        metrics["threshold"] = threshold
    return metrics


def family_mixture_metrics(family: str) -> list[dict[str, Any]]:
    rows = read_family_csv("layerwise_prompt_image_mixture", family)
    layers = np.asarray([int(row["layer"]) for row in rows], dtype=int)
    output = []
    for metric_name, threshold in [
        ("matched_image_weight", 0.5),
        ("prompt_weight", None),
        ("mixture_r2", 0.8),
    ]:
        values = np.asarray([safe_float(row[metric_name]) for row in rows], dtype=float)
        metrics = transition_metrics(layers, values, threshold)
        output.append({"family": family, "trajectory": metric_name, **metrics})
    image = np.asarray([safe_float(row["matched_image_weight"]) for row in rows], dtype=float)
    prompt = np.asarray([safe_float(row["prompt_weight"]) for row in rows], dtype=float)
    gap = image - prompt
    metrics = transition_metrics(layers, gap, 0.0)
    output.append({"family": family, "trajectory": "image_minus_prompt_weight", **metrics})
    return output


def family_retention_metrics(family: str) -> list[dict[str, Any]]:
    rows = [row for row in read_family_csv("layerwise_mismatched_identity_retention", family) if row["mismatch_mode"] == "all"]
    layers = np.asarray([int(row["layer"]) for row in rows], dtype=int)
    output = []
    for metric_name, threshold in [
        ("text_retention_rate", 0.99),
        ("image_hijack_rate", None),
        ("ambiguous_rate", None),
        ("mean_source_minus_target_distance", None),
    ]:
        values = np.asarray([safe_float(row[metric_name]) for row in rows], dtype=float)
        metrics = transition_metrics(layers, values, threshold)
        output.append({"family": family, "trajectory": metric_name, **metrics})
    return output


def rsa_transition_rows() -> list[dict[str, Any]]:
    path = metrics_path("layerwise_trajectory_summary.csv")
    if not path.exists():
        return []
    rows = [row for row in read_csv(path) if row["summary_type"] == "trajectory"]
    by_anchor_condition: dict[tuple[str, str], dict[int, float]] = {}
    for row in rows:
        by_anchor_condition.setdefault((row["anchor_name"], row["condition"]), {})[int(row["layer"])] = safe_float(row["rsa_score"])
    output = []
    anchors = sorted({anchor for anchor, _condition in by_anchor_condition})
    for anchor in anchors:
        matched = by_anchor_condition.get((anchor, "M_matched_image"), {})
        prompt = by_anchor_condition.get((anchor, "T_prompt_primary"), {})
        text = by_anchor_condition.get((anchor, "M_text_only"), {})
        for baseline_name, baseline in [("prompt", prompt), ("vlm_text_only", text)]:
            common_layers = np.asarray(sorted(set(matched) & set(baseline)), dtype=int)
            if common_layers.size == 0:
                continue
            gap = np.asarray([matched[int(layer)] - baseline[int(layer)] for layer in common_layers], dtype=float)
            output.append({"family": "qwen", "trajectory": f"matched_minus_{baseline_name}_{anchor}", **transition_metrics(common_layers, gap, 0.0)})
    return output


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for row in rows:
        family = row["family"]
        trajectory = row["trajectory"]
        summary.setdefault(family, {})[trajectory] = {
            "first_value": row["first_value"],
            "last_value": row["last_value"],
            "late_minus_early": row["late_minus_early"],
            "steepest_change_from_layer": row["steepest_change_from_layer"],
            "steepest_change_to_layer": row["steepest_change_to_layer"],
            "steepest_abs_delta": row["steepest_abs_delta"],
            "break_layer": row["break_layer"],
            "sse_reduction": row["sse_reduction"],
            "first_threshold_layer": row.get("first_threshold_layer", ""),
        }
    return summary


def report_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "# Layer Transition Diagnostics",
        "",
        "This report asks whether global-local integration emerges abruptly or gradually across layers.",
        "",
        "## Global Mixture",
        "",
        "| Family | Image weight first -> last | Late - early | First image > 0.5 | Steepest change | One-break layer | SSE reduction |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    lookup = {(row["family"], row["trajectory"]): row for row in rows}
    for family in FAMILIES:
        row = lookup[(family, "matched_image_weight")]
        lines.append(
            f"| {family} | {row['first_value']:.4f} -> {row['last_value']:.4f} | {row['late_minus_early']:+.4f} | "
            f"{row.get('first_threshold_layer', '')} | {row['steepest_change_from_layer']}->{row['steepest_change_to_layer']} "
            f"({row['steepest_abs_delta']:.4f}) | {row['break_layer']} | {row['sse_reduction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Local Identity Retention",
            "",
            "| Family | Retention first -> last | Late - early | First >= 0.99 | Steepest change | One-break layer | SSE reduction |",
            "|---|---:|---:|---:|---|---:|---:|",
        ]
    )
    for family in FAMILIES:
        row = lookup[(family, "text_retention_rate")]
        lines.append(
            f"| {family} | {row['first_value']:.4f} -> {row['last_value']:.4f} | {row['late_minus_early']:+.4f} | "
            f"{row.get('first_threshold_layer', '')} | {row['steepest_change_from_layer']}->{row['steepest_change_to_layer']} "
            f"({row['steepest_abs_delta']:.4f}) | {row['break_layer']} | {row['sse_reduction']:.3f} |"
        )
    qwen_rsa = [row for row in rows if row["family"] == "qwen" and row["trajectory"].startswith("matched_minus_prompt_")]
    if qwen_rsa:
        lines.extend(["", "## Qwen Matched-Minus-Prompt RSA Gaps", "", "| Anchor gap | First positive layer | Late - early | Steepest change | One-break layer |", "|---|---:|---:|---|---:|"])
        for row in qwen_rsa:
            label = row["trajectory"].replace("matched_minus_prompt_", "")
            lines.append(
                f"| {label} | {row.get('first_threshold_layer', '')} | {row['late_minus_early']:+.4f} | "
                f"{row['steepest_change_from_layer']}->{row['steepest_change_to_layer']} ({row['steepest_abs_delta']:.4f}) | {row['break_layer']} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "- A large one-break SSE reduction with a localized steepest change supports a phase-like transition.",
            "- Small late-minus-early changes and image dominance from the first layer support an early-established or gradual/stable regime.",
        ]
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute lightweight layer transition diagnostics from existing layerwise outputs.")
    parser.parse_args()

    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        rows.extend(family_mixture_metrics(family))
        rows.extend(family_retention_metrics(family))
    rows.extend(rsa_transition_rows())

    fieldnames = [
        "family",
        "trajectory",
        "first_layer",
        "last_layer",
        "first_value",
        "last_value",
        "mean_value",
        "max_value",
        "max_layer",
        "early_mean",
        "middle_mean",
        "late_mean",
        "late_minus_early",
        "first_threshold_layer",
        "threshold",
        "steepest_change_from_layer",
        "steepest_change_to_layer",
        "steepest_abs_delta",
        "break_layer",
        "linear_sse",
        "piecewise_sse",
        "sse_reduction",
    ]
    for row in rows:
        for field in fieldnames:
            row.setdefault(field, "")
    write_csv(metrics_path("layer_transition_diagnostics.csv"), rows, fieldnames)
    write_json(metrics_path("layer_transition_diagnostics_summary.json"), summarize(rows))
    write_text(ROOT / "reports" / "main_results" / "layer_transition_diagnostics_report.md", "\n".join(report_lines(rows)))
    append_run_log("Layer Transition Diagnostics", ["Computed lightweight layer transition diagnostics from existing layerwise outputs."])


if __name__ == "__main__":
    main()
