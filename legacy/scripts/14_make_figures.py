from __future__ import annotations

import argparse
from collections import defaultdict

import matplotlib.pyplot as plt

from common import ROOT, canonical_condition_name, figure_path, metrics_path, read_csv, append_run_log


BACKBONE_CONDITIONS = ["T_neutral", "T_prompt_primary", "M_matched_image", "M_degraded_image", "M_mismatched_image", "M_blank_image"]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def plot_matched_family_alignment(rows: list[dict[str, str]]) -> None:
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if row["bootstrap_id"] != "aggregate" or row["domain"] != "sensory":
            continue
        condition = canonical_condition_name(row["condition"])
        if condition not in {"T_neutral", "T_prompt_primary", "M_matched_image"}:
            continue
        grouped[(row.get("model", row.get("model_id", "")), condition)].append((int(row["layer"]), float(row["rsa_score"])))
    plt.figure(figsize=(10, 5))
    for (model, condition), pairs in sorted(grouped.items()):
        pairs.sort()
        plt.plot([x for x, _ in pairs], [y for _, y in pairs], label=f"{model} | {condition}")
    plt.xlabel("Layer")
    plt.ylabel("RSA")
    plt.title("Matched-family alignment")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(figure_path("fig1_matched_family_alignment.png"), dpi=180)
    plt.close()


def plot_perturbation_fingerprint(rows: list[dict[str, str]]) -> None:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("model", row.get("model_id", "")), row["perturbation"])].append(float(row["abs_drop"]))
    labels = []
    values = []
    for (model, perturbation), scores in sorted(grouped.items()):
        labels.append(f"{model}\n{perturbation}")
        values.append(mean(scores))
    plt.figure(figsize=(11, 5))
    plt.bar(labels, values)
    plt.ylabel("Mean absolute RSA drop")
    plt.title("Perturbation fingerprint")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(figure_path("fig2_perturbation_fingerprint.png"), dpi=180)
    plt.close()


def plot_anchor_robustness(rows: list[dict[str, str]]) -> None:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["condition"] not in {"T_prompt_primary", "M_matched_image"}:
            continue
        grouped[(row["anchor_name"], row["condition"])].append(float(row["mean_rsa"]))
    labels = []
    values = []
    for (anchor_name, condition), scores in sorted(grouped.items()):
        labels.append(f"{anchor_name}\n{condition}")
        values.append(mean(scores))
    plt.figure(figsize=(11, 5))
    plt.bar(labels, values)
    plt.ylabel("Mean RSA")
    plt.title("Anchor robustness")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(figure_path("fig3_anchor_robustness.png"), dpi=180)
    plt.close()


def plot_domain_control(rows: list[dict[str, str]]) -> None:
    labels = [f"{row['model']}\n{row['anchor_name']}" for row in rows]
    values = [float(row["sensory_minus_abstract_gap"]) for row in rows]
    plt.figure(figsize=(11, 5))
    plt.bar(labels, values)
    plt.ylabel("Sensory minus abstract prompt gap")
    plt.title("Domain control")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(figure_path("fig4_domain_control.png"), dpi=180)
    plt.close()


def plot_geometry(rows: list[dict[str, str]]) -> None:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["condition_a"], row["condition_b"])].append(float(row["mean_jaccard"]))
    labels = []
    values = []
    for (model, condition_a, condition_b), scores in sorted(grouped.items()):
        labels.append(f"{model}\n{condition_a}->{condition_b}")
        values.append(mean(scores))
    plt.figure(figsize=(11, 5))
    plt.bar(labels, values)
    plt.ylabel("Mean Jaccard")
    plt.title("Geometry restructuring")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(figure_path("fig5_geometry_restructuring.png"), dpi=180)
    plt.close()


def plot_human_local_geometry(rows: list[dict[str, str]]) -> None:
    ordered_groups = ["all_concepts", "appearance_color", "texture_material", "sound_linked", "smell_taste_proxy"]
    ordered_conditions = ["T_prompt_primary", "M_matched_image", "M_degraded_image"]
    lookup = {
        (row["group_name"], row["condition"]): float(row["mean_local_alignment"])
        for row in rows
        if row["granularity"] in {"all_concepts", "subtype"}
    }
    labels = []
    values = []
    for group_name in ordered_groups:
        for condition in ordered_conditions:
            value = lookup.get((group_name, condition))
            if value is None:
                continue
            labels.append(f"{group_name}\n{condition}")
            values.append(value)
    plt.figure(figsize=(12, 5))
    plt.bar(labels, values)
    plt.ylabel("Mean local alignment")
    plt.title("Human local geometry")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(figure_path("fig6_human_local_geometry.png"), dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.parse_args()

    figure_path("fig1_matched_family_alignment.png").parent.mkdir(parents=True, exist_ok=True)
    alignment_rows = read_csv(metrics_path("layerwise_alignment_full.csv"))
    stability_rows = read_csv(metrics_path("layerwise_stability_full.csv"))
    anchor_rows = read_csv(metrics_path("anchor_robustness.csv"))
    domain_rows = read_csv(metrics_path("domain_control_summary.csv"))
    geometry_rows = read_csv(metrics_path("neighbor_restructuring.csv")) if metrics_path("neighbor_restructuring.csv").exists() else []
    local_geometry_rows = read_csv(metrics_path("human_local_geometry.csv")) if metrics_path("human_local_geometry.csv").exists() else []

    plot_matched_family_alignment(alignment_rows)
    plot_perturbation_fingerprint(stability_rows)
    plot_anchor_robustness(anchor_rows)
    plot_domain_control(domain_rows)
    if geometry_rows:
        plot_geometry(geometry_rows)
    if local_geometry_rows:
        plot_human_local_geometry(local_geometry_rows)

    append_run_log(
        "Figures",
        [
            f"Wrote core figures to {figure_path('fig1_matched_family_alignment.png').parent.relative_to(ROOT)}.",
            "Generated fig1 through fig5 and fig6 when geometry metrics were available.",
        ],
    )


if __name__ == "__main__":
    main()
