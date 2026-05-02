from __future__ import annotations

import argparse
import csv

from common import ROOT, append_run_log, metrics_path, output_path, read_csv
from hardening_common import write_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.parse_args()

    multi_rows = read_csv(output_path("outputs", "tables", "multi_image_reversal_table.csv"))
    prototype_rows = read_csv(metrics_path("multi_image_prototype_summary.csv"))
    lancaster_rows = read_csv(output_path("outputs", "tables", "lancaster_main_result_table.csv"))
    trajectory_rows = read_csv(metrics_path("layerwise_trajectory_summary.csv"))

    stable = sum(row["stability_verdict"] == "concept_stable" for row in multi_rows)
    if stable >= max(1, int(0.6 * len(multi_rows))):
        multi_verdict = "supports concept-level grounding"
    elif stable <= int(0.3 * len(multi_rows)):
        multi_verdict = "supports image-specific grounding"
    else:
        multi_verdict = "mixed / concept-dependent"

    lancaster_prompt = []
    for row in lancaster_rows:
        if row["condition"] == "M_matched_image":
            lancaster_prompt.append(float(row["comparison_mean"]))
    if lancaster_prompt and all(value > 0 for value in lancaster_prompt):
        lancaster_verdict = "grounding advantage on sensorimotor/perceptual structure"
    elif lancaster_prompt and all(value <= 0 for value in lancaster_prompt):
        lancaster_verdict = "no clear advantage"
    else:
        lancaster_verdict = "mixed / anchor-dependent"

    summary_lookup = {(row["anchor_name"], row["summary_type"]): row["value"] for row in trajectory_rows if row["summary_type"] != "trajectory"}
    trajectory_hits = sum(
        value not in ("", "none")
        for key, value in summary_lookup.items()
        if key[1] in {"first_prompt_vs_neutral_layer", "first_matched_vs_prompt_layer", "peak_matched_vs_prompt_layer"}
    )
    if trajectory_hits >= 6:
        layerwise_verdict = "structured layerwise dissociation"
    elif trajectory_hits >= 3:
        layerwise_verdict = "partially structured layerwise dissociation"
    else:
        layerwise_verdict = "flat or weak layerwise structure"

    overall = "strengthens the dissociation paper" if multi_verdict != "supports image-specific grounding" or lancaster_verdict != "no clear advantage" else "adds constraints more than support"

    text = "\n".join(
        [
            "# Hardening Sprint Summary",
            "",
            "## 1. Multi-image verdict",
            multi_verdict,
            "",
            "## 2. Lancaster verdict",
            lancaster_verdict,
            "",
            "## 3. Layer-wise verdict",
            layerwise_verdict,
            "",
            "## 4. Overall paper impact",
            overall,
        ]
    )
    write_text(output_path("reports", "decision", "hardening_sprint_summary.md"), text)
    append_run_log(
        "Hardening Sprint Summary",
        [
            f"Wrote hardening sprint summary to {output_path('reports', 'decision', 'hardening_sprint_summary.md').relative_to(ROOT)}.",
        ],
    )


if __name__ == "__main__":
    main()
