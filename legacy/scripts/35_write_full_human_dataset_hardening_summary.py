from __future__ import annotations

import argparse

from common import ROOT, append_run_log, metrics_path, output_path, read_csv
from hardening_common import write_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.parse_args()

    variance_rows = read_csv(metrics_path("full_things_image_variance.csv"))
    comparison_rows = read_csv(output_path("outputs", "tables", "full_things_prototype_comparison_table.csv"))

    stable_count = sum(row["stability_verdict"] == "prototype_stable" for row in variance_rows)
    if stable_count >= max(1, int(0.6 * len(variance_rows))):
        full_things_verdict = "supports concept-level grounding"
    elif stable_count <= int(0.3 * len(variance_rows)):
        full_things_verdict = "too image-fragile to help materially"
    else:
        full_things_verdict = "supports image-sensitive but stable prototypes"

    comparison_lookup = {(row["anchor_name"], row["comparison_name"]): float(row["mean_gap"]) for row in comparison_rows}
    things_gain = comparison_lookup.get(("THINGS behavioral similarity", "prototype_minus_single"), 0.0)
    controlled_gain = comparison_lookup.get(("controlled_THINGS", "prototype_minus_single"), 0.0)
    siglip_gain = comparison_lookup.get(("SigLIP2", "prototype_minus_single"), 0.0)

    if full_things_verdict == "supports concept-level grounding" and (controlled_gain > 0 or things_gain > 0):
        paper_impact = "materially strengthens the dissociation paper"
        final_recommendation = "Use these additions in the NeurIPS paper"
    elif full_things_verdict != "too image-fragile to help materially":
        paper_impact = "adds modest support but not a cleaner story"
        final_recommendation = "Keep only the strongest subset and stop expanding"
    else:
        paper_impact = "does not justify more scope"
        final_recommendation = "Do not expand further; package the current paper as-is"

    bullets = [
        f"- Stable concepts: `{stable_count}/{len(variance_rows)}`",
        f"- Prototype minus single on raw THINGS: `{things_gain:.4f}`",
        f"- Prototype minus single on controlled THINGS: `{controlled_gain:.4f}`",
        f"- Prototype minus single on SigLIP2: `{siglip_gain:.4f}`",
        "- This branch addresses the single-JPEG critique directly with exhaustive local THINGS archive coverage.",
        "- The reduced v4 branch does not add new human datasets and should not be framed as a broader cross-dataset expansion.",
    ]

    text = "\n".join(
        [
            "# Full Human Dataset Hardening Summary",
            "",
            "## 1. Full THINGS image verdict",
            full_things_verdict,
            "",
            "## 2. Paper impact",
            paper_impact,
            "",
            "## 3. Final recommendation",
            final_recommendation,
            "",
            *bullets,
        ]
    )
    write_text(output_path("reports", "decision", "full_human_dataset_hardening_summary.md"), text)
    append_run_log(
        "Full Human Dataset Hardening Summary",
        [
            f"Wrote full human dataset hardening summary to {output_path('reports', 'decision', 'full_human_dataset_hardening_summary.md').relative_to(ROOT)}.",
        ],
    )


if __name__ == "__main__":
    main()
