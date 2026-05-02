from __future__ import annotations

import argparse

from common import append_run_log, load_project_config, metrics_path, midpoint_layer_start, write_csv
from hardening_common import load_layerwise_alignment_rows


BASELINE_FOR = {
    "T_prompt_primary": "T_neutral",
    "M_text_only": "M_matched_image",
    "M_degraded_image": "M_matched_image",
    "M_mismatched_image": "M_matched_image",
    "M_blank_image": "M_matched_image",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    args = parser.parse_args()

    config = load_project_config(args.config)
    rows = [row for row in load_layerwise_alignment_rows(args.config) if row["bootstrap_id"] == "aggregate"]
    grouped = {
        (row["family"], row["model_id"], row["condition"], row["domain"], int(row["layer"])): float(row["rsa_score"])
        for row in rows
    }

    output_rows = []
    for (family, model_id, condition, domain, layer), score in grouped.items():
        baseline_condition = BASELINE_FOR.get(condition)
        if baseline_condition is None:
            continue
        baseline_key = (family, model_id, baseline_condition, domain, layer)
        if baseline_key not in grouped:
            continue
        baseline = grouped[baseline_key]
        abs_drop = baseline - score
        pct_drop = 0.0 if baseline == 0 else abs_drop / abs(baseline)
        output_rows.append(
            {
                "family": family,
                "model_id": model_id,
                "condition": baseline_condition,
                "perturbation": condition,
                "domain": domain,
                "layer": layer,
                "baseline_rsa": baseline,
                "perturbed_rsa": score,
                "abs_drop": abs_drop,
                "pct_drop": pct_drop,
            }
        )

    write_csv(
        metrics_path("layerwise_stability_full.csv"),
        output_rows,
        ["family", "model_id", "condition", "perturbation", "domain", "layer", "baseline_rsa", "perturbed_rsa", "abs_drop", "pct_drop"],
    )
    layer_count = max(int(row["layer"]) for row in rows) + 1 if rows else 0
    append_run_log(
        "Full Stability",
        [
            f"Wrote stability metrics to {metrics_path('layerwise_stability_full.csv').relative_to(config['_resolved_root'])}.",
            f"Mid-to-late layers begin at index {midpoint_layer_start(layer_count, config['analysis']['analysis']['mid_to_late_fraction'])}.",
        ],
    )


if __name__ == "__main__":
    main()
