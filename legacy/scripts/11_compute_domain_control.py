from __future__ import annotations

import argparse
from collections import defaultdict

from common import append_run_log, canonical_condition_name, load_project_config, metrics_path, write_csv
from hardening_common import load_layerwise_alignment_rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    args = parser.parse_args()

    config = load_project_config(args.config)
    rows = [row for row in load_layerwise_alignment_rows(args.config) if row["bootstrap_id"] == "aggregate"]

    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("model", row.get("model_id", "")), row["domain"], canonical_condition_name(row["condition"]), row.get("anchor_name", row.get("anchor_model_id", "")))].append(float(row["rsa_score"]))

    models = sorted({key[0] for key in grouped})
    anchors = sorted({key[3] for key in grouped})
    output_rows = []
    for model in models:
        for anchor in anchors:
            sensory_neutral = mean(grouped.get((model, "sensory", "T_neutral", anchor), []))
            sensory_prompt = mean(grouped.get((model, "sensory", "T_prompt_primary", anchor), []))
            abstract_neutral = mean(grouped.get((model, "abstract", "T_neutral", anchor), []))
            abstract_prompt = mean(grouped.get((model, "abstract", "T_prompt_primary", anchor), []))
            sensory_gap = sensory_prompt - sensory_neutral
            abstract_gap = abstract_prompt - abstract_neutral
            sensory_matched = mean(grouped.get((model, "sensory", "M_matched_image", anchor), []))
            sensory_text_only = mean(grouped.get((model, "sensory", "M_text_only", anchor), []))
            output_rows.append(
                {
                    "model": model,
                    "anchor_name": anchor,
                    "sensory_prompt_gap": sensory_gap,
                    "abstract_prompt_gap": abstract_gap,
                    "sensory_minus_abstract_gap": sensory_gap - abstract_gap,
                    "multimodal_grounding_gap": sensory_matched - sensory_text_only,
                }
            )

    write_csv(
        metrics_path("domain_control_summary.csv"),
        output_rows,
        ["model", "anchor_name", "sensory_prompt_gap", "abstract_prompt_gap", "sensory_minus_abstract_gap", "multimodal_grounding_gap"],
    )
    append_run_log(
        "Domain Control",
        [
            f"Wrote domain-control summary to {metrics_path('domain_control_summary.csv').relative_to(config['_resolved_root'])}.",
            "Sensory and abstract prompt gaps are reported per model and per anchor.",
        ],
    )


if __name__ == "__main__":
    main()
