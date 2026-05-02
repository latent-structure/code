from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis_common import aggregate_condition_embedding, load_hierarchy_mapping, ordered_embedding_for_concepts
from common import ROOT, append_run_log, load_project_config, metrics_path, output_path, write_csv, write_json
from hardening_common import condition_model_id, load_embedding_bundle, load_project_backbone, selected_layers, write_text


CONDITIONS = [
    "T_neutral",
    "T_prompt_primary",
    "M_text_only",
    "M_matched_image",
    "M_prompt_plus_matched_image",
    "M_degraded_image",
    "M_mismatched_image",
    "M_blank_image",
]
TARGETS = ["coarse_category", "subtype"]


def label_baseline(y: np.ndarray) -> float:
    counts = Counter(y.tolist())
    if not counts:
        return 0.0
    return 1.0 / len(counts)


def cross_val_scores(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, float]:
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    predictions = []
    targets = []
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=500,
            solver="saga",
            n_jobs=1,
        ),
    )
    for train_idx, test_idx in splitter.split(X, y):
        fitted = model.fit(X[train_idx], y[train_idx])
        preds = fitted.predict(X[test_idx])
        predictions.extend(preds.tolist())
        targets.extend(y[test_idx].tolist())
    return float(f1_score(targets, predictions, average="macro")), float(balanced_accuracy_score(targets, predictions))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    args = parser.parse_args()

    config = load_project_config(args.config)
    _backbone_config, backbone_text, backbone_multimodal, mid_fraction = load_project_backbone(args.config)
    metadata_lookup, pooled, layers_by_model, _ = load_embedding_bundle()
    hierarchy_lookup, hierarchy_rows = load_hierarchy_mapping(args.config)
    selected_text_layers = selected_layers(layers_by_model[backbone_text], mid_fraction)
    selected_multimodal_layers = selected_layers(layers_by_model[backbone_multimodal], mid_fraction)
    seed = int(config["seeds"]["global"])

    probe_rows = []
    summary = {}
    for target in TARGETS:
        if target == "coarse_category":
            target_labels = {concept: row["coarse_category"] for concept, row in hierarchy_lookup.items()}
            eligible_concepts = sorted(target_labels)
        else:
            subtype_counts = Counter(row["subtype"] for row in hierarchy_rows)
            eligible_rows = [row for row in hierarchy_rows if subtype_counts[row["subtype"]] >= 3]
            target_labels = {row["concept"].lower(): row["subtype"] for row in eligible_rows}
            eligible_concepts = sorted(target_labels)

        summary[target] = {
            "num_concepts": len(eligible_concepts),
            "num_classes": len(set(target_labels.values())),
            "coverage_ratio": len(eligible_concepts) / len(hierarchy_rows),
            "condition_scores": {},
        }
        for condition in CONDITIONS:
            model_id = condition_model_id(backbone_text, backbone_multimodal, condition)
            layers = selected_text_layers if condition.startswith("T_") else selected_multimodal_layers
            embedding, concepts = aggregate_condition_embedding(metadata_lookup, pooled, model_id, condition, layers)
            ordered = ordered_embedding_for_concepts(embedding, concepts, eligible_concepts)
            y = np.asarray([target_labels[concept] for concept in eligible_concepts], dtype=object)
            macro_f1, bal_acc = cross_val_scores(ordered, y, seed)
            chance = label_baseline(y)
            summary[target]["condition_scores"][condition] = {
                "macro_f1": macro_f1,
                "balanced_accuracy": bal_acc,
                "chance": chance,
            }

        for metric_name in ["macro_f1", "balanced_accuracy"]:
            prompt_score = summary[target]["condition_scores"]["T_prompt_primary"][metric_name]
            matched_score = summary[target]["condition_scores"]["M_matched_image"][metric_name]
            for condition in CONDITIONS:
                metric_value = summary[target]["condition_scores"][condition][metric_name]
                probe_rows.append(
                    {
                        "condition": condition,
                        "target_label": target,
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                        "chance_value": summary[target]["condition_scores"][condition]["chance"],
                        "coverage_concepts": summary[target]["num_concepts"],
                        "coverage_classes": summary[target]["num_classes"],
                        "delta_vs_prompt": metric_value - prompt_score,
                        "delta_vs_matched": metric_value - matched_score,
                    }
                )

    write_csv(
        metrics_path("linear_probe_results.csv"),
        probe_rows,
        [
            "condition",
            "target_label",
            "metric_name",
            "metric_value",
            "chance_value",
            "coverage_concepts",
            "coverage_classes",
            "delta_vs_prompt",
            "delta_vs_matched",
        ],
    )
    write_json(metrics_path("linear_probe_summary.json"), summary)
    report_lines = [
        "# Linear Probe Report",
        "",
        "## Summary",
        "- Coarse category uses first-token superordinate bins derived from THINGS subtypes.",
        "- Subtype probe is evaluated only on labels with at least three examples.",
    ]
    for target in TARGETS:
        item = summary[target]
        report_lines.extend(
            [
                f"### {target}",
                f"- Concepts used: `{item['num_concepts']}`",
                f"- Classes used: `{item['num_classes']}`",
                f"- Coverage ratio: `{item['coverage_ratio']:.3f}`",
                f"- Prompt macro-F1: `{item['condition_scores']['T_prompt_primary']['macro_f1']:.4f}`",
                f"- Matched macro-F1: `{item['condition_scores']['M_matched_image']['macro_f1']:.4f}`",
                f"- Combined macro-F1: `{item['condition_scores']['M_prompt_plus_matched_image']['macro_f1']:.4f}`",
                f"- Prompt balanced accuracy: `{item['condition_scores']['T_prompt_primary']['balanced_accuracy']:.4f}`",
                f"- Matched balanced accuracy: `{item['condition_scores']['M_matched_image']['balanced_accuracy']:.4f}`",
                f"- Combined balanced accuracy: `{item['condition_scores']['M_prompt_plus_matched_image']['balanced_accuracy']:.4f}`",
            ]
        )
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "- If prompt-plus-image preserves or improves coarse-category decoding, the dissociation is about representational priority rather than information loss.",
            "- If matched-image decoding drops sharply on coarse labels, that would indicate deeper information loss in the grounded regime.",
        ]
    )
    write_text(output_path("reports", "main_results", "linear_probe_report.md"), "\n".join(report_lines))
    append_run_log(
        "Linear Probes",
        [
            f"Wrote linear-probe metrics to {metrics_path('linear_probe_results.csv').relative_to(ROOT)}.",
            f"Wrote linear-probe report to {output_path('reports', 'main_results', 'linear_probe_report.md').relative_to(ROOT)}.",
        ],
    )


if __name__ == "__main__":
    main()
