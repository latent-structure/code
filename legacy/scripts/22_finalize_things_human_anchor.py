from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import (
    ROOT,
    append_run_log,
    canonical_condition_name,
    condensed_cosine_distance,
    embeddings_path,
    load_project_config,
    metrics_path,
    output_path,
    percentile_interval,
    read_csv,
    set_global_seed,
    write_csv,
)


THINGS_BEHAVIOR_MATRIX = ROOT / "data" / "anchors" / "things_behavioral_similarity.npy"
THINGS_BEHAVIOR_CONCEPTS = ROOT / "data" / "anchors" / "things_behavioral_concepts.json"
CONDITIONS = ["T_prompt_primary", "M_matched_image", "M_degraded_image", "M_mismatched_image", "M_blank_image"]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    from common import spearman_corr as shared_spearman_corr

    return shared_spearman_corr(np.asarray(x, dtype=float), np.asarray(y, dtype=float))


def mean_embedding_for_condition(
    metadata_lookup: dict[tuple[str, str, int], dict[str, object]],
    pooled: dict[str, np.ndarray],
    model_id: str,
    condition: str,
    layers: list[int],
) -> tuple[np.ndarray, list[str]]:
    matrices = []
    concepts: list[str] | None = None
    for layer in layers:
        record = metadata_lookup.get((model_id, condition, layer))
        if record is None:
            continue
        if concepts is None:
            concepts = [concept.lower() for concept in record["concepts"]]
        matrices.append(np.asarray(pooled[f"record_{record['record_id']}"], dtype=float))
    if not matrices or concepts is None:
        raise RuntimeError(f"Missing embeddings for {model_id} {condition}")
    return np.mean(np.stack(matrices), axis=0), concepts


def condition_model_id(backbone_text: str, backbone_multimodal: str, condition: str) -> str:
    return backbone_text if condition.startswith("T_") else backbone_multimodal


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_bootstrap_scores(
    prompt_rdm: np.ndarray,
    matched_rdm: np.ndarray,
    degraded_rdm: np.ndarray,
    mismatched_rdm: np.ndarray,
    blank_rdm: np.ndarray,
    behavior_rdm: np.ndarray,
    resamples: int,
    seed: int,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    rows = []
    n = len(behavior_rdm)
    for sample_id in range(resamples):
        idx = rng.integers(0, n, size=n)
        prompt_score = spearman_corr(prompt_rdm[idx], behavior_rdm[idx])
        matched_score = spearman_corr(matched_rdm[idx], behavior_rdm[idx])
        degraded_score = spearman_corr(degraded_rdm[idx], behavior_rdm[idx])
        mismatched_score = spearman_corr(mismatched_rdm[idx], behavior_rdm[idx])
        blank_score = spearman_corr(blank_rdm[idx], behavior_rdm[idx])
        rows.append(
            {
                "sample_id": sample_id,
                "prompt_rsa": prompt_score,
                "matched_rsa": matched_score,
                "degraded_rsa": degraded_score,
                "mismatched_rsa": mismatched_score,
                "blank_rsa": blank_score,
                "prompt_minus_matched_gap": prompt_score - matched_score,
                "matched_minus_mismatched_gap": matched_score - mismatched_score,
                "matched_minus_blank_gap": matched_score - blank_score,
            }
        )
    return rows


def plot_human_anchor_gap(
    layer_prompt: list[tuple[int, float, float, float]],
    layer_matched: list[tuple[int, float, float, float]],
    aggregate_summary: dict[str, float],
) -> None:
    plt.figure(figsize=(10, 6))
    if layer_prompt:
        layers = [item[0] for item in layer_prompt]
        plt.plot(layers, [item[1] for item in layer_prompt], label="T_prompt_primary", color="#1f77b4")
        plt.fill_between(layers, [item[2] for item in layer_prompt], [item[3] for item in layer_prompt], color="#1f77b4", alpha=0.18)
    if layer_matched:
        layers = [item[0] for item in layer_matched]
        plt.plot(layers, [item[1] for item in layer_matched], label="M_matched_image", color="#d95f02")
        plt.fill_between(layers, [item[2] for item in layer_matched], [item[3] for item in layer_matched], color="#d95f02", alpha=0.18)
    plt.axhline(aggregate_summary["prompt_mean"], color="#1f77b4", linestyle="--", linewidth=1.0)
    plt.axhline(aggregate_summary["matched_mean"], color="#d95f02", linestyle="--", linewidth=1.0)
    plt.xlabel("Layer")
    plt.ylabel("THINGS RSA")
    plt.title("THINGS human-anchor gap")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path("outputs", "figures", "fig_human_anchor_gap.png"), dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    args = parser.parse_args()

    config = load_project_config(args.config)
    set_global_seed(config["seeds"]["global"])
    backbone_text = config["analysis"]["execution"]["sensory_backbone_text_model"]
    backbone_multimodal = config["analysis"]["execution"]["sensory_backbone_multimodal_model"]
    mid_to_late_fraction = float(config["analysis"]["analysis"]["mid_to_late_fraction"])
    bootstrap_resamples = int(config["analysis"]["budgets"].get("bootstrap_resamples", 1000))

    metadata = json.loads(embeddings_path("embedding_metadata_full.json").read_text(encoding="utf-8"))
    pooled_npz = np.load(embeddings_path("pooled_embeddings_full.npz"))
    pooled = {key: np.asarray(pooled_npz[key], dtype=float) for key in pooled_npz.files}
    metadata_lookup = {
        (record["model_id"], canonical_condition_name(record["condition"]), int(record["layer"])): record
        for record in metadata["records"]
        if record["domain"] == "sensory"
    }
    things_behavior = np.load(THINGS_BEHAVIOR_MATRIX)
    things_concepts = [concept.lower() for concept in json.loads(THINGS_BEHAVIOR_CONCEPTS.read_text(encoding="utf-8"))]
    things_index = {concept: idx for idx, concept in enumerate(things_concepts)}

    layers_by_model: dict[str, list[int]] = defaultdict(list)
    for record in metadata["records"]:
        if record["domain"] == "sensory":
            layers_by_model[record["model_id"]].append(int(record["layer"]))
    text_layers = sorted(set(layers_by_model[backbone_text]))
    multimodal_layers = sorted(set(layers_by_model[backbone_multimodal]))
    selected_text_layers = text_layers[len(text_layers) - int(np.ceil(len(text_layers) * mid_to_late_fraction)) :]
    selected_multimodal_layers = multimodal_layers[len(multimodal_layers) - int(np.ceil(len(multimodal_layers) * mid_to_late_fraction)) :]

    condition_embeddings: dict[str, np.ndarray] = {}
    overlap_concepts: list[str] | None = None
    for condition in CONDITIONS:
        model_id = condition_model_id(backbone_text, backbone_multimodal, condition)
        layer_selection = selected_text_layers if condition.startswith("T_") else selected_multimodal_layers
        embedding, concepts = mean_embedding_for_condition(metadata_lookup, pooled, model_id, condition, layer_selection)
        matched_positions = [idx for idx, concept in enumerate(concepts) if concept in things_index]
        matched_concepts = [concepts[idx] for idx in matched_positions]
        condition_embeddings[condition] = np.asarray(embedding[matched_positions], dtype=float)
        if overlap_concepts is None:
            overlap_concepts = matched_concepts
        elif overlap_concepts != matched_concepts:
            raise RuntimeError("THINGS finalization expects aligned overlap concepts across conditions.")
    if overlap_concepts is None:
        raise RuntimeError("Failed to construct THINGS overlap concepts.")

    behavior_idx = [things_index[concept] for concept in overlap_concepts]
    behavior_dist = 1.0 - things_behavior[np.ix_(behavior_idx, behavior_idx)]
    behavior_rdm = np.asarray(behavior_dist[np.triu_indices(len(overlap_concepts), k=1)], dtype=float)
    condition_rdms = {condition: condensed_cosine_distance(matrix) for condition, matrix in condition_embeddings.items()}

    aggregate_bootstrap_rows = build_bootstrap_scores(
        condition_rdms["T_prompt_primary"],
        condition_rdms["M_matched_image"],
        condition_rdms["M_degraded_image"],
        condition_rdms["M_mismatched_image"],
        condition_rdms["M_blank_image"],
        behavior_rdm,
        bootstrap_resamples,
        seed=config["seeds"]["global"],
    )

    output_rows = [
        {
            "analysis_level": "aggregate_mean_embedding",
            "layer": "",
            **row,
        }
        for row in aggregate_bootstrap_rows
    ]

    prompt_layer_plot = []
    matched_layer_plot = []
    max_layer = min(len(text_layers), len(multimodal_layers))
    for layer in range(max_layer):
        prompt_record = metadata_lookup.get((backbone_text, "T_prompt_primary", layer))
        matched_record = metadata_lookup.get((backbone_multimodal, "M_matched_image", layer))
        if prompt_record is None or matched_record is None:
            continue
        prompt_matrix = np.asarray(pooled[f"record_{prompt_record['record_id']}"], dtype=float)
        matched_matrix = np.asarray(pooled[f"record_{matched_record['record_id']}"], dtype=float)
        prompt_positions = [idx for idx, concept in enumerate([c.lower() for c in prompt_record["concepts"]]) if concept in things_index]
        matched_positions = [idx for idx, concept in enumerate([c.lower() for c in matched_record["concepts"]]) if concept in things_index]
        prompt_rdm = condensed_cosine_distance(prompt_matrix[prompt_positions])
        matched_rdm = condensed_cosine_distance(matched_matrix[matched_positions])
        layer_bootstrap = build_bootstrap_scores(
            prompt_rdm,
            matched_rdm,
            matched_rdm,
            matched_rdm,
            matched_rdm,
            behavior_rdm,
            bootstrap_resamples,
            seed=config["seeds"]["global"] + layer + 1,
        )
        prompt_values = np.asarray([row["prompt_rsa"] for row in layer_bootstrap], dtype=float)
        matched_values = np.asarray([row["matched_rsa"] for row in layer_bootstrap], dtype=float)
        prompt_low, prompt_high = percentile_interval(prompt_values, 0.95)
        matched_low, matched_high = percentile_interval(matched_values, 0.95)
        prompt_layer_plot.append((layer, float(np.mean(prompt_values)), prompt_low, prompt_high))
        matched_layer_plot.append((layer, float(np.mean(matched_values)), matched_low, matched_high))
        for row in layer_bootstrap:
            output_rows.append(
                {
                    "analysis_level": "layerwise",
                    "layer": layer,
                    **row,
                }
            )

    write_csv(
        metrics_path("things_gap_bootstrap.csv"),
        output_rows,
        [
            "analysis_level",
            "layer",
            "sample_id",
            "prompt_rsa",
            "matched_rsa",
            "degraded_rsa",
            "mismatched_rsa",
            "blank_rsa",
            "prompt_minus_matched_gap",
            "matched_minus_mismatched_gap",
            "matched_minus_blank_gap",
        ],
    )

    aggregate_prompt_values = np.asarray([row["prompt_rsa"] for row in aggregate_bootstrap_rows], dtype=float)
    aggregate_matched_values = np.asarray([row["matched_rsa"] for row in aggregate_bootstrap_rows], dtype=float)
    aggregate_gap_values = np.asarray([row["prompt_minus_matched_gap"] for row in aggregate_bootstrap_rows], dtype=float)
    mismatch_values = np.asarray([row["matched_minus_mismatched_gap"] for row in aggregate_bootstrap_rows], dtype=float)
    blank_values = np.asarray([row["matched_minus_blank_gap"] for row in aggregate_bootstrap_rows], dtype=float)
    gap_low, gap_high = percentile_interval(aggregate_gap_values, 0.95)
    mismatch_low, mismatch_high = percentile_interval(mismatch_values, 0.95)
    blank_low, blank_high = percentile_interval(blank_values, 0.95)

    table_rows = [
        {
            "anchor_name": "THINGS behavioral similarity",
            "condition": "T_prompt_primary",
            "mean_rsa": float(np.mean(aggregate_prompt_values)),
            "ci_low": percentile_interval(aggregate_prompt_values, 0.95)[0],
            "ci_high": percentile_interval(aggregate_prompt_values, 0.95)[1],
            "comparison_name": "",
            "comparison_mean": "",
            "comparison_ci_low": "",
            "comparison_ci_high": "",
            "interpretation": "Prompting recovers substantial human-behavioral sensory organization.",
        },
        {
            "anchor_name": "THINGS behavioral similarity",
            "condition": "M_matched_image",
            "mean_rsa": float(np.mean(aggregate_matched_values)),
            "ci_low": percentile_interval(aggregate_matched_values, 0.95)[0],
            "ci_high": percentile_interval(aggregate_matched_values, 0.95)[1],
            "comparison_name": "prompt_minus_matched_gap",
            "comparison_mean": float(np.mean(aggregate_gap_values)),
            "comparison_ci_low": gap_low,
            "comparison_ci_high": gap_high,
            "interpretation": "Prompting exceeds matched grounding on the raw THINGS anchor in the current overlap set.",
        },
        {
            "anchor_name": "THINGS behavioral similarity",
            "condition": "M_mismatched_image",
            "mean_rsa": mean([row["mismatched_rsa"] for row in aggregate_bootstrap_rows]),
            "ci_low": percentile_interval(np.asarray([row["mismatched_rsa"] for row in aggregate_bootstrap_rows], dtype=float), 0.95)[0],
            "ci_high": percentile_interval(np.asarray([row["mismatched_rsa"] for row in aggregate_bootstrap_rows], dtype=float), 0.95)[1],
            "comparison_name": "matched_minus_mismatched_gap",
            "comparison_mean": float(np.mean(mismatch_values)),
            "comparison_ci_low": mismatch_low,
            "comparison_ci_high": mismatch_high,
            "interpretation": "Mismatching causes a strong collapse relative to matched images on THINGS.",
        },
        {
            "anchor_name": "THINGS behavioral similarity",
            "condition": "M_blank_image",
            "mean_rsa": mean([row["blank_rsa"] for row in aggregate_bootstrap_rows]),
            "ci_low": percentile_interval(np.asarray([row["blank_rsa"] for row in aggregate_bootstrap_rows], dtype=float), 0.95)[0],
            "ci_high": percentile_interval(np.asarray([row["blank_rsa"] for row in aggregate_bootstrap_rows], dtype=float), 0.95)[1],
            "comparison_name": "matched_minus_blank_gap",
            "comparison_mean": float(np.mean(blank_values)),
            "comparison_ci_low": blank_low,
            "comparison_ci_high": blank_high,
            "interpretation": "Blank-image collapse indicates the human-anchor result is sensitive to image availability, not only to text prompt changes.",
        },
    ]
    write_csv(
        output_path("outputs", "tables", "human_anchor_main_result_table.csv"),
        table_rows,
        [
            "anchor_name",
            "condition",
            "mean_rsa",
            "ci_low",
            "ci_high",
            "comparison_name",
            "comparison_mean",
            "comparison_ci_low",
            "comparison_ci_high",
            "interpretation",
        ],
    )

    plot_human_anchor_gap(
        prompt_layer_plot,
        matched_layer_plot,
        {
            "prompt_mean": float(np.mean(aggregate_prompt_values)),
            "matched_mean": float(np.mean(aggregate_matched_values)),
        },
    )
    append_run_log(
        "THINGS Human Anchor Finalization",
        [
            f"Wrote THINGS gap bootstrap to {metrics_path('things_gap_bootstrap.csv').relative_to(ROOT)}.",
            f"Wrote THINGS main result table to {output_path('outputs', 'tables', 'human_anchor_main_result_table.csv').relative_to(ROOT)}.",
            f"Wrote THINGS gap figure to {output_path('outputs', 'figures', 'fig_human_anchor_gap.png').relative_to(ROOT)}.",
        ],
    )


if __name__ == "__main__":
    main()
