from __future__ import annotations

import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from common import (
    ROOT,
    append_run_log,
    canonical_condition_name,
    embeddings_path,
    load_project_config,
    metrics_path,
    output_path,
    read_csv,
    write_csv,
)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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


def top_neighbors(matrix: np.ndarray, k: int = 3) -> list[list[int]]:
    sims = matrix @ matrix.T
    np.fill_diagonal(sims, -np.inf)
    return [list(np.argsort(row)[-k:][::-1]) for row in sims]


def jaccard(a: list[int], b: list[int]) -> float:
    left = set(a)
    right = set(b)
    union = left | right
    return 0.0 if not union else len(left & right) / len(union)


def write_text(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def plot_geometry_support(neighbor_rows: list[dict[str, str]], procrustes_rows: list[dict[str, str]]) -> None:
    key_pairs = [
        ("T_neutral", "T_prompt_primary"),
        ("M_matched_image", "M_degraded_image"),
        ("M_matched_image", "M_mismatched_image"),
    ]
    neighbor_values = []
    procrustes_values = []
    labels = []
    for condition_a, condition_b in key_pairs:
        labels.append(f"{condition_a}->{condition_b}")
        neighbor_scores = [float(row["mean_jaccard"]) for row in neighbor_rows if row["condition_a"] == condition_a and row["condition_b"] == condition_b]
        procrustes_scores = [float(row["procrustes_disparity"]) for row in procrustes_rows if row["condition_a"] == condition_a and row["condition_b"] == condition_b]
        neighbor_values.append(mean(neighbor_scores))
        procrustes_values.append(mean(procrustes_scores))
    xs = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(10, 5))
    plt.bar(xs - width / 2, neighbor_values, width=width, label="Mean neighbor Jaccard")
    plt.bar(xs + width / 2, procrustes_values, width=width, label="Procrustes disparity")
    plt.xticks(xs, labels, rotation=25, ha="right")
    plt.ylabel("Support metric value")
    plt.title("Geometry support for dissociation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path("outputs", "figures", "fig_geometry_support.png"), dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    args = parser.parse_args()

    config = load_project_config(args.config)
    backbone_text = config["analysis"]["execution"]["sensory_backbone_text_model"]
    backbone_multimodal = config["analysis"]["execution"]["sensory_backbone_multimodal_model"]
    mid_to_late_fraction = float(config["analysis"]["analysis"]["mid_to_late_fraction"])

    neighbor_rows = [row for row in read_csv(metrics_path("neighbor_restructuring.csv")) if row["bootstrap_id"] == "aggregate"]
    procrustes_rows = [row for row in read_csv(metrics_path("procrustes_summary.csv")) if row["bootstrap_id"] == "aggregate"]

    metadata = json.loads(embeddings_path("embedding_metadata_full.json").read_text(encoding="utf-8"))
    pooled_npz = np.load(embeddings_path("pooled_embeddings_full.npz"))
    pooled = {key: np.asarray(pooled_npz[key], dtype=float) for key in pooled_npz.files}
    metadata_lookup = {
        (record["model_id"], canonical_condition_name(record["condition"]), int(record["layer"])): record
        for record in metadata["records"]
        if record["domain"] == "sensory"
    }
    layers_by_model: dict[str, list[int]] = defaultdict(list)
    for record in metadata["records"]:
        if record["domain"] == "sensory":
            layers_by_model[record["model_id"]].append(int(record["layer"]))
    text_layers = sorted(set(layers_by_model[backbone_text]))
    multimodal_layers = sorted(set(layers_by_model[backbone_multimodal]))
    selected_text_layers = text_layers[len(text_layers) - int(np.ceil(len(text_layers) * mid_to_late_fraction)) :]
    selected_multimodal_layers = multimodal_layers[len(multimodal_layers) - int(np.ceil(len(multimodal_layers) * mid_to_late_fraction)) :]

    prompt_emb, concepts = mean_embedding_for_condition(metadata_lookup, pooled, backbone_text, "T_prompt_primary", selected_text_layers)
    matched_emb, matched_concepts = mean_embedding_for_condition(metadata_lookup, pooled, backbone_multimodal, "M_matched_image", selected_multimodal_layers)
    mismatched_emb, mismatch_concepts = mean_embedding_for_condition(metadata_lookup, pooled, backbone_multimodal, "M_mismatched_image", selected_multimodal_layers)
    if concepts != matched_concepts or concepts != mismatch_concepts:
        raise RuntimeError("Geometry packaging expects aligned concept order across prompt, matched, and mismatched conditions.")
    prompt_neighbors = top_neighbors(np.asarray(prompt_emb, dtype=float), k=3)
    matched_neighbors = top_neighbors(np.asarray(matched_emb, dtype=float), k=3)
    mismatched_neighbors = top_neighbors(np.asarray(mismatched_emb, dtype=float), k=3)

    exemplar_rows = []
    for idx, concept in enumerate(concepts):
        mismatch_jaccard = jaccard(matched_neighbors[idx], mismatched_neighbors[idx])
        prompt_jaccard = jaccard(prompt_neighbors[idx], matched_neighbors[idx])
        exemplar_rows.append(
            {
                "concept": concept,
                "matched_vs_mismatched_neighbor_jaccard": mismatch_jaccard,
                "prompt_vs_matched_neighbor_jaccard": prompt_jaccard,
                "prompt_neighbors": ", ".join(concepts[item] for item in prompt_neighbors[idx]),
                "matched_neighbors": ", ".join(concepts[item] for item in matched_neighbors[idx]),
                "mismatched_neighbors": ", ".join(concepts[item] for item in mismatched_neighbors[idx]),
            }
        )
    exemplar_rows.sort(key=lambda row: float(row["matched_vs_mismatched_neighbor_jaccard"]))
    top_rows = exemplar_rows[:5]
    write_csv(
        output_path("outputs", "tables", "geometry_exemplar_neighbors.csv"),
        top_rows,
        [
            "concept",
            "matched_vs_mismatched_neighbor_jaccard",
            "prompt_vs_matched_neighbor_jaccard",
            "prompt_neighbors",
            "matched_neighbors",
            "mismatched_neighbors",
        ],
    )
    plot_geometry_support(neighbor_rows, procrustes_rows)

    report = "\n".join(
        [
            "# Geometry Support Report",
            "",
            "## Main Support Pattern",
            f"- `T_neutral -> T_prompt_primary` mean Jaccard: `{mean([float(row['mean_jaccard']) for row in neighbor_rows if row['condition_a'] == 'T_neutral' and row['condition_b'] == 'T_prompt_primary']):.4f}`",
            f"- `T_neutral -> T_prompt_primary` Procrustes disparity: `{mean([float(row['procrustes_disparity']) for row in procrustes_rows if row['condition_a'] == 'T_neutral' and row['condition_b'] == 'T_prompt_primary']):.4f}`",
            f"- `M_matched_image -> M_mismatched_image` mean Jaccard: `{mean([float(row['mean_jaccard']) for row in neighbor_rows if row['condition_a'] == 'M_matched_image' and row['condition_b'] == 'M_mismatched_image']):.4f}`",
            f"- `M_matched_image -> M_mismatched_image` Procrustes disparity: `{mean([float(row['procrustes_disparity']) for row in procrustes_rows if row['condition_a'] == 'M_matched_image' and row['condition_b'] == 'M_mismatched_image']):.4f}`",
            "",
            "## Exemplar Concepts",
            *[
                f"- `{row['concept']}` prompt_neighbors=`{row['prompt_neighbors']}` matched_neighbors=`{row['matched_neighbors']}` mismatched_neighbors=`{row['mismatched_neighbors']}`"
                for row in top_rows
            ],
            "",
            "## Interpretation",
            "- Prompting induces a moderate neighborhood shift with a smaller global reshape.",
            "- Mismatched grounding causes the strongest local rupture and the strongest global disruption.",
            "- Geometry should be read as support for dissociation, not as proof of ontology.",
        ]
    )
    write_text(output_path("reports", "main_results", "geometry_support_report.md"), report)
    append_run_log(
        "Geometry Support Packaging",
        [
            f"Wrote geometry support figure to {output_path('outputs', 'figures', 'fig_geometry_support.png').relative_to(ROOT)}.",
            f"Wrote geometry exemplar neighbors to {output_path('outputs', 'tables', 'geometry_exemplar_neighbors.csv').relative_to(ROOT)}.",
            f"Wrote geometry support report to {output_path('reports', 'main_results', 'geometry_support_report.md').relative_to(ROOT)}.",
        ],
    )


if __name__ == "__main__":
    main()
