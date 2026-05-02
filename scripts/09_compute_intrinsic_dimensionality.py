from __future__ import annotations

import argparse
import json

import numpy as np

from common import ROOT, append_run_log, canonical_condition_name, embeddings_path, load_project_config, metrics_path, read_csv, write_csv


def load_subset_concepts(path: str) -> list[str]:
    subset_rows = read_csv((ROOT / path).resolve())
    return [row["concept"].lower() for row in subset_rows if row["domain"] == "sensory"]


def default_concept_subset(config: dict) -> str:
    return str(config["analysis"]["execution"]["default_concept_subset"])


def reorder_to_subset(matrix: np.ndarray, concepts: list[str], subset: list[str]) -> np.ndarray:
    index = {concept.lower(): idx for idx, concept in enumerate(concepts)}
    missing = [concept for concept in subset if concept not in index]
    if missing:
        raise RuntimeError(f"Requested subset concepts missing from embedding record: {', '.join(missing)}")
    ordered_idx = [index[concept] for concept in subset]
    return np.asarray(matrix[ordered_idx], dtype=float)


def participation_ratio(matrix: np.ndarray) -> float:
    centered = np.asarray(matrix, dtype=float) - np.asarray(matrix, dtype=float).mean(axis=0, keepdims=True)
    if centered.shape[0] <= 1:
        return 0.0
    gram = centered @ centered.T
    trace = float(np.trace(gram))
    frob_sq = float(np.square(gram).sum())
    if trace <= 1e-12 or frob_sq <= 1e-12:
        return 0.0
    return float((trace ** 2) / frob_sq)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--concept-subset", default=None)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()

    config = load_project_config(args.config)
    arrays = np.load(embeddings_path("pooled_embeddings_full.npz"), mmap_mode="r")
    metadata = json.loads(embeddings_path("embedding_metadata_full.json").read_text(encoding="utf-8"))
    active_subset = args.concept_subset or default_concept_subset(config)
    output_name = args.output_name or ("intrinsic_dimensionality_things38.csv" if args.concept_subset else "intrinsic_dimensionality.csv")
    subset_concepts = load_subset_concepts(active_subset)

    rows = []
    for record in metadata["records"]:
        matrix = np.asarray(arrays[f"record_{record['record_id']}"], dtype=float)
        domain = record["domain"]
        concepts = [concept.lower() for concept in record["concepts"]]
        if domain == "sensory":
            matrix = reorder_to_subset(matrix, concepts, subset_concepts)
            num_concepts = len(subset_concepts)
            concept_subset = active_subset
        else:
            num_concepts = matrix.shape[0]
            concept_subset = ""
        rows.append(
            {
                "model": record["model_id"],
                "layer": int(record["layer"]),
                "domain": domain,
                "condition": canonical_condition_name(record["condition"]),
                "participation_ratio": participation_ratio(matrix),
                "num_concepts": num_concepts,
                "feature_dim": matrix.shape[1],
                "concept_subset": concept_subset,
            }
        )

    write_csv(
        metrics_path(output_name),
        rows,
        ["model", "layer", "domain", "condition", "participation_ratio", "num_concepts", "feature_dim", "concept_subset"],
    )
    append_run_log(
        "Intrinsic Dimensionality",
        [
            f"Wrote intrinsic dimensionality metrics to {metrics_path(output_name).relative_to(config['_resolved_root'])}.",
            f"Participation ratio was computed on the active sensory subset {active_subset}.",
        ],
    )


if __name__ == "__main__":
    main()
