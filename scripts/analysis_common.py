from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from common import ROOT, load_project_config, read_csv
from hardening_common import load_active_concept_rows, mean_embedding_for_condition, selected_layers


HIERARCHY_MAPPING_PATH = ROOT / "data" / "manifests" / "things_hierarchy_mapping.csv"


def coarse_category_from_subtype(subtype: str) -> str:
    return subtype.split(",")[0].strip().lower()


def build_hierarchy_mapping_rows(concept_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in concept_rows:
        subtype = row["subtype"]
        rows.append(
            {
                "concept": row["concept"],
                "subtype": subtype,
                "coarse_category": coarse_category_from_subtype(subtype),
                "hierarchy_level_available": "coarse_category,subtype",
            }
        )
    return rows


def load_hierarchy_mapping(config_path: str = "config/analysis.yaml") -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    if HIERARCHY_MAPPING_PATH.exists():
        rows = read_csv(HIERARCHY_MAPPING_PATH)
    else:
        config = load_project_config(config_path)
        rows = build_hierarchy_mapping_rows(load_active_concept_rows(config_path, domain="sensory"))
    lookup = {row["concept"].lower(): row for row in rows}
    return lookup, rows


def active_concepts_for_config(config_path: str = "config/analysis.yaml") -> list[dict[str, str]]:
    return load_active_concept_rows(config_path, domain="sensory")


def ordered_embedding_for_concepts(embedding: np.ndarray, concepts: list[str], target_concepts: list[str]) -> np.ndarray:
    index = {concept: idx for idx, concept in enumerate(concepts)}
    return np.asarray([embedding[index[concept]] for concept in target_concepts], dtype=float)


def aggregate_condition_embedding(
    metadata_lookup: dict[tuple[str, str, int], dict[str, Any]],
    pooled: dict[str, np.ndarray],
    model_id: str,
    condition: str,
    layers: list[int],
) -> tuple[np.ndarray, list[str]]:
    return mean_embedding_for_condition(metadata_lookup, pooled, model_id, condition, layers)
