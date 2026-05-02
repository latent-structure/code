from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    ROOT,
    canonical_condition_name,
    condensed_cosine_distance,
    embeddings_path,
    load_project_config,
    metrics_path,
    rankdata,
    read_csv,
)


THINGS_BEHAVIOR_MATRIX = ROOT / "data" / "anchors" / "things_behavioral_similarity.npy"
THINGS_BEHAVIOR_CONCEPTS = ROOT / "data" / "anchors" / "things_behavioral_concepts.json"
LANCASTER_SENSORIMOTOR = ROOT / "Lancaster_sensorimotor.csv"
LANCASTER_PERCEPTUAL = ROOT / "Lancaster_perceptual.csv"
LANCASTER_ACTION = ROOT / "Lancaster_action.csv"

LANCASTER_SPACES: dict[str, list[str]] = {
    "lancaster_full_sensorimotor": [
        "Auditory.mean",
        "Gustatory.mean",
        "Haptic.mean",
        "Interoceptive.mean",
        "Olfactory.mean",
        "Visual.mean",
        "Foot_leg.mean",
        "Hand_arm.mean",
        "Head.mean",
        "Mouth.mean",
        "Torso.mean",
    ],
    "lancaster_perceptual": [
        "Auditory.mean",
        "Gustatory.mean",
        "Haptic.mean",
        "Interoceptive.mean",
        "Olfactory.mean",
        "Visual.mean",
    ],
    "lancaster_haptic_material": [
        "Haptic.mean",
        "Hand_arm.mean",
    ],
}

CONDITION_ORDER = [
    "T_neutral",
    "T_prompt_primary",
    "M_text_only",
    "M_prompt_only",
    "M_matched_image",
    "M_prompt_plus_matched_image",
    "M_degraded_image",
    "M_mismatched_image",
    "M_blank_image",
]


def load_stage01_module():
    script_path = ROOT / "scripts" / "01_extract_hidden_states.py"
    spec = importlib.util.spec_from_file_location("stage01_extract_hidden_states", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load helper module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


load_stage05_module = load_stage01_module


def resolve_cached_snapshot(model_id: str, cache_root: str) -> Path:
    org, name = model_id.split("/", 1)
    candidates = [
        Path(cache_root) / f"models--{org}--{name}",
        Path(cache_root) / "hub" / f"models--{org}--{name}",
    ]
    tried_refs = []
    for model_dir in candidates:
        refs_main = model_dir / "refs" / "main"
        tried_refs.append(str(refs_main))
        if not refs_main.exists():
            continue
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot_dir = model_dir / "snapshots" / revision
        if not snapshot_dir.exists():
            raise RuntimeError(f"Snapshot missing for {model_id}: {snapshot_dir}")
        return snapshot_dir
    raise RuntimeError(f"Cache ref missing for {model_id}: {', '.join(tried_refs)}")


def load_embedding_bundle(domain: str = "sensory") -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[str, np.ndarray], dict[str, list[int]], dict[str, Any]]:
    metadata = json.loads(embeddings_path("embedding_metadata_full.json").read_text(encoding="utf-8"))
    pooled_npz = np.load(embeddings_path("pooled_embeddings_full.npz"))
    pooled = {key: np.asarray(pooled_npz[key], dtype=np.float32) for key in pooled_npz.files}
    metadata_lookup = {
        (record["model_id"], canonical_condition_name(record["condition"]), int(record["layer"])): record
        for record in metadata["records"]
        if record["domain"] == domain
    }
    layers_by_model: dict[str, list[int]] = {}
    for record in metadata["records"]:
        if record["domain"] != domain:
            continue
        layers_by_model.setdefault(record["model_id"], []).append(int(record["layer"]))
    for model_id in list(layers_by_model):
        layers_by_model[model_id] = sorted(set(layers_by_model[model_id]))
    return metadata_lookup, pooled, layers_by_model, metadata


def load_active_concept_rows(config_path: str = "config/analysis.yaml", domain: str | None = None) -> list[dict[str, str]]:
    config = load_project_config(config_path)
    subset_path = config["analysis"]["execution"].get("default_concept_subset", "")
    if subset_path:
        subset_file = ROOT / subset_path
        if subset_file.exists():
            rows = read_csv(subset_file)
        else:
            rows = read_csv(ROOT / "data" / "concepts" / "full_concept_list.csv")
    else:
        rows = read_csv(ROOT / "data" / "concepts" / "full_concept_list.csv")
    if domain is not None:
        rows = [row for row in rows if row["domain"] == domain]
    return rows


def load_layerwise_alignment_rows(config_path: str = "config/analysis.yaml") -> list[dict[str, str]]:
    config = load_project_config(config_path)
    primary = metrics_path("layerwise_alignment_full.csv")
    fallback = metrics_path("reference_space_alignment.csv")
    if primary.exists():
        rows = read_csv(primary)
        if rows:
            return rows
    if fallback.exists():
        rows = read_csv(fallback)
        for row in rows:
            if "model_id" not in row or not row["model_id"]:
                row["model_id"] = row.get("model", "")
            if "family" not in row or not row["family"]:
                condition = row.get("condition", "")
                row["family"] = "text" if condition.startswith("T_") else "multimodal"
        return rows
    return []


def selected_layers(layers: list[int], fraction: float) -> list[int]:
    count = int(np.ceil(len(layers) * fraction))
    return layers[len(layers) - count :]


def mean_embedding_for_condition(
    metadata_lookup: dict[tuple[str, str, int], dict[str, Any]],
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
        matrices.append(np.asarray(pooled[f"record_{record['record_id']}"], dtype=np.float32))
    if not matrices or concepts is None:
        raise RuntimeError(f"Missing embeddings for {model_id} {condition}")
    return np.mean(np.stack(matrices), axis=0, dtype=np.float32), concepts


def condition_model_id(backbone_text: str, backbone_multimodal: str, condition: str) -> str:
    return backbone_text if condition.startswith("T_") else backbone_multimodal


def load_things_reference() -> tuple[np.ndarray, list[str], dict[str, int]]:
    things_behavior = np.load(THINGS_BEHAVIOR_MATRIX)
    things_concepts = [concept.lower() for concept in json.loads(THINGS_BEHAVIOR_CONCEPTS.read_text(encoding="utf-8"))]
    return things_behavior, things_concepts, {concept: idx for idx, concept in enumerate(things_concepts)}


def load_siglip_reference(
    metadata_lookup: dict[tuple[str, str, int], dict[str, Any]],
    pooled: dict[str, np.ndarray],
    layers_by_model: dict[str, list[int]],
    metadata: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    siglip_model_ids = sorted(
        {
            record["model_id"]
            for record in metadata["records"]
            if record["family"] == "anchor" and "siglip" in record["model_id"].lower() and record["domain"] == "sensory"
        }
    )
    if not siglip_model_ids:
        raise RuntimeError("Could not locate SigLIP2 anchor rows in embedding metadata.")
    siglip_model_id = siglip_model_ids[0]
    return mean_embedding_for_condition(
        metadata_lookup,
        pooled,
        siglip_model_id,
        "reference_anchor_image",
        layers_by_model[siglip_model_id],
    )


def lexical_distance(a: str, b: str) -> float:
    padded_a = f"__{a.lower()}__"
    padded_b = f"__{b.lower()}__"
    trigrams_a = {padded_a[idx : idx + 3] for idx in range(len(padded_a) - 2)}
    trigrams_b = {padded_b[idx : idx + 3] for idx in range(len(padded_b) - 2)}
    union = trigrams_a | trigrams_b
    if not union:
        return 1.0
    return 1.0 - (len(trigrams_a & trigrams_b) / len(union))


def build_proxy_rdms(concepts: list[str]) -> dict[str, np.ndarray]:
    concept_rows = {row["concept"].lower(): row for row in load_active_concept_rows(domain="sensory")}
    subtype_values = []
    coarse_values = []
    sound_values = []
    lexical_values = []
    for idx, left in enumerate(concepts):
        for right in concepts[idx + 1 :]:
            same_subtype = concept_rows[left]["subtype"] == concept_rows[right]["subtype"]
            subtype_values.append(0.0 if same_subtype else 1.0)
            coarse_values.append(0.0 if same_subtype else 1.0)
            sound_left = concept_rows[left]["subtype"] == "sound_linked"
            sound_right = concept_rows[right]["subtype"] == "sound_linked"
            sound_values.append(0.0 if sound_left == sound_right else 1.0)
            lexical_values.append(lexical_distance(left, right))
    return {
        "subtype_membership": np.asarray(subtype_values, dtype=float),
        "coarse_category_structure": np.asarray(coarse_values, dtype=float),
        "sound_linked_vs_other": np.asarray(sound_values, dtype=float),
        "lexical_trigram_distance": np.asarray(lexical_values, dtype=float),
    }


def residualize(values: np.ndarray, controls: list[np.ndarray]) -> np.ndarray:
    target = rankdata(np.asarray(values, dtype=float))
    target = target - target.mean()
    if not controls:
        return target
    design = np.column_stack([rankdata(np.asarray(control, dtype=float)) for control in controls])
    design = design - design.mean(axis=0, keepdims=True)
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    return target - design @ beta


def residual_rsa(model_rdm: np.ndarray, behavior_rdm: np.ndarray, controls: list[np.ndarray]) -> float:
    from common import spearman_corr

    return spearman_corr(residualize(model_rdm, controls), residualize(behavior_rdm, controls))


def load_lancaster_rows(path: Path) -> list[dict[str, str]]:
    return read_csv(path)


def normalize_word(word: str) -> str:
    return re.sub(r"\s+", " ", word.strip().lower())


def load_lancaster_lookup() -> dict[str, dict[str, str]]:
    rows = load_lancaster_rows(LANCASTER_SENSORIMOTOR)
    return {normalize_word(row["Word"]): row for row in rows}


def lancaster_matrix_for_concepts(concepts: list[str], dimensions: list[str]) -> np.ndarray:
    lookup = load_lancaster_lookup()
    vectors = []
    for concept in concepts:
        row = lookup[normalize_word(concept)]
        vectors.append([float(row[dim]) for dim in dimensions])
    return np.asarray(vectors, dtype=float)


def rdm_for_feature_matrix(matrix: np.ndarray) -> np.ndarray:
    return condensed_cosine_distance(matrix)


def subset_embedding_matrix(embedding: np.ndarray, concepts: list[str], subset: list[str]) -> np.ndarray:
    index = {concept: idx for idx, concept in enumerate(concepts)}
    return np.asarray(embedding[[index[concept] for concept in subset]], dtype=float)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_project_backbone(config_path: str = "config/analysis.yaml") -> tuple[dict[str, Any], str, str, float]:
    config = load_project_config(config_path)
    return (
        config,
        config["analysis"]["execution"]["sensory_backbone_text_model"],
        config["analysis"]["execution"]["sensory_backbone_multimodal_model"],
        float(config["analysis"]["analysis"]["mid_to_late_fraction"]),
    )
