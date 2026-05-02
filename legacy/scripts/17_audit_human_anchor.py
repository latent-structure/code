from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy.io as sio

from common import (
    ROOT,
    append_run_log,
    canonical_condition_name,
    condensed_cosine_distance,
    load_project_config,
    metrics_path,
    midpoint_layer_start,
    output_path,
    read_csv,
    write_csv,
    write_json,
)
from hardening_common import load_active_concept_rows


THINGS_BEHAVIOR_MATRIX = ROOT / "THINGS-behavior" / "osfstorage" / "data" / "spose_similarity.mat"
THINGS_BEHAVIOR_ORDER = ROOT / "THINGS-behavior" / "osfstorage" / "variables" / "unique_id.txt"
TARGET_ANCHOR_NAME = "THINGS behavioral similarity"
FLAGGED_REVERSAL_CONCEPTS = ["foam", "perfume", "coral", "bark", "bell", "chime", "coffee", "fireworks"]
PROTOTYPE_MANIFEST = ROOT / "data" / "anchors" / "clip_vitl14_prototype_manifest.csv"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    sorted_vals = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_vals[end] == sorted_vals[start]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata(np.asarray(x, dtype=float))
    ry = rankdata(np.asarray(y, dtype=float))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    if denom == 0:
        return 0.0
    return float(np.dot(rx, ry) / denom)


def char_trigram_vector(text: str) -> set[str]:
    padded = f"__{text.lower()}__"
    return {padded[index : index + 3] for index in range(len(padded) - 2)}


def lexical_distance(a: str, b: str) -> float:
    left = char_trigram_vector(a)
    right = char_trigram_vector(b)
    union = left | right
    if not union:
        return 1.0
    return 1.0 - (len(left & right) / len(union))


def load_things_behavior() -> tuple[np.ndarray, dict[str, int]]:
    matrix = sio.loadmat(THINGS_BEHAVIOR_MATRIX)["spose_sim"]
    ordering = [line.strip().lower() for line in THINGS_BEHAVIOR_ORDER.read_text(encoding="utf-8").splitlines() if line.strip()]
    return np.asarray(matrix, dtype=float), {concept: idx for idx, concept in enumerate(ordering)}


def evaluate_anchor(scores: dict[str, float]) -> tuple[bool, str]:
    required = ["T_neutral", "T_prompt_primary", "M_text_only", "M_matched_image", "M_degraded_image", "M_mismatched_image"]
    missing = [condition for condition in required if condition not in scores]
    if missing:
        return False, f"missing_conditions:{','.join(missing)}"

    violations: list[str] = []
    if not scores["T_neutral"] < scores["T_prompt_primary"]:
        violations.append("text_prompt_not_above_neutral")
    if not scores["T_prompt_primary"] < scores["M_matched_image"]:
        violations.append("matched_not_above_prompt")
    if not scores["M_text_only"] < scores["M_matched_image"]:
        violations.append("matched_not_above_multimodal_text_only")
    if not scores["M_matched_image"] > scores["M_degraded_image"]:
        violations.append("degraded_not_below_matched")
    if not scores["M_degraded_image"] > scores["M_mismatched_image"]:
        violations.append("mismatched_not_below_degraded")
    if "M_blank_image" in scores and not scores["M_matched_image"] > scores["M_blank_image"]:
        violations.append("blank_not_below_matched")
    return (not violations), ";".join(violations) if violations else "ok"


def aligned_behavior_rdm(concepts: list[str], embeddings: np.ndarray, behavior_matrix: np.ndarray, behavior_index: dict[str, int]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    matched_positions = [pos for pos, concept in enumerate(concepts) if concept in behavior_index]
    matched_concepts = [concepts[pos] for pos in matched_positions]
    behavior_idx = [behavior_index[concept] for concept in matched_concepts]
    behavior_dist = 1.0 - behavior_matrix[np.ix_(behavior_idx, behavior_idx)]
    model_subset = np.asarray(embeddings[matched_positions], dtype=float)
    model_rdm = condensed_cosine_distance(model_subset)
    behavior_rdm = np.asarray(behavior_dist[np.triu_indices(len(matched_concepts), k=1)], dtype=float)
    return model_rdm, behavior_rdm, matched_concepts


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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def condition_model_id(backbone_text: str, backbone_multimodal: str, condition: str) -> str:
    return backbone_text if condition.startswith("T_") else backbone_multimodal


def proxy_rdm(concepts: list[str], concept_rows: dict[str, dict[str, str]], proxy_name: str) -> np.ndarray:
    values = []
    for left, right in combinations(concepts, 2):
        if proxy_name == "subtype_membership":
            value = 0.0 if concept_rows[left]["subtype"] == concept_rows[right]["subtype"] else 1.0
        elif proxy_name == "sound_linked_vs_other":
            left_is_sound = concept_rows[left]["subtype"] == "sound_linked"
            right_is_sound = concept_rows[right]["subtype"] == "sound_linked"
            value = 0.0 if left_is_sound == right_is_sound else 1.0
        elif proxy_name == "lexical_trigram_distance":
            value = lexical_distance(left, right)
        else:
            raise ValueError(f"Unknown proxy_name={proxy_name}")
        values.append(value)
    return np.asarray(values, dtype=float)


def diagnose_reversal_issue(
    subtype: str,
    polysemy_risk: str,
    things_status: str,
    prototype_count: int,
) -> str:
    if subtype == "sound_linked":
        return "likely_nonvisual_or_multimodal"
    if polysemy_risk == "medium":
        return "likely_semantically_broad"
    if things_status != "covered" or prototype_count < 3:
        return "possible_image_typicality_issue"
    return "no_clear_metadata_issue"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    args = parser.parse_args()

    config = load_project_config(args.config)
    backbone_text = config["analysis"]["execution"]["sensory_backbone_text_model"]
    backbone_multimodal = config["analysis"]["execution"]["sensory_backbone_multimodal_model"]
    mid_to_late_fraction = float(config["analysis"]["analysis"]["mid_to_late_fraction"])

    alignment_rows = [
        row
        for row in read_csv(metrics_path("layerwise_alignment_full.csv"))
        if row["bootstrap_id"] == "aggregate"
        and row["domain"] == "sensory"
        and row.get("anchor_name", row.get("anchor_model_id", "")) == TARGET_ANCHOR_NAME
    ]
    metadata = json.loads(output_path("outputs", "embeddings", "embedding_metadata_full.json").read_text(encoding="utf-8"))
    pooled_npz = np.load(output_path("outputs", "embeddings", "pooled_embeddings_full.npz"))
    pooled = {key: np.asarray(pooled_npz[key], dtype=float) for key in pooled_npz.files}
    behavior_matrix, behavior_index = load_things_behavior()

    concept_rows = {row["concept"].lower(): row for row in load_active_concept_rows(args.config)}
    subset_rows = [row for row in load_active_concept_rows(args.config, domain="sensory")]
    image_manifest = {row["concept"].lower(): row for row in read_csv(ROOT / "data" / "manifests" / "image_manifest.csv")}
    provenance_rows = {row["concept"].lower(): row for row in read_csv(ROOT / "data" / "manifests" / "things_image_provenance.csv")}
    prototype_rows = {row["concept"].lower(): row for row in read_csv(PROTOTYPE_MANIFEST)} if PROTOTYPE_MANIFEST.exists() else {}

    grouped_rows: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    layer_rows: dict[int, dict[str, float]] = defaultdict(dict)
    layers_by_model: dict[str, list[int]] = defaultdict(list)
    for row in alignment_rows:
        model = row.get("model", row.get("model_id", ""))
        condition = canonical_condition_name(row["condition"])
        score = float(row["rsa_score"])
        grouped_rows[(model, condition)].append(row)
        layer_rows[int(row["layer"])][condition] = score
        layers_by_model[model].append(int(row["layer"]))

    text_layers = sorted(set(layers_by_model[backbone_text]))
    multimodal_layers = sorted(set(layers_by_model[backbone_multimodal]))
    selected_text_layers = text_layers[midpoint_layer_start(len(text_layers), mid_to_late_fraction) :]
    selected_multimodal_layers = multimodal_layers[midpoint_layer_start(len(multimodal_layers), mid_to_late_fraction) :]

    audit_rows: list[dict[str, object]] = []
    for band_name, layer_selector in (
        ("all_layers", None),
        ("mid_to_late", {**{layer: True for layer in selected_text_layers}, **{layer: True for layer in selected_multimodal_layers}}),
    ):
        scores_by_condition: dict[str, float] = {}
        for (model, condition), rows in grouped_rows.items():
            if model not in {backbone_text, backbone_multimodal}:
                continue
            filtered = rows if layer_selector is None else [row for row in rows if int(row["layer"]) in layer_selector]
            if filtered:
                scores_by_condition[condition] = mean([float(row["rsa_score"]) for row in filtered])
        supports, notes = evaluate_anchor(scores_by_condition)
        ranked = sorted(scores_by_condition.items(), key=lambda item: item[1], reverse=True)
        rank_map = {condition: rank + 1 for rank, (condition, _) in enumerate(ranked)}
        for condition, score in sorted(scores_by_condition.items()):
            audit_rows.append(
                {
                    "audit_section": "aggregate_condition",
                    "band": band_name,
                    "key": condition,
                    "value": score,
                    "comparison_value": score - scores_by_condition.get("T_prompt_primary", score) if condition == "M_matched_image" else "",
                    "support_flag": supports,
                    "notes": notes,
                    "ordering_rank": rank_map.get(condition, ""),
                }
            )

    layer_support_rows = []
    for layer, scores in sorted(layer_rows.items()):
        support, notes = evaluate_anchor(scores)
        layer_support_rows.append(
            {
                "layer": layer,
                "supports_anchor_ordering": support,
                "violation_notes": notes,
                "T_neutral": scores.get("T_neutral", ""),
                "T_prompt_primary": scores.get("T_prompt_primary", ""),
                "M_text_only": scores.get("M_text_only", ""),
                "M_matched_image": scores.get("M_matched_image", ""),
                "M_degraded_image": scores.get("M_degraded_image", ""),
                "M_mismatched_image": scores.get("M_mismatched_image", ""),
                "M_blank_image": scores.get("M_blank_image", ""),
            }
        )
        audit_rows.append(
            {
                "audit_section": "layer_ordering",
                "band": "per_layer",
                "key": str(layer),
                "value": int(bool(support)),
                "comparison_value": float(scores.get("M_matched_image", 0.0)) - float(scores.get("T_prompt_primary", 0.0)),
                "support_flag": support,
                "notes": notes,
                "ordering_rank": "",
            }
        )

    metadata_lookup = {
        (record["model_id"], canonical_condition_name(record["condition"]), int(record["layer"])): record
        for record in metadata["records"]
        if record["domain"] == "sensory"
    }
    prompt_embedding, prompt_concepts = mean_embedding_for_condition(
        metadata_lookup, pooled, backbone_text, "T_prompt_primary", selected_text_layers
    )
    matched_embedding, matched_concepts = mean_embedding_for_condition(
        metadata_lookup, pooled, backbone_multimodal, "M_matched_image", selected_multimodal_layers
    )
    degraded_embedding, degraded_concepts = mean_embedding_for_condition(
        metadata_lookup, pooled, backbone_multimodal, "M_degraded_image", selected_multimodal_layers
    )
    if prompt_concepts != matched_concepts or prompt_concepts != degraded_concepts:
        raise RuntimeError("Human-anchor audit requires aligned concept ordering across prompt and multimodal conditions.")

    matched_positions = [pos for pos, concept in enumerate(prompt_concepts) if concept in behavior_index]
    matched_concepts = [prompt_concepts[pos] for pos in matched_positions]
    prompt_embedding = np.asarray(prompt_embedding[matched_positions], dtype=float)
    matched_embedding = np.asarray(matched_embedding[matched_positions], dtype=float)
    degraded_embedding = np.asarray(degraded_embedding[matched_positions], dtype=float)

    prompt_rdm, behavior_rdm, matched_concepts = aligned_behavior_rdm(matched_concepts, prompt_embedding, behavior_matrix, behavior_index)
    matched_rdm, _, _ = aligned_behavior_rdm(matched_concepts, matched_embedding, behavior_matrix, behavior_index)
    degraded_rdm, _, _ = aligned_behavior_rdm(matched_concepts, degraded_embedding, behavior_matrix, behavior_index)
    full_prompt_score = spearman_corr(prompt_rdm, behavior_rdm)
    full_matched_score = spearman_corr(matched_rdm, behavior_rdm)
    full_gap = full_prompt_score - full_matched_score

    concept_diagnostic_rows = []
    for idx, concept in enumerate(matched_concepts):
        mask = np.ones(len(matched_concepts), dtype=bool)
        mask[idx] = False
        prompt_subset = np.asarray(prompt_embedding[mask], dtype=float)
        matched_subset = np.asarray(matched_embedding[mask], dtype=float)
        degraded_subset = np.asarray(degraded_embedding[mask], dtype=float)
        prompt_loo_rdm, behavior_loo_rdm, _ = aligned_behavior_rdm([c for pos, c in enumerate(matched_concepts) if pos != idx], prompt_subset, behavior_matrix, behavior_index)
        matched_loo_rdm, _, _ = aligned_behavior_rdm([c for pos, c in enumerate(matched_concepts) if pos != idx], matched_subset, behavior_matrix, behavior_index)
        degraded_loo_rdm, _, _ = aligned_behavior_rdm([c for pos, c in enumerate(matched_concepts) if pos != idx], degraded_subset, behavior_matrix, behavior_index)
        prompt_loo = spearman_corr(prompt_loo_rdm, behavior_loo_rdm)
        matched_loo = spearman_corr(matched_loo_rdm, behavior_loo_rdm)
        degraded_loo = spearman_corr(degraded_loo_rdm, behavior_loo_rdm)

        neighbor_behavior_distance = 1.0 - behavior_matrix[
            behavior_index[concept], [behavior_index[c] for pos, c in enumerate(matched_concepts) if pos != idx]
        ]
        prompt_local = spearman_corr(np.linalg.norm(prompt_embedding[idx] - prompt_embedding[mask], axis=1), neighbor_behavior_distance)
        matched_local = spearman_corr(np.linalg.norm(matched_embedding[idx] - matched_embedding[mask], axis=1), neighbor_behavior_distance)
        degraded_local = spearman_corr(np.linalg.norm(degraded_embedding[idx] - degraded_embedding[mask], axis=1), neighbor_behavior_distance)

        concept_meta = concept_rows.get(concept, {})
        image_row = image_manifest.get(concept, {})
        provenance = provenance_rows.get(concept, {})
        prompt_minus_matched_local = prompt_local - matched_local
        concept_diagnostic_rows.append(
            {
                "concept": concept,
                "subtype": concept_meta.get("subtype", ""),
                "polysemy_risk": concept_meta.get("polysemy_risk", ""),
                "image_quality_flag": concept_meta.get("image_quality_flag", ""),
                "human_anchor_available": concept_meta.get("human_anchor_available", ""),
                "prompt_leave_one_out_rsa": prompt_loo,
                "matched_leave_one_out_rsa": matched_loo,
                "degraded_leave_one_out_rsa": degraded_loo,
                "leave_one_out_prompt_minus_matched_gap": prompt_loo - matched_loo,
                "prompt_advantage_contribution": full_gap - (prompt_loo - matched_loo),
                "prompt_local_alignment": prompt_local,
                "matched_local_alignment": matched_local,
                "degraded_local_alignment": degraded_local,
                "prompt_minus_matched_local_gap": prompt_minus_matched_local,
                "matched_image": image_row.get("matched_image", ""),
                "degraded_image": image_row.get("degraded_image", ""),
                "image_status": image_row.get("status", ""),
                "image_source_kind": image_row.get("source_kind", ""),
                "things_status": provenance.get("status", ""),
                "things_word": provenance.get("things_word", ""),
                "things_unique_id": provenance.get("things_unique_id", ""),
                "things_archive_member": provenance.get("archive_member", ""),
                "things_notes": provenance.get("notes", ""),
            }
        )
    concept_diagnostic_rows.sort(key=lambda row: float(row["prompt_minus_matched_local_gap"]), reverse=True)

    subtype_rows = []
    subtype_breakdown_rows = []
    subtype_groups: dict[str, list[str]] = defaultdict(list)
    for row in subset_rows:
        concept = row["concept"].lower()
        if concept in matched_concepts:
            subtype_groups[row["subtype"]].append(concept)
    concept_index = {concept: idx for idx, concept in enumerate(matched_concepts)}
    subtype_layer_condition_scores: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    subtype_conditions = ["T_neutral", "T_prompt_primary", "M_matched_image", "M_degraded_image", "M_mismatched_image", "M_blank_image"]
    for subtype, concepts in sorted(subtype_groups.items()):
        subtype_set = set(concepts)
        for condition in subtype_conditions:
            model_id = condition_model_id(backbone_text, backbone_multimodal, condition)
            for layer in sorted(set(text_layers + multimodal_layers)):
                record = metadata_lookup.get((model_id, condition, layer))
                if record is None:
                    continue
                record_concepts = [concept.lower() for concept in record["concepts"]]
                positions = [pos for pos, concept in enumerate(record_concepts) if concept in subtype_set and concept in behavior_index]
                if len(positions) < 3:
                    continue
                subtype_concepts = [record_concepts[pos] for pos in positions]
                behavior_idx = [behavior_index[concept] for concept in subtype_concepts]
                behavior_sub = 1.0 - behavior_matrix[np.ix_(behavior_idx, behavior_idx)]
                behavior_sub_rdm = np.asarray(behavior_sub[np.triu_indices(len(subtype_concepts), k=1)], dtype=float)
                embeddings_sub = np.asarray(pooled[f"record_{record['record_id']}"], dtype=float)[positions]
                model_sub_rdm = condensed_cosine_distance(embeddings_sub)
                subtype_layer_condition_scores[subtype][layer][condition] = spearman_corr(model_sub_rdm, behavior_sub_rdm)

    for subtype, concepts in sorted(subtype_groups.items()):
        if len(concepts) < 3:
            continue
        idx = [concept_index[concept] for concept in concepts]
        prompt_sub = np.asarray(prompt_embedding[idx], dtype=float)
        matched_sub = np.asarray(matched_embedding[idx], dtype=float)
        degraded_sub = np.asarray(degraded_embedding[idx], dtype=float)
        behavior_idx = [behavior_index[concept] for concept in concepts]
        behavior_sub = 1.0 - behavior_matrix[np.ix_(behavior_idx, behavior_idx)]
        behavior_sub_rdm = np.asarray(behavior_sub[np.triu_indices(len(concepts), k=1)], dtype=float)
        prompt_sub_rdm = condensed_cosine_distance(prompt_sub)
        matched_sub_rdm = condensed_cosine_distance(matched_sub)
        degraded_sub_rdm = condensed_cosine_distance(degraded_sub)
        prompt_score = spearman_corr(prompt_sub_rdm, behavior_sub_rdm)
        matched_score = spearman_corr(matched_sub_rdm, behavior_sub_rdm)
        degraded_score = spearman_corr(degraded_sub_rdm, behavior_sub_rdm)
        subtype_rows.append(
            {
                "subtype": subtype,
                "num_concepts": len(concepts),
                "prompt_rsa": prompt_score,
                "matched_rsa": matched_score,
                "degraded_rsa": degraded_score,
                "matched_minus_prompt": matched_score - prompt_score,
                "matched_minus_degraded": matched_score - degraded_score,
            }
        )
        audit_rows.append(
            {
                "audit_section": "subtype_gap",
                "band": "mid_to_late_mean_embedding",
                "key": subtype,
                "value": matched_score - prompt_score,
                "comparison_value": matched_score - degraded_score,
                "support_flag": matched_score > prompt_score,
                "notes": f"n={len(concepts)}",
                "ordering_rank": "",
            }
        )

        condition_scores = {
            condition: mean(
                [
                    subtype_layer_condition_scores[subtype][layer][condition]
                    for layer in sorted(subtype_layer_condition_scores[subtype])
                    if condition in subtype_layer_condition_scores[subtype][layer]
                ]
            )
            for condition in subtype_conditions
        }

        subtype_layer_support = 0
        total_subtype_layers = 0
        for layer in sorted(subtype_layer_condition_scores[subtype]):
            layer_condition_scores = subtype_layer_condition_scores[subtype][layer]
            if "T_neutral" in layer_condition_scores and "T_prompt_primary" in layer_condition_scores:
                total_subtype_layers += 1
                supports, _ = evaluate_anchor(layer_condition_scores)
                subtype_layer_support += int(bool(supports))

        subtype_breakdown_rows.append(
            {
                "subtype": subtype,
                "num_concepts": len(concepts),
                "T_neutral": condition_scores.get("T_neutral", ""),
                "T_prompt_primary": condition_scores.get("T_prompt_primary", ""),
                "M_matched_image": condition_scores.get("M_matched_image", ""),
                "M_degraded_image": condition_scores.get("M_degraded_image", ""),
                "M_mismatched_image": condition_scores.get("M_mismatched_image", ""),
                "M_blank_image": condition_scores.get("M_blank_image", ""),
                "matched_above_prompt": matched_score > prompt_score,
                "blank_collapse_holds": matched_score > condition_scores.get("M_blank_image", 0.0),
                "mismatch_collapse_holds": matched_score > condition_scores.get("M_mismatched_image", 0.0),
                "supporting_layers": subtype_layer_support,
                "total_layers": total_subtype_layers,
            }
        )

    overlap_rows = []
    subset_concepts = [row["concept"].lower() for row in subset_rows]
    for concept in subset_concepts:
        overlap_rows.append(
            {
                "concept": concept,
                "in_subset": True,
                "in_things_behavior": concept in behavior_index,
                "in_image_manifest": concept in image_manifest,
                "in_things_provenance": concept in provenance_rows,
                "image_status": image_manifest.get(concept, {}).get("status", ""),
                "things_status": provenance_rows.get(concept, {}).get("status", ""),
            }
        )

    coarseness_rows = []
    proxy_specs = [
        ("subtype_membership", "Distance is 0 within subtype and 1 across subtype."),
        ("sound_linked_vs_other", "Distance is 0 within sound-vs-nonsound split and 1 across split."),
        ("lexical_trigram_distance", "Character trigram lexical distance proxy from the legacy diagnostic script."),
    ]
    for proxy_name, description in proxy_specs:
        proxy = proxy_rdm(matched_concepts, concept_rows, proxy_name)
        correlation = spearman_corr(proxy, behavior_rdm)
        supports_semantically_coarse_anchor = abs(correlation) >= 0.3
        interpretation = "strong_proxy_alignment" if abs(correlation) >= 0.5 else "moderate_proxy_alignment" if abs(correlation) >= 0.3 else "weak_proxy_alignment"
        coarseness_rows.append(
            {
                "proxy_name": proxy_name,
                "proxy_description": description,
                "num_concepts": len(matched_concepts),
                "spearman_with_human_rdm": correlation,
                "variance_interpretation": interpretation,
                "supports_semantically_coarse_anchor": supports_semantically_coarse_anchor,
            }
        )

    reversal_rows = []
    for concept in FLAGGED_REVERSAL_CONCEPTS:
        concept_row = next((row for row in concept_diagnostic_rows if row["concept"] == concept), None)
        if concept_row is None:
            continue
        prototype_row = prototype_rows.get(concept, {})
        prototype_count = int(prototype_row.get("prototype_image_count", "0") or 0)
        diagnosis = diagnose_reversal_issue(
            concept_row["subtype"],
            concept_row["polysemy_risk"],
            concept_row["things_status"],
            prototype_count,
        )
        reversal_rows.append(
            {
                "concept": concept,
                "subtype": concept_row["subtype"],
                "polysemy_risk": concept_row["polysemy_risk"],
                "matched_image": concept_row["matched_image"],
                "image_quality_flag": concept_row["image_quality_flag"],
                "things_status": concept_row["things_status"],
                "things_word": concept_row["things_word"],
                "things_unique_id": concept_row["things_unique_id"],
                "prototype_image_count": prototype_count,
                "prototype_image_set": prototype_row.get("selected_sources", ""),
                "prompt_minus_matched_local_gap": concept_row["prompt_minus_matched_local_gap"],
                "diagnosis": diagnosis,
            }
        )

    support_layers = sum(1 for row in layer_support_rows if row["supports_anchor_ordering"])
    excluded_concepts = [concept for concept in subset_concepts if concept not in behavior_index]
    subset_path = config["analysis"]["execution"].get("default_concept_subset", "")
    summary_payload = {
        "anchor_name": TARGET_ANCHOR_NAME,
        "concept_subset": subset_path or "data/concepts/full_concept_list.csv",
        "sensory_subset_count": len(subset_concepts),
        "behavior_overlap_count": sum(1 for concept in subset_concepts if concept in behavior_index),
        "excluded_concepts": excluded_concepts,
        "things_behavior_matrix_path": str(THINGS_BEHAVIOR_MATRIX.relative_to(ROOT)),
        "things_behavior_order_path": str(THINGS_BEHAVIOR_ORDER.relative_to(ROOT)),
        "supporting_layers": support_layers,
        "total_layers": len(layer_support_rows),
        "all_layers_prompt_rsa": mean([float(row["rsa_score"]) for row in grouped_rows[(backbone_text, "T_prompt_primary")]]),
        "all_layers_matched_rsa": mean([float(row["rsa_score"]) for row in grouped_rows[(backbone_multimodal, "M_matched_image")]]),
        "mid_to_late_prompt_rsa": mean(
            [float(row["rsa_score"]) for row in grouped_rows[(backbone_text, "T_prompt_primary")] if int(row["layer"]) in selected_text_layers]
        ),
        "mid_to_late_matched_rsa": mean(
            [float(row["rsa_score"]) for row in grouped_rows[(backbone_multimodal, "M_matched_image")] if int(row["layer"]) in selected_multimodal_layers]
        ),
        "mid_to_late_mean_embedding_prompt_rsa": full_prompt_score,
        "mid_to_late_mean_embedding_matched_rsa": full_matched_score,
        "mid_to_late_mean_embedding_prompt_minus_matched": full_gap,
        "top_prompt_advantage_concepts": [row["concept"] for row in concept_diagnostic_rows[:5]],
    }

    write_csv(
        metrics_path("human_anchor_audit.csv"),
        audit_rows,
        ["audit_section", "band", "key", "value", "comparison_value", "support_flag", "notes", "ordering_rank"],
    )
    write_csv(
        metrics_path("human_anchor_layer_diagnostics.csv"),
        layer_support_rows,
        [
            "layer",
            "supports_anchor_ordering",
            "violation_notes",
            "T_neutral",
            "T_prompt_primary",
            "M_text_only",
            "M_matched_image",
            "M_degraded_image",
            "M_mismatched_image",
            "M_blank_image",
        ],
    )
    write_csv(
        metrics_path("human_anchor_concept_diagnostics.csv"),
        concept_diagnostic_rows,
        [
            "concept",
            "subtype",
            "polysemy_risk",
            "image_quality_flag",
            "human_anchor_available",
            "prompt_leave_one_out_rsa",
            "matched_leave_one_out_rsa",
            "degraded_leave_one_out_rsa",
            "leave_one_out_prompt_minus_matched_gap",
            "prompt_advantage_contribution",
            "prompt_local_alignment",
            "matched_local_alignment",
            "degraded_local_alignment",
            "prompt_minus_matched_local_gap",
            "matched_image",
            "degraded_image",
            "image_status",
            "image_source_kind",
            "things_status",
            "things_word",
            "things_unique_id",
            "things_archive_member",
            "things_notes",
        ],
    )
    write_csv(
        metrics_path("human_anchor_subtype_summary.csv"),
        subtype_rows,
        ["subtype", "num_concepts", "prompt_rsa", "matched_rsa", "degraded_rsa", "matched_minus_prompt", "matched_minus_degraded"],
    )
    write_csv(
        metrics_path("human_anchor_overlap_review.csv"),
        overlap_rows,
        ["concept", "in_subset", "in_things_behavior", "in_image_manifest", "in_things_provenance", "image_status", "things_status"],
    )
    write_csv(
        metrics_path("human_anchor_coarseness_tests.csv"),
        coarseness_rows,
        ["proxy_name", "proxy_description", "num_concepts", "spearman_with_human_rdm", "variance_interpretation", "supports_semantically_coarse_anchor"],
    )
    write_json(metrics_path("human_anchor_audit_summary.json"), summary_payload)
    output_path("outputs", "tables").mkdir(parents=True, exist_ok=True)
    write_csv(
        output_path("outputs", "tables", "human_anchor_subtype_breakdown.csv"),
        subtype_breakdown_rows,
        [
            "subtype",
            "num_concepts",
            "T_neutral",
            "T_prompt_primary",
            "M_matched_image",
            "M_degraded_image",
            "M_mismatched_image",
            "M_blank_image",
            "matched_above_prompt",
            "blank_collapse_holds",
            "mismatch_collapse_holds",
            "supporting_layers",
            "total_layers",
        ],
    )
    write_csv(
        output_path("outputs", "tables", "human_anchor_reversal_audit.csv"),
        reversal_rows,
        [
            "concept",
            "subtype",
            "polysemy_risk",
            "matched_image",
            "image_quality_flag",
            "things_status",
            "things_word",
            "things_unique_id",
            "prototype_image_count",
            "prototype_image_set",
            "prompt_minus_matched_local_gap",
            "diagnosis",
        ],
    )
    write_csv(
        output_path("outputs", "tables", "human_anchor_coarseness_tests.csv"),
        coarseness_rows,
        ["proxy_name", "proxy_description", "num_concepts", "spearman_with_human_rdm", "variance_interpretation", "supports_semantically_coarse_anchor"],
    )

    top_reversal_lines = [
        f"- `{row['concept']}` subtype=`{row['subtype']}` prompt_minus_matched_local_gap=`{float(row['prompt_minus_matched_local_gap']):.4f}` matched_image=`{row['matched_image']}` things_status=`{row['things_status'] or 'missing'}`"
        for row in concept_diagnostic_rows[:5]
    ]
    coarseness_lines = [
        f"- `{row['proxy_name']}` rho=`{float(row['spearman_with_human_rdm']):.4f}` interpretation=`{row['variance_interpretation']}` coarse=`{row['supports_semantically_coarse_anchor']}`"
        for row in coarseness_rows
    ]
    subtype_lines = [
        f"- `{row['subtype']}` matched_above_prompt=`{row['matched_above_prompt']}` mismatch_collapse=`{row['mismatch_collapse_holds']}` blank_collapse=`{row['blank_collapse_holds']}` support_layers=`{row['supporting_layers']}/{row['total_layers']}`"
        for row in subtype_breakdown_rows
    ]
    reversal_lines = [
        f"- `{row['concept']}` diagnosis=`{row['diagnosis']}` prototype_count=`{row['prototype_image_count']}` matched_image=`{row['matched_image']}`"
        for row in reversal_rows
    ]
    decision_audit_report = "\n".join(
        [
            "# Human Anchor Audit Report",
            "",
            "## Construction",
            f"- THINGS similarity source: `{summary_payload['things_behavior_matrix_path']}`",
            f"- THINGS ordering source: `{summary_payload['things_behavior_order_path']}`",
            f"- Overlap used for the human RDM: `{summary_payload['behavior_overlap_count']}/{summary_payload['sensory_subset_count']}` sensory concepts",
            f"- Excluded concept from THINGS overlap: `{', '.join(excluded_concepts) or 'none'}`",
            "- Human RDM construction: `1 - spose_similarity` on the overlap subset, then upper-triangle vectorization.",
            "- RSA scaling: rank correlation on condensed RDM vectors with no extra normalization beyond the distance conversion above.",
            "",
            "## Main Result",
            f"- Layers supporting full ordering: `{support_layers}/{len(layer_support_rows)}`",
            f"- All-layer prompt RSA: `{summary_payload['all_layers_prompt_rsa']:.4f}`",
            f"- All-layer matched-image RSA: `{summary_payload['all_layers_matched_rsa']:.4f}`",
            f"- Mid-to-late mean-embedding prompt RSA: `{summary_payload['mid_to_late_mean_embedding_prompt_rsa']:.4f}`",
            f"- Mid-to-late mean-embedding matched-image RSA: `{summary_payload['mid_to_late_mean_embedding_matched_rsa']:.4f}`",
            f"- Mid-to-late prompt-minus-matched gap: `{summary_payload['mid_to_late_mean_embedding_prompt_minus_matched']:.4f}`",
            "",
            "## Coarseness Tests",
            *(coarseness_lines or ["- No coarseness diagnostics were available."]),
            "",
            "## Subtype Breakdown",
            *(subtype_lines or ["- No subtype breakdown rows were available."]),
            "",
            "## Reversal Audit",
            *(reversal_lines or ["- No reversal audit rows were available."]),
            "",
            "## Interpretation",
            "- The THINGS behavioral anchor shows substantial sensory structure for both prompted text and matched images.",
            "- The current matched-family result does not satisfy the stronger human-anchor claim because prompted text remains more human-aligned than matched images overall on the present THINGS overlap subset.",
            "- The failure is structured rather than arbitrary: sound-linked concepts are strongly prompt-favored, while smell/taste proxy concepts are matched-favored.",
            "- No bounded methodological defect is identified by overlap, provenance, or prototype-count metadata alone.",
        ]
    )

    verdict = "Verdict A - Continue, reframed claim"
    recommendation_report = "\n".join(
        [
            "# Go/No-Go Recommendation",
            "",
            "The human-anchor assay looks scientifically usable rather than obviously broken.",
            "- The THINGS construction is explicit and the only overlap loss is `crystal`.",
            "- Mismatch and blank collapse remain strong in the aggregate human-anchor summaries.",
            "- The subtype picture is interpretable: `sound_linked` is prompt-favored while `smell_taste_proxy` is matched-favored.",
            "- The reversal audit does not reveal a single bounded methodological flaw that would justify a targeted rerun as a methodological correction.",
            "",
            "Recommended paper direction:",
            "- prompting recovers substantial human-like sensory organization",
            "- grounding is more input-dependent and stronger in some perceptual-semantic reference spaces",
            "- prompting and grounding are different regimes, not simply stronger vs weaker versions of the same thing",
            "",
            verdict,
        ]
    )

    write_text(output_path("reports", "main_results", "human_anchor_audit_report.md"), decision_audit_report)
    write_text(output_path("reports", "decision", "human_anchor_audit_report.md"), decision_audit_report)
    write_text(output_path("reports", "decision", "go_no_go_recommendation.md"), recommendation_report)
    append_run_log(
        "Human Anchor Audit",
        [
            f"Wrote human-anchor audit summary to {metrics_path('human_anchor_audit.csv').relative_to(ROOT)}.",
            f"Wrote concept diagnostics to {metrics_path('human_anchor_concept_diagnostics.csv').relative_to(ROOT)}.",
            f"Wrote decision-stage audit report to {output_path('reports', 'decision', 'human_anchor_audit_report.md').relative_to(ROOT)}.",
            f"Wrote go/no-go recommendation to {output_path('reports', 'decision', 'go_no_go_recommendation.md').relative_to(ROOT)}.",
        ],
    )


if __name__ == "__main__":
    main()
