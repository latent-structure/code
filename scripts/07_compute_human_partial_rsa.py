from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations

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
    rankdata,
    read_csv,
    write_csv,
    write_json,
)
from hardening_common import load_active_concept_rows


THINGS_BEHAVIOR_MATRIX = ROOT / "data" / "anchors" / "things_behavioral_similarity.npy"
THINGS_BEHAVIOR_CONCEPTS = ROOT / "data" / "anchors" / "things_behavioral_concepts.json"
CONDITIONS = ["T_prompt_primary", "M_matched_image", "M_degraded_image"]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    left = np.asarray(x, dtype=float) - np.mean(x)
    right = np.asarray(y, dtype=float) - np.mean(y)
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    if denom == 0:
        return 0.0
    return float(np.dot(left, right) / denom)


def spearman_from_vectors(x: np.ndarray, y: np.ndarray) -> float:
    return pearson_corr(rankdata(np.asarray(x, dtype=float)), rankdata(np.asarray(y, dtype=float)))


def residualize(y: np.ndarray, control_vectors: list[np.ndarray]) -> np.ndarray:
    target = rankdata(np.asarray(y, dtype=float))
    if not control_vectors:
        return target - np.mean(target)
    design = np.column_stack([rankdata(np.asarray(vec, dtype=float)) for vec in control_vectors])
    design = np.column_stack([np.ones(len(target), dtype=float), design])
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - (design @ beta)
    return residual - np.mean(residual)


def partial_rsa(model_rdm: np.ndarray, behavior_rdm: np.ndarray, controls: list[np.ndarray]) -> float:
    model_resid = residualize(model_rdm, controls)
    behavior_resid = residualize(behavior_rdm, controls)
    return pearson_corr(model_resid, behavior_resid)


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


def coarse_category_for_subtype(subtype: str) -> str:
    if subtype in {"appearance_color", "texture_material"}:
        return "visual_surface"
    if subtype in {"sound_linked", "smell_taste_proxy"}:
        return "cross_modal"
    return "other"


def lexical_distance(a: str, b: str) -> float:
    padded_a = f"__{a.lower()}__"
    padded_b = f"__{b.lower()}__"
    trigrams_a = {padded_a[idx : idx + 3] for idx in range(len(padded_a) - 2)}
    trigrams_b = {padded_b[idx : idx + 3] for idx in range(len(padded_b) - 2)}
    union = trigrams_a | trigrams_b
    if not union:
        return 1.0
    return 1.0 - (len(trigrams_a & trigrams_b) / len(union))


def build_proxy_rdms(concepts: list[str], concept_rows: dict[str, dict[str, str]]) -> dict[str, np.ndarray]:
    subtype_values = []
    sound_values = []
    lexical_values = []
    coarse_values = []
    for left, right in combinations(concepts, 2):
        left_subtype = concept_rows[left]["subtype"]
        right_subtype = concept_rows[right]["subtype"]
        subtype_values.append(0.0 if left_subtype == right_subtype else 1.0)
        left_sound = left_subtype == "sound_linked"
        right_sound = right_subtype == "sound_linked"
        sound_values.append(0.0 if left_sound == right_sound else 1.0)
        lexical_values.append(lexical_distance(left, right))
        coarse_values.append(
            0.0
            if coarse_category_for_subtype(left_subtype) == coarse_category_for_subtype(right_subtype)
            else 1.0
        )
    return {
        "subtype_membership": np.asarray(subtype_values, dtype=float),
        "sound_linked_vs_other": np.asarray(sound_values, dtype=float),
        "lexical_trigram_distance": np.asarray(lexical_values, dtype=float),
        "coarse_category_structure": np.asarray(coarse_values, dtype=float),
    }


def write_text(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    args = parser.parse_args()

    config = load_project_config(args.config)
    backbone_text = config["analysis"]["execution"]["sensory_backbone_text_model"]
    backbone_multimodal = config["analysis"]["execution"]["sensory_backbone_multimodal_model"]
    mid_to_late_fraction = float(config["analysis"]["analysis"]["mid_to_late_fraction"])

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

    things_behavior = np.load(THINGS_BEHAVIOR_MATRIX)
    things_concepts = [concept.lower() for concept in json.loads(THINGS_BEHAVIOR_CONCEPTS.read_text(encoding="utf-8"))]
    things_index = {concept: idx for idx, concept in enumerate(things_concepts)}
    concept_rows = {row["concept"].lower(): row for row in load_active_concept_rows(args.config)}

    model_rdms: dict[str, np.ndarray] = {}
    overlap_concepts: list[str] | None = None
    for condition in CONDITIONS:
        model_id = condition_model_id(backbone_text, backbone_multimodal, condition)
        layers = selected_text_layers if condition.startswith("T_") else selected_multimodal_layers
        embeddings, concepts = mean_embedding_for_condition(metadata_lookup, pooled, model_id, condition, layers)
        matched_positions = [pos for pos, concept in enumerate(concepts) if concept in things_index]
        matched_concepts = [concepts[pos] for pos in matched_positions]
        matched_embeddings = np.asarray(embeddings[matched_positions], dtype=float)
        if overlap_concepts is None:
            overlap_concepts = matched_concepts
        elif overlap_concepts != matched_concepts:
            raise RuntimeError("Expected aligned THINGS overlap ordering across partial-RSA conditions.")
        model_rdms[condition] = condensed_cosine_distance(matched_embeddings)

    if overlap_concepts is None:
        raise RuntimeError("Failed to construct THINGS-overlap concept set for partial RSA.")
    behavior_idx = [things_index[concept] for concept in overlap_concepts]
    behavior_dist = 1.0 - things_behavior[np.ix_(behavior_idx, behavior_idx)]
    behavior_rdm = np.asarray(behavior_dist[np.triu_indices(len(overlap_concepts), k=1)], dtype=float)
    proxy_rdms = build_proxy_rdms(overlap_concepts, concept_rows)

    control_sets = {
        "none": [],
        "subtype_membership": ["subtype_membership"],
        "sound_linked_vs_other": ["sound_linked_vs_other"],
        "lexical_trigram_distance": ["lexical_trigram_distance"],
        "coarse_category_structure": ["coarse_category_structure"],
        "all_proxies_joint": [
            "subtype_membership",
            "sound_linked_vs_other",
            "lexical_trigram_distance",
            "coarse_category_structure",
        ],
    }

    partial_rows = []
    raw_scores: dict[str, float] = {}
    joint_scores: dict[str, float] = {}
    reduction_by_control: dict[str, float] = {}
    num_pairs = len(behavior_rdm)
    for condition in CONDITIONS:
        raw_score = spearman_from_vectors(model_rdms[condition], behavior_rdm)
        raw_scores[condition] = raw_score
        for control_name, control_keys in control_sets.items():
            score = raw_score if control_name == "none" else partial_rsa(
                model_rdms[condition],
                behavior_rdm,
                [proxy_rdms[key] for key in control_keys],
            )
            partial_rows.append(
                {
                    "analysis_level": "full_set",
                    "condition": condition,
                    "comparison": "vs_things_behavioral",
                    "control_name": control_name,
                    "rsa_score": score,
                    "delta_vs_uncontrolled": score - raw_score,
                    "num_concepts": len(overlap_concepts),
                    "num_pairs": num_pairs,
                }
            )
            if control_name == "all_proxies_joint":
                joint_scores[condition] = score
            if condition == "T_prompt_primary" and control_name != "none":
                reduction_by_control[control_name] = raw_score - score

    blockwise_rows = []
    concept_to_subtype = {concept: concept_rows[concept]["subtype"] for concept in overlap_concepts}
    index_pairs = list(combinations(range(len(overlap_concepts)), 2))
    pair_subtypes = [(concept_to_subtype[overlap_concepts[i]], concept_to_subtype[overlap_concepts[j]]) for i, j in index_pairs]
    subtypes = sorted({concept_to_subtype[concept] for concept in overlap_concepts})

    for subtype in subtypes:
        mask = [left == subtype and right == subtype for left, right in pair_subtypes]
        pair_count = int(sum(mask))
        if pair_count == 0:
            continue
        for condition in CONDITIONS:
            score = spearman_from_vectors(model_rdms[condition][mask], behavior_rdm[mask])
            blockwise_rows.append(
                {
                    "block_type": "within_subtype",
                    "block_name": subtype,
                    "condition": condition,
                    "rsa_score": score,
                    "num_concepts": sum(1 for concept in overlap_concepts if concept_to_subtype[concept] == subtype),
                    "num_pairs": pair_count,
                }
            )

    between_mask = [left != right for left, right in pair_subtypes]
    between_pair_count = int(sum(between_mask))
    for condition in CONDITIONS:
        blockwise_rows.append(
            {
                "block_type": "between_subtypes",
                "block_name": "all_between_subtypes",
                "condition": condition,
                "rsa_score": spearman_from_vectors(model_rdms[condition][between_mask], behavior_rdm[between_mask]),
                "num_concepts": len(overlap_concepts),
                "num_pairs": between_pair_count,
            }
        )

    for left_subtype, right_subtype in combinations(subtypes, 2):
        pair_name = f"{left_subtype}__vs__{right_subtype}"
        mask = [
            {left, right} == {left_subtype, right_subtype}
            for left, right in pair_subtypes
        ]
        pair_count = int(sum(mask))
        if pair_count == 0:
            continue
        for condition in CONDITIONS:
            blockwise_rows.append(
                {
                    "block_type": "between_subtypes",
                    "block_name": pair_name,
                    "condition": condition,
                    "rsa_score": spearman_from_vectors(model_rdms[condition][mask], behavior_rdm[mask]),
                    "num_concepts": len(overlap_concepts),
                    "num_pairs": pair_count,
                }
            )

    summary_payload = {
        "num_concepts": len(overlap_concepts),
        "num_pairs": num_pairs,
        "raw_scores": raw_scores,
        "joint_control_scores": joint_scores,
        "joint_control_prompt_minus_matched": joint_scores["T_prompt_primary"] - joint_scores["M_matched_image"],
        "raw_prompt_minus_matched": raw_scores["T_prompt_primary"] - raw_scores["M_matched_image"],
        "largest_prompt_reduction_control": max(reduction_by_control, key=reduction_by_control.get),
        "largest_prompt_reduction_value": reduction_by_control[max(reduction_by_control, key=reduction_by_control.get)],
    }

    write_csv(
        metrics_path("human_partial_rsa.csv"),
        partial_rows,
        ["analysis_level", "condition", "comparison", "control_name", "rsa_score", "delta_vs_uncontrolled", "num_concepts", "num_pairs"],
    )
    write_csv(
        metrics_path("human_blockwise_rsa.csv"),
        blockwise_rows,
        ["block_type", "block_name", "condition", "rsa_score", "num_concepts", "num_pairs"],
    )
    write_json(metrics_path("human_partial_rsa_summary.json"), summary_payload)

    output_path("outputs", "tables").mkdir(parents=True, exist_ok=True)
    write_csv(
        output_path("outputs", "tables", "human_partial_rsa_table.csv"),
        partial_rows,
        ["analysis_level", "condition", "comparison", "control_name", "rsa_score", "delta_vs_uncontrolled", "num_concepts", "num_pairs"],
    )
    write_csv(
        output_path("outputs", "tables", "human_blockwise_rsa_table.csv"),
        blockwise_rows,
        ["block_type", "block_name", "condition", "rsa_score", "num_concepts", "num_pairs"],
    )

    within_lines = [
        f"- `{row['block_name']}` prompt=`{next(r for r in blockwise_rows if r['block_type'] == 'within_subtype' and r['block_name'] == row['block_name'] and r['condition'] == 'T_prompt_primary')['rsa_score']:.4f}` matched=`{next(r for r in blockwise_rows if r['block_type'] == 'within_subtype' and r['block_name'] == row['block_name'] and r['condition'] == 'M_matched_image')['rsa_score']:.4f}`"
        for row in blockwise_rows
        if row["block_type"] == "within_subtype" and row["condition"] == "T_prompt_primary"
    ]
    report = "\n".join(
        [
            "# Human Partial RSA Report",
            "",
            "## Full-Set Residual Human Alignment",
            f"- Raw prompt RSA: `{raw_scores['T_prompt_primary']:.4f}`",
            f"- Raw matched-image RSA: `{raw_scores['M_matched_image']:.4f}`",
            f"- Raw degraded-image RSA: `{raw_scores['M_degraded_image']:.4f}`",
            f"- Joint-controlled prompt RSA: `{joint_scores['T_prompt_primary']:.4f}`",
            f"- Joint-controlled matched-image RSA: `{joint_scores['M_matched_image']:.4f}`",
            f"- Joint-controlled degraded-image RSA: `{joint_scores['M_degraded_image']:.4f}`",
            f"- Joint-controlled prompt-minus-matched gap: `{summary_payload['joint_control_prompt_minus_matched']:.4f}`",
            f"- Largest prompt reduction control: `{summary_payload['largest_prompt_reduction_control']}` delta=`{summary_payload['largest_prompt_reduction_value']:.4f}`",
            "",
            "## Blockwise Human Alignment",
            *(within_lines or ["- No within-subtype rows were available."]),
            "",
            "## Interpretation",
            "- This analysis tests whether the prompt advantage on THINGS survives removal of coarse structure and lexical structure.",
            "- Within-subtype and between-subtype blockwise scores show whether the prompt advantage is global or concentrated in coarse category structure.",
        ]
    )
    write_text(output_path("reports", "main_results", "human_partial_rsa_report.md"), report)
    append_run_log(
        "Human Partial RSA",
        [
            f"Wrote partial human-RSA metrics to {metrics_path('human_partial_rsa.csv').relative_to(ROOT)}.",
            f"Wrote blockwise human-RSA metrics to {metrics_path('human_blockwise_rsa.csv').relative_to(ROOT)}.",
            f"Wrote partial human-RSA report to {output_path('reports', 'main_results', 'human_partial_rsa_report.md').relative_to(ROOT)}.",
        ],
    )


if __name__ == "__main__":
    main()
