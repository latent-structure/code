from __future__ import annotations

import argparse
import json
from itertools import combinations
from typing import Any

import numpy as np

from analysis_common import load_hierarchy_mapping
from common import (
    ROOT,
    append_run_log,
    condensed_cosine_distance,
    embeddings_path,
    load_project_config,
    metrics_path,
    output_path,
    rankdata,
    read_csv,
    spearman_corr,
    write_csv,
    write_json,
)
from hardening_common import (
    THINGS_BEHAVIOR_CONCEPTS,
    THINGS_BEHAVIOR_MATRIX,
    canonical_condition_name,
    condition_model_id,
    load_active_concept_rows,
    load_project_backbone,
    selected_layers,
    write_text,
)


CONDITIONS = [
    "T_prompt_primary",
    "M_text_only",
    "M_matched_image",
    "M_prompt_plus_matched_image",
    "M_mismatched_image",
    "M_blank_image",
]
VISUAL_ANCHORS = ["SigLIP2", "CLIP ViT-L/14", "DINOv2"]
PROXY_CONTROLS = ["subtype_membership", "coarse_category_structure", "lexical_trigram_distance"]


def suffix(limit: int) -> str:
    return "_smoke" if limit else ""


def build_metadata_lookup(metadata: dict[str, Any]) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[str, list[int]]]:
    lookup = {
        (record["model_id"], canonical_condition_name(record["condition"]), int(record["layer"])): record
        for record in metadata["records"]
        if record["domain"] == "sensory"
    }
    layers_by_model: dict[str, list[int]] = {}
    for record in metadata["records"]:
        if record["domain"] != "sensory":
            continue
        layers_by_model.setdefault(record["model_id"], []).append(int(record["layer"]))
    for model_id in list(layers_by_model):
        layers_by_model[model_id] = sorted(set(layers_by_model[model_id]))
    return lookup, layers_by_model


def mean_embedding_for_condition(
    arrays: Any,
    lookup: dict[tuple[str, str, int], dict[str, Any]],
    model_id: str,
    condition: str,
    layers: list[int],
) -> tuple[np.ndarray, list[str]]:
    matrices = []
    concepts: list[str] | None = None
    for layer in layers:
        record = lookup.get((model_id, condition, int(layer)))
        if record is None:
            continue
        if concepts is None:
            concepts = [concept.lower() for concept in record["concepts"]]
        matrices.append(np.asarray(arrays[f"record_{record['record_id']}"], dtype=np.float32))
    if not matrices or concepts is None:
        raise RuntimeError(f"Missing embeddings for {model_id} {condition}")
    return np.mean(np.stack(matrices), axis=0, dtype=np.float32), concepts


def ordered_matrix(matrix: np.ndarray, concepts: list[str], target_concepts: list[str]) -> np.ndarray:
    index = {concept.lower(): idx for idx, concept in enumerate(concepts)}
    missing = [concept for concept in target_concepts if concept not in index]
    if missing:
        raise RuntimeError(f"Missing concepts from embedding matrix: {', '.join(missing[:20])}")
    return np.asarray(matrix[[index[concept] for concept in target_concepts]], dtype=np.float32)


def load_static_anchor(name: str, target_concepts: list[str]) -> np.ndarray:
    mapping = {
        "CLIP ViT-L/14": ("clip_vitl14_embeddings.npy", "clip_vitl14_concepts.json"),
        "DINOv2": ("dinov2_embeddings.npy", "dinov2_concepts.json"),
    }
    emb_name, concepts_name = mapping[name]
    matrix = np.load(ROOT / "data" / "anchors" / emb_name)
    concepts = [concept.lower() for concept in json.loads((ROOT / "data" / "anchors" / concepts_name).read_text(encoding="utf-8"))]
    return ordered_matrix(matrix, concepts, target_concepts)


def load_siglip_anchor(
    arrays: Any,
    metadata: dict[str, Any],
    lookup: dict[tuple[str, str, int], dict[str, Any]],
    layers_by_model: dict[str, list[int]],
    target_concepts: list[str],
) -> np.ndarray:
    siglip_model_ids = sorted(
        {
            record["model_id"]
            for record in metadata["records"]
            if record["domain"] == "sensory" and record["family"] == "anchor" and "siglip" in record["model_id"].lower()
        }
    )
    if not siglip_model_ids:
        raise RuntimeError("Could not locate SigLIP2 anchor rows in embedding metadata.")
    matrix, concepts = mean_embedding_for_condition(
        arrays,
        lookup,
        siglip_model_ids[0],
        "reference_anchor_image",
        layers_by_model[siglip_model_ids[0]],
    )
    return ordered_matrix(matrix, concepts, target_concepts)


def build_proxy_rdms(target_concepts: list[str], config_path: str) -> dict[str, np.ndarray]:
    active_rows = {row["concept"].lower(): row for row in load_active_concept_rows(config_path, domain="sensory")}
    hierarchy_lookup, _ = load_hierarchy_mapping(config_path)
    subtype_values = []
    coarse_values = []
    lexical_values = []
    for left, right in combinations(target_concepts, 2):
        subtype_values.append(0.0 if active_rows[left]["subtype"] == active_rows[right]["subtype"] else 1.0)
        coarse_values.append(
            0.0
            if hierarchy_lookup[left]["coarse_category"] == hierarchy_lookup[right]["coarse_category"]
            else 1.0
        )
        lexical_values.append(lexical_distance(left, right))
    return {
        "subtype_membership": np.asarray(subtype_values, dtype=float),
        "coarse_category_structure": np.asarray(coarse_values, dtype=float),
        "lexical_trigram_distance": np.asarray(lexical_values, dtype=float),
    }


def lexical_distance(left: str, right: str) -> float:
    padded_left = f"__{left.lower()}__"
    padded_right = f"__{right.lower()}__"
    left_trigrams = {padded_left[idx : idx + 3] for idx in range(len(padded_left) - 2)}
    right_trigrams = {padded_right[idx : idx + 3] for idx in range(len(padded_right) - 2)}
    union = left_trigrams | right_trigrams
    if not union:
        return 1.0
    return 1.0 - (len(left_trigrams & right_trigrams) / len(union))


def residualize(values: np.ndarray, controls: list[np.ndarray]) -> np.ndarray:
    target = rankdata(np.asarray(values, dtype=float))
    target = target - target.mean()
    if not controls:
        return target
    columns = [rankdata(np.asarray(control, dtype=float)) for control in controls]
    design = np.column_stack(columns)
    design = design - design.mean(axis=0, keepdims=True)
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ beta
    return residual - residual.mean()


def standardized_regression(target: np.ndarray, predictors: list[np.ndarray]) -> tuple[list[float], float, float]:
    y = rankdata(np.asarray(target, dtype=float))
    y = (y - y.mean()) / (y.std() if y.std() else 1.0)
    columns = []
    for predictor in predictors:
        x = rankdata(np.asarray(predictor, dtype=float))
        columns.append((x - x.mean()) / (x.std() if x.std() else 1.0))
    design = np.column_stack(columns)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ beta
    denom = float(np.dot(y, y))
    r2 = 0.0 if denom == 0 else float(np.dot(fitted, fitted) / denom)
    residual_norm = float(np.linalg.norm(y - fitted) / np.sqrt(len(y)))
    return [float(value) for value in beta], max(0.0, min(1.0, r2)), residual_norm


def integration_label(prompt_weight: float, image_weight: float, r2: float) -> str:
    if r2 < 0.25:
        return "nonlinear_or_poor_mixture_fit"
    if abs(prompt_weight) < 0.1 and image_weight > 0.25:
        return "image_dominant"
    if abs(image_weight) < 0.1 and prompt_weight > 0.25:
        return "prompt_dominant"
    if image_weight > prompt_weight * 1.5:
        return "image_dominant"
    if prompt_weight > image_weight * 1.5:
        return "prompt_dominant"
    return "additive_balanced"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test concept limit. Writes *_smoke outputs.")
    args = parser.parse_args()

    _config = load_project_config(args.config)
    _backbone_config, backbone_text, backbone_multimodal, mid_fraction = load_project_backbone(args.config)
    metadata = json.loads(embeddings_path("embedding_metadata_full.json").read_text(encoding="utf-8"))
    arrays = np.load(embeddings_path("pooled_embeddings_full.npz"))
    lookup, layers_by_model = build_metadata_lookup(metadata)

    active_concepts = [row["concept"].lower() for row in load_active_concept_rows(args.config, domain="sensory")]
    things_concepts = [concept.lower() for concept in json.loads(THINGS_BEHAVIOR_CONCEPTS.read_text(encoding="utf-8"))]
    things_index = {concept: idx for idx, concept in enumerate(things_concepts)}
    target_concepts = [concept for concept in active_concepts if concept in things_index]
    if args.limit:
        target_concepts = target_concepts[: args.limit]

    behavior = np.load(THINGS_BEHAVIOR_MATRIX)
    behavior_idx = [things_index[concept] for concept in target_concepts]
    behavior_dist = 1.0 - behavior[np.ix_(behavior_idx, behavior_idx)]
    human_rdm = np.asarray(behavior_dist[np.triu_indices(len(target_concepts), k=1)], dtype=float)

    selected_text_layers = selected_layers(layers_by_model[backbone_text], mid_fraction)
    selected_multimodal_layers = selected_layers(layers_by_model[backbone_multimodal], mid_fraction)
    model_rdms = {}
    for condition in CONDITIONS:
        model_id = condition_model_id(backbone_text, backbone_multimodal, condition)
        layers = selected_text_layers if condition.startswith("T_") else selected_multimodal_layers
        matrix, concepts = mean_embedding_for_condition(arrays, lookup, model_id, condition, layers)
        model_rdms[condition] = condensed_cosine_distance(ordered_matrix(matrix, concepts, target_concepts))

    anchor_matrices = {
        "SigLIP2": load_siglip_anchor(arrays, metadata, lookup, layers_by_model, target_concepts),
        "CLIP ViT-L/14": load_static_anchor("CLIP ViT-L/14", target_concepts),
        "DINOv2": load_static_anchor("DINOv2", target_concepts),
    }
    anchor_rdms = {name: condensed_cosine_distance(matrix) for name, matrix in anchor_matrices.items()}
    proxy_rdms = build_proxy_rdms(target_concepts, args.config)

    visual_controls = [anchor_rdms[name] for name in VISUAL_ANCHORS] + [proxy_rdms[name] for name in PROXY_CONTROLS]
    human_residual = residualize(human_rdm, visual_controls)

    rows = []
    summary: dict[str, Any] = {"num_concepts": len(target_concepts), "conditions": {}, "visual_residual_anchors": {}}
    for condition, model_rdm in model_rdms.items():
        human_resid_score = spearman_corr(residualize(model_rdm, visual_controls), human_residual)
        raw_human_score = spearman_corr(model_rdm, human_rdm)
        rows.append(
            {
                "analysis": "human_residual_after_visual_control",
                "anchor_name": "THINGS_residual_after_visual_anchors",
                "condition": condition,
                "rsa_score": human_resid_score,
                "raw_reference_rsa": raw_human_score,
                "num_concepts": len(target_concepts),
                "num_pairs": len(human_rdm),
            }
        )
        summary["conditions"].setdefault(condition, {})["human_residual_rsa"] = human_resid_score
        summary["conditions"][condition]["raw_things_rsa"] = raw_human_score

    visual_control_base = [human_rdm] + [proxy_rdms[name] for name in PROXY_CONTROLS]
    for anchor_name, anchor_rdm in anchor_rdms.items():
        anchor_residual = residualize(anchor_rdm, visual_control_base)
        summary["visual_residual_anchors"][anchor_name] = {}
        for condition, model_rdm in model_rdms.items():
            score = spearman_corr(residualize(model_rdm, visual_control_base), anchor_residual)
            rows.append(
                {
                    "analysis": "visual_residual_after_human_category_control",
                    "anchor_name": f"{anchor_name}_residual_after_THINGS_and_proxies",
                    "condition": condition,
                    "rsa_score": score,
                    "raw_reference_rsa": spearman_corr(model_rdm, anchor_rdm),
                    "num_concepts": len(target_concepts),
                    "num_pairs": len(anchor_rdm),
                }
            )
            summary["visual_residual_anchors"][anchor_name][condition] = score

    beta, r2, residual_norm = standardized_regression(
        model_rdms["M_prompt_plus_matched_image"],
        [model_rdms["T_prompt_primary"], model_rdms["M_matched_image"]],
    )
    mixture_rows = [
        {
            "target_condition": "M_prompt_plus_matched_image",
            "predictor_condition": "T_prompt_primary",
            "standardized_weight": beta[0],
            "mixture_r2": r2,
            "residual_norm": residual_norm,
            "integration_label": integration_label(beta[0], beta[1], r2),
            "num_concepts": len(target_concepts),
            "num_pairs": len(human_rdm),
        },
        {
            "target_condition": "M_prompt_plus_matched_image",
            "predictor_condition": "M_matched_image",
            "standardized_weight": beta[1],
            "mixture_r2": r2,
            "residual_norm": residual_norm,
            "integration_label": integration_label(beta[0], beta[1], r2),
            "num_concepts": len(target_concepts),
            "num_pairs": len(human_rdm),
        },
    ]
    summary["prompt_image_mixture"] = {
        "prompt_weight": beta[0],
        "matched_image_weight": beta[1],
        "mixture_r2": r2,
        "residual_norm": residual_norm,
        "integration_label": integration_label(beta[0], beta[1], r2),
    }

    prompt_human = summary["conditions"]["T_prompt_primary"]["human_residual_rsa"]
    matched_human = summary["conditions"]["M_matched_image"]["human_residual_rsa"]
    summary["primary_contrasts"] = {
        "human_residual_prompt_minus_matched": prompt_human - matched_human,
        "human_residual_leader": "T_prompt_primary" if prompt_human >= matched_human else "M_matched_image",
    }
    for anchor_name in VISUAL_ANCHORS:
        prompt_visual = summary["visual_residual_anchors"][anchor_name]["T_prompt_primary"]
        matched_visual = summary["visual_residual_anchors"][anchor_name]["M_matched_image"]
        summary["primary_contrasts"][f"{anchor_name}_visual_residual_matched_minus_prompt"] = matched_visual - prompt_visual

    out_suffix = suffix(args.limit)
    write_csv(
        metrics_path(f"residual_reference_alignment{out_suffix}.csv"),
        rows,
        ["analysis", "anchor_name", "condition", "rsa_score", "raw_reference_rsa", "num_concepts", "num_pairs"],
    )
    write_csv(
        metrics_path(f"prompt_image_mixture_decomposition{out_suffix}.csv"),
        mixture_rows,
        [
            "target_condition",
            "predictor_condition",
            "standardized_weight",
            "mixture_r2",
            "residual_norm",
            "integration_label",
            "num_concepts",
            "num_pairs",
        ],
    )
    write_json(metrics_path(f"residual_interaction_summary{out_suffix}.json"), summary)

    lines = [
        "# Residual Interaction Report",
        "",
        "## Primary Contrasts",
    ]
    for key, value in summary["primary_contrasts"].items():
        lines.append(f"- {key}: `{value}`" if isinstance(value, str) else f"- {key}: `{value:.4f}`")
    mix = summary["prompt_image_mixture"]
    lines.extend(
        [
            "",
            "## Prompt+Image Mixture",
            f"- Prompt weight: `{mix['prompt_weight']:.4f}`",
            f"- Matched-image weight: `{mix['matched_image_weight']:.4f}`",
            f"- Mixture R2: `{mix['mixture_r2']:.4f}`",
            f"- Integration label: `{mix['integration_label']}`",
            "",
            "## Interpretation",
            "- Human-residual RSA asks whether prompting or grounding better captures THINGS structure not explained by visual anchors and simple controls.",
            "- Visual-residual RSA asks whether grounding uniquely captures visual-anchor structure after removing human/category organization.",
            "- The mixture model tests whether prompt+image geometry is additive, image-dominant, prompt-dominant, or poorly explained by the two base regimes.",
        ]
    )
    write_text(output_path("reports", "main_results", f"residual_interaction_report{out_suffix}.md"), "\n".join(lines))
    append_run_log(
        "Residual Interaction Analyses",
        [
            f"Wrote residual-reference alignment to {metrics_path(f'residual_reference_alignment{out_suffix}.csv').relative_to(ROOT)}.",
            f"Wrote prompt-image mixture decomposition to {metrics_path(f'prompt_image_mixture_decomposition{out_suffix}.csv').relative_to(ROOT)}.",
        ],
    )


if __name__ == "__main__":
    main()
