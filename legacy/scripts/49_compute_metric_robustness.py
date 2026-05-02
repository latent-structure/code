from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from analysis_common import ordered_embedding_for_concepts
from common import (
    ROOT,
    append_run_log,
    condensed_cosine_distance,
    load_project_config,
    metrics_path,
    output_path,
    rankdata,
    write_csv,
    write_json,
)
from hardening_common import (
    build_proxy_rdms,
    lancaster_matrix_for_concepts,
    load_embedding_bundle,
    load_project_backbone,
    load_siglip_reference,
    load_things_reference,
    mean_embedding_for_condition,
    residual_rsa,
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
    "M_degraded_image",
]
ANCHORS = ["THINGS", "controlled_THINGS", "SigLIP2", "CLIP_ViT_L_14", "DINOv2", "lancaster_perceptual"]
CKA_PAIRS = [
    ("T_prompt_primary", "M_matched_image"),
    ("M_text_only", "M_matched_image"),
    ("M_matched_image", "M_prompt_plus_matched_image"),
    ("M_mismatched_image", "M_matched_image"),
    ("M_blank_image", "M_matched_image"),
]


def family_specs(config: dict[str, Any]) -> list[dict[str, str]]:
    return [dict(row) for row in config["analysis"]["analysis"].get("cross_family_families", [])]


def model_for_condition(family: dict[str, str], condition: str) -> str:
    return family["text_model"] if condition.startswith("T_") else family["multimodal_model"]


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x, y) / denom)


def spearman_corr_local(x: np.ndarray, y: np.ndarray) -> float:
    return pearson_corr(rankdata(np.asarray(x, dtype=float)), rankdata(np.asarray(y, dtype=float)))


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    denom = np.linalg.norm(matrix, axis=1, keepdims=True)
    denom[denom <= 1e-12] = 1.0
    return matrix / denom


def condensed_euclidean_l2_distance(matrix: np.ndarray) -> np.ndarray:
    normed = l2_normalize(matrix)
    gram = np.clip(normed @ normed.T, -1.0, 1.0)
    sq = np.maximum(2.0 - 2.0 * gram, 0.0)
    dist = np.sqrt(sq)
    return dist[np.triu_indices(dist.shape[0], k=1)]


def square_from_condensed(condensed: np.ndarray, n: int) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=float)
    matrix[np.triu_indices(n, k=1)] = condensed
    return matrix + matrix.T


def centered_gram(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    gram = x @ x.T
    row_mean = gram.mean(axis=1, keepdims=True)
    col_mean = gram.mean(axis=0, keepdims=True)
    return gram - row_mean - col_mean + gram.mean()


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    left_gram = centered_gram(left)
    right_gram = centered_gram(right)
    numerator = float(np.sum(left_gram * right_gram))
    denom = float(np.linalg.norm(left_gram) * np.linalg.norm(right_gram))
    if denom <= 1e-12:
        return 0.0
    return numerator / denom


def ordered_static_anchor(name: str, target_concepts: list[str]) -> np.ndarray:
    mapping = {
        "CLIP_ViT_L_14": ("clip_vitl14_embeddings.npy", "clip_vitl14_concepts.json"),
        "DINOv2": ("dinov2_embeddings.npy", "dinov2_concepts.json"),
    }
    emb_name, concept_name = mapping[name]
    matrix = np.load(ROOT / "data" / "anchors" / emb_name)
    concepts = [concept.lower() for concept in json.loads((ROOT / "data" / "anchors" / concept_name).read_text(encoding="utf-8"))]
    return ordered_embedding_for_concepts(matrix, concepts, target_concepts)


def rdm_for_metric(matrix: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        return condensed_cosine_distance(matrix)
    if metric == "euclidean_l2":
        return condensed_euclidean_l2_distance(matrix)
    raise ValueError(f"Unknown RDM metric: {metric}")


def rsa_for_method(model_rdm: np.ndarray, anchor_rdm: np.ndarray, method: str) -> float:
    if method == "pearson":
        return pearson_corr(model_rdm, anchor_rdm)
    if method == "spearman":
        return spearman_corr_local(model_rdm, anchor_rdm)
    raise ValueError(f"Unknown RSA method: {method}")


def add_score_row(
    rows: list[dict[str, Any]],
    family: dict[str, str],
    condition: str,
    anchor_name: str,
    robustness_metric: str,
    rdm_metric: str,
    rsa_correlation: str,
    score: float,
    num_concepts: int,
    num_pairs: int,
) -> None:
    rows.append(
        {
            "family_name": family["family_name"],
            "family_role": family.get("family_role", ""),
            "text_model": family["text_model"],
            "multimodal_model": family["multimodal_model"],
            "condition": condition,
            "anchor_name": anchor_name,
            "robustness_metric": robustness_metric,
            "rdm_metric": rdm_metric,
            "rsa_correlation": rsa_correlation,
            "score": score,
            "num_concepts": num_concepts,
            "num_pairs": num_pairs,
        }
    )


def add_contrast_rows(
    rows: list[dict[str, Any]],
    family: dict[str, str],
    scores: dict[tuple[str, str, str], float],
) -> None:
    contrasts = [
        ("matched_minus_prompt", "T_prompt_primary", "M_matched_image"),
        ("matched_minus_text_only", "M_text_only", "M_matched_image"),
        ("prompt_plus_image_minus_matched", "M_matched_image", "M_prompt_plus_matched_image"),
        ("matched_minus_mismatched", "M_mismatched_image", "M_matched_image"),
        ("matched_minus_blank", "M_blank_image", "M_matched_image"),
        ("matched_minus_degraded", "M_degraded_image", "M_matched_image"),
    ]
    metric_anchor_pairs = sorted({(key[0], key[1]) for key in scores})
    for robustness_metric, anchor_name in metric_anchor_pairs:
        for contrast_name, left, right in contrasts:
            left_key = (robustness_metric, anchor_name, left)
            right_key = (robustness_metric, anchor_name, right)
            if left_key not in scores or right_key not in scores:
                continue
            rows.append(
                {
                    "family_name": family["family_name"],
                    "family_role": family.get("family_role", ""),
                    "anchor_name": anchor_name,
                    "robustness_metric": robustness_metric,
                    "contrast_name": contrast_name,
                    "left_condition": left,
                    "right_condition": right,
                    "delta": scores[right_key] - scores[left_key],
                }
            )


def copy_to_figures_data(paths: list[Path]) -> None:
    target_dir = ROOT / "figures_data" / "derived"
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists():
            shutil.copy2(path, target_dir / path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute RSA/CKA robustness metrics.")
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--limit", type=int, default=0, help="Optional concept limit for smoke tests.")
    parser.add_argument("--no-copy-figures-data", action="store_true")
    args = parser.parse_args()

    config, _primary_text, _primary_multimodal, mid_fraction = load_project_backbone(args.config)
    metadata_lookup, pooled, layers_by_model, metadata = load_embedding_bundle()
    things_behavior, things_concepts, _ = load_things_reference()
    target_concepts = things_concepts[: args.limit] if args.limit else things_concepts
    target_idx = [things_concepts.index(concept) for concept in target_concepts]
    things_subset = things_behavior[np.ix_(target_idx, target_idx)]
    things_rdm = np.asarray(1.0 - things_subset[np.triu_indices(len(target_concepts), k=1)], dtype=float)
    proxy_rdms = build_proxy_rdms(target_concepts)
    proxy_controls = [proxy_rdms[name] for name in ["subtype_membership", "coarse_category_structure", "sound_linked_vs_other", "lexical_trigram_distance"]]

    siglip_matrix, siglip_concepts = load_siglip_reference(metadata_lookup, pooled, layers_by_model, metadata)
    siglip_ordered = ordered_embedding_for_concepts(siglip_matrix, siglip_concepts, target_concepts)
    anchor_features = {
        "SigLIP2": siglip_ordered,
        "CLIP_ViT_L_14": ordered_static_anchor("CLIP_ViT_L_14", target_concepts),
        "DINOv2": ordered_static_anchor("DINOv2", target_concepts),
    }

    lancaster_all = [concept.lower() for concept in json.loads((ROOT / "data" / "anchors" / "lancaster_perceptual_concepts.json").read_text(encoding="utf-8"))]
    target_set = set(target_concepts)
    lancaster_concepts = [concept for concept in lancaster_all if concept in target_set]
    lancaster_reference = lancaster_matrix_for_concepts(
        lancaster_concepts,
        ["Auditory.mean", "Gustatory.mean", "Haptic.mean", "Interoceptive.mean", "Olfactory.mean", "Visual.mean"],
    )

    score_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    cka_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "num_concepts": len(target_concepts),
        "num_lancaster_concepts": len(lancaster_concepts),
        "conditions": CONDITIONS,
        "families": {},
    }

    robustness_specs = [
        ("cosine_pearson", "cosine", "pearson"),
        ("euclidean_l2_spearman", "euclidean_l2", "spearman"),
    ]

    for family in family_specs(config):
        family_name = family["family_name"]
        missing_models = [model for model in {family["text_model"], family["multimodal_model"]} if model not in layers_by_model]
        if missing_models:
            summary["families"][family_name] = {"status": "blocked", "missing_models": missing_models}
            continue

        selected_by_model = {
            model: selected_layers(layers_by_model[model], mid_fraction)
            for model in {family["text_model"], family["multimodal_model"]}
        }
        condition_matrices: dict[str, np.ndarray] = {}
        lancaster_matrices: dict[str, np.ndarray] = {}
        for condition in CONDITIONS:
            model_id = model_for_condition(family, condition)
            matrix, concepts = mean_embedding_for_condition(metadata_lookup, pooled, model_id, condition, selected_by_model[model_id])
            ordered = ordered_embedding_for_concepts(matrix, concepts, target_concepts)
            condition_matrices[condition] = ordered
            lancaster_matrices[condition] = ordered_embedding_for_concepts(matrix, concepts, lancaster_concepts)

        family_scores: dict[tuple[str, str, str], float] = {}
        for robustness_metric, rdm_metric, rsa_method in robustness_specs:
            anchor_rdms = {
                "SigLIP2": rdm_for_metric(anchor_features["SigLIP2"], rdm_metric),
                "CLIP_ViT_L_14": rdm_for_metric(anchor_features["CLIP_ViT_L_14"], rdm_metric),
                "DINOv2": rdm_for_metric(anchor_features["DINOv2"], rdm_metric),
                "lancaster_perceptual": rdm_for_metric(lancaster_reference, rdm_metric),
            }
            for condition in CONDITIONS:
                model_rdm = rdm_for_metric(condition_matrices[condition], rdm_metric)
                lancaster_model_rdm = rdm_for_metric(lancaster_matrices[condition], rdm_metric)
                anchor_scores = {
                    "THINGS": rsa_for_method(model_rdm, things_rdm, rsa_method),
                    "controlled_THINGS": residual_rsa(model_rdm, things_rdm, proxy_controls),
                    "SigLIP2": rsa_for_method(model_rdm, anchor_rdms["SigLIP2"], rsa_method),
                    "CLIP_ViT_L_14": rsa_for_method(model_rdm, anchor_rdms["CLIP_ViT_L_14"], rsa_method),
                    "DINOv2": rsa_for_method(model_rdm, anchor_rdms["DINOv2"], rsa_method),
                    "lancaster_perceptual": rsa_for_method(lancaster_model_rdm, anchor_rdms["lancaster_perceptual"], rsa_method),
                }
                for anchor_name, score in anchor_scores.items():
                    num_concepts = len(lancaster_concepts) if anchor_name == "lancaster_perceptual" else len(target_concepts)
                    num_pairs = len(lancaster_model_rdm) if anchor_name == "lancaster_perceptual" else len(model_rdm)
                    family_scores[(robustness_metric, anchor_name, condition)] = score
                    add_score_row(
                        score_rows,
                        family,
                        condition,
                        anchor_name,
                        robustness_metric,
                        rdm_metric,
                        rsa_method,
                        score,
                        num_concepts,
                        num_pairs,
                    )
        add_contrast_rows(contrast_rows, family, family_scores)

        for left, right in CKA_PAIRS:
            if left not in condition_matrices or right not in condition_matrices:
                continue
            cka_rows.append(
                {
                    "family_name": family_name,
                    "family_role": family.get("family_role", ""),
                    "left_condition": left,
                    "right_condition": right,
                    "cka": linear_cka(condition_matrices[left], condition_matrices[right]),
                    "num_concepts": len(target_concepts),
                    "left_model": model_for_condition(family, left),
                    "right_model": model_for_condition(family, right),
                }
            )

        summary["families"][family_name] = {
            "status": "ok",
            "headline": {
                "|".join(key): value
                for key, value in family_scores.items()
                if key[1] in {"THINGS", "SigLIP2"} and key[2] in {"T_prompt_primary", "M_text_only", "M_matched_image"}
            },
        }

    suffix = "_smoke" if args.limit else ""
    rsa_path = metrics_path(f"metric_robustness_rsa{suffix}.csv")
    contrast_path = metrics_path(f"metric_robustness_contrasts{suffix}.csv")
    cka_path = metrics_path(f"metric_robustness_cka{suffix}.csv")
    summary_path = metrics_path(f"metric_robustness_summary{suffix}.json")
    report_path = output_path("reports", "main_results", f"metric_robustness_report{suffix}.md")

    write_csv(
        rsa_path,
        score_rows,
        [
            "family_name",
            "family_role",
            "text_model",
            "multimodal_model",
            "condition",
            "anchor_name",
            "robustness_metric",
            "rdm_metric",
            "rsa_correlation",
            "score",
            "num_concepts",
            "num_pairs",
        ],
    )
    write_csv(
        contrast_path,
        contrast_rows,
        [
            "family_name",
            "family_role",
            "anchor_name",
            "robustness_metric",
            "contrast_name",
            "left_condition",
            "right_condition",
            "delta",
        ],
    )
    write_csv(
        cka_path,
        cka_rows,
        ["family_name", "family_role", "left_condition", "right_condition", "cka", "num_concepts", "left_model", "right_model"],
    )
    write_json(summary_path, summary)

    lines = [
        "# Metric Robustness Report",
        "",
        "Primary analyses remain cosine-distance RDMs with Spearman RSA. This report checks Pearson RSA, L2-normalized Euclidean RSA, and linear CKA.",
        "",
        "## Headline contrasts",
    ]
    for row in contrast_rows:
        if row["anchor_name"] in {"THINGS", "SigLIP2"} and row["contrast_name"] in {"matched_minus_prompt", "matched_minus_text_only"}:
            lines.append(
                f"- `{row['family_name']}` `{row['robustness_metric']}` `{row['anchor_name']}` "
                f"`{row['contrast_name']}`: `{row['delta']:+.4f}`"
            )
    lines.append("")
    lines.append("## CKA condition similarities")
    for row in cka_rows:
        lines.append(
            f"- `{row['family_name']}` `{row['left_condition']}` vs `{row['right_condition']}`: `{row['cka']:.4f}`"
        )
    write_text(report_path, "\n".join(lines))

    if not args.no_copy_figures_data and not args.limit:
        copy_to_figures_data([rsa_path, contrast_path, cka_path])

    append_run_log("Metric Robustness", [f"Wrote metric robustness outputs with suffix `{suffix}`."])
    print(f"Wrote {rsa_path.relative_to(ROOT)}")
    print(f"Wrote {contrast_path.relative_to(ROOT)}")
    print(f"Wrote {cka_path.relative_to(ROOT)}")
    print(f"Wrote {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
