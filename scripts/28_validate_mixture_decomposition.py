from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from analysis_common import ordered_embedding_for_concepts
from common import ROOT, append_run_log, condensed_cosine_distance, embeddings_path, load_project_config, metrics_path, output_path, rankdata, write_csv, write_json
from hardening_common import condition_model_id, load_active_concept_rows, load_project_backbone, mean_embedding_for_condition, selected_layers, write_text


CONDITIONS = ["T_prompt_primary", "M_matched_image", "M_prompt_plus_matched_image"]
NORMALIZATIONS = ["raw_centered", "zscore", "unit_norm", "rank_zscore"]


def build_lookup(metadata: dict[str, Any]) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[str, list[int]]]:
    from common import canonical_condition_name

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
    return lookup, {model: sorted(set(layers)) for model, layers in layers_by_model.items()}


def normalize_vector(values: np.ndarray, mode: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if mode == "rank_zscore":
        vector = rankdata(vector)
        vector = vector - vector.mean()
        scale = vector.std() if vector.std() else 1.0
        return vector / scale
    if mode == "raw_centered":
        return vector - vector.mean()
    if mode == "zscore":
        vector = vector - vector.mean()
        scale = vector.std() if vector.std() else 1.0
        return vector / scale
    if mode == "unit_norm":
        vector = vector - vector.mean()
        scale = np.linalg.norm(vector)
        return vector / (scale if scale else 1.0)
    raise ValueError(f"Unknown normalization mode: {mode}")


def mixture_regression(target: np.ndarray, predictors: list[np.ndarray], mode: str = "rank_zscore") -> tuple[list[float], float, float]:
    y = normalize_vector(target, mode)
    columns = []
    for predictor in predictors:
        columns.append(normalize_vector(predictor, mode))
    design = np.column_stack(columns)
    return regression_from_normalized(y, design)


def regression_from_normalized(y: np.ndarray, design: np.ndarray) -> tuple[list[float], float, float]:
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ beta
    denom = float(np.dot(y, y))
    r2 = 0.0 if denom == 0 else float(np.dot(fitted, fitted) / denom)
    residual_norm = float(np.linalg.norm(y - fitted) / np.sqrt(len(y)))
    return [float(value) for value in beta], max(0.0, min(1.0, r2)), residual_norm


def pearson_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left = left - left.mean()
    right = right - right.mean()
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denom)


def spearman_corr(left: np.ndarray, right: np.ndarray) -> float:
    return pearson_corr(rankdata(np.asarray(left, dtype=float)), rankdata(np.asarray(right, dtype=float)))


def single_predictor_r2(target: np.ndarray, predictor: np.ndarray, mode: str = "rank_zscore") -> float:
    y = normalize_vector(target, mode)
    x = normalize_vector(predictor, mode)
    _beta, r2, _residual = regression_from_normalized(y, x[:, None])
    return r2


def collinearity_and_unique_variance(
    target: np.ndarray,
    prompt: np.ndarray,
    matched: np.ndarray,
    full_r2: float,
    mode: str = "rank_zscore",
) -> dict[str, float]:
    pearson = pearson_corr(normalize_vector(prompt, "zscore"), normalize_vector(matched, "zscore"))
    spearman = spearman_corr(prompt, matched)
    vif = float(1.0 / max(1.0 - pearson**2, 1e-12))
    prompt_only_r2 = single_predictor_r2(target, prompt, mode)
    matched_only_r2 = single_predictor_r2(target, matched, mode)
    unique_prompt = max(0.0, full_r2 - matched_only_r2)
    unique_matched = max(0.0, full_r2 - prompt_only_r2)
    shared = max(0.0, full_r2 - unique_prompt - unique_matched)
    residual = max(0.0, 1.0 - full_r2)
    return {
        "predictor_pearson_r": pearson,
        "predictor_spearman_rho": spearman,
        "predictor_vif": vif,
        "prompt_only_r2": prompt_only_r2,
        "matched_only_r2": matched_only_r2,
        "full_model_r2": full_r2,
        "unique_prompt_r2": unique_prompt,
        "unique_matched_image_r2": unique_matched,
        "shared_r2": shared,
        "residual_unexplained": residual,
    }


def standardized_regression(target: np.ndarray, predictors: list[np.ndarray]) -> tuple[list[float], float, float]:
    return mixture_regression(target, predictors, "rank_zscore")


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


def p_ge(null_values: list[float], observed: float) -> float:
    return (sum(value >= observed for value in null_values) + 1.0) / (len(null_values) + 1.0)


def diagnostic_decision(diagnostics: dict[str, dict[str, float]]) -> dict[str, Any]:
    required_modes = ["zscore", "unit_norm", "rank_zscore"]
    robust = True
    for mode in required_modes:
        values = diagnostics[mode]
        prompt_weight = abs(values["prompt_weight"])
        image_weight = values["matched_image_weight"]
        if image_weight <= 0.25 or image_weight <= 1.5 * prompt_weight:
            robust = False
    return {
        "robust_image_dominance": robust,
        "qualification_needed": not robust,
        "decision_rule": (
            "Image-dominance is treated as robust only if the matched-image coefficient remains >0.25 "
            "and at least 1.5x the absolute prompt coefficient under z-score, unit-norm, and rank-z-score "
            "RDM regressions. If this fails, the mixture result should be reported as normalization-sensitive."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test concept limit.")
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260424)
    args = parser.parse_args()

    _config, backbone_text, backbone_multimodal, mid_fraction = load_project_backbone(args.config)
    metadata = json.loads(embeddings_path("embedding_metadata_full.json").read_text(encoding="utf-8"))
    arrays = np.load(embeddings_path("pooled_embeddings_full.npz"))
    lookup, layers_by_model = build_lookup(metadata)
    target_concepts = [row["concept"].lower() for row in load_active_concept_rows(args.config, domain="sensory")]
    if args.limit:
        target_concepts = target_concepts[: args.limit]

    selected_text_layers = selected_layers(layers_by_model[backbone_text], mid_fraction)
    selected_multimodal_layers = selected_layers(layers_by_model[backbone_multimodal], mid_fraction)
    rdms = {}
    for condition in CONDITIONS:
        model_id = condition_model_id(backbone_text, backbone_multimodal, condition)
        layers = selected_text_layers if condition.startswith("T_") else selected_multimodal_layers
        matrix, concepts = mean_embedding_for_condition(lookup, arrays, model_id, condition, layers)
        rdms[condition] = condensed_cosine_distance(ordered_embedding_for_concepts(matrix, concepts, target_concepts))

    beta, r2, residual_norm = standardized_regression(
        rdms["M_prompt_plus_matched_image"],
        [rdms["T_prompt_primary"], rdms["M_matched_image"]],
    )
    label = integration_label(beta[0], beta[1], r2)
    diagnostics: dict[str, dict[str, float]] = {}
    for mode in NORMALIZATIONS:
        mode_beta, mode_r2, mode_residual_norm = mixture_regression(
            rdms["M_prompt_plus_matched_image"],
            [rdms["T_prompt_primary"], rdms["M_matched_image"]],
            mode,
        )
        diagnostics[mode] = {
            "prompt_weight": mode_beta[0],
            "matched_image_weight": mode_beta[1],
            "mixture_r2": mode_r2,
            "residual_norm": mode_residual_norm,
        }
    decision = diagnostic_decision(diagnostics)
    collinearity = collinearity_and_unique_variance(
        rdms["M_prompt_plus_matched_image"],
        rdms["T_prompt_primary"],
        rdms["M_matched_image"],
        r2,
        "rank_zscore",
    )
    rng = np.random.default_rng(args.seed)
    null_r2 = []
    null_image_weight = []
    null_prompt_weight = []
    target = np.asarray(rdms["M_prompt_plus_matched_image"], dtype=float)
    null_y = normalize_vector(target, "rank_zscore")
    null_design = np.column_stack(
        [
            normalize_vector(rdms["T_prompt_primary"], "rank_zscore"),
            normalize_vector(rdms["M_matched_image"], "rank_zscore"),
        ]
    )
    for _ in range(args.permutations):
        null_beta, null_fit, _ = regression_from_normalized(null_y[rng.permutation(len(null_y))], null_design)
        null_prompt_weight.append(null_beta[0])
        null_image_weight.append(null_beta[1])
        null_r2.append(null_fit)

    rows = [
        {
            "analysis": "observed",
            "target_condition": "M_prompt_plus_matched_image",
            "predictor_condition": "T_prompt_primary",
            "standardized_weight": beta[0],
            "mixture_r2": r2,
            "residual_norm": residual_norm,
            "integration_label": label,
            "p_ge_null_r2": p_ge(null_r2, r2),
            "num_concepts": len(target_concepts),
            "num_pairs": len(target),
            "normalization": "rank_zscore",
        },
        {
            "analysis": "observed",
            "target_condition": "M_prompt_plus_matched_image",
            "predictor_condition": "M_matched_image",
            "standardized_weight": beta[1],
            "mixture_r2": r2,
            "residual_norm": residual_norm,
            "integration_label": label,
            "p_ge_null_r2": p_ge(null_r2, r2),
            "num_concepts": len(target_concepts),
            "num_pairs": len(target),
            "normalization": "rank_zscore",
        },
    ]
    for mode, mode_values in diagnostics.items():
        mode_label = integration_label(mode_values["prompt_weight"], mode_values["matched_image_weight"], mode_values["mixture_r2"])
        rows.extend(
            [
                {
                    "analysis": "normalization_diagnostic",
                    "target_condition": "M_prompt_plus_matched_image",
                    "predictor_condition": "T_prompt_primary",
                    "standardized_weight": mode_values["prompt_weight"],
                    "mixture_r2": mode_values["mixture_r2"],
                    "residual_norm": mode_values["residual_norm"],
                    "integration_label": mode_label,
                    "p_ge_null_r2": "",
                    "num_concepts": len(target_concepts),
                    "num_pairs": len(target),
                    "normalization": mode,
                },
                {
                    "analysis": "normalization_diagnostic",
                    "target_condition": "M_prompt_plus_matched_image",
                    "predictor_condition": "M_matched_image",
                    "standardized_weight": mode_values["matched_image_weight"],
                    "mixture_r2": mode_values["mixture_r2"],
                    "residual_norm": mode_values["residual_norm"],
                    "integration_label": mode_label,
                    "p_ge_null_r2": "",
                    "num_concepts": len(target_concepts),
                    "num_pairs": len(target),
                    "normalization": mode,
                },
            ]
        )
    rows.extend(
        [
            {
                "analysis": "predictor_collinearity",
                "target_condition": "M_prompt_plus_matched_image",
                "predictor_condition": "T_prompt_primary_vs_M_matched_image",
                "standardized_weight": "",
                "mixture_r2": "",
                "residual_norm": "",
                "integration_label": (
                    f"pearson_r={collinearity['predictor_pearson_r']:.6f};"
                    f"spearman_rho={collinearity['predictor_spearman_rho']:.6f};"
                    f"vif={collinearity['predictor_vif']:.6f}"
                ),
                "p_ge_null_r2": "",
                "num_concepts": len(target_concepts),
                "num_pairs": len(target),
                "normalization": "rank_zscore",
            },
            {
                "analysis": "unique_variance",
                "target_condition": "M_prompt_plus_matched_image",
                "predictor_condition": "T_prompt_primary",
                "standardized_weight": "",
                "mixture_r2": collinearity["unique_prompt_r2"],
                "residual_norm": "",
                "integration_label": "unique_prompt_r2",
                "p_ge_null_r2": "",
                "num_concepts": len(target_concepts),
                "num_pairs": len(target),
                "normalization": "rank_zscore",
            },
            {
                "analysis": "unique_variance",
                "target_condition": "M_prompt_plus_matched_image",
                "predictor_condition": "M_matched_image",
                "standardized_weight": "",
                "mixture_r2": collinearity["unique_matched_image_r2"],
                "residual_norm": "",
                "integration_label": "unique_matched_image_r2",
                "p_ge_null_r2": "",
                "num_concepts": len(target_concepts),
                "num_pairs": len(target),
                "normalization": "rank_zscore",
            },
            {
                "analysis": "unique_variance",
                "target_condition": "M_prompt_plus_matched_image",
                "predictor_condition": "shared",
                "standardized_weight": "",
                "mixture_r2": collinearity["shared_r2"],
                "residual_norm": "",
                "integration_label": "shared_r2",
                "p_ge_null_r2": "",
                "num_concepts": len(target_concepts),
                "num_pairs": len(target),
                "normalization": "rank_zscore",
            },
        ]
    )
    summary = {
        "num_concepts": len(target_concepts),
        "num_pairs": len(target),
        "permutations": args.permutations,
        "seed": args.seed,
        "observed": {
            "prompt_weight": beta[0],
            "matched_image_weight": beta[1],
            "mixture_r2": r2,
            "residual_norm": residual_norm,
            "integration_label": label,
            "p_ge_null_r2": p_ge(null_r2, r2),
        },
        "normalization_diagnostics": diagnostics,
        "predictor_collinearity": {
            "pearson_r": collinearity["predictor_pearson_r"],
            "spearman_rho": collinearity["predictor_spearman_rho"],
            "vif": collinearity["predictor_vif"],
            "note": "Computed between prompt-only and matched-image predictor RDM vectors.",
        },
        "unique_variance_diagnostics": {
            "prompt_only_r2": collinearity["prompt_only_r2"],
            "matched_only_r2": collinearity["matched_only_r2"],
            "full_model_r2": collinearity["full_model_r2"],
            "unique_prompt_r2": collinearity["unique_prompt_r2"],
            "unique_matched_image_r2": collinearity["unique_matched_image_r2"],
            "shared_r2": collinearity["shared_r2"],
            "residual_unexplained": collinearity["residual_unexplained"],
            "note": "Unique terms are full R2 minus the single-predictor R2 of the other predictor.",
        },
        "diagnostic_decision": decision,
        "null": {
            "mean_r2": float(np.mean(null_r2)) if null_r2 else 0.0,
            "p95_r2": float(np.quantile(null_r2, 0.95)) if null_r2 else 0.0,
            "mean_prompt_weight": float(np.mean(null_prompt_weight)) if null_prompt_weight else 0.0,
            "mean_image_weight": float(np.mean(null_image_weight)) if null_image_weight else 0.0,
        },
        "method_note": "Descriptive rank-RDM regression: prompt+image RDM is predicted from prompt-only and matched-image RDMs.",
    }
    suffix = "_smoke" if args.limit else ""
    write_csv(
        metrics_path(f"mixture_decomposition_validation{suffix}.csv"),
        rows,
        [
            "analysis",
            "target_condition",
            "predictor_condition",
            "standardized_weight",
            "mixture_r2",
            "residual_norm",
            "integration_label",
            "p_ge_null_r2",
            "num_concepts",
            "num_pairs",
            "normalization",
        ],
    )
    write_json(metrics_path(f"mixture_decomposition_validation_summary{suffix}.json"), summary)
    lines = [
        "# Mixture Decomposition Validation Report",
        "",
        f"- Prompt weight: `{beta[0]:.4f}`",
        f"- Matched-image weight: `{beta[1]:.4f}`",
        f"- R2: `{r2:.4f}`",
        f"- Permutation p(R2 >= observed): `{summary['observed']['p_ge_null_r2']:.4f}`",
        f"- Label: `{label}`",
        f"- Predictor Pearson r: `{collinearity['predictor_pearson_r']:.4f}`",
        f"- Predictor Spearman rho: `{collinearity['predictor_spearman_rho']:.4f}`",
        f"- Predictor VIF: `{collinearity['predictor_vif']:.2f}`",
        f"- Prompt-only R2: `{collinearity['prompt_only_r2']:.4f}`",
        f"- Matched-image-only R2: `{collinearity['matched_only_r2']:.4f}`",
        f"- Unique prompt R2: `{collinearity['unique_prompt_r2']:.4f}`",
        f"- Unique matched-image R2: `{collinearity['unique_matched_image_r2']:.4f}`",
        f"- Shared R2: `{collinearity['shared_r2']:.4f}`",
        f"- Normalization diagnostic robust image dominance: `{decision['robust_image_dominance']}`",
        f"- Decision rule: {decision['decision_rule']}",
        "- This is a descriptive rank-RDM regression, not a causal estimator.",
    ]
    write_text(output_path("reports", "main_results", f"mixture_decomposition_validation_report{suffix}.md"), "\n".join(lines))
    append_run_log("Mixture Decomposition Validation", [f"Wrote mixture validation outputs with suffix `{suffix}`."])


if __name__ == "__main__":
    main()
