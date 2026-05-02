from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

import numpy as np

from common import (
    ROOT,
    append_run_log,
    canonical_condition_name,
    condensed_cosine_distance,
    embeddings_path,
    metrics_path,
    output_path,
    rankdata,
    read_csv,
    write_csv,
    write_json,
)
from hardening_common import load_active_concept_rows, load_project_backbone, write_text


MIXTURE_CONDITIONS = ["T_prompt_primary", "M_matched_image", "M_prompt_plus_matched_image"]
RETENTION_CONDITIONS = ["M_matched_image", "M_mismatched_image"]
FAMILY_MODELS = {
    "qwen": {
        "text": "Qwen/Qwen3.5-9B",
        "multimodal": "Qwen/Qwen3-VL-8B-Instruct",
    },
    "mistral": {
        "text": "mistralai/Mistral-Small-24B-Instruct-2501",
        "multimodal": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    },
    "llama": {
        "text": "meta-llama/Llama-3.1-8B-Instruct",
        "multimodal": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    },
}


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


def layer_embedding(
    arrays: Any,
    lookup: dict[tuple[str, str, int], dict[str, Any]],
    model_id: str,
    condition: str,
    layer: int,
) -> tuple[np.ndarray, list[str]]:
    record = lookup.get((model_id, condition, int(layer)))
    if record is None:
        raise RuntimeError(f"Missing embeddings for {model_id} {condition} layer={layer}")
    matrix = np.asarray(arrays[f"record_{record['record_id']}"], dtype=np.float32)
    concepts = [concept.lower() for concept in record["concepts"]]
    return matrix, concepts


def ordered_matrix(matrix: np.ndarray, concepts: list[str], target_concepts: list[str]) -> np.ndarray:
    index = {concept.lower(): idx for idx, concept in enumerate(concepts)}
    missing = [concept for concept in target_concepts if concept not in index]
    if missing:
        raise RuntimeError(f"Missing concepts from layer matrix: {', '.join(missing[:20])}")
    return np.asarray(matrix[[index[concept] for concept in target_concepts]], dtype=np.float32)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(1.0 - np.clip(np.dot(left, right), -1.0, 1.0))


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


def classify_shift(target_distance: float, source_distance: float, margin: float) -> str:
    if source_distance + margin < target_distance:
        return "image_hijack"
    if target_distance + margin < source_distance:
        return "text_retention"
    return "ambiguous"


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "first_half_mean": 0.0, "last_half_mean": 0.0}
    mid = len(values) // 2
    return {
        "mean": float(np.mean(values)),
        "first_half_mean": float(np.mean(values[:mid])) if mid else float(np.mean(values)),
        "last_half_mean": float(np.mean(values[mid:])),
    }


def families_to_run(family: str) -> list[str]:
    if family == "all":
        return list(FAMILY_MODELS)
    if family not in FAMILY_MODELS:
        raise RuntimeError(f"Unknown family={family}. Expected one of: {', '.join(sorted(FAMILY_MODELS))}, all")
    return [family]


def family_models(family: str, config_path: str) -> tuple[str, str]:
    if family == "qwen":
        _config, backbone_text, backbone_multimodal, _mid_fraction = load_project_backbone(config_path)
        return backbone_text, backbone_multimodal
    spec = FAMILY_MODELS[family]
    return spec["text"], spec["multimodal"]


def run_family(
    *,
    family: str,
    config_path: str,
    limit: int,
    margin: float,
    metadata: dict[str, Any],
    arrays: Any,
    lookup: dict[tuple[str, str, int], dict[str, Any]],
    layers_by_model: dict[str, list[int]],
    target_concepts: list[str],
) -> dict[str, Any]:
    backbone_text, backbone_multimodal = family_models(family, config_path)
    mixture_layers = sorted(
        set(layers_by_model[backbone_multimodal])
        & set(layers_by_model[backbone_text])
    )
    for condition in MIXTURE_CONDITIONS:
        model_id = backbone_text if condition.startswith("T_") else backbone_multimodal
        mixture_layers = sorted(set(mixture_layers) & {layer for model, cond, layer in lookup if model == model_id and cond == condition})
    if not mixture_layers:
        raise RuntimeError(f"No common {family} layers found for layerwise global-local dissociation.")

    retention_layers = sorted(set(layers_by_model[backbone_multimodal]))
    for condition in RETENTION_CONDITIONS:
        retention_layers = sorted(
            set(retention_layers) & {layer for model, cond, layer in lookup if model == backbone_multimodal and cond == condition}
        )
    if not retention_layers:
        raise RuntimeError(f"No {family} VLM layers found for layerwise mismatched identity retention.")

    mixture_rows = []
    retention_rows = []
    mismatch_rows = read_csv(ROOT / "data" / "manifests" / "mismatch_map.csv")
    if limit:
        mismatch_rows = mismatch_rows[:limit]

    image_weights: list[float] = []
    prompt_weights: list[float] = []
    retention_rates: list[float] = []
    hijack_rates: list[float] = []

    for layer in mixture_layers:
        prompt_matrix, prompt_concepts = layer_embedding(arrays, lookup, backbone_text, "T_prompt_primary", layer)
        matched_matrix, matched_concepts = layer_embedding(arrays, lookup, backbone_multimodal, "M_matched_image", layer)
        combined_matrix, combined_concepts = layer_embedding(arrays, lookup, backbone_multimodal, "M_prompt_plus_matched_image", layer)

        prompt_rdm = condensed_cosine_distance(ordered_matrix(prompt_matrix, prompt_concepts, target_concepts))
        matched_rdm = condensed_cosine_distance(ordered_matrix(matched_matrix, matched_concepts, target_concepts))
        combined_rdm = condensed_cosine_distance(ordered_matrix(combined_matrix, combined_concepts, target_concepts))
        beta, r2, residual_norm = standardized_regression(combined_rdm, [prompt_rdm, matched_rdm])
        prompt_weight, image_weight = beta
        prompt_weights.append(prompt_weight)
        image_weights.append(image_weight)
        mixture_rows.append(
            {
                "layer": layer,
                "prompt_weight": prompt_weight,
                "matched_image_weight": image_weight,
                "mixture_r2": r2,
                "residual_norm": residual_norm,
                "integration_label": integration_label(prompt_weight, image_weight, r2),
                "num_concepts": len(target_concepts),
                "num_pairs": len(combined_rdm),
            }
        )

    for layer in retention_layers:
        matched_for_retention, matched_retention_concepts = layer_embedding(
            arrays, lookup, backbone_multimodal, "M_matched_image", layer
        )
        mismatched_matrix, mismatched_concepts = layer_embedding(
            arrays, lookup, backbone_multimodal, "M_mismatched_image", layer
        )
        matched_for_retention = normalize_rows(matched_for_retention)
        mismatched_matrix = normalize_rows(mismatched_matrix)
        matched_index = {concept.lower(): idx for idx, concept in enumerate(matched_retention_concepts)}
        mismatched_index = {concept.lower(): idx for idx, concept in enumerate(mismatched_concepts)}

        counts_by_mode: dict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
        margins_by_mode: dict[str, list[float]] = defaultdict(list)
        for item in mismatch_rows:
            target = item["concept"].lower()
            source = item["mismatch_concept"].lower()
            mode = item.get("mismatch_mode", "")
            if target not in matched_index or target not in mismatched_index or source not in matched_index:
                continue
            mismatched_vec = mismatched_matrix[mismatched_index[target]]
            target_vec = matched_for_retention[matched_index[target]]
            source_vec = matched_for_retention[matched_index[source]]
            target_distance = cosine_distance(mismatched_vec, target_vec)
            source_distance = cosine_distance(mismatched_vec, source_vec)
            source_minus_target = source_distance - target_distance
            label = classify_shift(target_distance, source_distance, margin)
            counts_by_mode[mode][label] += 1
            counts_by_mode["all"][label] += 1
            margins_by_mode[mode].append(source_minus_target)
            margins_by_mode["all"].append(source_minus_target)

        for mode, counts in sorted(counts_by_mode.items()):
            total = sum(counts.values())
            margins = np.asarray(margins_by_mode[mode], dtype=float)
            text_retention_rate = 0.0 if total == 0 else counts["text_retention"] / total
            image_hijack_rate = 0.0 if total == 0 else counts["image_hijack"] / total
            ambiguous_rate = 0.0 if total == 0 else counts["ambiguous"] / total
            if mode == "all":
                retention_rates.append(text_retention_rate)
                hijack_rates.append(image_hijack_rate)
            retention_rows.append(
                {
                    "layer": layer,
                    "mismatch_mode": mode,
                    "num_pairs": total,
                    "text_retention_rate": text_retention_rate,
                    "image_hijack_rate": image_hijack_rate,
                    "ambiguous_rate": ambiguous_rate,
                    "mean_source_minus_target_distance": 0.0 if len(margins) == 0 else float(margins.mean()),
                    "median_source_minus_target_distance": 0.0 if len(margins) == 0 else float(np.median(margins)),
                }
            )

    summary = {
        "model_id": backbone_multimodal,
        "text_model_id": backbone_text,
        "family": family,
        "num_mixture_layers": len(mixture_layers),
        "num_retention_layers": len(retention_layers),
        "mixture_layers": mixture_layers,
        "retention_layers": retention_layers,
        "num_concepts": len(target_concepts),
        "mixture": {
            "prompt_weight": summarize(prompt_weights),
            "matched_image_weight": summarize(image_weights),
        },
        "identity_retention": {
            "text_retention_rate": summarize(retention_rates),
            "image_hijack_rate": summarize(hijack_rates),
        },
        "global_local_dissociation_present": bool(
            np.mean(image_weights[len(image_weights) // 2 :]) > 0.5
            and np.mean(retention_rates[len(retention_rates) // 2 :]) > 0.95
        ),
    }

    out_suffix = f"_{family}{suffix(limit)}"
    write_csv(
        metrics_path(f"layerwise_prompt_image_mixture{out_suffix}.csv"),
        mixture_rows,
        [
            "layer",
            "prompt_weight",
            "matched_image_weight",
            "mixture_r2",
            "residual_norm",
            "integration_label",
            "num_concepts",
            "num_pairs",
        ],
    )
    write_csv(
        metrics_path(f"layerwise_mismatched_identity_retention{out_suffix}.csv"),
        retention_rows,
        [
            "layer",
            "mismatch_mode",
            "num_pairs",
            "text_retention_rate",
            "image_hijack_rate",
            "ambiguous_rate",
            "mean_source_minus_target_distance",
            "median_source_minus_target_distance",
        ],
    )
    write_json(metrics_path(f"layerwise_global_local_summary{out_suffix}.json"), summary)

    lines = [
        "# Layerwise Global-Local Dissociation Report",
        "",
        "## Summary",
        f"- Family: `{summary['family']}`",
        f"- Model: `{summary['model_id']}`",
        f"- Mixture layers analyzed: `{summary['num_mixture_layers']}`",
        f"- Retention layers analyzed: `{summary['num_retention_layers']}`",
        f"- Concepts: `{summary['num_concepts']}`",
        f"- Global-local dissociation present: `{summary['global_local_dissociation_present']}`",
        f"- Last-half matched-image mixture weight: `{summary['mixture']['matched_image_weight']['last_half_mean']:.4f}`",
        f"- Last-half prompt mixture weight: `{summary['mixture']['prompt_weight']['last_half_mean']:.4f}`",
        f"- Last-half text-retention rate: `{summary['identity_retention']['text_retention_rate']['last_half_mean']:.4f}`",
        f"- Last-half image-hijack rate: `{summary['identity_retention']['image_hijack_rate']['last_half_mean']:.4f}`",
        "",
        "## Interpretation",
        "- The mixture analysis tracks whether prompt+image global geometry is better explained by prompt-only or matched-image geometry at each layer.",
        "- The mismatched-image analysis tracks whether local text identity remains closer to the text target than to the mismatched image source.",
        "- A positive dissociation means image-dominant global geometry coexists with stable local text identity.",
    ]
    write_text(output_path("reports", "main_results", f"layerwise_global_local_dissociation_report{out_suffix}.md"), "\n".join(lines))
    append_run_log(
        f"Layerwise Global-Local Dissociation ({family})",
        [
            f"Wrote layerwise prompt-image mixture to {metrics_path(f'layerwise_prompt_image_mixture{out_suffix}.csv').relative_to(ROOT)}.",
            f"Wrote layerwise identity retention to {metrics_path(f'layerwise_mismatched_identity_retention{out_suffix}.csv').relative_to(ROOT)}.",
        ],
    )
    return summary


def combined_summary_row(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "family": summary["family"],
        "text_model_id": summary["text_model_id"],
        "multimodal_model_id": summary["model_id"],
        "num_concepts": summary["num_concepts"],
        "num_mixture_layers": summary["num_mixture_layers"],
        "num_retention_layers": summary["num_retention_layers"],
        "mean_prompt_weight": summary["mixture"]["prompt_weight"]["mean"],
        "last_half_prompt_weight": summary["mixture"]["prompt_weight"]["last_half_mean"],
        "mean_matched_image_weight": summary["mixture"]["matched_image_weight"]["mean"],
        "last_half_matched_image_weight": summary["mixture"]["matched_image_weight"]["last_half_mean"],
        "mean_text_retention_rate": summary["identity_retention"]["text_retention_rate"]["mean"],
        "last_half_text_retention_rate": summary["identity_retention"]["text_retention_rate"]["last_half_mean"],
        "mean_image_hijack_rate": summary["identity_retention"]["image_hijack_rate"]["mean"],
        "last_half_image_hijack_rate": summary["identity_retention"]["image_hijack_rate"]["last_half_mean"],
        "global_local_dissociation_present": summary["global_local_dissociation_present"],
    }


def write_combined_summary(summaries: list[dict[str, Any]], limit: int) -> None:
    rows = [combined_summary_row(summary) for summary in summaries]
    out_suffix = suffix(limit)
    write_csv(
        metrics_path(f"cross_family_global_local_summary{out_suffix}.csv"),
        rows,
        [
            "family",
            "text_model_id",
            "multimodal_model_id",
            "num_concepts",
            "num_mixture_layers",
            "num_retention_layers",
            "mean_prompt_weight",
            "last_half_prompt_weight",
            "mean_matched_image_weight",
            "last_half_matched_image_weight",
            "mean_text_retention_rate",
            "last_half_text_retention_rate",
            "mean_image_hijack_rate",
            "last_half_image_hijack_rate",
            "global_local_dissociation_present",
        ],
    )
    write_json(
        metrics_path(f"cross_family_global_local_summary{out_suffix}.json"),
        {
            "families": {summary["family"]: summary for summary in summaries},
            "rows": rows,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--family", choices=["qwen", "mistral", "llama", "all"], default="qwen")
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test concept/row limit. Writes *_smoke outputs.")
    parser.add_argument("--margin", type=float, default=0.01)
    args = parser.parse_args()

    metadata = json.loads(embeddings_path("embedding_metadata_full.json").read_text(encoding="utf-8"))
    arrays = np.load(embeddings_path("pooled_embeddings_full.npz"))
    lookup, layers_by_model = build_metadata_lookup(metadata)

    target_concepts = [row["concept"].lower() for row in load_active_concept_rows(args.config, domain="sensory")]
    if args.limit:
        target_concepts = target_concepts[: args.limit]

    summaries = []
    for family in families_to_run(args.family):
        summaries.append(
            run_family(
                family=family,
                config_path=args.config,
                limit=args.limit,
                margin=args.margin,
                metadata=metadata,
                arrays=arrays,
                lookup=lookup,
                layers_by_model=layers_by_model,
                target_concepts=target_concepts,
            )
        )
    if len(summaries) > 1:
        write_combined_summary(summaries, args.limit)


if __name__ == "__main__":
    main()
