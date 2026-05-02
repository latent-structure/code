from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from typing import Any

import numpy as np

from analysis_common import ordered_embedding_for_concepts
from common import (
    append_run_log,
    canonical_condition_name,
    embeddings_path,
    metrics_path,
    output_path,
    percentile_interval,
    rankdata,
    read_csv,
    write_csv,
    write_json,
)
from hardening_common import condition_model_id, load_project_backbone, selected_layers, write_text


MATCHED_PREDICTORS = [
    "matched_vs_text_geometry_shift",
    "matched_vs_blank_geometry_shift",
    "matched_vs_text_rdm_disruption",
    "matched_vs_blank_rdm_disruption",
]

MATCHED_ENDPOINTS = [
    "matched_minus_text_visual_word_rate_per_100",
    "matched_minus_blank_visual_word_rate_per_100",
    "matched_minus_text_lancaster_visual_mean",
    "matched_minus_blank_lancaster_visual_mean",
    "matched_minus_text_exemplar_specific_rate_per_100",
    "matched_minus_blank_exemplar_specific_rate_per_100",
]

MISMATCH_PREDICTORS = [
    "source_attraction",
    "source_minus_target_margin",
    "source_description_similarity",
    "source_minus_target_description_similarity",
]

MISMATCH_ENDPOINTS = [
    "visual_word_rate_per_100",
    "lancaster_visual_mean",
    "exemplar_specific_rate_per_100",
    "mismatched_source_leakage",
]

TEXT_COLUMNS = ["generated_text_stripped"]


def cosine_distance_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom[denom == 0] = 1.0
    return 1.0 - np.sum(a * b, axis=1) / denom


def cosine_distance_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = matrix / norms
    return (1.0 - np.clip(normed @ normed.T, -1.0, 1.0)).astype(np.float32)


def rowwise_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=1, kind="mergesort")
    ranks = np.empty(order.shape, dtype=np.float32)
    ranks[np.arange(values.shape[0])[:, None], order] = np.arange(values.shape[1], dtype=np.float32)
    return ranks


def rowwise_spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    eye = np.eye(a.shape[0], dtype=bool)
    a_rows = a[~eye].reshape(a.shape[0], a.shape[0] - 1)
    b_rows = b[~eye].reshape(b.shape[0], b.shape[0] - 1)
    ar = rowwise_rank(a_rows)
    br = rowwise_rank(b_rows)
    ar = ar - ar.mean(axis=1, keepdims=True)
    br = br - br.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(ar, axis=1) * np.linalg.norm(br, axis=1)
    denom[denom == 0] = 1.0
    return np.sum(ar * br, axis=1) / denom


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return 0.0
    return float(np.dot(x, y) / denom)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return pearson_corr(rankdata(np.asarray(x, dtype=float)), rankdata(np.asarray(y, dtype=float)))


def bootstrap_ci(x: np.ndarray, y: np.ndarray, fn: Any, n_bootstrap: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(x), size=len(x))
        values.append(fn(x[idx], y[idx]))
    return percentile_interval(np.asarray(values, dtype=float), 0.95)


def permutation_p(x: np.ndarray, y: np.ndarray, observed: float, fn: Any, n_permutations: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_permutations):
        if abs(fn(x, rng.permutation(y))) >= abs(observed):
            count += 1
    return float((count + 1.0) / (n_permutations + 1.0))


def family_models(config_path: str, family_name: str) -> tuple[str, str, float]:
    config, backbone_text, backbone_multimodal, mid_fraction = load_project_backbone(config_path)
    if family_name == "qwen":
        return backbone_text, backbone_multimodal, mid_fraction
    for family in config["analysis"]["analysis"].get("cross_family_families", []):
        if str(family.get("family_name")) == family_name:
            return str(family["text_model"]), str(family["multimodal_model"]), mid_fraction
    raise RuntimeError(f"Unknown family `{family_name}` in config cross_family_families.")


def load_condition_embedding(condition: str, config_path: str, target_concepts: list[str], family_name: str) -> np.ndarray:
    backbone_text, backbone_multimodal, mid_fraction = family_models(config_path, family_name)
    metadata = json.loads(embeddings_path("embedding_metadata_full.json").read_text(encoding="utf-8"))
    layers_by_model: dict[str, list[int]] = {}
    records_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for record in metadata["records"]:
        if record["domain"] != "sensory":
            continue
        model_id_record = record["model_id"]
        layer = int(record["layer"])
        layers_by_model.setdefault(model_id_record, []).append(layer)
        records_by_key[(model_id_record, canonical_condition_name(record["condition"]), layer)] = record
    layers_by_model = {model_id: sorted(set(layers)) for model_id, layers in layers_by_model.items()}
    model_id = condition_model_id(backbone_text, backbone_multimodal, condition)
    matrices = []
    concepts: list[str] | None = None
    with np.load(embeddings_path("pooled_embeddings_full.npz"), allow_pickle=False) as pooled:
        for layer in selected_layers(layers_by_model[model_id], mid_fraction):
            record = records_by_key.get((model_id, condition, layer))
            if record is None:
                continue
            if concepts is None:
                concepts = [concept.lower() for concept in record["concepts"]]
            matrices.append(np.asarray(pooled[f"record_{record['record_id']}"], dtype=np.float32))
    if concepts is None or not matrices:
        raise RuntimeError(f"Missing embeddings for {model_id} {condition}")
    return ordered_embedding_for_concepts(np.mean(np.stack(matrices), axis=0, dtype=np.float32), concepts, target_concepts)


def token_counts(text: str) -> Counter[str]:
    import re

    return Counter(token for token in re.sub(r"[^a-z0-9 ]+", " ", text.lower().replace("_", " ").replace("-", " ")).split() if token)


def text_cosine(left: str, right: str) -> float:
    left_counts = token_counts(left)
    right_counts = token_counts(right)
    if not left_counts or not right_counts:
        return 0.0
    vocab = set(left_counts) | set(right_counts)
    dot = sum(left_counts[token] * right_counts[token] for token in vocab)
    left_norm = sum(value * value for value in left_counts.values()) ** 0.5
    right_norm = sum(value * value for value in right_counts.values()) ** 0.5
    denom = left_norm * right_norm
    return 0.0 if denom == 0 else float(dot / denom)


def index_scores(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["concept"].lower(), row["condition"]): row for row in rows}


def concept_rows(score_rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    concepts = {}
    for row in score_rows:
        concepts.setdefault(row["concept"].lower(), row)
    rows = [concepts[concept] for concept in sorted(concepts)]
    return rows[:limit] if limit else rows


def build_matched_gain_rows(score_rows: list[dict[str, str]], config_path: str, limit: int, family_name: str) -> list[dict[str, Any]]:
    rows = concept_rows(score_rows, limit)
    concepts = [row["concept"].lower() for row in rows]
    scores = index_scores(score_rows)
    matched = load_condition_embedding("M_matched_image", config_path, concepts, family_name)
    text_only = load_condition_embedding("M_text_only", config_path, concepts, family_name)
    blank = load_condition_embedding("M_blank_image", config_path, concepts, family_name)
    matched_rdm = cosine_distance_matrix(matched)
    text_rdm = cosine_distance_matrix(text_only)
    blank_rdm = cosine_distance_matrix(blank)
    matched_text_disruption = 1.0 - rowwise_spearman(matched_rdm, text_rdm)
    matched_blank_disruption = 1.0 - rowwise_spearman(matched_rdm, blank_rdm)
    matched_text_shift = cosine_distance_rows(matched, text_only)
    matched_blank_shift = cosine_distance_rows(matched, blank)

    output_rows = []
    for idx, row in enumerate(rows):
        concept = row["concept"].lower()
        matched_row = scores[(concept, "M_matched_image")]
        text_row = scores[(concept, "M_text_only")]
        blank_row = scores[(concept, "M_blank_image")]

        def diff(metric: str, right: dict[str, str]) -> float:
            return float(matched_row[metric]) - float(right[metric])

        output_rows.append(
            {
                "concept": row["concept"],
                "subtype": row["subtype"],
                "matched_vs_text_geometry_shift": float(matched_text_shift[idx]),
                "matched_vs_blank_geometry_shift": float(matched_blank_shift[idx]),
                "matched_vs_text_rdm_disruption": float(matched_text_disruption[idx]),
                "matched_vs_blank_rdm_disruption": float(matched_blank_disruption[idx]),
                "matched_minus_text_visual_word_rate_per_100": diff("visual_word_rate_per_100", text_row),
                "matched_minus_blank_visual_word_rate_per_100": diff("visual_word_rate_per_100", blank_row),
                "matched_minus_text_lancaster_visual_mean": diff("lancaster_visual_mean", text_row),
                "matched_minus_blank_lancaster_visual_mean": diff("lancaster_visual_mean", blank_row),
                "matched_minus_text_exemplar_specific_rate_per_100": diff("exemplar_specific_rate_per_100", text_row),
                "matched_minus_blank_exemplar_specific_rate_per_100": diff("exemplar_specific_rate_per_100", blank_row),
            }
        )
    return output_rows


def build_implicit_leakage_rows(score_rows: list[dict[str, str]], config_path: str, limit: int, family_name: str) -> list[dict[str, Any]]:
    rows = [row for row in score_rows if row["condition"] == "M_mismatched_image"]
    rows.sort(key=lambda row: row["concept"].lower())
    if limit:
        rows = rows[:limit]
    concepts = [row["concept"].lower() for row in rows]
    sources = [row["mismatch_source"].lower() for row in rows]
    all_needed = sorted(set(concepts) | set(sources))
    all_index = {concept: idx for idx, concept in enumerate(all_needed)}
    scores = index_scores(score_rows)

    matched_all = load_condition_embedding("M_matched_image", config_path, all_needed, family_name)
    mismatched_targets = load_condition_embedding("M_mismatched_image", config_path, concepts, family_name)
    matched_targets = matched_all[[all_index[concept] for concept in concepts]]
    matched_sources = matched_all[[all_index[source] for source in sources]]
    target_distance = cosine_distance_rows(mismatched_targets, matched_targets)
    source_distance = cosine_distance_rows(mismatched_targets, matched_sources)

    output_rows = []
    for idx, row in enumerate(rows):
        concept = row["concept"].lower()
        source = row["mismatch_source"].lower()
        mismatched_text = row["generated_text_stripped"]
        source_text = scores.get((source, "M_matched_image"), {}).get("generated_text_stripped", "")
        target_text = scores.get((concept, "M_matched_image"), {}).get("generated_text_stripped", "")
        source_similarity = text_cosine(mismatched_text, source_text)
        target_similarity = text_cosine(mismatched_text, target_text)
        output_rows.append(
            {
                "concept": row["concept"],
                "subtype": row["subtype"],
                "mismatch_source": row["mismatch_source"],
                "source_attraction": float(-source_distance[idx]),
                "source_minus_target_margin": float(source_distance[idx] - target_distance[idx]),
                "source_description_similarity": source_similarity,
                "target_description_similarity": target_similarity,
                "source_minus_target_description_similarity": source_similarity - target_similarity,
                "visual_word_rate_per_100": float(row["visual_word_rate_per_100"]),
                "lancaster_visual_mean": float(row["lancaster_visual_mean"]),
                "exemplar_specific_rate_per_100": float(row["exemplar_specific_rate_per_100"]),
                "mismatched_source_leakage": int(float(row["mismatched_source_leakage"])),
            }
        )
    return output_rows


def correlate_rows(
    rows: list[dict[str, Any]],
    predictors: list[str],
    endpoints: list[str],
    *,
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
    binary_endpoints: set[str] | None = None,
) -> list[dict[str, Any]]:
    binary_endpoints = binary_endpoints or set()
    correlation_rows = []
    for predictor_idx, predictor in enumerate(predictors):
        x = np.asarray([float(row[predictor]) for row in rows], dtype=float)
        for endpoint_idx, endpoint in enumerate(endpoints):
            y = np.asarray([float(row[endpoint]) for row in rows], dtype=float)
            is_binary = endpoint in binary_endpoints
            fn = pearson_corr if is_binary else spearman_corr
            estimate = fn(x, y)
            ci_low, ci_high = bootstrap_ci(x, y, fn, n_bootstrap, seed + 100 * predictor_idx + endpoint_idx)
            p_value = permutation_p(x, y, estimate, fn, n_permutations, seed + 10000 + 100 * predictor_idx + endpoint_idx)
            correlation_rows.append(
                {
                    "predictor": predictor,
                    "endpoint": endpoint,
                    "statistic": "point_biserial_r" if is_binary else "spearman_rho",
                    "n": len(rows),
                    "estimate": estimate,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "permutation_p": p_value,
                }
            )
    return correlation_rows


def report_lines(title: str, summary: dict[str, Any], correlations: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"# {title}",
        "",
        f"- Concepts: `{summary['n_concepts']}`",
        f"- Bootstrap resamples: `{summary['n_bootstrap']}`",
        f"- Permutations: `{summary['n_permutations']}`",
        f"- Note: {summary['method_note']}",
        "",
        "| Predictor | Endpoint | Statistic | Estimate | 95% CI | p |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in correlations:
        lines.append(
            f"| `{row['predictor']}` | `{row['endpoint']}` | `{row['statistic']}` | "
            f"{row['estimate']:.4f} | [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | {row['permutation_p']:.4f} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute matched-gain and implicit-leakage behavior bridge extensions.")
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--scores", default="outputs/metrics/behavior_probe_v2_full_scores.csv")
    parser.add_argument("--output-stem", default="behavior_bridge_extensions_full")
    parser.add_argument("--family", default="qwen", choices=["qwen", "mistral", "llama"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260424)
    args = parser.parse_args()

    score_rows = read_csv(output_path(args.scores))
    suffix = "_smoke" if args.limit else ""

    matched_rows = build_matched_gain_rows(score_rows, args.config, args.limit, args.family)
    matched_correlations = correlate_rows(
        matched_rows,
        MATCHED_PREDICTORS,
        MATCHED_ENDPOINTS,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        seed=args.seed,
    )
    matched_summary = {
        "n_concepts": len(matched_rows),
        "n_bootstrap": args.bootstrap,
        "n_permutations": args.permutations,
        "method_note": "Per-concept matched-image geometry shift is correlated with matched-minus-control generated-output visualness gains.",
        "correlations": matched_correlations,
    }
    write_csv(
        metrics_path(f"{args.output_stem}_matched_gain{suffix}.csv"),
        matched_rows,
        ["concept", "subtype", *MATCHED_PREDICTORS, *MATCHED_ENDPOINTS],
    )
    write_csv(
        metrics_path(f"{args.output_stem}_matched_gain_correlations{suffix}.csv"),
        matched_correlations,
        ["predictor", "endpoint", "statistic", "n", "estimate", "ci95_low", "ci95_high", "permutation_p"],
    )
    write_json(metrics_path(f"{args.output_stem}_matched_gain_summary{suffix}.json"), matched_summary)
    write_text(
        output_path("reports", "main_results", f"{args.output_stem}_matched_gain_report{suffix}.md"),
        "\n".join(report_lines("Matched-Image Behavior Gain Bridge", matched_summary, matched_correlations)),
    )

    leakage_rows = build_implicit_leakage_rows(score_rows, args.config, args.limit, args.family)
    leakage_correlations = correlate_rows(
        leakage_rows,
        MISMATCH_PREDICTORS,
        MISMATCH_ENDPOINTS,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        seed=args.seed + 5000,
        binary_endpoints={"mismatched_source_leakage"},
    )
    leakage_summary = {
        "n_concepts": len(leakage_rows),
        "n_source_leakage_positive": sum(int(row["mismatched_source_leakage"]) for row in leakage_rows),
        "source_leakage_rate": float(np.mean([int(row["mismatched_source_leakage"]) for row in leakage_rows])) if leakage_rows else 0.0,
        "n_bootstrap": args.bootstrap,
        "n_permutations": args.permutations,
        "method_note": "Implicit leakage is approximated by bag-of-words cosine similarity between mismatched output and the source concept's matched-image output.",
        "correlations": leakage_correlations,
    }
    write_csv(
        metrics_path(f"{args.output_stem}_implicit_leakage{suffix}.csv"),
        leakage_rows,
        [
            "concept",
            "subtype",
            "mismatch_source",
            *MISMATCH_PREDICTORS,
            "target_description_similarity",
            *MISMATCH_ENDPOINTS,
        ],
    )
    write_csv(
        metrics_path(f"{args.output_stem}_implicit_leakage_correlations{suffix}.csv"),
        leakage_correlations,
        ["predictor", "endpoint", "statistic", "n", "estimate", "ci95_low", "ci95_high", "permutation_p"],
    )
    write_json(metrics_path(f"{args.output_stem}_implicit_leakage_summary{suffix}.json"), leakage_summary)
    write_text(
        output_path("reports", "main_results", f"{args.output_stem}_implicit_leakage_report{suffix}.md"),
        "\n".join(report_lines("Implicit Mismatched-Image Leakage Bridge", leakage_summary, leakage_correlations)),
    )
    append_run_log(
        "Behavior Bridge Extensions",
        [
            f"Wrote matched-gain bridge for {len(matched_rows)} concepts.",
            f"Wrote implicit-leakage bridge for {len(leakage_rows)} concepts.",
        ],
    )


if __name__ == "__main__":
    main()
