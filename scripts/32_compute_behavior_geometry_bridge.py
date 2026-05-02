from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from analysis_common import ordered_embedding_for_concepts
from common import append_run_log, canonical_condition_name, embeddings_path, metrics_path, output_path, percentile_interval, rankdata, read_csv, write_csv, write_json
from hardening_common import condition_model_id, load_project_backbone, selected_layers, write_text


PREDICTORS = [
    "target_perturbation",
    "source_attraction",
    "source_minus_target_margin",
    "rdm_disruption",
]

CONTINUOUS_ENDPOINTS = [
    "visual_word_rate_per_100",
    "lancaster_visual_mean",
    "lancaster_visual_per_100",
    "exemplar_specific_rate_per_100",
]

BINARY_ENDPOINTS = ["mismatched_source_leakage", "target_retention"]


def rowwise_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=1, kind="mergesort")
    ranks = np.empty(order.shape, dtype=np.float32)
    row_ids = np.arange(values.shape[0])[:, None]
    ranks[row_ids, order] = np.arange(values.shape[1], dtype=np.float32)
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


def cosine_distance_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = matrix / norms
    return (1.0 - np.clip(normed @ normed.T, -1.0, 1.0)).astype(np.float32)


def cosine_distance_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    denom = a_norm * b_norm
    denom[denom == 0] = 1.0
    return 1.0 - np.sum(a * b, axis=1) / denom


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
        permuted = rng.permutation(y)
        if abs(fn(x, permuted)) >= abs(observed):
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
    layers = selected_layers(layers_by_model[model_id], mid_fraction)
    matrices = []
    concepts: list[str] | None = None
    with np.load(embeddings_path("pooled_embeddings_full.npz"), allow_pickle=False) as pooled:
        for layer in layers:
            record = records_by_key.get((model_id, condition, layer))
            if record is None:
                continue
            if concepts is None:
                concepts = [concept.lower() for concept in record["concepts"]]
            matrices.append(np.asarray(pooled[f"record_{record['record_id']}"], dtype=np.float32))
    if concepts is None or not matrices:
        raise RuntimeError(f"Missing embeddings for {model_id} {condition}")
    matrix = np.mean(np.stack(matrices), axis=0, dtype=np.float32)
    return ordered_embedding_for_concepts(matrix, concepts, target_concepts)


def build_concept_rows(score_rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    rows = [row for row in score_rows if row["condition"] == "M_mismatched_image"]
    rows.sort(key=lambda row: row["concept"].lower())
    if limit:
        rows = rows[:limit]
    return rows


def bridge_rows(score_rows: list[dict[str, str]], config_path: str, limit: int, family_name: str) -> list[dict[str, Any]]:
    rows = build_concept_rows(score_rows, limit)
    concepts = [row["concept"].lower() for row in rows]
    sources = [row["mismatch_source"].lower() for row in rows]
    all_needed = sorted(set(concepts) | set(sources))
    all_index = {concept: idx for idx, concept in enumerate(all_needed)}

    matched_all = load_condition_embedding("M_matched_image", config_path, all_needed, family_name)
    mismatched_targets = load_condition_embedding("M_mismatched_image", config_path, concepts, family_name)
    matched_targets = matched_all[[all_index[concept] for concept in concepts]]
    matched_sources = matched_all[[all_index[source] for source in sources]]

    target_distance = cosine_distance_rows(mismatched_targets, matched_targets)
    source_distance = cosine_distance_rows(mismatched_targets, matched_sources)
    matched_target_rdm = cosine_distance_matrix(matched_targets)
    mismatched_target_rdm = cosine_distance_matrix(mismatched_targets)
    row_similarity = rowwise_spearman(matched_target_rdm, mismatched_target_rdm)

    output_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        output_rows.append(
            {
                "concept": row["concept"],
                "subtype": row["subtype"],
                "mismatch_source": row["mismatch_source"],
                "target_perturbation": float(target_distance[idx]),
                "source_attraction": float(-source_distance[idx]),
                "source_minus_target_margin": float(source_distance[idx] - target_distance[idx]),
                "rdm_disruption": float(1.0 - row_similarity[idx]),
                **{endpoint: float(row[endpoint]) for endpoint in CONTINUOUS_ENDPOINTS},
                **{endpoint: int(float(row[endpoint])) for endpoint in BINARY_ENDPOINTS},
            }
        )
    return output_rows


def summarize(rows: list[dict[str, Any]], n_bootstrap: int, n_permutations: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    correlation_rows: list[dict[str, Any]] = []
    endpoints = CONTINUOUS_ENDPOINTS + BINARY_ENDPOINTS
    for predictor_idx, predictor in enumerate(PREDICTORS):
        x = np.asarray([float(row[predictor]) for row in rows], dtype=float)
        for endpoint_idx, endpoint in enumerate(endpoints):
            y = np.asarray([float(row[endpoint]) for row in rows], dtype=float)
            is_binary = endpoint in BINARY_ENDPOINTS
            fn = pearson_corr if is_binary else spearman_corr
            stat_name = "point_biserial_r" if is_binary else "spearman_rho"
            value = fn(x, y)
            ci_low, ci_high = bootstrap_ci(x, y, fn, n_bootstrap, seed + 100 * predictor_idx + endpoint_idx)
            p_value = permutation_p(x, y, value, fn, n_permutations, seed + 10000 + 100 * predictor_idx + endpoint_idx)
            correlation_rows.append(
                {
                    "predictor": predictor,
                    "endpoint": endpoint,
                    "statistic": stat_name,
                    "n": len(rows),
                    "estimate": value,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "permutation_p": p_value,
                }
            )

    leakage_count = sum(int(row["mismatched_source_leakage"]) for row in rows)
    summary = {
        "n_concepts": len(rows),
        "n_source_leakage_positive": leakage_count,
        "source_leakage_rate": leakage_count / len(rows) if rows else 0.0,
        "primary_endpoints": CONTINUOUS_ENDPOINTS,
        "secondary_endpoints": BINARY_ENDPOINTS,
        "predictors": PREDICTORS,
        "precision_note": (
            "The primary bridge uses continuous lexical and Lancaster endpoints because explicit source leakage can be sparse. "
            "Rowwise RDM disruption is computed by ranking rows of full matched and mismatched distance matrices and correlating "
            "1,853 off-diagonal distances per concept in a vectorized pass."
        ),
        "interpretation_rule": (
            "Continuous endpoint correlations are the primary behavior bridge. Binary source leakage is interpreted only as "
            "secondary support unless the number of positive leakage cases is large enough for stable estimates."
        ),
    }
    return correlation_rows, summary


def report_lines(summary: dict[str, Any], correlations: list[dict[str, Any]]) -> list[str]:
    lines = [
        "# Behavior-Geometry Bridge Report",
        "",
        f"- Concepts: `{summary['n_concepts']}`",
        f"- Explicit source-leakage positives: `{summary['n_source_leakage_positive']}`",
        f"- Explicit source-leakage rate: `{summary['source_leakage_rate']:.4f}`",
        f"- Precision note: {summary['precision_note']}",
        "",
        "## Correlations",
        "",
        "| Predictor | Endpoint | Statistic | Estimate | 95% CI | p |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in correlations:
        lines.append(
            f"| `{row['predictor']}` | `{row['endpoint']}` | `{row['statistic']}` | "
            f"{row['estimate']:.4f} | [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | {row['permutation_p']:.4f} |"
        )
    lines.extend(["", f"## Interpretation Rule", "", summary["interpretation_rule"]])
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Link mismatched-image geometry perturbation to output-level behavior.")
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--scores", default="outputs/metrics/behavior_probe_v2_full_scores.csv")
    parser.add_argument("--output-stem", default="behavior_geometry_bridge_full")
    parser.add_argument("--family", default="qwen", choices=["qwen", "mistral", "llama"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260424)
    args = parser.parse_args()

    score_rows = read_csv(output_path(args.scores))
    rows = bridge_rows(score_rows, args.config, args.limit, args.family)
    correlations, summary = summarize(rows, args.bootstrap, args.permutations, args.seed)
    suffix = "_smoke" if args.limit else ""
    write_csv(
        metrics_path(f"{args.output_stem}{suffix}.csv"),
        rows,
        ["concept", "subtype", "mismatch_source", *PREDICTORS, *CONTINUOUS_ENDPOINTS, *BINARY_ENDPOINTS],
    )
    write_csv(
        metrics_path(f"{args.output_stem}_correlations{suffix}.csv"),
        correlations,
        ["predictor", "endpoint", "statistic", "n", "estimate", "ci95_low", "ci95_high", "permutation_p"],
    )
    write_json(metrics_path(f"{args.output_stem}_summary{suffix}.json"), {**summary, "correlations": correlations})
    write_text(output_path("reports", "main_results", f"{args.output_stem}_report{suffix}.md"), "\n".join(report_lines(summary, correlations)))
    append_run_log(
        "Behavior-Geometry Bridge",
        [
            f"Wrote behavior-geometry bridge for {len(rows)} mismatched concepts.",
            f"Bootstrap resamples: {args.bootstrap}; permutations: {args.permutations}.",
        ],
    )


if __name__ == "__main__":
    main()
