from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from typing import Any, Callable

import numpy as np

from common import ROOT, append_run_log, metrics_path, percentile_interval, rankdata, read_csv, write_csv, write_json
from hardening_common import write_text


PREDICTOR = "source_attraction"
CONTINUOUS_ENDPOINTS = ["clip_target_margin", "visual_word_rate_per_100", "lancaster_visual_mean"]
BINARY_ENDPOINTS = ["clip_source_choice", "clip_target_choice", "mismatched_source_leakage"]
ALL_ENDPOINTS = CONTINUOUS_ENDPOINTS + BINARY_ENDPOINTS
CONTROL_NAMES = [
    "pair_image_similarity",
    "clip_anchor_pair_similarity",
    "dinov2_anchor_pair_similarity",
    "lancaster_pair_similarity",
    "lexical_similarity",
    "same_subtype",
    "same_coarse_category",
    "mismatch_mode_code",
]


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


def residualize(values: np.ndarray, covariates: list[np.ndarray]) -> np.ndarray:
    y = rankdata(np.asarray(values, dtype=float))
    y = y - y.mean()
    if not covariates:
        return y
    design = np.column_stack([rankdata(np.asarray(cov, dtype=float)) for cov in covariates])
    design = design - design.mean(axis=0, keepdims=True)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def normalize_concept(text: str) -> str:
    text = text.lower().replace("_", " ").replace("-", " ")
    return re.sub(r"[^a-z0-9 ]+", " ", text).strip()


def trigram_similarity(a: str, b: str) -> float:
    def trigrams(text: str) -> set[str]:
        padded = f"__{normalize_concept(text)}__"
        return {padded[idx : idx + 3] for idx in range(max(0, len(padded) - 2))}

    left = trigrams(a)
    right = trigrams(b)
    union = left | right
    if not union:
        return 0.0
    return float(len(left & right) / len(union))


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denom)


def load_anchor_similarity(filename: str, concepts_filename: str) -> Callable[[str, str], float]:
    matrix = np.load(ROOT / "data" / "anchors" / filename)
    concepts = [concept.lower() for concept in json.loads((ROOT / "data" / "anchors" / concepts_filename).read_text(encoding="utf-8"))]
    index = {concept: idx for idx, concept in enumerate(concepts)}

    def similarity(left: str, right: str) -> float:
        if left not in index or right not in index:
            return 0.0
        return cosine_similarity(matrix[index[left]], matrix[index[right]])

    return similarity


def load_lancaster_similarity() -> Callable[[str, str], float]:
    matrix = np.load(ROOT / "data" / "anchors" / "lancaster_perceptual_matrix.npy")
    concepts = [concept.lower() for concept in json.loads((ROOT / "data" / "anchors" / "lancaster_perceptual_concepts.json").read_text(encoding="utf-8"))]
    index = {concept: idx for idx, concept in enumerate(concepts)}

    def similarity(left: str, right: str) -> float:
        if left not in index or right not in index:
            return 0.0
        return cosine_similarity(matrix[index[left]], matrix[index[right]])

    return similarity


def load_metadata_maps() -> tuple[dict[str, dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    hierarchy = {row["concept"].lower(): row for row in read_csv(ROOT / "data" / "manifests" / "things_hierarchy_mapping.csv")}
    mismatch = {
        (row["concept"].lower(), row["mismatch_concept"].lower()): row
        for row in read_csv(ROOT / "data" / "manifests" / "mismatch_map.csv")
    }
    return hierarchy, mismatch


def bootstrap_ci(x: np.ndarray, y: np.ndarray, fn: Callable[[np.ndarray, np.ndarray], float], n_bootstrap: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(x), size=len(x))
        values.append(fn(x[idx], y[idx]))
    return percentile_interval(np.asarray(values, dtype=float), 0.95)


def join_rows() -> list[dict[str, Any]]:
    bridge = {row["concept"].lower(): row for row in read_csv(metrics_path("behavior_geometry_bridge_full.csv"))}
    clip = {
        row["concept"].lower(): row
        for row in read_csv(metrics_path("clip_forced_choice_behavior.csv"))
        if row["condition"] == "M_mismatched_image"
    }
    implicit = {row["concept"].lower(): row for row in read_csv(metrics_path("behavior_bridge_extensions_full_implicit_leakage.csv"))}
    hierarchy, mismatch = load_metadata_maps()
    clip_similarity = load_anchor_similarity("clip_vitl14_embeddings.npy", "clip_vitl14_concepts.json")
    dinov2_similarity = load_anchor_similarity("dinov2_embeddings.npy", "dinov2_concepts.json")
    lancaster_similarity = load_lancaster_similarity()
    concepts = sorted(set(bridge) & set(clip) & set(implicit))
    rows = []
    for concept in concepts:
        b = bridge[concept]
        c = clip[concept]
        i = implicit[concept]
        source = b["mismatch_source"].lower()
        target_h = hierarchy.get(concept, {})
        source_h = hierarchy.get(source, {})
        mismatch_row = mismatch.get((concept, source), {})
        same_subtype = int(bool(target_h) and bool(source_h) and target_h.get("subtype") == source_h.get("subtype"))
        same_coarse = int(bool(target_h) and bool(source_h) and target_h.get("coarse_category") == source_h.get("coarse_category"))
        mismatch_mode = mismatch_row.get("mismatch_mode", "")
        mismatch_mode_code = {"within_subtype": 1.0, "cross_subtype": 0.0}.get(mismatch_mode, 0.5)
        rows.append(
            {
                "concept": concept,
                "subtype": b["subtype"],
                "mismatch_source": b["mismatch_source"],
                "source_attraction": float(b["source_attraction"]),
                "source_minus_target_margin": float(b["source_minus_target_margin"]),
                "rdm_disruption": float(b["rdm_disruption"]),
                "pair_image_similarity": float(c["pair_image_similarity"]),
                "clip_anchor_pair_similarity": clip_similarity(concept, source),
                "dinov2_anchor_pair_similarity": dinov2_similarity(concept, source),
                "lancaster_pair_similarity": lancaster_similarity(concept, source),
                "lexical_similarity": trigram_similarity(concept, source),
                "same_subtype": same_subtype,
                "same_coarse_category": same_coarse,
                "mismatch_mode_code": mismatch_mode_code,
                "pair_difficulty": c["pair_difficulty"],
                "clip_target_margin": float(c["target_margin"]),
                "clip_target_choice": int(float(c["target_choice"])),
                "clip_source_choice": int(float(c["source_choice"])),
                "visual_word_rate_per_100": float(b["visual_word_rate_per_100"]),
                "lancaster_visual_mean": float(b["lancaster_visual_mean"]),
                "exemplar_specific_rate_per_100": float(b["exemplar_specific_rate_per_100"]),
                "mismatched_source_leakage": int(float(b["mismatched_source_leakage"])),
                "source_description_similarity": float(i["source_description_similarity"]),
                "source_minus_target_description_similarity": float(i["source_minus_target_description_similarity"]),
            }
        )
    return rows


def quartile_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = np.asarray([row[PREDICTOR] for row in rows], dtype=float)
    thresholds = np.quantile(values, [0.25, 0.5, 0.75])
    output = []
    groups = [
        ("bottom_quartile", values <= thresholds[0]),
        ("bottom_half", values <= thresholds[1]),
        ("top_half", values >= thresholds[1]),
        ("top_quartile", values >= thresholds[2]),
    ]
    for label, mask in groups:
        subset = [row for row, keep in zip(rows, mask) if keep]
        item = {"group": label, "n": len(subset), "source_attraction_min": min(row[PREDICTOR] for row in subset), "source_attraction_max": max(row[PREDICTOR] for row in subset)}
        for endpoint in ALL_ENDPOINTS:
            item[f"mean_{endpoint}"] = float(np.mean([row[endpoint] for row in subset]))
        output.append(item)
    by_label = {row["group"]: row for row in output}
    top = by_label["top_quartile"]
    bottom = by_label["bottom_quartile"]
    diff = {"group": "top_minus_bottom_quartile", "n": min(top["n"], bottom["n"]), "source_attraction_min": "", "source_attraction_max": ""}
    for endpoint in ALL_ENDPOINTS:
        diff[f"mean_{endpoint}"] = float(top[f"mean_{endpoint}"] - bottom[f"mean_{endpoint}"])
    output.append(diff)
    return output


def correlation_rows(rows: list[dict[str, Any]], n_bootstrap: int, seed: int) -> list[dict[str, Any]]:
    x = np.asarray([row[PREDICTOR] for row in rows], dtype=float)
    pair_sim = np.asarray([row["pair_image_similarity"] for row in rows], dtype=float)
    output = []
    for idx, endpoint in enumerate(ALL_ENDPOINTS):
        y = np.asarray([row[endpoint] for row in rows], dtype=float)
        is_binary = endpoint in BINARY_ENDPOINTS
        raw_fn = pearson_corr if is_binary else spearman_corr
        raw = raw_fn(x, y)
        raw_low, raw_high = bootstrap_ci(x, y, raw_fn, n_bootstrap, seed + idx)
        x_resid = residualize(x, [pair_sim])
        y_resid = residualize(y, [pair_sim])
        partial = pearson_corr(x_resid, y_resid)
        partial_low, partial_high = bootstrap_ci(x_resid, y_resid, pearson_corr, n_bootstrap, seed + 100 + idx)
        output.append(
            {
                "predictor": PREDICTOR,
                "endpoint": endpoint,
                "statistic": "point_biserial_r" if is_binary else "spearman_rho",
                "n": len(rows),
                "raw_estimate": raw,
                "raw_ci95_low": raw_low,
                "raw_ci95_high": raw_high,
                "controlled_for": "pair_image_similarity",
                "partial_estimate": partial,
                "partial_ci95_low": partial_low,
                "partial_ci95_high": partial_high,
            }
        )
    return output


def controlled_correlation_rows(rows: list[dict[str, Any]], n_bootstrap: int, seed: int) -> list[dict[str, Any]]:
    x = np.asarray([row[PREDICTOR] for row in rows], dtype=float)
    control_sets = [
        ("pair_image_similarity", ["pair_image_similarity"]),
        ("semantic_visual_anchors", ["clip_anchor_pair_similarity", "dinov2_anchor_pair_similarity", "lancaster_pair_similarity"]),
        ("lexical_category", ["lexical_similarity", "same_subtype", "same_coarse_category", "mismatch_mode_code"]),
        ("all_simple_proxies", CONTROL_NAMES),
    ]
    output = []
    for endpoint_idx, endpoint in enumerate(ALL_ENDPOINTS):
        y = np.asarray([row[endpoint] for row in rows], dtype=float)
        for control_idx, (control_label, control_names) in enumerate(control_sets):
            controls = [np.asarray([row[name] for row in rows], dtype=float) for name in control_names]
            x_resid = residualize(x, controls)
            y_resid = residualize(y, controls)
            partial = pearson_corr(x_resid, y_resid)
            low, high = bootstrap_ci(x_resid, y_resid, pearson_corr, n_bootstrap, seed + 1000 + endpoint_idx * 17 + control_idx)
            output.append(
                {
                    "predictor": PREDICTOR,
                    "endpoint": endpoint,
                    "controlled_for": control_label,
                    "control_terms": ",".join(control_names),
                    "n": len(rows),
                    "partial_estimate": partial,
                    "partial_ci95_low": low,
                    "partial_ci95_high": high,
                }
            )
    return output


def competing_predictor_rows(rows: list[dict[str, Any]], n_bootstrap: int, seed: int) -> list[dict[str, Any]]:
    predictors = [PREDICTOR, *CONTROL_NAMES]
    output = []
    for endpoint_idx, endpoint in enumerate(ALL_ENDPOINTS):
        y = np.asarray([row[endpoint] for row in rows], dtype=float)
        is_binary = endpoint in BINARY_ENDPOINTS
        fn = pearson_corr if is_binary else spearman_corr
        for predictor_idx, predictor in enumerate(predictors):
            x = np.asarray([row[predictor] for row in rows], dtype=float)
            estimate = fn(x, y)
            low, high = bootstrap_ci(x, y, fn, n_bootstrap, seed + 2000 + endpoint_idx * 31 + predictor_idx)
            output.append(
                {
                    "predictor": predictor,
                    "endpoint": endpoint,
                    "statistic": "point_biserial_r" if is_binary else "spearman_rho",
                    "n": len(rows),
                    "estimate": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return output


def stratified_rows(rows: list[dict[str, Any]], group_key: str, min_n: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if group_key == "subtype_collapsed":
        counts = Counter(row["subtype"] for row in rows)
        for row in rows:
            label = row["subtype"] if counts[row["subtype"]] >= min_n else "other"
            grouped[label].append(row)
    else:
        for row in rows:
            grouped[row[group_key]].append(row)
    output = []
    for group, subset in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(subset) < min_n and group != "other":
            continue
        x = np.asarray([row[PREDICTOR] for row in subset], dtype=float)
        for endpoint in ["clip_source_choice", "clip_target_margin", "mismatched_source_leakage", "lancaster_visual_mean"]:
            y = np.asarray([row[endpoint] for row in subset], dtype=float)
            fn = pearson_corr if endpoint in BINARY_ENDPOINTS else spearman_corr
            output.append(
                {
                    "moderator": group_key,
                    "group": group,
                    "endpoint": endpoint,
                    "n": len(subset),
                    "estimate": fn(x, y),
                    "mean_source_attraction": float(np.mean(x)),
                    "mean_endpoint": float(np.mean(y)),
                }
            )
    return output


def report_lines(summary: dict[str, Any]) -> list[str]:
    lines = [
        "# Behavior Moderator Analysis",
        "",
        f"- Joined concepts: `{summary['n_concepts']}`",
        "- Primary predictor: `source_attraction`.",
        "- Main covariate: `pair_image_similarity`.",
        "",
        "## Top-vs-Bottom Source-Attraction Quartiles",
        "",
        "| Endpoint | Bottom quartile | Top quartile | Top - bottom |",
        "|---|---:|---:|---:|",
    ]
    quartiles = {row["group"]: row for row in summary["quartile_rows"]}
    bottom = quartiles["bottom_quartile"]
    top = quartiles["top_quartile"]
    diff = quartiles["top_minus_bottom_quartile"]
    for endpoint in ALL_ENDPOINTS:
        lines.append(
            f"| `{endpoint}` | {bottom[f'mean_{endpoint}']:.4f} | {top[f'mean_{endpoint}']:.4f} | {diff[f'mean_{endpoint}']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Pair-Similarity Controlled Correlations",
            "",
            "| Endpoint | Raw | 95% CI | Controlled | 95% CI |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["partial_correlations"]:
        lines.append(
            f"| `{row['endpoint']}` | {row['raw_estimate']:+.4f} | "
            f"[{row['raw_ci95_low']:+.4f}, {row['raw_ci95_high']:+.4f}] | "
            f"{row['partial_estimate']:+.4f} | [{row['partial_ci95_low']:+.4f}, {row['partial_ci95_high']:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Full Simple-Proxy Controls",
            "",
            "| Endpoint | Pair image | Semantic/visual anchors | Lexical/category | All simple proxies |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    by_endpoint: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summary["full_control_partial_correlations"]:
        by_endpoint[row["endpoint"]][row["controlled_for"]] = row
    for endpoint in ALL_ENDPOINTS:
        values = []
        for label in ["pair_image_similarity", "semantic_visual_anchors", "lexical_category", "all_simple_proxies"]:
            row = by_endpoint[endpoint][label]
            values.append(f"{row['partial_estimate']:+.4f}")
        lines.append(f"| `{endpoint}` | {' | '.join(values)} |")
    lines.extend(["", "## Pair-Difficulty Moderator", "", "| Group | Endpoint | n | Estimate | Mean endpoint |", "|---|---|---:|---:|---:|"])
    for row in summary["pair_difficulty_rows"]:
        lines.append(f"| {row['group']} | `{row['endpoint']}` | {row['n']} | {row['estimate']:+.4f} | {row['mean_endpoint']:.4f} |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute behavior moderator and pair-similarity covariate analyses.")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--min-subtype-n", type=int, default=30)
    args = parser.parse_args()

    rows = join_rows()
    if len(rows) != 1854:
        raise RuntimeError(f"Expected 1854 joined concepts, got {len(rows)}")
    quartiles = quartile_rows(rows)
    partials = correlation_rows(rows, args.bootstrap, args.seed)
    full_control_partials = controlled_correlation_rows(rows, args.bootstrap, args.seed)
    competing_predictors = competing_predictor_rows(rows, args.bootstrap, args.seed)
    pair_difficulty = stratified_rows(rows, "pair_difficulty", min_n=1)
    subtypes = stratified_rows(rows, "subtype_collapsed", args.min_subtype_n)
    summary = {
        "n_concepts": len(rows),
        "predictor": PREDICTOR,
        "covariate": "pair_image_similarity",
        "quartile_rows": quartiles,
        "partial_correlations": partials,
        "full_control_partial_correlations": full_control_partials,
        "competing_predictor_correlations": competing_predictors,
        "pair_difficulty_rows": pair_difficulty,
        "subtype_rows": subtypes,
        "simple_proxy_controls": CONTROL_NAMES,
        "interpretation_note": "Moderator analyses are exploratory; the primary use is to show that the modest aggregate behavior bridge is concentrated in high source-attraction concepts and is not explained away by target-source CLIP image similarity, anchor-based semantic similarity, lexical overlap, or category relation.",
    }
    write_csv(
        metrics_path("behavior_moderator_quartiles.csv"),
        quartiles,
        ["group", "n", "source_attraction_min", "source_attraction_max", *[f"mean_{endpoint}" for endpoint in ALL_ENDPOINTS]],
    )
    write_csv(
        metrics_path("behavior_moderator_partial_correlations.csv"),
        partials,
        [
            "predictor",
            "endpoint",
            "statistic",
            "n",
            "raw_estimate",
            "raw_ci95_low",
            "raw_ci95_high",
            "controlled_for",
            "partial_estimate",
            "partial_ci95_low",
            "partial_ci95_high",
        ],
    )
    write_csv(
        metrics_path("behavior_moderator_full_control_partials.csv"),
        full_control_partials,
        ["predictor", "endpoint", "controlled_for", "control_terms", "n", "partial_estimate", "partial_ci95_low", "partial_ci95_high"],
    )
    write_csv(
        metrics_path("behavior_moderator_competing_predictors.csv"),
        competing_predictors,
        ["predictor", "endpoint", "statistic", "n", "estimate", "ci95_low", "ci95_high"],
    )
    write_csv(
        metrics_path("behavior_moderator_pair_difficulty.csv"),
        pair_difficulty,
        ["moderator", "group", "endpoint", "n", "estimate", "mean_source_attraction", "mean_endpoint"],
    )
    write_csv(
        metrics_path("behavior_moderator_subtypes.csv"),
        subtypes,
        ["moderator", "group", "endpoint", "n", "estimate", "mean_source_attraction", "mean_endpoint"],
    )
    write_json(metrics_path("behavior_moderator_summary.json"), summary)
    write_text(ROOT / "reports" / "main_results" / "behavior_moderator_report.md", "\n".join(report_lines(summary)))
    append_run_log(
        "Behavior Moderator Analysis",
        [
            f"Computed behavior moderator analysis for {len(rows)} concepts.",
            f"Bootstrap resamples: {args.bootstrap}.",
        ],
    )


if __name__ == "__main__":
    main()
