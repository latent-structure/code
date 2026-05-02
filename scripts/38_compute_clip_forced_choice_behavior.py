from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from common import (
    ROOT,
    append_run_log,
    metrics_path,
    percentile_interval,
    rankdata,
    read_csv,
    write_csv,
    write_json,
)
from hardening_common import resolve_cached_snapshot, write_text


PRIMARY_CONDITIONS = ["M_matched_image", "M_mismatched_image"]
DEFAULT_CONDITIONS = [
    "M_text_only",
    "M_matched_image",
    "M_prompt_plus_matched_image",
    "M_mismatched_image",
    "M_blank_image",
]


def normalize_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["concept"].lower(), row["condition"]): row for row in rows}


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0.0:
        return 0.0
    return float(np.dot(x, y) / denom)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return pearson_corr(rankdata(np.asarray(x, dtype=float)), rankdata(np.asarray(y, dtype=float)))


def bootstrap_ci(values: np.ndarray, fn: Callable[[np.ndarray], float], n_bootstrap: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(values), size=len(values))
        estimates.append(fn(values[idx]))
    return percentile_interval(np.asarray(estimates, dtype=float), 0.95)


def bootstrap_corr_ci(
    x: np.ndarray,
    y: np.ndarray,
    fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(x), size=len(x))
        estimates.append(fn(x[idx], y[idx]))
    return percentile_interval(np.asarray(estimates, dtype=float), 0.95)


def paired_permutation_p(differences: np.ndarray, observed: float, n_permutations: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(differences))
        if abs(float(np.mean(differences * signs))) >= abs(observed):
            count += 1
    return float((count + 1.0) / (n_permutations + 1.0))


def permutation_p(
    x: np.ndarray,
    y: np.ndarray,
    observed: float,
    fn: Callable[[np.ndarray, np.ndarray], float],
    n_permutations: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_permutations):
        if abs(fn(x, rng.permutation(y))) >= abs(observed):
            count += 1
    return float((count + 1.0) / (n_permutations + 1.0))


def load_clip_image_embeddings() -> tuple[list[str], np.ndarray]:
    concepts = [concept.lower() for concept in json.loads((ROOT / "data" / "anchors" / "clip_vitl14_concepts.json").read_text(encoding="utf-8"))]
    embeddings = np.load(ROOT / "data" / "anchors" / "clip_vitl14_embeddings.npy")
    if len(concepts) != embeddings.shape[0]:
        raise RuntimeError(f"CLIP concept count {len(concepts)} did not match image embeddings {embeddings.shape}.")
    return concepts, l2_normalize(embeddings)


def load_image_manifest() -> dict[str, Path]:
    rows = read_csv(ROOT / "data" / "manifests" / "image_manifest.csv")
    return {row["concept"].lower(): ROOT / row["matched_image"] for row in rows if row["status"] == "ready"}


def resolve_clip_model_source(model_id: str, explicit_path: str | None, cache_root: str) -> str:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise RuntimeError(f"Explicit CLIP model path does not exist: {path}")
        return str(path)
    local_path = Path(cache_root).expanduser() / "models--openai--clip-vit-large-patch14"
    if local_path.exists():
        return str(local_path)
    return str(resolve_cached_snapshot(model_id, cache_root))


def tensor_from_clip_output(features: Any, *, output_name: str) -> Any:
    if hasattr(features, "detach"):
        return features
    if getattr(features, f"{output_name}_embeds", None) is not None:
        return getattr(features, f"{output_name}_embeds")
    if getattr(features, "pooler_output", None) is not None:
        return features.pooler_output
    if getattr(features, "last_hidden_state", None) is not None:
        return features.last_hidden_state[:, 0]
    raise RuntimeError(f"Unsupported CLIP {output_name} feature return type: {type(features).__name__}")


def load_clip_components(
    model_id: str,
    model_path: str | None,
    cache_root: str,
) -> tuple[Any, Any, Any, Any, str]:
    import torch
    import transformers

    source = resolve_clip_model_source(model_id, model_path, cache_root)
    processor = transformers.AutoProcessor.from_pretrained(source, local_files_only=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(source, local_files_only=True)
    model = transformers.CLIPModel.from_pretrained(source, local_files_only=True).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return processor, tokenizer, model, device, source


def encode_texts(
    texts: list[str],
    *,
    tokenizer: Any,
    model: Any,
    device: Any,
    batch_size: int,
) -> np.ndarray:
    vectors = []
    import torch

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch = tokenizer(batch_texts, padding=True, truncation=True, max_length=77, return_tensors="pt")
            batch = {key: value.to(device) for key, value in batch.items()}
            features = model.get_text_features(**batch)
            features = tensor_from_clip_output(features, output_name="text")
            vectors.append(features.detach().float().cpu().numpy().astype(np.float32))
    return l2_normalize(np.vstack(vectors))


def encode_images(
    concepts: list[str],
    *,
    processor: Any,
    model: Any,
    device: Any,
    batch_size: int,
) -> tuple[list[str], np.ndarray]:
    from PIL import Image
    import torch

    image_manifest = load_image_manifest()
    vectors = []
    for start in range(0, len(concepts), batch_size):
        batch_concepts = concepts[start : start + batch_size]
        images = []
        for concept in batch_concepts:
            if concept not in image_manifest:
                raise RuntimeError(f"Missing matched image for CLIP forced choice concept: {concept}")
            images.append(Image.open(image_manifest[concept]).convert("RGB"))
        batch = processor(images=images, return_tensors="pt")
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            features = model.get_image_features(**batch)
        features = tensor_from_clip_output(features, output_name="image")
        vectors.append(features.detach().float().cpu().numpy().astype(np.float32))
    return concepts, l2_normalize(np.vstack(vectors))


def difficulty_labels(pair_similarities: np.ndarray) -> list[str]:
    low, high = np.quantile(pair_similarities, [1.0 / 3.0, 2.0 / 3.0])
    labels = []
    for value in pair_similarities:
        if value <= low:
            labels.append("distant")
        elif value <= high:
            labels.append("medium")
        else:
            labels.append("similar")
    return labels


def build_forced_choice_rows(
    score_rows: list[dict[str, str]],
    conditions: list[str],
    text_vectors: np.ndarray,
    clip_concepts: list[str],
    image_vectors: np.ndarray,
) -> list[dict[str, Any]]:
    score_index = normalize_rows(score_rows)
    mismatch_rows = [row for row in score_rows if row["condition"] == "M_mismatched_image"]
    mismatch_source = {row["concept"].lower(): row["mismatch_source"].lower() for row in mismatch_rows}
    concept_index = {concept: idx for idx, concept in enumerate(clip_concepts)}
    ordered = []
    text_idx = 0
    for condition in conditions:
        rows = [row for row in score_rows if row["condition"] == condition]
        rows.sort(key=lambda row: row["concept"].lower())
        for row in rows:
            concept = row["concept"].lower()
            source = row["mismatch_source"].lower() if condition == "M_mismatched_image" else mismatch_source.get(concept, "")
            if concept not in concept_index or source not in concept_index:
                raise RuntimeError(f"Missing CLIP image embedding for concept/source: {concept} / {source}")
            target_vec = image_vectors[concept_index[concept]]
            source_vec = image_vectors[concept_index[source]]
            text_vec = text_vectors[text_idx]
            target_sim = float(np.dot(text_vec, target_vec))
            source_sim = float(np.dot(text_vec, source_vec))
            pair_sim = float(np.dot(target_vec, source_vec))
            target_margin = target_sim - source_sim
            ordered.append(
                {
                    "concept": row["concept"],
                    "subtype": row["subtype"],
                    "condition": condition,
                    "mismatch_source": source,
                    "target_similarity": target_sim,
                    "source_similarity": source_sim,
                    "target_margin": target_margin,
                    "target_choice": int(target_margin > 0),
                    "source_choice": int(target_margin < 0),
                    "pair_image_similarity": pair_sim,
                    "pair_difficulty": "",
                }
            )
            text_idx += 1
    labels_by_concept = {
        row["concept"]: label
        for row, label in zip(
            [row for row in ordered if row["condition"] == conditions[0]],
            difficulty_labels(np.asarray([row["pair_image_similarity"] for row in ordered if row["condition"] == conditions[0]], dtype=float)),
        )
    }
    for row in ordered:
        row["pair_difficulty"] = labels_by_concept[row["concept"]]
    return ordered


def summarize_condition(rows: list[dict[str, Any]], n_bootstrap: int, seed: int) -> list[dict[str, Any]]:
    summaries = []
    for condition in sorted({row["condition"] for row in rows}):
        subset = [row for row in rows if row["condition"] == condition]
        target_choice = np.asarray([row["target_choice"] for row in subset], dtype=float)
        source_choice = np.asarray([row["source_choice"] for row in subset], dtype=float)
        margins = np.asarray([row["target_margin"] for row in subset], dtype=float)
        target_ci = bootstrap_ci(target_choice, lambda values: float(np.mean(values)), n_bootstrap, seed + len(summaries))
        margin_ci = bootstrap_ci(margins, lambda values: float(np.mean(values)), n_bootstrap, seed + 100 + len(summaries))
        summaries.append(
            {
                "condition": condition,
                "pair_difficulty": "all",
                "n": len(subset),
                "target_choice_rate": float(np.mean(target_choice)),
                "target_choice_ci95_low": target_ci[0],
                "target_choice_ci95_high": target_ci[1],
                "source_choice_rate": float(np.mean(source_choice)),
                "mean_target_margin": float(np.mean(margins)),
                "target_margin_ci95_low": margin_ci[0],
                "target_margin_ci95_high": margin_ci[1],
                "median_target_margin": float(np.median(margins)),
                "chance_target_choice_rate": 0.5,
            }
        )
        for difficulty in ["distant", "medium", "similar"]:
            diff_subset = [row for row in subset if row["pair_difficulty"] == difficulty]
            if not diff_subset:
                continue
            diff_choice = np.asarray([row["target_choice"] for row in diff_subset], dtype=float)
            diff_margin = np.asarray([row["target_margin"] for row in diff_subset], dtype=float)
            summaries.append(
                {
                    "condition": condition,
                    "pair_difficulty": difficulty,
                    "n": len(diff_subset),
                    "target_choice_rate": float(np.mean(diff_choice)),
                    "target_choice_ci95_low": "",
                    "target_choice_ci95_high": "",
                    "source_choice_rate": float(np.mean([row["source_choice"] for row in diff_subset])),
                    "mean_target_margin": float(np.mean(diff_margin)),
                    "target_margin_ci95_low": "",
                    "target_margin_ci95_high": "",
                    "median_target_margin": float(np.median(diff_margin)),
                    "chance_target_choice_rate": 0.5,
                }
            )
    return summaries


def paired_contrasts(rows: list[dict[str, Any]], n_bootstrap: int, n_permutations: int, seed: int) -> list[dict[str, Any]]:
    by_key = {(row["concept"].lower(), row["condition"]): row for row in rows}
    concepts = sorted({row["concept"].lower() for row in rows if row["condition"] == "M_matched_image"})
    output = []
    for metric in ["target_choice", "target_margin"]:
        differences = []
        for concept in concepts:
            differences.append(float(by_key[(concept, "M_matched_image")][metric]) - float(by_key[(concept, "M_mismatched_image")][metric]))
        diffs = np.asarray(differences, dtype=float)
        estimate = float(np.mean(diffs))
        ci_low, ci_high = bootstrap_ci(diffs, lambda values: float(np.mean(values)), n_bootstrap, seed)
        p_value = paired_permutation_p(diffs, estimate, n_permutations, seed + 1000)
        output.append(
            {
                "contrast": "M_matched_image_minus_M_mismatched_image",
                "metric": metric,
                "n": len(diffs),
                "estimate": estimate,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "permutation_p": p_value,
            }
        )
    return output


def geometry_correlations(
    rows: list[dict[str, Any]],
    bridge_rows: list[dict[str, str]],
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
) -> list[dict[str, Any]]:
    mismatch = {row["concept"].lower(): row for row in rows if row["condition"] == "M_mismatched_image"}
    predictors = ["source_attraction", "source_minus_target_margin"]
    endpoints = ["target_margin", "target_choice", "source_choice"]
    output = []
    for predictor_idx, predictor in enumerate(predictors):
        x_values = []
        y_by_endpoint = {endpoint: [] for endpoint in endpoints}
        for bridge in bridge_rows:
            concept = bridge["concept"].lower()
            if concept not in mismatch:
                continue
            x_values.append(float(bridge[predictor]))
            for endpoint in endpoints:
                y_by_endpoint[endpoint].append(float(mismatch[concept][endpoint]))
        x = np.asarray(x_values, dtype=float)
        for endpoint_idx, endpoint in enumerate(endpoints):
            y = np.asarray(y_by_endpoint[endpoint], dtype=float)
            is_binary = endpoint in {"target_choice", "source_choice"}
            fn = pearson_corr if is_binary else spearman_corr
            estimate = fn(x, y)
            ci_low, ci_high = bootstrap_corr_ci(x, y, fn, n_bootstrap, seed + 100 * predictor_idx + endpoint_idx)
            p_value = permutation_p(x, y, estimate, fn, n_permutations, seed + 5000 + 100 * predictor_idx + endpoint_idx)
            output.append(
                {
                    "predictor": predictor,
                    "endpoint": endpoint,
                    "statistic": "point_biserial_r" if is_binary else "spearman_rho",
                    "n": len(x),
                    "estimate": estimate,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "permutation_p": p_value,
                }
            )
    return output


def report_lines(summary: dict[str, Any]) -> list[str]:
    lines = [
        "# CLIP Forced-Choice Behavior",
        "",
        f"- CLIP text model source: `{summary['clip_model_source']}`",
        f"- CLIP dimension: `{summary['clip_dimension']}`",
        f"- Chance target-choice baseline: `{summary['chance_target_choice_rate']:.2f}`",
        f"- Bootstrap resamples: `{summary['n_bootstrap']}`",
        f"- Permutations: `{summary['n_permutations']}`",
        "",
        "## Condition Summaries",
        "",
        "| Condition | Difficulty | n | Target choice | Source choice | Mean margin | 95% CI margin |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["condition_summaries"]:
        margin_ci = (
            ""
            if row["target_margin_ci95_low"] == ""
            else f"[{row['target_margin_ci95_low']:+.4f}, {row['target_margin_ci95_high']:+.4f}]"
        )
        lines.append(
            f"| `{row['condition']}` | {row['pair_difficulty']} | {row['n']} | "
            f"{row['target_choice_rate']:.4f} | {row['source_choice_rate']:.4f} | "
            f"{row['mean_target_margin']:+.4f} | {margin_ci} |"
        )
    lines.extend(
        [
            "",
            "## Primary Paired Contrasts",
            "",
            "| Contrast | Metric | Estimate | 95% CI | p |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in summary["primary_contrasts"]:
        lines.append(
            f"| `{row['contrast']}` | `{row['metric']}` | {row['estimate']:+.4f} | "
            f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | {row['permutation_p']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Geometry Correlations",
            "",
            "| Predictor | Endpoint | Statistic | Estimate | 95% CI | p |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in summary["geometry_correlations"]:
        lines.append(
            f"| `{row['predictor']}` | `{row['endpoint']}` | `{row['statistic']}` | "
            f"{row['estimate']:+.4f} | [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | {row['permutation_p']:.4f} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CLIP forced-choice perceptual consistency for generated descriptions.")
    parser.add_argument("--scores", default="outputs/metrics/behavior_probe_v2_full_scores.csv")
    parser.add_argument("--bridge", default="outputs/metrics/behavior_bridge_extensions_full_implicit_leakage.csv")
    parser.add_argument("--output-stem", default="clip_forced_choice_behavior")
    parser.add_argument("--model-id", default="openai/clip-vit-large-patch14")
    parser.add_argument("--model-path")
    parser.add_argument("--cache-root", default=os.environ.get("HF_HOME", ".cache/hf"))
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260429)
    args = parser.parse_args()

    score_rows = read_csv(ROOT / args.scores)
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    if not all(condition in conditions for condition in PRIMARY_CONDITIONS):
        raise RuntimeError(f"Conditions must include primary conditions: {PRIMARY_CONDITIONS}")
    concepts = sorted({row["concept"].lower() for row in score_rows})
    if args.limit:
        rng = np.random.default_rng(args.seed)
        concepts = sorted(rng.choice(np.asarray(concepts, dtype=object), size=args.limit, replace=False).tolist())
    concept_set = set(concepts)
    selected_rows = [
        row
        for condition in conditions
        for row in sorted(
            [item for item in score_rows if item["condition"] == condition and item["concept"].lower() in concept_set],
            key=lambda item: item["concept"].lower(),
        )
    ]
    processor, tokenizer, model, device, clip_source = load_clip_components(
        model_id=args.model_id,
        model_path=args.model_path,
        cache_root=args.cache_root,
    )
    texts = [row["generated_text_stripped"] for row in selected_rows]
    text_vectors = encode_texts(
        texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )
    mismatch_sources = {
        row["concept"].lower(): row["mismatch_source"].lower()
        for row in score_rows
        if row["condition"] == "M_mismatched_image" and row["concept"].lower() in concept_set
    }
    image_concepts = sorted(concept_set | set(mismatch_sources.values()))
    clip_concepts, image_vectors = encode_images(
        image_concepts,
        processor=processor,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )
    if text_vectors.shape[1] != image_vectors.shape[1]:
        raise RuntimeError(f"CLIP dimension mismatch: text={text_vectors.shape[1]} image={image_vectors.shape[1]}")
    forced_rows = build_forced_choice_rows(selected_rows, conditions, text_vectors, clip_concepts, image_vectors)
    condition_summaries = summarize_condition(forced_rows, args.bootstrap, args.seed)
    contrasts = paired_contrasts(forced_rows, args.bootstrap, args.permutations, args.seed + 2000)
    bridge_rows = [row for row in read_csv(ROOT / args.bridge) if row["concept"].lower() in concept_set]
    correlations = geometry_correlations(forced_rows, bridge_rows, args.bootstrap, args.permutations, args.seed + 3000)
    suffix = "_smoke" if args.limit else ""
    summary = {
        "n_concepts": len(concepts),
        "conditions": conditions,
        "clip_model_id": args.model_id,
        "clip_model_source": clip_source,
        "clip_dimension": int(text_vectors.shape[1]),
        "chance_target_choice_rate": 0.5,
        "n_bootstrap": args.bootstrap,
        "n_permutations": args.permutations,
        "condition_summaries": condition_summaries,
        "primary_contrasts": contrasts,
        "geometry_correlations": correlations,
    }
    write_csv(
        metrics_path(f"{args.output_stem}{suffix}.csv"),
        forced_rows,
        [
            "concept",
            "subtype",
            "condition",
            "mismatch_source",
            "target_similarity",
            "source_similarity",
            "target_margin",
            "target_choice",
            "source_choice",
            "pair_image_similarity",
            "pair_difficulty",
        ],
    )
    write_csv(
        metrics_path(f"{args.output_stem}_summary_rows{suffix}.csv"),
        condition_summaries,
        [
            "condition",
            "pair_difficulty",
            "n",
            "target_choice_rate",
            "target_choice_ci95_low",
            "target_choice_ci95_high",
            "source_choice_rate",
            "mean_target_margin",
            "target_margin_ci95_low",
            "target_margin_ci95_high",
            "median_target_margin",
            "chance_target_choice_rate",
        ],
    )
    write_csv(
        metrics_path(f"{args.output_stem}_correlations{suffix}.csv"),
        correlations,
        ["predictor", "endpoint", "statistic", "n", "estimate", "ci95_low", "ci95_high", "permutation_p"],
    )
    write_csv(
        metrics_path(f"{args.output_stem}_contrasts{suffix}.csv"),
        contrasts,
        ["contrast", "metric", "n", "estimate", "ci95_low", "ci95_high", "permutation_p"],
    )
    write_json(metrics_path(f"{args.output_stem}_summary{suffix}.json"), summary)
    write_text(ROOT / "reports" / "main_results" / f"{args.output_stem}_report{suffix}.md", "\n".join(report_lines(summary)))
    append_run_log(
        "CLIP Forced-Choice Behavior",
        [
            f"Computed CLIP forced-choice behavior for {len(concepts)} concepts.",
            f"Conditions: {', '.join(conditions)}.",
            f"CLIP source: {clip_source}.",
        ],
    )


if __name__ == "__main__":
    main()
