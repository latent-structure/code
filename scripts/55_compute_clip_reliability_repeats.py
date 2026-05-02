from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import ROOT, append_run_log, percentile_interval, rankdata, write_csv, write_json
from hardening_common import write_text


def load_clip_script() -> Any:
    path = ROOT / "scripts" / "38_compute_clip_forced_choice_behavior.py"
    spec = importlib.util.spec_from_file_location("clip_forced_choice_behavior", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return 0.0
    return float(np.dot(x, y) / denom)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return pearson_corr(rankdata(np.asarray(x, dtype=float)), rankdata(np.asarray(y, dtype=float)))


def split_half_reliability(
    scored: pd.DataFrame,
    metric: str,
    *,
    statistic: str,
    n_splits: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    concepts = sorted(scored["concept"].unique())
    by_concept = {
        concept: group.sort_values("repeat_id")[["repeat_id", metric]].copy()
        for concept, group in scored.groupby("concept")
    }
    estimates = []
    fn = spearman_corr if statistic == "spearman" else pearson_corr
    for _ in range(n_splits):
        left_values = []
        right_values = []
        for concept in concepts:
            group = by_concept[concept]
            repeats = group["repeat_id"].to_numpy()
            values = group[metric].to_numpy(dtype=float)
            if len(values) < 2:
                continue
            perm = rng.permutation(len(values))
            midpoint = len(values) // 2
            left_idx = perm[:midpoint]
            right_idx = perm[midpoint:]
            if len(left_idx) == 0 or len(right_idx) == 0:
                continue
            left_values.append(float(np.mean(values[left_idx])))
            right_values.append(float(np.mean(values[right_idx])))
        estimates.append(fn(np.asarray(left_values, dtype=float), np.asarray(right_values, dtype=float)))
    arr = np.asarray(estimates, dtype=float)
    low, high = percentile_interval(arr, 0.95)
    return float(np.mean(arr)), low, high


def score_repeated_generations(args: argparse.Namespace) -> pd.DataFrame:
    clip = load_clip_script()
    repeats = pd.read_csv(ROOT / args.generations)
    if args.limit:
        concepts = sorted(repeats["concept"].str.lower().unique())[: args.limit]
        repeats = repeats[repeats["concept"].str.lower().isin(concepts)].copy()
    repeats["concept"] = repeats["concept"].str.lower()
    repeats["mismatch_source"] = repeats["mismatch_source"].str.lower()

    processor, tokenizer, model, device, clip_source = clip.load_clip_components(
        model_id=args.model_id,
        model_path=args.model_path,
        cache_root=args.cache_root,
    )
    text_vectors = clip.encode_texts(
        repeats["generated_text"].astype(str).tolist(),
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )
    image_concepts = sorted(set(repeats["concept"]) | set(repeats["mismatch_source"]))
    clip_concepts, image_vectors = clip.encode_images(
        image_concepts,
        processor=processor,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )
    concept_index = {concept: idx for idx, concept in enumerate(clip_concepts)}
    rows: list[dict[str, Any]] = []
    for idx, row in repeats.reset_index(drop=True).iterrows():
        concept = str(row["concept"])
        source = str(row["mismatch_source"])
        text_vec = text_vectors[idx]
        target_vec = image_vectors[concept_index[concept]]
        source_vec = image_vectors[concept_index[source]]
        target_similarity = float(np.dot(text_vec, target_vec))
        source_similarity = float(np.dot(text_vec, source_vec))
        target_margin = target_similarity - source_similarity
        rows.append(
            {
                "concept": concept,
                "subtype": row["subtype"],
                "mismatch_source": source,
                "repeat_id": int(row["repeat_id"]),
                "seed": int(row["seed"]),
                "target_similarity": target_similarity,
                "source_similarity": source_similarity,
                "target_margin": target_margin,
                "target_choice": int(target_margin > 0),
                "source_choice": int(target_margin < 0),
                "clip_model_source": clip_source,
            }
        )
    return pd.DataFrame(rows)


def report_lines(summary_rows: list[dict[str, Any]], n_concepts: int, n_rows: int, n_splits: int) -> list[str]:
    lines = [
        "# CLIP Forced-Choice Reliability Over Repeated Generations",
        "",
        f"- Concepts: `{n_concepts}`",
        f"- Generation rows: `{n_rows}`",
        f"- Split-half resamples: `{n_splits}`",
        "- Method: for each concept, repeated mismatched-image generations are randomly split into two halves; each endpoint is averaged within each half and correlated across concepts.",
        "",
        "| Endpoint | Statistic | Split-half reliability | 95% CI |",
        "|---|---|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['metric']}` | `{row['statistic']}` | {row['split_half_reliability']:.4f} | "
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate CLIP forced-choice endpoint reliability from repeated stochastic generations.")
    parser.add_argument("--generations", default="outputs/generations/behavior_repeats_mismatched_1854x5.csv")
    parser.add_argument("--model-id", default="openai/clip-vit-large-patch14")
    parser.add_argument("--model-path")
    parser.add_argument("--cache-root", default=os.environ.get("HF_HOME", ".cache/hf"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--splits", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-stem", default="clip_forced_choice_reliability_repeats")
    args = parser.parse_args()

    scored = score_repeated_generations(args)
    metrics = [
        ("target_margin", "spearman"),
        ("target_choice", "pearson"),
        ("source_choice", "pearson"),
        ("target_similarity", "spearman"),
        ("source_similarity", "spearman"),
    ]
    summary_rows = []
    for idx, (metric, statistic) in enumerate(metrics):
        estimate, low, high = split_half_reliability(
            scored,
            metric,
            statistic=statistic,
            n_splits=args.splits,
            seed=args.seed + idx * 97,
        )
        summary_rows.append(
            {
                "metric": metric,
                "statistic": statistic,
                "n_concepts": int(scored["concept"].nunique()),
                "n_generation_rows": int(len(scored)),
                "split_half_reliability": estimate,
                "ci95_low": low,
                "ci95_high": high,
            }
        )

    suffix = "_smoke" if args.limit else ""
    score_path = ROOT / "outputs" / "metrics" / f"{args.output_stem}{suffix}_scores.csv"
    summary_path = ROOT / "outputs" / "metrics" / f"{args.output_stem}{suffix}.csv"
    json_path = ROOT / "outputs" / "metrics" / f"{args.output_stem}{suffix}.json"
    report_path = ROOT / "reports" / "main_results" / f"{args.output_stem}{suffix}_report.md"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(score_path, index=False)
    write_csv(summary_path, summary_rows, list(summary_rows[0].keys()))
    write_json(
        json_path,
        {
            "n_concepts": int(scored["concept"].nunique()),
            "n_generation_rows": int(len(scored)),
            "n_splits": args.splits,
            "summary_rows": summary_rows,
        },
    )
    write_text(report_path, "\n".join(report_lines(summary_rows, int(scored["concept"].nunique()), len(scored), args.splits)))
    append_run_log(
        "CLIP Forced-Choice Reliability Repeats",
        [f"Wrote {summary_path.relative_to(ROOT)} for {scored['concept'].nunique()} concepts."],
    )
    print(f"Wrote {score_path.relative_to(ROOT)}")
    print(f"Wrote {summary_path.relative_to(ROOT)}")
    print(f"Wrote {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
