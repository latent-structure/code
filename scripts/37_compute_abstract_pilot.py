from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from analysis_common import ordered_embedding_for_concepts
from common import (
    ROOT,
    append_run_log,
    canonical_condition_name,
    condensed_cosine_distance,
    embeddings_path,
    metrics_path,
    percentile_interval,
    read_csv,
    spearman_corr,
    write_csv,
    write_json,
)
from hardening_common import LANCASTER_SPACES, lancaster_matrix_for_concepts, selected_layers, write_text


TEXT_CONDITIONS = ["T_neutral", "T_prompt_primary", "T_prompt_para_1", "T_prompt_para_2"]
REFERENCE_SPACES = {
    "lancaster_full_sensorimotor": LANCASTER_SPACES["lancaster_full_sensorimotor"],
    "lancaster_perceptual": LANCASTER_SPACES["lancaster_perceptual"],
    "lancaster_action_body": ["Foot_leg.mean", "Hand_arm.mean", "Head.mean", "Mouth.mean", "Torso.mean"],
}
FAMILY_SPECS = {
    "qwen": {
        "label": "Qwen",
        "model_id": "Qwen/Qwen3.5-9B",
        "abstract_tag": "abstract_lancaster_204",
    },
    "mistral": {
        "label": "Mistral",
        "model_id": "mistralai/Mistral-Small-24B-Instruct-2501",
        "abstract_tag": "abstract_lancaster_204_mistral",
    },
    "llama": {
        "label": "Llama",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "abstract_tag": "abstract_lancaster_204_llama",
    },
}


def bundle_paths(tag: str) -> tuple[Path, Path]:
    return embeddings_path(f"pooled_embeddings_{tag}.npz"), embeddings_path(f"embedding_metadata_{tag}.json")


def load_bundle(tag: str, domain: str) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[str, np.ndarray], dict[str, list[int]], dict[str, Any]]:
    pooled_path, metadata_path = bundle_paths(tag)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    pooled_npz = np.load(pooled_path)
    pooled = {key: np.asarray(pooled_npz[key], dtype=np.float32) for key in pooled_npz.files}
    lookup = {
        (record["model_id"], canonical_condition_name(record["condition"]), int(record["layer"])): record
        for record in metadata["records"]
        if record["domain"] == domain
    }
    layers_by_model: dict[str, list[int]] = {}
    for record in metadata["records"]:
        if record["domain"] != domain:
            continue
        layers_by_model.setdefault(record["model_id"], []).append(int(record["layer"]))
    layers_by_model = {model_id: sorted(set(layers)) for model_id, layers in layers_by_model.items()}
    return lookup, pooled, layers_by_model, metadata


def aggregate_embedding(
    lookup: dict[tuple[str, str, int], dict[str, Any]],
    pooled: dict[str, np.ndarray],
    model_id: str,
    condition: str,
    layers: list[int],
    target_concepts: list[str],
) -> np.ndarray:
    matrices = []
    concepts: list[str] | None = None
    for layer in layers:
        record = lookup.get((model_id, condition, layer))
        if record is None:
            continue
        if concepts is None:
            concepts = [concept.lower() for concept in record["concepts"]]
        matrices.append(np.asarray(pooled[f"record_{record['record_id']}"], dtype=np.float32))
    if concepts is None or not matrices:
        raise RuntimeError(f"Missing embedding rows for {model_id} {condition}")
    return ordered_embedding_for_concepts(np.mean(np.stack(matrices), axis=0, dtype=np.float32), concepts, target_concepts)


def participation_ratio(matrix: np.ndarray) -> float:
    centered = np.asarray(matrix, dtype=float) - np.asarray(matrix, dtype=float).mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    eigenvalues = singular_values**2
    denom = float(np.sum(eigenvalues**2))
    return 0.0 if denom == 0.0 else float((np.sum(eigenvalues) ** 2) / denom)


def concept_bootstrap_gap(
    left: np.ndarray,
    right: np.ndarray,
    reference: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    n = left.shape[0]
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        ref_rdm = condensed_cosine_distance(reference[idx])
        left_rdm = condensed_cosine_distance(left[idx])
        right_rdm = condensed_cosine_distance(right[idx])
        values.append(spearman_corr(left_rdm, ref_rdm) - spearman_corr(right_rdm, ref_rdm))
    return percentile_interval(np.asarray(values, dtype=float), 0.95)


def compute_for_group(
    *,
    group: str,
    tag: str,
    domain: str,
    concept_path: Path,
    model_id: str,
    mid_fraction: float,
    n_bootstrap: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    concept_rows = read_csv(concept_path)
    concepts = [row["concept"].lower() for row in concept_rows]
    lookup, pooled, layers_by_model, metadata = load_bundle(tag, domain)
    layers = selected_layers(layers_by_model[model_id], mid_fraction)
    embeddings = {
        condition: aggregate_embedding(lookup, pooled, model_id, condition, layers, concepts)
        for condition in TEXT_CONDITIONS
    }
    rsa_rows = []
    pr_rows = []
    for condition, matrix in embeddings.items():
        pr_rows.append(
            {
                "group": group,
                "condition": condition,
                "n_concepts": len(concepts),
                "participation_ratio": participation_ratio(matrix),
            }
        )
    for reference_name, dimensions in REFERENCE_SPACES.items():
        reference_matrix = lancaster_matrix_for_concepts(concepts, dimensions)
        reference_rdm = condensed_cosine_distance(reference_matrix)
        condition_scores = {}
        for condition, matrix in embeddings.items():
            score = spearman_corr(condensed_cosine_distance(matrix), reference_rdm)
            condition_scores[condition] = score
            rsa_rows.append(
                {
                    "group": group,
                    "reference_space": reference_name,
                    "condition": condition,
                    "n_concepts": len(concepts),
                    "rsa": score,
                    "prompt_minus_neutral": "",
                    "ci95_low": "",
                    "ci95_high": "",
                }
            )
        prompt = embeddings["T_prompt_primary"]
        neutral = embeddings["T_neutral"]
        ci_low, ci_high = concept_bootstrap_gap(
            prompt,
            neutral,
            reference_matrix,
            n_bootstrap=n_bootstrap,
            seed=seed + len(rsa_rows),
        )
        rsa_rows.append(
            {
                "group": group,
                "reference_space": reference_name,
                "condition": "T_prompt_primary_minus_T_neutral",
                "n_concepts": len(concepts),
                "rsa": "",
                "prompt_minus_neutral": condition_scores["T_prompt_primary"] - condition_scores["T_neutral"],
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
        )
    summary = {
        "group": group,
        "tag": tag,
        "domain": domain,
        "concept_path": str(concept_path.relative_to(ROOT)),
        "n_concepts": len(concepts),
        "model_id": model_id,
        "selected_layers": layers,
        "metadata_record_count": metadata.get("record_count", len(metadata.get("records", []))),
    }
    return rsa_rows, pr_rows, summary


def report_lines(summary: dict[str, Any], rsa_rows: list[dict[str, Any]], pr_rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "# Abstract Lancaster Pilot",
        "",
        f"- Families analyzed: `{', '.join(summary['families'].keys())}`",
        f"- Abstract concepts per family: `{summary['abstract']['n_concepts']}`",
        f"- Concrete control concepts per family: `{summary['concrete_control']['n_concepts']}`",
        f"- Bootstrap resamples for prompt-minus-neutral gaps: `{summary['n_bootstrap']}`",
        "",
        "## RSA",
        "",
        "| Family | Group | Reference | Neutral | Prompt | Prompt - neutral | 95% CI |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {
        (row["family"], row["group"], row["reference_space"], row["condition"]): row for row in rsa_rows
    }
    for family in summary["families"].keys():
        for group in ["abstract", "concrete_control"]:
            for reference in REFERENCE_SPACES:
                neutral = by_key[(family, group, reference, "T_neutral")]
                prompt = by_key[(family, group, reference, "T_prompt_primary")]
                gap = by_key[(family, group, reference, "T_prompt_primary_minus_T_neutral")]
                lines.append(
                    f"| {family} | {group} | {reference} | {float(neutral['rsa']):.4f} | {float(prompt['rsa']):.4f} | "
                    f"{float(gap['prompt_minus_neutral']):+.4f} | [{float(gap['ci95_low']):+.4f}, {float(gap['ci95_high']):+.4f}] |"
                )
    lines.extend(["", "## Participation Ratio", "", "| Group | Condition | PR |", "|---|---|---:|"])
    for row in pr_rows:
        if row["condition"] in {"T_neutral", "T_prompt_primary"}:
            lines.append(f"| {row['family']} | {row['group']} | {row['condition']} | {float(row['participation_ratio']):.2f} |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute abstract Lancaster pilot RSA and PR.")
    parser.add_argument("--concrete-tag", default="full")
    parser.add_argument("--abstract-concepts", default="data/concepts/abstract_lancaster_204_concepts.csv")
    parser.add_argument("--concrete-concepts", default="data/concepts/concrete_lancaster_control_204_concepts.csv")
    parser.add_argument("--families", default="qwen", help="Comma-separated families to analyze: qwen,mistral,llama")
    parser.add_argument("--mid-fraction", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260429)
    args = parser.parse_args()

    families = [item.strip() for item in args.families.split(",") if item.strip()]
    rsa_rows: list[dict[str, Any]] = []
    pr_rows: list[dict[str, Any]] = []
    family_summaries: dict[str, dict[str, Any]] = {}
    for idx, family in enumerate(families):
        spec = FAMILY_SPECS.get(family)
        if spec is None:
            raise KeyError(f"Unknown family {family!r}; expected one of {sorted(FAMILY_SPECS)}")
        abstract_rsa, abstract_pr, abstract_summary = compute_for_group(
            group="abstract",
            tag=spec["abstract_tag"],
            domain="abstract",
            concept_path=ROOT / args.abstract_concepts,
            model_id=spec["model_id"],
            mid_fraction=args.mid_fraction,
            n_bootstrap=args.bootstrap,
            seed=args.seed + idx * 1000,
        )
        concrete_rsa, concrete_pr, concrete_summary = compute_for_group(
            group="concrete_control",
            tag=args.concrete_tag,
            domain="sensory",
            concept_path=ROOT / args.concrete_concepts,
            model_id=spec["model_id"],
            mid_fraction=args.mid_fraction,
            n_bootstrap=args.bootstrap,
            seed=args.seed + idx * 1000 + 10000,
        )
        for row in abstract_rsa + concrete_rsa:
            row["family"] = family
        for row in abstract_pr + concrete_pr:
            row["family"] = family
        rsa_rows.extend(abstract_rsa + concrete_rsa)
        pr_rows.extend(abstract_pr + concrete_pr)
        family_summaries[family] = {
            "label": spec["label"],
            "abstract": abstract_summary,
            "concrete_control": concrete_summary,
        }
    summary = {
        "families": family_summaries,
        "n_bootstrap": args.bootstrap,
        "reference_spaces": REFERENCE_SPACES,
    }
    write_csv(
        metrics_path("abstract_pilot_rsa.csv"),
        rsa_rows,
        ["family", "group", "reference_space", "condition", "n_concepts", "rsa", "prompt_minus_neutral", "ci95_low", "ci95_high"],
    )
    write_csv(metrics_path("abstract_pilot_pr.csv"), pr_rows, ["family", "group", "condition", "n_concepts", "participation_ratio"])
    write_json(metrics_path("abstract_pilot_summary.json"), summary)
    write_text(ROOT / "reports" / "main_results" / "abstract_pilot_report.md", "\n".join(report_lines(summary, rsa_rows, pr_rows)))
    append_run_log(
        "Abstract Lancaster Pilot",
        [f"Computed abstract pilot for families: {', '.join(families)}."],
    )


if __name__ == "__main__":
    main()
