from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import ROOT, append_run_log, metrics_path, output_path, read_csv, spearman_corr, write_csv
from hardening_common import (
    build_proxy_rdms,
    condition_model_id,
    lancaster_matrix_for_concepts,
    load_embedding_bundle,
    load_project_backbone,
    load_siglip_reference,
    load_stage05_module,
    load_things_reference,
    mean_embedding_for_condition,
    residual_rsa,
    resolve_cached_snapshot,
    selected_layers,
    subset_embedding_matrix,
    write_text,
)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def extract_image_embeddings(config_path: str, manifest_rows: list[dict[str, str]]) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    config, _, backbone_multimodal, mid_fraction = load_project_backbone(config_path)
    stage05 = load_stage05_module()
    stage05.configure_hf_cache(config)
    cache_root = config["analysis"]["runtime"].get("hf_cache_dir", "")

    import torch
    import transformers
    from PIL import Image

    multimodal_snapshot = resolve_cached_snapshot(backbone_multimodal, cache_root)
    processor = transformers.AutoProcessor.from_pretrained(str(multimodal_snapshot), local_files_only=True)
    model = stage05.load_multimodal_model(transformers, str(multimodal_snapshot), stage05.multimodal_load_kwargs(torch, config)).eval()
    selected = selected_layers(list(range(model.config.num_hidden_layers + 1 if hasattr(model.config, "num_hidden_layers") else 37)), mid_fraction)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError(f"{backbone_multimodal} processor does not expose a tokenizer")

    embeddings: dict[str, np.ndarray] = {}
    paths: dict[str, str] = {}
    prompt_template = config["prompts"]["multimodal"]["neutral"]
    max_side = 448
    try:
        with torch.no_grad():
            for row in manifest_rows:
                concept = row["concept"]
                image_id = row["image_id"]
                image_path = ROOT / row["image_path"]
                prompt = prompt_template.format(concept=concept)
                image = Image.open(image_path).convert("RGB")
                image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                rendered_text = stage05.render_multimodal_text(processor, prompt, image)
                batch = stage05.build_multimodal_inputs(processor, prompt, image)
                span_start, span_end = stage05.resolve_text_span(
                    tokenizer,
                    batch["input_ids"][0].tolist(),
                    rendered_text,
                    prompt,
                    concept,
                    model_id=backbone_multimodal,
                    condition="M_matched_image",
                )
                batch = stage05.move_batch_to_device(batch, stage05.first_model_device(model))
                outputs = model(**batch, output_hidden_states=True)
                pooled = stage05.pool_text_hidden_states(stage05.extract_hidden_states(outputs), span_start, span_end)
                selected_vectors = [np.asarray(pooled[layer], dtype=float) for layer in selected if layer < len(pooled)]
                embeddings[image_id] = np.mean(np.stack(selected_vectors), axis=0)
                paths[image_id] = str(image_path.relative_to(ROOT))
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return embeddings, paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    args = parser.parse_args()

    manifest_rows = read_csv(ROOT / "data" / "manifests" / "multi_image_manifest.csv")
    diagnostic_rows = read_csv(ROOT / "data" / "concepts" / "multi_image_diagnostic_concepts.csv")
    diagnostic_concepts = [row["concept"] for row in diagnostic_rows]
    diagnostic_roles = {row["concept"]: row["diagnostic_role"] for row in diagnostic_rows}

    config, backbone_text, backbone_multimodal, mid_fraction = load_project_backbone(args.config)
    metadata_lookup, pooled, layers_by_model, metadata = load_embedding_bundle()
    image_embeddings, _ = extract_image_embeddings(args.config, manifest_rows)

    text_layers = selected_layers(layers_by_model[backbone_text], mid_fraction)
    multimodal_layers = selected_layers(layers_by_model[backbone_multimodal], mid_fraction)
    prompt_embedding, prompt_concepts = mean_embedding_for_condition(metadata_lookup, pooled, backbone_text, "T_prompt_primary", text_layers)
    prompt_index = {concept: idx for idx, concept in enumerate(prompt_concepts)}
    single_embedding, single_concepts = mean_embedding_for_condition(metadata_lookup, pooled, backbone_multimodal, "M_matched_image", multimodal_layers)
    single_index = {concept: idx for idx, concept in enumerate(single_concepts)}

    things_behavior, things_concepts, things_index = load_things_reference()
    siglip_embedding, siglip_concepts = load_siglip_reference(metadata_lookup, pooled, layers_by_model, metadata)
    siglip_index = {concept: idx for idx, concept in enumerate(siglip_concepts)}
    proxy_rdms = build_proxy_rdms(diagnostic_concepts)
    from common import condensed_cosine_distance

    per_image_rows = []
    per_concept_rows = []
    prototype_vectors = []
    single_vectors = []
    prompt_vectors = []
    available_for_anchor = []

    images_by_concept: dict[str, list[str]] = defaultdict(list)
    for row in manifest_rows:
        images_by_concept[row["concept"]].append(row["image_id"])

    for concept in diagnostic_concepts:
        image_ids = images_by_concept[concept]
        vectors = [image_embeddings[image_id] for image_id in image_ids]
        prototype = np.mean(np.stack(vectors), axis=0)
        prototype_vectors.append(prototype)
        single_vectors.append(np.asarray(single_embedding[single_index[concept]], dtype=float))
        prompt_vectors.append(np.asarray(prompt_embedding[prompt_index[concept]], dtype=float))
        available_for_anchor.append(concept)

        within_values = []
        between_best = []
        prompt_values = []
        for image_id in image_ids:
            vec = image_embeddings[image_id]
            peers = [cosine_similarity(vec, image_embeddings[peer]) for peer in image_ids if peer != image_id]
            within_mean = float(np.mean(peers)) if peers else 0.0
            other_values = [
                cosine_similarity(vec, image_embeddings[other_image_id])
                for other_concept, other_ids in images_by_concept.items()
                if other_concept != concept
                for other_image_id in other_ids
            ]
            nearest_other = max(other_values) if other_values else 0.0
            prompt_sim = cosine_similarity(vec, prompt_embedding[prompt_index[concept]])
            within_values.append(within_mean)
            between_best.append(nearest_other)
            prompt_values.append(prompt_sim)
            per_image_rows.append(
                {
                    "concept": concept,
                    "diagnostic_role": diagnostic_roles[concept],
                    "image_id": image_id,
                    "within_concept_similarity": within_mean,
                    "nearest_between_similarity": nearest_other,
                    "image_to_prompt_similarity": prompt_sim,
                }
            )

        prototype_prompt = cosine_similarity(prototype, prompt_embedding[prompt_index[concept]])
        single_prompt = cosine_similarity(single_embedding[single_index[concept]], prompt_embedding[prompt_index[concept]])
        prototype_single = cosine_similarity(prototype, single_embedding[single_index[concept]])
        verdict = "concept_stable" if np.mean(within_values) > np.mean(between_best) else "image_unstable"
        per_concept_rows.append(
            {
                "concept": concept,
                "diagnostic_role": diagnostic_roles[concept],
                "image_count": len(image_ids),
                "mean_within_concept_similarity": float(np.mean(within_values)),
                "mean_nearest_between_similarity": float(np.mean(between_best)),
                "mean_image_to_prompt_similarity": float(np.mean(prompt_values)),
                "prototype_to_prompt_similarity": prototype_prompt,
                "single_to_prompt_similarity": single_prompt,
                "prototype_to_single_similarity": prototype_single,
                "stability_verdict": verdict,
            }
        )

    prototype_vectors = np.asarray(prototype_vectors, dtype=float)
    single_vectors = np.asarray(single_vectors, dtype=float)
    prompt_vectors = np.asarray(prompt_vectors, dtype=float)
    things_idx = [things_index[concept] for concept in available_for_anchor]
    behavior_dist = 1.0 - things_behavior[np.ix_(things_idx, things_idx)]
    behavior_rdm = np.asarray(behavior_dist[np.triu_indices(len(available_for_anchor), k=1)], dtype=float)
    siglip_rdm = condensed_cosine_distance(np.asarray([siglip_embedding[siglip_index[concept]] for concept in available_for_anchor], dtype=float))
    prototype_rdm = condensed_cosine_distance(prototype_vectors)
    single_rdm = condensed_cosine_distance(single_vectors)
    prompt_rdm = condensed_cosine_distance(prompt_vectors)
    controls = [
        proxy_rdms["subtype_membership"],
        proxy_rdms["coarse_category_structure"],
        proxy_rdms["sound_linked_vs_other"],
        proxy_rdms["lexical_trigram_distance"],
    ]

    prototype_summary_rows = [
        {
            "anchor_name": "THINGS behavioral similarity",
            "representation": "multi_image_prototype",
            "rsa_score": spearman_corr(prototype_rdm, behavior_rdm),
        },
        {
            "anchor_name": "THINGS behavioral similarity",
            "representation": "single_image_grounding",
            "rsa_score": spearman_corr(single_rdm, behavior_rdm),
        },
        {
            "anchor_name": "THINGS behavioral similarity",
            "representation": "T_prompt_primary",
            "rsa_score": spearman_corr(prompt_rdm, behavior_rdm),
        },
        {
            "anchor_name": "controlled_THINGS",
            "representation": "multi_image_prototype",
            "rsa_score": residual_rsa(prototype_rdm, behavior_rdm, controls),
        },
        {
            "anchor_name": "controlled_THINGS",
            "representation": "single_image_grounding",
            "rsa_score": residual_rsa(single_rdm, behavior_rdm, controls),
        },
        {
            "anchor_name": "controlled_THINGS",
            "representation": "T_prompt_primary",
            "rsa_score": residual_rsa(prompt_rdm, behavior_rdm, controls),
        },
        {
            "anchor_name": "SigLIP2",
            "representation": "multi_image_prototype",
            "rsa_score": spearman_corr(prototype_rdm, siglip_rdm),
        },
        {
            "anchor_name": "SigLIP2",
            "representation": "single_image_grounding",
            "rsa_score": spearman_corr(single_rdm, siglip_rdm),
        },
        {
            "anchor_name": "SigLIP2",
            "representation": "T_prompt_primary",
            "rsa_score": spearman_corr(prompt_rdm, siglip_rdm),
        },
    ]

    plt.figure(figsize=(10, 5))
    concepts = [row["concept"] for row in per_concept_rows]
    within = [float(row["mean_within_concept_similarity"]) for row in per_concept_rows]
    between = [float(row["mean_nearest_between_similarity"]) for row in per_concept_rows]
    x = np.arange(len(concepts))
    plt.bar(x - 0.18, within, width=0.36, label="within-concept")
    plt.bar(x + 0.18, between, width=0.36, label="nearest different concept")
    plt.xticks(x, concepts, rotation=60, ha="right")
    plt.ylabel("Cosine similarity")
    plt.title("Multi-image concept consistency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path("outputs", "figures", "fig_multi_image_consistency.png"), dpi=180)
    plt.close()

    write_csv(
        metrics_path("multi_image_consistency.csv"),
        per_image_rows,
        ["concept", "diagnostic_role", "image_id", "within_concept_similarity", "nearest_between_similarity", "image_to_prompt_similarity"],
    )
    write_csv(
        metrics_path("multi_image_prototype_summary.csv"),
        prototype_summary_rows,
        ["anchor_name", "representation", "rsa_score"],
    )
    write_csv(
        output_path("outputs", "tables", "multi_image_reversal_table.csv"),
        per_concept_rows,
        [
            "concept",
            "diagnostic_role",
            "image_count",
            "mean_within_concept_similarity",
            "mean_nearest_between_similarity",
            "mean_image_to_prompt_similarity",
            "prototype_to_prompt_similarity",
            "single_to_prompt_similarity",
            "prototype_to_single_similarity",
            "stability_verdict",
        ],
    )

    stable_count = sum(row["stability_verdict"] == "concept_stable" for row in per_concept_rows)
    if stable_count >= max(1, int(0.6 * len(per_concept_rows))):
        verdict = "supports concept-level grounding"
    elif stable_count <= int(0.3 * len(per_concept_rows)):
        verdict = "supports image-specific grounding"
    else:
        verdict = "mixed / concept-dependent"

    report_lines = [
        "# Multi-Image Consistency Report",
        "",
        "## Concept Stability",
        f"- Diagnostic concepts evaluated: `{len(per_concept_rows)}`",
        f"- Concepts with clear within-concept clustering: `{stable_count}`",
        f"- Verdict: `{verdict}`",
        "",
        "## Prototype Anchor Summary",
    ]
    for row in prototype_summary_rows:
        report_lines.append(f"- `{row['anchor_name']}` `{row['representation']}` RSA=`{float(row['rsa_score']):.4f}`")
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "- This analysis tests whether grounding is stable across multiple images of the same concept or dominated by image-specific composition.",
            "- Prototype averaging is compared against the current single-image grounding and prompted concept embeddings on THINGS, controlled THINGS, and SigLIP2.",
        ]
    )
    write_text(output_path("reports", "main_results", "multi_image_consistency_report.md"), "\n".join(report_lines))
    append_run_log(
        "Multi-Image Consistency",
        [
            f"Wrote multi-image consistency metrics to {metrics_path('multi_image_consistency.csv').relative_to(ROOT)}.",
            f"Wrote multi-image report to {output_path('reports', 'main_results', 'multi_image_consistency_report.md').relative_to(ROOT)}.",
        ],
    )


if __name__ == "__main__":
    main()
