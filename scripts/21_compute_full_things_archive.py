from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import ROOT, append_run_log, condensed_cosine_distance, metrics_path, output_path, percentile_interval, read_csv, spearman_corr, write_csv
from hardening_common import (
    build_proxy_rdms,
    load_embedding_bundle,
    load_project_backbone,
    load_siglip_reference,
    load_stage05_module,
    load_things_reference,
    mean_embedding_for_condition,
    residual_rsa,
    resolve_cached_snapshot,
    selected_layers,
    write_text,
)


PROTOTYPE_SIZES = [3, 5, 10]
PROTOTYPE_SEEDS = [11, 23, 37, 41, 53]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def bootstrap_gap(
    left_matrix: np.ndarray,
    right_matrix: np.ndarray,
    anchor_rdm: np.ndarray,
    mode: str,
    controls: list[np.ndarray] | None,
    n_resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    scores = []
    n = left_matrix.shape[0]
    for _ in range(n_resamples):
        sample_idx = rng.integers(0, n, size=n)
        left_rdm = condensed_cosine_distance(left_matrix[sample_idx])
        right_rdm = condensed_cosine_distance(right_matrix[sample_idx])
        anchor = anchor_rdm[np.ix_(sample_idx, sample_idx)]
        anchor_condensed = anchor[np.triu_indices(n, k=1)]
        if mode == "residual":
            sampled_controls = []
            for control in controls or []:
                matrix = np.zeros((n, n), dtype=float)
                matrix[np.triu_indices(n, k=1)] = control
                matrix = matrix + matrix.T
                sampled_matrix = matrix[np.ix_(sample_idx, sample_idx)]
                sampled_controls.append(sampled_matrix[np.triu_indices(n, k=1)])
            gap = residual_rsa(left_rdm, anchor_condensed, sampled_controls) - residual_rsa(right_rdm, anchor_condensed, sampled_controls)
        else:
            gap = spearman_corr(left_rdm, anchor_condensed) - spearman_corr(right_rdm, anchor_condensed)
        scores.append(gap)
    values = np.asarray(scores, dtype=float)
    low, high = percentile_interval(values, 0.95)
    return float(values.mean()), low, high


def select_equal_image_manifest(
    manifest_rows: list[dict[str, str]],
    concept_rows: list[dict[str, str]],
    *,
    min_images: int,
    images_per_concept: int,
    limit_concepts: int,
) -> list[dict[str, str]]:
    rows_by_concept: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        rows_by_concept[row["concept"]].append(row)
    for rows in rows_by_concept.values():
        rows.sort(key=lambda row: int(row["image_index"]))

    selected_rows: list[dict[str, str]] = []
    selected_concepts = 0
    for concept_row in concept_rows:
        concept = concept_row["concept"]
        rows = rows_by_concept.get(concept, [])
        if min_images and len(rows) < min_images:
            continue
        if images_per_concept and len(rows) < images_per_concept:
            continue
        selected = rows[:images_per_concept] if images_per_concept else rows
        if not selected:
            continue
        selected_rows.extend(selected)
        selected_concepts += 1
        if limit_concepts and selected_concepts >= limit_concepts:
            break
    return selected_rows


def extract_image_embeddings(
    config_path: str,
    manifest_rows: list[dict[str, str]],
    cache_path: Path,
    force_extract: bool,
    save_every: int,
) -> dict[str, np.ndarray]:
    required_ids = [row["image_id"] for row in manifest_rows]
    embeddings: dict[str, np.ndarray] = {}
    if cache_path.exists() and not force_extract:
        cached = np.load(cache_path)
        embeddings = {image_id: np.asarray(cached[image_id], dtype=float) for image_id in required_ids if image_id in cached.files}
        if len(embeddings) == len(required_ids):
            return embeddings

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
    layer_count = model.config.num_hidden_layers + 1 if hasattr(model.config, "num_hidden_layers") else 37
    selected = selected_layers(list(range(layer_count)), mid_fraction)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError(f"{backbone_multimodal} processor does not expose a tokenizer")

    prompt_template = config["prompts"]["multimodal"]["neutral"]
    max_side = 448
    extracted_since_save = 0
    try:
        with torch.no_grad():
            for row in manifest_rows:
                concept = row["concept"]
                image_id = row["image_id"]
                if image_id in embeddings:
                    continue
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
                pooled = stage05.pool_text_hidden_states(
                    stage05.extract_hidden_states(outputs),
                    span_start,
                    span_end,
                )
                selected_vectors = [np.asarray(pooled[layer], dtype=float) for layer in selected if layer < len(pooled)]
                embeddings[image_id] = np.mean(np.stack(selected_vectors), axis=0)
                extracted_since_save += 1
                if save_every and extracted_since_save >= save_every:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(cache_path, **embeddings)
                    extracted_since_save = 0
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **embeddings)
    return embeddings


def prototype_from_sample(vectors: list[np.ndarray], size: int) -> np.ndarray:
    if len(vectors) <= size:
        return np.mean(np.stack(vectors), axis=0)
    sampled = []
    for seed in PROTOTYPE_SEEDS:
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(len(vectors), size=size, replace=False)
        sampled.append(np.mean(np.stack([vectors[idx] for idx in sample_idx]), axis=0))
    return np.mean(np.stack(sampled), axis=0)


def build_anchor_rdm_matrix(condensed: np.ndarray, size: int) -> np.ndarray:
    matrix = np.zeros((size, size), dtype=float)
    matrix[np.triu_indices(size, k=1)] = condensed
    return matrix + matrix.T


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--images-per-concept", type=int, default=0, help="Use the first N archive images for every included concept. Default uses all available images.")
    parser.add_argument("--min-images", type=int, default=0, help="Require at least this many archive images per concept.")
    parser.add_argument("--limit-concepts", type=int, default=0, help="Optional smoke/debug limit after image-count filtering.")
    parser.add_argument("--output-prefix", default="full_things", help="Prefix for output artifacts.")
    parser.add_argument("--force-extract", action="store_true", help="Ignore cached image embeddings and recompute.")
    parser.add_argument("--save-every", type=int, default=1000, help="Save partial image-embedding cache after this many newly extracted images.")
    parser.add_argument("--dry-run", action="store_true", help="Only report selected concept/image counts.")
    args = parser.parse_args()

    manifest_rows = read_csv(ROOT / "data" / "manifests" / "full_things_archive_manifest.csv")
    concept_rows = read_csv(ROOT / "data" / "concepts" / "full_things_archive_concepts.csv")
    if not manifest_rows or not concept_rows:
        raise RuntimeError("Full THINGS archive manifest is empty. Run scripts/20_prepare_full_things_archive.py first.")
    manifest_rows = select_equal_image_manifest(
        manifest_rows,
        concept_rows,
        min_images=args.min_images,
        images_per_concept=args.images_per_concept,
        limit_concepts=args.limit_concepts,
    )
    selected_counts: dict[str, int] = defaultdict(int)
    for row in manifest_rows:
        selected_counts[row["concept"]] += 1
    if args.dry_run:
        counts = sorted(selected_counts.values())
        payload = {
            "selected_concepts": len(selected_counts),
            "selected_images": len(manifest_rows),
            "min_selected_images": min(counts) if counts else 0,
            "max_selected_images": max(counts) if counts else 0,
            "images_per_concept": args.images_per_concept,
            "min_images": args.min_images,
            "limit_concepts": args.limit_concepts,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    config, backbone_text, backbone_multimodal, mid_fraction = load_project_backbone(args.config)
    metadata_lookup, pooled, layers_by_model, metadata = load_embedding_bundle()
    image_embeddings = extract_image_embeddings(
        args.config,
        manifest_rows,
        output_path("outputs", "embeddings", f"{args.output_prefix}_image_embeddings.npz"),
        args.force_extract,
        args.save_every,
    )

    text_layers = selected_layers(layers_by_model[backbone_text], mid_fraction)
    multimodal_layers = selected_layers(layers_by_model[backbone_multimodal], mid_fraction)
    prompt_embedding, prompt_concepts = mean_embedding_for_condition(metadata_lookup, pooled, backbone_text, "T_prompt_primary", text_layers)
    prompt_index = {concept: idx for idx, concept in enumerate(prompt_concepts)}
    single_embedding, single_concepts = mean_embedding_for_condition(metadata_lookup, pooled, backbone_multimodal, "M_matched_image", multimodal_layers)
    single_index = {concept: idx for idx, concept in enumerate(single_concepts)}

    things_behavior, things_concepts, things_index = load_things_reference()
    siglip_embedding, siglip_concepts = load_siglip_reference(metadata_lookup, pooled, layers_by_model, metadata)
    siglip_index = {concept: idx for idx, concept in enumerate(siglip_concepts)}

    concept_meta = {row["concept"]: row for row in concept_rows}
    images_by_concept: dict[str, list[str]] = defaultdict(list)
    for row in manifest_rows:
        images_by_concept[row["concept"]].append(row["image_id"])

    available_concepts = [
        concept
        for concept in concept_meta
        if concept in images_by_concept and concept in prompt_index and concept in single_index and concept in things_index and concept in siglip_index
    ]
    proxy_rdms = build_proxy_rdms(available_concepts)
    scalable_mode = len(manifest_rows) > 5000

    per_concept_rows = []
    representation_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    representation_labels = {
        "single_image_grounding": "single_image_grounding",
        "prototype_all_images": "prototype_all_images",
        "prototype_size_3": "prototype_size_3",
        "prototype_size_5": "prototype_size_5",
        "prototype_size_10": "prototype_size_10",
        "T_prompt_primary": "T_prompt_primary",
    }

    for concept in available_concepts:
        image_ids = images_by_concept[concept]
        archive_vectors = [image_embeddings[image_id] for image_id in image_ids]
        prototype_all = np.mean(np.stack(archive_vectors), axis=0)
        size_vectors = {
            3: prototype_from_sample(archive_vectors, 3),
            5: prototype_from_sample(archive_vectors, 5),
            10: prototype_from_sample(archive_vectors, 10),
        }
        single_vec = np.asarray(single_embedding[single_index[concept]], dtype=float)
        prompt_vec = np.asarray(prompt_embedding[prompt_index[concept]], dtype=float)

        within_values = []
        between_best = []
        prompt_values = []
        prototype_values = []
        prototype_to_single = cosine_similarity(prototype_all, single_vec)
        prototype_to_prompt = cosine_similarity(prototype_all, prompt_vec)
        single_to_prompt = cosine_similarity(single_vec, prompt_vec)

        for image_id in image_ids:
            vec = image_embeddings[image_id]
            peers = [cosine_similarity(vec, image_embeddings[peer]) for peer in image_ids if peer != image_id]
            within_mean = float(np.mean(peers)) if peers else 1.0
            if scalable_mode:
                nearest_other = 0.0
            else:
                other_values = [
                    cosine_similarity(vec, image_embeddings[other_image_id])
                    for other_concept, other_ids in images_by_concept.items()
                    if other_concept != concept and other_concept in available_concepts
                    for other_image_id in other_ids
                ]
                nearest_other = max(other_values) if other_values else 0.0
            prompt_sim = cosine_similarity(vec, prompt_vec)
            proto_sim = cosine_similarity(vec, prototype_all)
            within_values.append(within_mean)
            if not scalable_mode:
                between_best.append(nearest_other)
            prompt_values.append(prompt_sim)
            prototype_values.append(proto_sim)

        margin = prototype_to_single if scalable_mode else float(np.mean(within_values) - np.mean(between_best))
        if scalable_mode and margin >= 0.95:
            verdict = "prototype_stable"
        elif scalable_mode and margin <= 0.90:
            verdict = "image_fragile"
        elif margin >= 0.05:
            verdict = "prototype_stable"
        elif margin <= 0.0:
            verdict = "image_fragile"
        else:
            verdict = "mixed"

        per_concept_rows.append(
            {
                "concept": concept,
                "subtype": concept_meta[concept]["subtype"],
                "archive_image_count": len(image_ids),
                "mean_within_concept_similarity": float(np.mean(within_values)),
                "mean_nearest_between_similarity": 0.0 if scalable_mode else float(np.mean(between_best)),
                "mean_image_to_prompt_similarity": float(np.mean(prompt_values)),
                "mean_image_to_prototype_similarity": float(np.mean(prototype_values)),
                "prototype_to_prompt_similarity": prototype_to_prompt,
                "prototype_to_single_similarity": prototype_to_single,
                "single_to_prompt_similarity": single_to_prompt,
                "stability_margin": margin,
                "stability_verdict": verdict,
            }
        )

        representation_vectors["single_image_grounding"].append(single_vec)
        representation_vectors["prototype_all_images"].append(prototype_all)
        representation_vectors["prototype_size_3"].append(size_vectors[3])
        representation_vectors["prototype_size_5"].append(size_vectors[5])
        representation_vectors["prototype_size_10"].append(size_vectors[10])
        representation_vectors["T_prompt_primary"].append(prompt_vec)

    concept_count = len(available_concepts)
    things_idx = [things_index[concept] for concept in available_concepts]
    behavior_dist = 1.0 - things_behavior[np.ix_(things_idx, things_idx)]
    behavior_rdm = np.asarray(behavior_dist[np.triu_indices(concept_count, k=1)], dtype=float)
    behavior_rdm_matrix = behavior_dist.astype(float)
    siglip_rdm = condensed_cosine_distance(
        np.asarray([siglip_embedding[siglip_index[concept]] for concept in available_concepts], dtype=float)
    )
    siglip_rdm_matrix = build_anchor_rdm_matrix(siglip_rdm, concept_count)
    controls = [
        proxy_rdms["subtype_membership"],
        proxy_rdms["coarse_category_structure"],
        proxy_rdms["sound_linked_vs_other"],
        proxy_rdms["lexical_trigram_distance"],
    ]

    anchor_rsa_rows = []
    summary_rows = []
    comparison_rows = []
    prompt_matrix = np.asarray(representation_vectors["T_prompt_primary"], dtype=float)
    single_matrix = np.asarray(representation_vectors["single_image_grounding"], dtype=float)

    representation_order = [
        "single_image_grounding",
        "prototype_size_3",
        "prototype_size_5",
        "prototype_size_10",
        "prototype_all_images",
        "T_prompt_primary",
    ]
    size_lookup = {
        "single_image_grounding": 1,
        "prototype_size_3": 3,
        "prototype_size_5": 5,
        "prototype_size_10": 10,
        "prototype_all_images": args.images_per_concept if args.images_per_concept else 999,
        "T_prompt_primary": 0,
    }

    for representation in representation_order:
        matrix = np.asarray(representation_vectors[representation], dtype=float)
        rdm = condensed_cosine_distance(matrix)
        things_score = spearman_corr(rdm, behavior_rdm)
        controlled_score = residual_rsa(rdm, behavior_rdm, controls)
        siglip_score = spearman_corr(rdm, siglip_rdm)
        anchor_rsa_rows.extend(
            [
                {
                    "anchor_name": "THINGS behavioral similarity",
                    "representation": representation_labels[representation],
                    "sample_size": size_lookup[representation],
                    "rsa_score": things_score,
                },
                {
                    "anchor_name": "controlled_THINGS",
                    "representation": representation_labels[representation],
                    "sample_size": size_lookup[representation],
                    "rsa_score": controlled_score,
                },
                {
                    "anchor_name": "SigLIP2",
                    "representation": representation_labels[representation],
                    "sample_size": size_lookup[representation],
                    "rsa_score": siglip_score,
                },
            ]
        )

        if representation == "T_prompt_primary":
            stable_fraction = 0.0
            mean_proto_prompt = 0.0
            mean_proto_single = 0.0
        else:
            stable_fraction = float(sum(row["stability_verdict"] == "prototype_stable" for row in per_concept_rows) / len(per_concept_rows))
            mean_proto_prompt = float(np.mean([cosine_similarity(vec, prompt_matrix[idx]) for idx, vec in enumerate(matrix)]))
            mean_proto_single = float(np.mean([cosine_similarity(vec, single_matrix[idx]) for idx, vec in enumerate(matrix)]))
        summary_rows.append(
            {
                "representation": representation_labels[representation],
                "sample_size": size_lookup[representation],
                "concept_count": concept_count,
                "mean_archive_image_count": float(np.mean([float(row["archive_image_count"]) for row in per_concept_rows])),
                "stable_concept_fraction": stable_fraction,
                "mean_representation_to_prompt_similarity": mean_proto_prompt,
                "mean_representation_to_single_similarity": mean_proto_single,
            }
        )

    def fetch(anchor: str, representation: str) -> float:
        for row in anchor_rsa_rows:
            if row["anchor_name"] == anchor and row["representation"] == representation:
                return float(row["rsa_score"])
        return 0.0

    gap_specs = [
        ("THINGS behavioral similarity", "prototype_all_images", "T_prompt_primary", "prototype_minus_prompt"),
        ("THINGS behavioral similarity", "prototype_all_images", "single_image_grounding", "prototype_minus_single"),
        ("controlled_THINGS", "prototype_all_images", "T_prompt_primary", "prototype_minus_prompt"),
        ("controlled_THINGS", "prototype_all_images", "single_image_grounding", "prototype_minus_single"),
        ("SigLIP2", "prototype_all_images", "single_image_grounding", "prototype_minus_single"),
    ]
    for anchor_name, left_rep, right_rep, comparison_name in gap_specs:
        left_matrix = np.asarray(representation_vectors[left_rep], dtype=float)
        right_matrix = np.asarray(representation_vectors[right_rep], dtype=float)
        if anchor_name == "controlled_THINGS":
            mean_gap, ci_low, ci_high = bootstrap_gap(
                left_matrix,
                right_matrix,
                behavior_rdm_matrix,
                "residual",
                controls,
                int(config["analysis"]["budgets"]["bootstrap_resamples"]),
                7,
            )
        elif anchor_name == "SigLIP2":
            mean_gap, ci_low, ci_high = bootstrap_gap(
                left_matrix,
                right_matrix,
                siglip_rdm_matrix,
                "spearman",
                None,
                int(config["analysis"]["budgets"]["bootstrap_resamples"]),
                13,
            )
        else:
            mean_gap, ci_low, ci_high = bootstrap_gap(
                left_matrix,
                right_matrix,
                behavior_rdm_matrix,
                "spearman",
                None,
                int(config["analysis"]["budgets"]["bootstrap_resamples"]),
                3,
            )
        comparison_rows.append(
            {
                "anchor_name": anchor_name,
                "comparison_name": comparison_name,
                "left_representation": left_rep,
                "right_representation": right_rep,
                "left_rsa": fetch(anchor_name, left_rep),
                "right_rsa": fetch(anchor_name, right_rep),
                "mean_gap": mean_gap,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    plt.figure(figsize=(10, 5))
    anchors = ["THINGS behavioral similarity", "controlled_THINGS", "SigLIP2"]
    plot_representations = ["single_image_grounding", "prototype_all_images", "T_prompt_primary"]
    x = np.arange(len(anchors))
    width = 0.24
    for offset, representation in enumerate(plot_representations):
        values = [fetch(anchor, representation) for anchor in anchors]
        plt.bar(x + (offset - 1) * width, values, width=width, label=representation)
    plt.xticks(x, anchors, rotation=12, ha="right")
    plt.ylabel("RSA")
    plt.title("Full THINGS prototypes versus single-image grounding")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path("outputs", "figures", f"fig_{args.output_prefix}_prototype_vs_single.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    curve_representations = ["prototype_size_3", "prototype_size_5", "prototype_size_10", "prototype_all_images"]
    curve_x = [3, 5, 10, max(int(row["archive_image_count"]) for row in per_concept_rows)]
    for anchor in anchors:
        values = [fetch(anchor, representation) for representation in curve_representations]
        plt.plot(curve_x, values, marker="o", label=anchor)
    plt.xlabel("Prototype size")
    plt.ylabel("RSA")
    plt.title("Prototype size curve across anchors")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path("outputs", "figures", f"fig_{args.output_prefix}_prototype_size_curve.png"), dpi=180)
    plt.close()

    write_csv(
        metrics_path(f"{args.output_prefix}_image_variance.csv"),
        per_concept_rows,
        [
            "concept",
            "subtype",
            "archive_image_count",
            "mean_within_concept_similarity",
            "mean_nearest_between_similarity",
            "mean_image_to_prompt_similarity",
            "mean_image_to_prototype_similarity",
            "prototype_to_prompt_similarity",
            "prototype_to_single_similarity",
            "single_to_prompt_similarity",
            "stability_margin",
            "stability_verdict",
        ],
    )
    write_csv(
        metrics_path(f"{args.output_prefix}_prototype_summary.csv"),
        summary_rows,
        [
            "representation",
            "sample_size",
            "concept_count",
            "mean_archive_image_count",
            "stable_concept_fraction",
            "mean_representation_to_prompt_similarity",
            "mean_representation_to_single_similarity",
        ],
    )
    write_csv(
        metrics_path(f"{args.output_prefix}_prototype_anchor_rsa.csv"),
        anchor_rsa_rows,
        ["anchor_name", "representation", "sample_size", "rsa_score"],
    )
    write_csv(
        output_path("outputs", "tables", f"{args.output_prefix}_prototype_comparison_table.csv"),
        comparison_rows,
        ["anchor_name", "comparison_name", "left_representation", "right_representation", "left_rsa", "right_rsa", "mean_gap", "ci_low", "ci_high"],
    )
    write_csv(
        output_path("outputs", "tables", f"{args.output_prefix}_concept_stability_table.csv"),
        sorted(per_concept_rows, key=lambda row: (row["stability_verdict"], float(row["stability_margin"]), row["concept"]), reverse=True),
        [
            "concept",
            "subtype",
            "archive_image_count",
            "mean_within_concept_similarity",
            "mean_nearest_between_similarity",
            "mean_image_to_prompt_similarity",
            "mean_image_to_prototype_similarity",
            "prototype_to_prompt_similarity",
            "prototype_to_single_similarity",
            "single_to_prompt_similarity",
            "stability_margin",
            "stability_verdict",
        ],
    )

    stable_count = sum(row["stability_verdict"] == "prototype_stable" for row in per_concept_rows)
    if stable_count >= max(1, int(0.6 * len(per_concept_rows))):
        verdict = "supports concept-level grounding"
    elif stable_count <= int(0.3 * len(per_concept_rows)):
        verdict = "too image-fragile to help materially"
    else:
        verdict = "supports image-sensitive but stable prototypes"

    report_lines = [
        "# Full THINGS Image Archive Report",
        "",
        "## Coverage",
        f"- Concepts included: `{len(per_concept_rows)}`",
        f"- Mean archive images per concept: `{np.mean([float(row['archive_image_count']) for row in per_concept_rows]):.2f}`",
        f"- Stable concepts: `{stable_count}/{len(per_concept_rows)}`",
        f"- Verdict: `{verdict}`",
        "",
        "## Anchor Summary",
    ]
    for anchor_name in ["THINGS behavioral similarity", "controlled_THINGS", "SigLIP2"]:
        for representation in ["single_image_grounding", "prototype_all_images", "T_prompt_primary"]:
            report_lines.append(f"- `{anchor_name}` `{representation}` RSA=`{fetch(anchor_name, representation):.4f}`")
    report_lines.extend(
        [
            "",
            "## Prototype Size Curve",
        ]
    )
    for anchor_name in ["THINGS behavioral similarity", "controlled_THINGS", "SigLIP2"]:
        curve_bits = ", ".join(f"{representation}=`{fetch(anchor_name, representation):.4f}`" for representation in ["prototype_size_3", "prototype_size_5", "prototype_size_10", "prototype_all_images"])
        report_lines.append(f"- `{anchor_name}` {curve_bits}")
    report_lines.extend(
        [
            "",
            "## Comparison Summary",
        ]
    )
    for row in comparison_rows:
        report_lines.append(
            f"- `{row['anchor_name']}` `{row['comparison_name']}` mean_gap=`{float(row['mean_gap']):.4f}` CI=`[{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}]`"
        )
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "- This analysis tests whether grounding remains concept-level when the local THINGS archive is used exhaustively rather than a single selected JPEG.",
            "- Prototype averaging is compared against the current single-image matched condition and prompted text on raw THINGS, controlled THINGS, and SigLIP2.",
        ]
    )
    write_text(output_path("reports", "main_results", f"{args.output_prefix}_image_archive_report.md"), "\n".join(report_lines))

    append_run_log(
        "Full THINGS Image Archive",
        [
            f"Wrote image variance metrics to {metrics_path(f'{args.output_prefix}_image_variance.csv').relative_to(ROOT)}.",
            f"Wrote prototype summary to {metrics_path(f'{args.output_prefix}_prototype_summary.csv').relative_to(ROOT)}.",
            f"Wrote archive report to {output_path('reports', 'main_results', f'{args.output_prefix}_image_archive_report.md').relative_to(ROOT)}.",
        ],
    )


if __name__ == "__main__":
    main()
