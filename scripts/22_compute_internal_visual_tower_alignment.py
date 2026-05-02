from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from analysis_common import aggregate_condition_embedding, ordered_embedding_for_concepts
from common import (
    ROOT,
    append_run_log,
    condensed_cosine_distance,
    embeddings_path,
    load_project_config,
    metrics_path,
    output_path,
    spearman_corr,
    write_csv,
    write_json,
)
from hardening_common import (
    condition_model_id,
    load_embedding_bundle,
    selected_layers,
    write_text,
)


CONDITIONS = [
    "T_neutral",
    "T_prompt_primary",
    "M_text_only",
    "M_matched_image",
    "M_prompt_plus_matched_image",
    "M_degraded_image",
    "M_mismatched_image",
    "M_blank_image",
]

FAMILY_SPECS: dict[str, dict[str, str]] = {
    "qwen": {
        "text_model_id": "Qwen/Qwen3.5-9B",
        "multimodal_model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "display_name": "Qwen",
    },
    "mistral": {
        "text_model_id": "mistralai/Mistral-Small-24B-Instruct-2501",
        "multimodal_model_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "display_name": "Mistral",
    },
    "llama": {
        "text_model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "multimodal_model_id": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "display_name": "Llama",
    },
}


def load_stage05_module() -> Any:
    script_path = ROOT / "scripts" / "01_extract_hidden_states.py"
    spec = importlib.util.spec_from_file_location("stage01_extract_hidden_states", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load helper module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def visual_module(model: Any) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        return model.model.visual
    if hasattr(model, "visual"):
        return model.visual
    if hasattr(model, "vision_model"):
        return model.vision_model
    if hasattr(model, "model") and hasattr(model.model, "vision_model"):
        return model.model.vision_model
    if hasattr(model, "model") and hasattr(model.model, "vision_tower"):
        return model.model.vision_tower
    if hasattr(model, "vision_tower"):
        return model.vision_tower
    raise RuntimeError("Could not locate an internal visual tower on the multimodal model.")


def module_device_and_dtype(module: Any) -> tuple[Any, Any]:
    parameter = next(module.parameters())
    return parameter.device, parameter.dtype


def move_visual_inputs(batch: Any, device: Any, dtype: Any) -> tuple[Any, dict[str, Any]]:
    pixel_values = batch.get("pixel_values")
    image_grid_thw = batch.get("image_grid_thw")
    image_sizes = batch.get("image_sizes")
    aspect_ratio_ids = batch.get("aspect_ratio_ids")
    aspect_ratio_mask = batch.get("aspect_ratio_mask")
    if pixel_values is None:
        keys = ", ".join(sorted(batch.keys()))
        raise RuntimeError(f"Visual extraction expected pixel_values; got keys: {keys}")
    if hasattr(pixel_values, "is_floating_point") and pixel_values.is_floating_point():
        pixel_values = pixel_values.to(device=device, dtype=dtype)
    else:
        pixel_values = pixel_values.to(device=device)
    aux: dict[str, Any] = {}
    if image_grid_thw is not None:
        aux["grid_thw"] = image_grid_thw.to(device=device)
    if image_sizes is not None:
        aux["image_sizes"] = image_sizes.to(device=device)
    if aspect_ratio_ids is not None:
        aux["aspect_ratio_ids"] = aspect_ratio_ids.to(device=device)
    if aspect_ratio_mask is not None:
        aux["aspect_ratio_mask"] = aspect_ratio_mask.to(device=device)
    return pixel_values, aux


def pool_visual_tokens(tensor: Any) -> np.ndarray:
    if tensor.ndim == 3:
        tokens = tensor[0]
    elif tensor.ndim == 2:
        tokens = tensor
    elif tensor.ndim == 1:
        return tensor.detach().float().cpu().numpy().astype(np.float32)
    else:
        tokens = tensor.reshape(-1, tensor.shape[-1])
    return tokens.detach().float().mean(dim=0).cpu().numpy().astype(np.float32)


def family_display_name(family: str) -> str:
    return FAMILY_SPECS[family]["display_name"]


def family_artifact_suffix(requested_family: str, family: str, limit: int) -> str:
    parts: list[str] = []
    if requested_family == "all" or family != "qwen":
        parts.append(family)
    if limit:
        parts.append("smoke")
    return f"_{'_'.join(parts)}" if parts else ""


def extract_internal_visual_vector(model: Any, batch: Any) -> tuple[np.ndarray, str]:
    visual = visual_module(model)
    device, dtype = module_device_and_dtype(visual)
    pixel_values, aux = move_visual_inputs(batch, device, dtype)
    outputs = visual(pixel_values, return_dict=True, **aux)
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return pool_visual_tokens(outputs.pooler_output), "pooler_output"
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return pool_visual_tokens(outputs.last_hidden_state), "last_hidden_state"
    if isinstance(outputs, (tuple, list)) and outputs:
        return pool_visual_tokens(outputs[0]), "tuple_0"
    raise RuntimeError("Visual tower returned no usable tensor output.")


def add_llama_aspect_ratio_inputs(batch: Any, processor: Any, image: Any) -> Any:
    if "aspect_ratio_ids" in batch and "aspect_ratio_mask" in batch:
        return batch
    try:
        from transformers.models.mllama.image_processing_mllama import (
            build_aspect_ratio_mask,
            convert_aspect_ratios_to_ids,
            get_optimal_tiled_canvas,
        )
    except Exception as exc:  # pragma: no cover - import depends on installed transformers
        raise RuntimeError("Could not import Mllama aspect-ratio helpers.") from exc

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise RuntimeError("Llama visual extraction requires processor.image_processor.")
    max_image_tiles = int(getattr(image_processor, "max_image_tiles", 4))
    size = getattr(image_processor, "size", {})
    tile_size = int(size.get("height") or size.get("width") or 448)
    canvas_h, canvas_w = get_optimal_tiled_canvas(image.height, image.width, max_image_tiles, tile_size)
    num_tiles_h = max(1, canvas_h // tile_size)
    num_tiles_w = max(1, canvas_w // tile_size)
    aspect_ratios = [[(num_tiles_w, num_tiles_h)]]
    device = batch["pixel_values"].device
    batch["aspect_ratio_ids"] = convert_aspect_ratios_to_ids(aspect_ratios, max_image_tiles=max_image_tiles, device=device)
    batch["aspect_ratio_mask"] = build_aspect_ratio_mask(aspect_ratios, max_image_tiles=max_image_tiles, device=device)
    return batch


def load_or_extract_internal_visual_embeddings(
    args: argparse.Namespace,
    family: str,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    suffix = family_artifact_suffix(args.family, family, args.limit)
    cache_npz = embeddings_path(f"internal_visual_tower{suffix}.npz")
    cache_json = embeddings_path(f"internal_visual_tower{suffix}.json")
    if cache_npz.exists() and cache_json.exists() and not args.overwrite_vision_cache:
        payload = np.load(cache_npz, allow_pickle=False)
        metadata = json.loads(cache_json.read_text(encoding="utf-8"))
        concepts = [str(concept) for concept in payload["concepts"].tolist()]
        embeddings = np.asarray(payload["embeddings"], dtype=float)
        return embeddings, concepts, metadata

    stage05 = load_stage05_module()
    config = load_project_config(args.config)
    stage05.configure_hf_cache(config)

    import torch
    import transformers
    from PIL import Image

    spec = FAMILY_SPECS[family]
    backbone_multimodal = spec["multimodal_model_id"]
    cache_root = config["analysis"]["runtime"].get("hf_cache_dir", "")
    model_source = str(stage05.resolve_cached_snapshot(backbone_multimodal, cache_root)) if cache_root else backbone_multimodal
    processor = transformers.AutoProcessor.from_pretrained(model_source, **stage05.tokenizer_load_kwargs(model_source))
    model = stage05.load_multimodal_model(transformers, model_source, stage05.multimodal_load_kwargs(torch, config))
    model.eval()

    concept_rows = [row for row in stage05.load_concepts(config, None) if row["domain"] == "sensory"]
    if args.limit:
        concept_rows = concept_rows[: args.limit]
    ready_images = stage05.load_ready_image_map()
    missing = [row["concept"] for row in concept_rows if row["concept"] not in ready_images]
    if missing:
        raise RuntimeError(f"Missing ready matched images for internal visual extraction: {', '.join(missing[:20])}")

    prompt_template = config["prompts"]["multimodal"]["neutral"]
    max_side = int(config.get("analysis", {}).get("image_policy", {}).get("multimodal_max_side", 384))
    vectors: list[np.ndarray] = []
    concepts: list[str] = []
    vision_source = ""
    with torch.no_grad():
        for row in concept_rows:
            concept = row["concept"]
            image_path = ROOT / ready_images[concept]["matched_image"]
            image = stage05.prepare_multimodal_image(Image.open(image_path).convert("RGB"), max_side)
            prompt = prompt_template.format(concept=concept)
            batch = stage05.build_multimodal_inputs(processor, prompt, image)
            if family == "llama":
                batch = add_llama_aspect_ratio_inputs(batch, processor, image)
            vector, source = extract_internal_visual_vector(model, batch)
            if vision_source and source != vision_source:
                raise RuntimeError(f"Inconsistent visual source: saw {source}, expected {vision_source}")
            vision_source = source
            concepts.append(concept.lower())
            vectors.append(vector)

    embeddings = np.asarray(vectors, dtype=np.float32)
    metadata = {
        "family": family,
        "model_id": backbone_multimodal,
        "num_concepts": len(concepts),
        "vision_source": vision_source,
        "image_policy_max_side": max_side,
        "description": "Image-level vectors pooled from the model's internal visual tower before language-side RSA.",
    }
    cache_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_npz, concepts=np.asarray(concepts), embeddings=embeddings)
    write_json(cache_json, metadata)
    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embeddings, concepts, metadata


def condition_alignment_rows(
    internal_embeddings: np.ndarray,
    internal_concepts: list[str],
    internal_metadata: dict[str, Any],
    config_path: str,
    family: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = FAMILY_SPECS[family]
    backbone_text = spec["text_model_id"]
    backbone_multimodal = spec["multimodal_model_id"]
    mid_fraction = float(load_project_config(config_path)["analysis"]["analysis"]["mid_to_late_fraction"])
    try:
        metadata_lookup, pooled, layers_by_model, metadata = load_embedding_bundle()
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            "Could not read outputs/embeddings/pooled_embeddings_full.npz. "
            "The merged embedding bundle is missing, incomplete, or currently being written; "
            "rerun this analysis after the merge phase finishes successfully."
        ) from exc
    internal_rdm = condensed_cosine_distance(internal_embeddings)

    rows: list[dict[str, Any]] = []
    scores: dict[str, float] = {}
    skipped: dict[str, str] = {}
    for condition in CONDITIONS:
        model_id = condition_model_id(backbone_text, backbone_multimodal, condition)
        available_layers = layers_by_model.get(model_id, [])
        if not available_layers:
            skipped[condition] = f"no layers for {model_id}"
            continue
        layers = selected_layers(available_layers, mid_fraction)
        try:
            embedding, concepts = aggregate_condition_embedding(metadata_lookup, pooled, model_id, condition, layers)
            ordered = ordered_embedding_for_concepts(embedding, concepts, internal_concepts)
        except Exception as exc:
            skipped[condition] = f"{type(exc).__name__}: {exc}"
            continue
        score = spearman_corr(condensed_cosine_distance(ordered), internal_rdm)
        scores[condition] = score
        rows.append(
            {
                "family": family,
                "anchor_name": f"{family_display_name(family)}_internal_visual_tower",
                "condition": condition,
                "model_id": model_id,
                "rsa_score": score,
                "comparison_to_prompt": "",
                "comparison_to_matched": "",
                "num_concepts": len(internal_concepts),
                "language_layers": ",".join(str(layer) for layer in layers),
                "vision_model_id": internal_metadata["model_id"],
                "vision_source": internal_metadata["vision_source"],
            }
        )

    prompt_score = scores.get("T_prompt_primary")
    matched_score = scores.get("M_matched_image")
    for row in rows:
        condition = str(row["condition"])
        if prompt_score is not None:
            row["comparison_to_prompt"] = scores[condition] - prompt_score
        if matched_score is not None:
            row["comparison_to_matched"] = scores[condition] - matched_score

    summary = {
        "anchor_name": f"{family_display_name(family)}_internal_visual_tower",
        "family": family,
        "vision_model_id": internal_metadata["model_id"],
        "vision_source": internal_metadata["vision_source"],
        "num_concepts": len(internal_concepts),
        "scores": scores,
        "skipped_conditions": skipped,
        "matched_minus_prompt": None if prompt_score is None or matched_score is None else matched_score - prompt_score,
        "prompt_plus_image_minus_matched": None
        if matched_score is None or "M_prompt_plus_matched_image" not in scores
        else scores["M_prompt_plus_matched_image"] - matched_score,
        "matched_minus_text_only": None
        if matched_score is None or "M_text_only" not in scores
        else matched_score - scores["M_text_only"],
        "degraded_minus_matched": None
        if matched_score is None or "M_degraded_image" not in scores
        else scores["M_degraded_image"] - matched_score,
        "mismatched_minus_matched": None
        if matched_score is None or "M_mismatched_image" not in scores
        else scores["M_mismatched_image"] - matched_score,
        "blank_minus_matched": None
        if matched_score is None or "M_blank_image" not in scores
        else scores["M_blank_image"] - matched_score,
        "embedding_metadata_source": metadata.get("source_tags", metadata.get("output_tag", "")),
    }
    if scores:
        summary["best_condition"] = max(scores, key=scores.get)
    return rows, summary


def write_report(summary: dict[str, Any], suffix: str, family: str) -> None:
    scores = summary["scores"]
    report_lines = [
        "# Internal Visual-Tower Alignment",
        "",
        "## Summary",
        f"- Internal anchor: `{summary['anchor_name']}`",
        f"- Family: `{family}`",
        f"- Vision model: `{summary['vision_model_id']}`",
        f"- Vision representation: `{summary['vision_source']}`",
        f"- Concepts: `{summary['num_concepts']}`",
    ]
    if summary.get("best_condition"):
        report_lines.append(f"- Best-aligned condition: `{summary['best_condition']}`")
    for key in [
        "matched_minus_prompt",
        "prompt_plus_image_minus_matched",
        "matched_minus_text_only",
        "degraded_minus_matched",
        "mismatched_minus_matched",
        "blank_minus_matched",
    ]:
        value = summary.get(key)
        if value is not None:
            report_lines.append(f"- {key}: `{value:.4f}`")
    report_lines.extend(["", "## Condition Scores"])
    for condition, score in sorted(scores.items()):
        report_lines.append(f"- `{condition}`: `{score:.4f}`")
    if summary.get("skipped_conditions"):
        report_lines.extend(["", "## Skipped Conditions"])
        for condition, reason in sorted(summary["skipped_conditions"].items()):
            report_lines.append(f"- `{condition}`: {reason}")
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "- This analysis asks whether language-side concept geometry aligns with the VLM's own image-side geometry for the same matched THINGS images.",
            "- Higher matched-image alignment than prompt alignment supports the interpretation that visual input acts as an image-contingent constraint rather than merely a generic semantic enrichment.",
        ]
    )
    write_text(output_path("reports", "main_results", f"internal_visual_tower_report{suffix}.md"), "\n".join(report_lines))


def process_family(args: argparse.Namespace, family: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    suffix = family_artifact_suffix(args.family, family, args.limit)
    internal_embeddings, internal_concepts, internal_metadata = load_or_extract_internal_visual_embeddings(args, family)
    rows, summary = condition_alignment_rows(internal_embeddings, internal_concepts, internal_metadata, args.config, family)
    write_csv(
        metrics_path(f"internal_visual_tower_alignment{suffix}.csv"),
        rows,
        [
            "family",
            "anchor_name",
            "condition",
            "model_id",
            "rsa_score",
            "comparison_to_prompt",
            "comparison_to_matched",
            "num_concepts",
            "language_layers",
            "vision_model_id",
            "vision_source",
        ],
    )
    write_json(metrics_path(f"internal_visual_tower_summary{suffix}.json"), summary)
    write_report(summary, suffix, family)
    append_run_log(
        f"{family_display_name(family)} Internal Visual-Tower Alignment",
        [
            f"Wrote internal visual-tower alignment to {metrics_path(f'internal_visual_tower_alignment{suffix}.csv').relative_to(ROOT)}.",
            f"Wrote internal visual-tower report to {output_path('reports', 'main_results', f'internal_visual_tower_report{suffix}.md').relative_to(ROOT)}.",
        ],
    )
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--family", choices=sorted(FAMILY_SPECS) + ["all"], default="qwen")
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test concept limit. Writes *_smoke outputs.")
    parser.add_argument("--overwrite-vision-cache", action="store_true")
    args = parser.parse_args()

    families = list(FAMILY_SPECS) if args.family == "all" else [args.family]
    combined_rows: list[dict[str, Any]] = []
    combined_summaries: dict[str, Any] = {}
    for family in families:
        rows, summary = process_family(args, family)
        combined_rows.extend(rows)
        combined_summaries[family] = summary
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    if args.family == "all":
        combined_suffix = family_artifact_suffix("all", "all", args.limit)
        write_csv(
            metrics_path(f"internal_visual_tower_alignment{combined_suffix}.csv"),
            combined_rows,
            [
                "family",
                "anchor_name",
                "condition",
                "model_id",
                "rsa_score",
                "comparison_to_prompt",
                "comparison_to_matched",
                "num_concepts",
                "language_layers",
                "vision_model_id",
                "vision_source",
            ],
        )
        write_json(metrics_path(f"internal_visual_tower_summary{combined_suffix}.json"), combined_summaries)
        append_run_log(
            "Internal Visual-Tower Alignment",
            [
                f"Wrote combined internal visual-tower alignment to {metrics_path(f'internal_visual_tower_alignment{combined_suffix}.csv').relative_to(ROOT)}.",
                f"Wrote combined internal visual-tower summary to {metrics_path(f'internal_visual_tower_summary{combined_suffix}.json').relative_to(ROOT)}.",
            ],
        )


if __name__ == "__main__":
    main()
