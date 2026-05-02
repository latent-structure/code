from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    ROOT,
    append_run_log,
    canonical_condition_name,
    load_project_config,
    read_csv,
    require,
    set_global_seed,
    write_json,
)


def configure_hf_cache(config: dict[str, Any]) -> None:
    cache_dir = config["analysis"]["runtime"].get("hf_cache_dir")
    if not cache_dir:
        cache_dir = None
    if cache_dir:
        os.environ.setdefault("HF_HOME", cache_dir)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(cache_dir) / "transformers"))
    # Qwen3-VL can fragment GPU memory badly during repeated per-concept passes.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def tokenizer_load_kwargs(model_source: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"local_files_only": True}
    # Mistral-Small tokenizers need the regex fix to avoid incorrect tokenization.
    if "mistral" in model_source.lower():
        kwargs["fix_mistral_regex"] = True
    return kwargs


def load_concepts(config: dict[str, Any], subset_path: str | None) -> list[dict[str, str]]:
    effective_subset = subset_path
    if not effective_subset:
        default_subset = config["analysis"].get("execution", {}).get("default_concept_subset", "")
        if default_subset:
            default_path = ROOT / default_subset
            if default_path.exists():
                effective_subset = str(default_path)
    if not effective_subset:
        return read_csv(ROOT / "data/concepts/full_concept_list.csv")
    subset_file = (ROOT / effective_subset).resolve() if not Path(effective_subset).is_absolute() else Path(effective_subset)
    subset_rows = read_csv(subset_file)
    require(subset_rows, f"Subset file was empty: {subset_file}")
    require(all("concept" in row and "domain" in row for row in subset_rows), f"Subset file is missing required metadata columns: {subset_file}")
    return subset_rows


def load_ready_image_map() -> dict[str, dict[str, str]]:
    rows = read_csv(ROOT / "data/manifests/image_manifest.csv")
    return {row["concept"]: row for row in rows if row["status"] == "ready"}


def load_mismatch_map() -> dict[str, str]:
    rows = read_csv(ROOT / "data/manifests/mismatch_map.csv")
    return {row["concept"]: row["mismatch_concept"] for row in rows}


def validate_image_coverage(config: dict[str, Any], concepts: list[dict[str, str]], ready_images: dict[str, dict[str, str]], subset_path: str | None) -> None:
    sensory = [row for row in concepts if row["domain"] == "sensory"]
    missing = sorted(row["concept"] for row in sensory if row["concept"] not in ready_images)
    if subset_path:
        require(not missing, f"Subset extraction requires ready images for all sensory concepts. Missing: {', '.join(missing)}")
        return
    if config["analysis"]["analysis"].get("require_full_ready_sensory", True):
        require(not missing, f"Full extraction is blocked until all sensory concepts have matched images. Missing: {', '.join(missing)}")


def parse_requested_models(config: dict[str, Any], value: str | None) -> list[dict[str, str]]:
    roster: dict[str, tuple[str, str]] = {}
    for family in ("text", "multimodal", "anchor"):
        for priority in ("primary", "fallback"):
            for model_id in config["models"][family][priority]:
                roster[model_id] = (family, priority)
    for model_id in config["models"].get("optional_anchor", []):
        roster[model_id] = ("optional_anchor", "optional")

    if value is None:
        execution = config["analysis"].get("execution", {})
        model_ids = [
            execution.get("sensory_backbone_text_model"),
            execution.get("sensory_backbone_multimodal_model"),
            execution.get("primary_anchor_model"),
        ]
        model_ids = [model_id for model_id in model_ids if model_id]
    elif value == "all":
        probe_log = read_csv(ROOT / "outputs/logs/model_probe_log.csv")
        model_ids = [row["model_id"] for row in probe_log if row["status"] == "ok"]
    else:
        model_ids = [item.strip() for item in value.split(",") if item.strip()]

    require(model_ids, "No models were selected for extraction.")
    rows = []
    for model_id in model_ids:
        require(model_id in roster, f"Requested model {model_id} is not in config/models.yaml")
        family, priority = roster[model_id]
        rows.append({"model_id": model_id, "family": family, "priority": priority})
    return rows


def pick_text_template_map(config: dict[str, Any]) -> dict[str, str]:
    prompts = config["prompts"]["text_only"]
    return {
        "T_neutral": prompts["neutral"],
        "T_prompt_primary": prompts["sensory_prompt_1"],
        "T_prompt_para_1": prompts["sensory_prompt_2"],
        "T_prompt_para_2": prompts["sensory_prompt_3"],
    }


def pick_multimodal_prompt_map(config: dict[str, Any]) -> dict[str, str]:
    prompts = config["prompts"]["text_only"]
    neutral = config["prompts"]["multimodal"]["neutral"]
    return {
        "M_text_only": neutral,
        "M_prompt_only": prompts["sensory_prompt_1"],
        "M_matched_image": neutral,
        "M_prompt_plus_matched_image": prompts["sensory_prompt_1"],
        "M_degraded_image": neutral,
        "M_mismatched_image": neutral,
        "M_blank_image": neutral,
    }


def select_precision(torch: Any, config: dict[str, Any]) -> tuple[str, Any]:
    runtime = config["analysis"]["runtime"]
    if torch.cuda.is_available():
        if runtime["default_precision"] == "bf16" and torch.cuda.is_bf16_supported():
            return "bf16", torch.bfloat16
        if runtime["fallback_precision"] == "fp16":
            return "fp16", torch.float16
    return "fp32", torch.float32


def model_load_kwargs(torch: Any, config: dict[str, Any]) -> dict[str, Any]:
    _, dtype = select_precision(torch, config)
    kwargs: dict[str, Any] = {"dtype": dtype, "local_files_only": True}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    return kwargs


def multimodal_load_kwargs(torch: Any, config: dict[str, Any]) -> dict[str, Any]:
    _, dtype = select_precision(torch, config)
    kwargs: dict[str, Any] = {"dtype": dtype, "local_files_only": True}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    return kwargs


def first_model_device(model: Any) -> Any:
    return next(model.parameters()).device


def move_batch_to_device(batch: Any, device: Any) -> Any:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def find_subsequence(sequence: list[int], subsequence: list[int]) -> tuple[int, int] | None:
    if not subsequence:
        return None
    width = len(subsequence)
    for start in range(0, len(sequence) - width + 1):
        if sequence[start : start + width] == subsequence:
            return start, start + width
    return None


def prompt_anchored_concept_char_span(rendered_text: str, prompt_text: str, concept_text: str) -> tuple[int, int]:
    prompt_start = rendered_text.find(prompt_text)
    if prompt_start < 0:
        raise RuntimeError(
            f"Rendered text did not contain the prompt text. Prompt={prompt_text!r} rendered_prefix={rendered_text[:200]!r}"
        )
    concept_relative = prompt_text.find(concept_text)
    if concept_relative < 0:
        raise RuntimeError(f"Prompt text did not contain concept {concept_text!r}. Prompt={prompt_text!r}")
    start = prompt_start + concept_relative
    return start, start + len(concept_text)


def token_span_from_offsets(offset_mapping: list[tuple[int, int]], char_start: int, char_end: int) -> tuple[int, int] | None:
    token_indices = [
        idx
        for idx, (start, end) in enumerate(offset_mapping)
        if end > start and start < char_end and end > char_start
    ]
    if not token_indices:
        return None
    return token_indices[0], token_indices[-1] + 1


def render_multimodal_text(processor: Any, prompt: str, image: Any | None) -> str:
    if hasattr(processor, "apply_chat_template"):
        content = []
        if image is not None:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def unwrap_tokenized_batch(tokenized: Any) -> tuple[list[int], list[tuple[int, int]] | None]:
    input_ids = tokenized["input_ids"]
    offset_mapping = tokenized.get("offset_mapping")
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
        if offset_mapping is not None:
            offset_mapping = offset_mapping[0]
    return input_ids, offset_mapping


def resolve_prompt_subsequence_span(
    tokenizer: Any,
    sequence_ids: list[int],
    prompt_text: str,
    concept_text: str,
    *,
    model_id: str,
    condition: str,
) -> tuple[int, int]:
    tokenized = tokenizer(prompt_text, add_special_tokens=False, return_offsets_mapping=True)
    prompt_ids, offset_mapping = unwrap_tokenized_batch(tokenized)
    if offset_mapping is None:
        raise RuntimeError(f"{model_id} tokenizer did not return offset mappings for prompt-subsequence alignment.")
    prompt_loc = find_subsequence(sequence_ids, prompt_ids)
    if prompt_loc is None:
        raise RuntimeError(
            f"Prompt token subsequence did not align with extraction input ids for model={model_id} "
            f"condition={condition} concept={concept_text!r} prompt={prompt_text!r}"
        )
    local_char_start, local_char_end = prompt_anchored_concept_char_span(prompt_text, prompt_text, concept_text)
    local_span = token_span_from_offsets(offset_mapping, local_char_start, local_char_end)
    if local_span is None:
        raise RuntimeError(
            f"Failed to map concept span within prompt tokens for model={model_id} "
            f"condition={condition} concept={concept_text!r} prompt={prompt_text!r}"
        )
    return prompt_loc[0] + local_span[0], prompt_loc[0] + local_span[1]


def resolve_text_span(
    tokenizer: Any,
    sequence_ids: list[int],
    rendered_text: str,
    prompt_text: str,
    concept_text: str,
    *,
    model_id: str,
    condition: str,
) -> tuple[int, int]:
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(f"{model_id} tokenizer is not fast; offset-based concept span alignment is unavailable.")
    if rendered_text != prompt_text:
        return resolve_prompt_subsequence_span(
            tokenizer,
            sequence_ids,
            prompt_text,
            concept_text,
            model_id=model_id,
            condition=condition,
        )
    tokenized = tokenizer(rendered_text, add_special_tokens=True, return_offsets_mapping=True)
    rendered_ids, offset_mapping = unwrap_tokenized_batch(tokenized)
    if offset_mapping is None:
        raise RuntimeError(f"{model_id} tokenizer did not return offset mappings for concept span alignment.")
    located = find_subsequence(sequence_ids, rendered_ids)
    if located is None:
        raise RuntimeError(
            f"Tokenized rendered text did not align with extraction input ids for model={model_id} "
            f"condition={condition} concept={concept_text!r} rendered_prefix={rendered_text[:200]!r}"
        )
    char_start, char_end = prompt_anchored_concept_char_span(rendered_text, prompt_text, concept_text)
    local_span = token_span_from_offsets(offset_mapping, char_start, char_end)
    if local_span is None:
        raise RuntimeError(
            f"Failed to map concept character span onto tokens for model={model_id} "
            f"condition={condition} concept={concept_text!r} rendered_prefix={rendered_text[:200]!r}"
        )
    return located[0] + local_span[0], located[0] + local_span[1]


def extract_hidden_states(outputs: Any) -> Any:
    if getattr(outputs, "hidden_states", None) is not None:
        return outputs.hidden_states
    nested = getattr(outputs, "language_model_outputs", None)
    if nested is not None and getattr(nested, "hidden_states", None) is not None:
        return nested.hidden_states
    raise RuntimeError("Model outputs did not expose hidden states in a supported location.")


def pool_text_hidden_states(hidden_states: Any, span_start: int, span_end: int) -> list[np.ndarray]:
    pooled = []
    require(span_end > span_start, f"Invalid span for text pooling: start={span_start}, end={span_end}")
    for layer_tensor in hidden_states:
        layer = layer_tensor[0]
        span = layer[span_start:span_end]
        pooled.append(span.mean(dim=0).detach().float().cpu().numpy().astype(np.float32))
    return pooled


def pool_vision_hidden_states(hidden_states: Any) -> list[np.ndarray]:
    pooled = []
    for layer_tensor in hidden_states:
        layer = layer_tensor[0]
        if layer.shape[0] > 1:
            layer = layer[1:]
        pooled.append(layer.mean(dim=0).detach().float().cpu().numpy().astype(np.float32))
    return pooled


def pool_single_vision_state(state: Any) -> np.ndarray:
    layer = state[0]
    if layer.shape[0] > 1:
        layer = layer[1:]
    return layer.mean(dim=0).detach().float().cpu().numpy().astype(np.float32)


def build_multimodal_inputs(processor: Any, prompt: str, image: Any | None) -> Any:
    if hasattr(processor, "apply_chat_template"):
        rendered = render_multimodal_text(processor, prompt, image)
        if image is None:
            return processor(text=rendered, return_tensors="pt")
        return processor(text=rendered, images=image, return_tensors="pt")
    if image is None:
        return processor(text=prompt, return_tensors="pt")
    return processor(images=image, text=prompt, return_tensors="pt")


def prepare_multimodal_image(image: Any, max_side: int) -> Any:
    if image is None:
        return None
    prepared = image.copy()
    prepared.thumbnail((max_side, max_side))
    return prepared


def load_multimodal_model(transformers: Any, model_id: str, kwargs: dict[str, Any]) -> Any:
    local_kwargs = dict(kwargs)
    local_kwargs.setdefault("attn_implementation", "eager")
    constructors = [
        getattr(transformers, "AutoModelForImageTextToText", None),
        getattr(transformers, "AutoModelForVision2Seq", None),
        getattr(transformers, "AutoModel", None),
    ]
    errors = []
    for constructor in constructors:
        if constructor is None:
            continue
        try:
            return constructor.from_pretrained(model_id, **local_kwargs)
        except Exception as exc:
            errors.append(f"{constructor.__name__}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors))


def resolve_cached_snapshot(model_id: str, cache_root: str) -> Path:
    org, name = model_id.split("/", 1)
    candidates = [
        Path(cache_root) / f"models--{org}--{name}",
        Path(cache_root) / "hub" / f"models--{org}--{name}",
    ]
    tried_refs = []
    for model_dir in candidates:
        refs_main = model_dir / "refs" / "main"
        tried_refs.append(str(refs_main))
        if not refs_main.exists():
            continue
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot_dir = model_dir / "snapshots" / revision
        require(snapshot_dir.exists(), f"Snapshot missing for {model_id}: {snapshot_dir}")
        return snapshot_dir
    raise RuntimeError(f"Cache ref missing for {model_id}: {', '.join(tried_refs)}")


def extract_anchor_hidden_states(model: Any, batch: Any) -> tuple[Any, str]:
    if hasattr(model, "vision_model"):
        outputs = model.vision_model(**batch, output_hidden_states=True)
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is not None and len(hidden_states) > 0:
            return hidden_states, "hidden_states"
        if getattr(outputs, "last_hidden_state", None) is not None:
            return [outputs.last_hidden_state], "last_hidden_state_fallback"
        raise RuntimeError("anchor model did not expose usable vision outputs")

    outputs = model(**batch, output_hidden_states=True)
    vision_output = getattr(outputs, "vision_model_output", None)
    hidden_states = getattr(vision_output, "hidden_states", None) if vision_output is not None else None
    if hidden_states is not None and len(hidden_states) > 0:
        return hidden_states, "hidden_states"
    if vision_output is not None and getattr(vision_output, "last_hidden_state", None) is not None:
        return [vision_output.last_hidden_state], "last_hidden_state_fallback"
    raise RuntimeError("anchor model did not expose usable vision outputs")


def remap_mismatch_for_subset(
    concepts: list[dict[str, str]],
    ready_images: dict[str, dict[str, str]],
    mismatch_map: dict[str, str],
) -> dict[str, str]:
    ready_sensory = [row for row in concepts if row["domain"] == "sensory" and row["concept"] in ready_images]
    ready_names = {row["concept"] for row in ready_sensory}
    by_subtype: dict[str, list[str]] = {}
    for row in ready_sensory:
        by_subtype.setdefault(row["subtype"], []).append(row["concept"])
    remapped = {}
    all_ready = sorted(ready_names)
    for row in ready_sensory:
        concept = row["concept"]
        candidate = mismatch_map.get(concept)
        if candidate in ready_names:
            remapped[concept] = candidate
            continue
        same_subtype = [name for name in sorted(by_subtype[row["subtype"]]) if name != concept]
        cross_subtype = [name for name in all_ready if name != concept and name not in by_subtype[row["subtype"]]]
        pool = same_subtype or cross_subtype
        require(pool, f"No mismatch candidate available for {concept}")
        remapped[concept] = pool[0]
    return remapped


def offset_records(records: list[dict[str, Any]], arrays: dict[str, np.ndarray], start: int) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    adjusted_records = []
    adjusted_arrays = {}
    for record in records:
        old_id = record["record_id"]
        new_id = start + old_id
        item = dict(record)
        item["record_id"] = new_id
        adjusted_records.append(item)
        adjusted_arrays[f"record_{new_id}"] = arrays[f"record_{old_id}"]
    return adjusted_records, adjusted_arrays


def embedding_output_paths(output_tag: str | None) -> tuple[Path, Path]:
    suffix = f"_{output_tag}" if output_tag else "_full"
    base = ROOT / "outputs" / "embeddings"
    return base / f"pooled_embeddings{suffix}.npz", base / f"embedding_metadata{suffix}.json"


def build_text_records(
    concepts: list[dict[str, str]],
    model_id: str,
    tokenizer: Any,
    model: Any,
    template_map: dict[str, str],
    torch: Any,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    arrays: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    by_domain: dict[str, list[dict[str, str]]] = {}
    for row in concepts:
        by_domain.setdefault(row["domain"], []).append(row)

    for condition, template in template_map.items():
        for domain, rows in by_domain.items():
            concept_names = [row["concept"] for row in rows]
            pooled_by_layer = None
            matched_spans = 0
            for concept in concept_names:
                prompt = template.format(concept=concept)
                batch = tokenizer(prompt, return_tensors="pt")
                span_start, span_end = resolve_text_span(
                    tokenizer,
                    batch["input_ids"][0].tolist(),
                    prompt,
                    prompt,
                    concept,
                    model_id=model_id,
                    condition=condition,
                )
                batch = move_batch_to_device(batch, first_model_device(model))
                with torch.no_grad():
                    outputs = model(**batch, output_hidden_states=True)
                pooled = pool_text_hidden_states(extract_hidden_states(outputs), span_start, span_end)
                matched_spans += 1
                if pooled_by_layer is None:
                    pooled_by_layer = [[] for _ in range(len(pooled))]
                for layer_index, vector in enumerate(pooled):
                    pooled_by_layer[layer_index].append(vector)
            for layer_index, vectors in enumerate(pooled_by_layer or []):
                record_id = len(records)
                arrays[f"record_{record_id}"] = np.stack(vectors).astype(np.float32)
                records.append(
                    {
                        "record_id": record_id,
                        "family": "text",
                        "model_id": model_id,
                        "condition": condition,
                        "domain": domain,
                        "layer": layer_index,
                        "num_concepts": len(concept_names),
                        "concepts": concept_names,
                    }
                )
            diagnostics.append(
                {
                    "model_id": model_id,
                    "family": "text",
                    "condition": condition,
                    "domain": domain,
                    "attempted_spans": len(concept_names),
                    "matched_spans": matched_spans,
                    "pooling_target": "concept_span",
                }
            )
    return arrays, records, diagnostics


def build_multimodal_records(
    concepts: list[dict[str, str]],
    model_id: str,
    config: dict[str, Any],
    processor: Any,
    model: Any,
    prompt_templates: dict[str, str],
    ready_images: dict[str, dict[str, str]],
    mismatch_map: dict[str, str],
    torch: Any,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    from PIL import Image

    arrays: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    sensory = [row for row in concepts if row["domain"] == "sensory"]
    concept_names = [row["concept"] for row in sensory]
    tokenizer = getattr(processor, "tokenizer", None)
    require(tokenizer is not None, f"{model_id} processor does not expose a tokenizer")
    max_image_side = int(config.get("analysis", {}).get("image_policy", {}).get("multimodal_max_side", 384))

    def image_for(condition: str, concept: str) -> Path | None:
        if condition in {"M_text_only", "M_prompt_only"}:
            return None
        if condition == "M_matched_image":
            return ROOT / ready_images[concept]["matched_image"]
        if condition == "M_degraded_image":
            return ROOT / ready_images[concept]["degraded_image"]
        if condition == "M_mismatched_image":
            return ROOT / ready_images[mismatch_map[concept]]["matched_image"]
        return ROOT / ready_images[concept]["matched_image"]

    multimodal_conditions = config["analysis"]["execution"].get(
        "multimodal_conditions",
        ["M_text_only", "M_prompt_only", "M_matched_image", "M_prompt_plus_matched_image", "M_degraded_image", "M_mismatched_image", "M_blank_image"],
    )
    requested_conditions = config.get("_requested_multimodal_conditions")
    if requested_conditions:
        multimodal_conditions = requested_conditions
    for condition in multimodal_conditions:
        pooled_by_layer = None
        matched_spans = 0
        for concept in concept_names:
            prompt = prompt_templates.get(condition, prompt_templates["M_text_only"]).format(concept=concept)
            image_path = image_for(condition, concept)
            if condition == "M_blank_image" and image_path is not None:
                template_image = Image.open(image_path).convert("RGB")
                image = Image.new("RGB", template_image.size, color=(128, 128, 128))
            else:
                image = Image.open(image_path).convert("RGB") if image_path is not None else None
            image = prepare_multimodal_image(image, max_image_side)
            rendered_text = render_multimodal_text(processor, prompt, image)
            batch = build_multimodal_inputs(processor, prompt, image)
            span_start, span_end = resolve_text_span(
                tokenizer,
                batch["input_ids"][0].tolist(),
                rendered_text,
                prompt,
                concept,
                model_id=model_id,
                condition=condition,
            )
            batch = move_batch_to_device(batch, first_model_device(model))
            with torch.no_grad():
                outputs = model(**batch, output_hidden_states=True)
            pooled = pool_text_hidden_states(extract_hidden_states(outputs), span_start, span_end)
            matched_spans += 1
            if pooled_by_layer is None:
                pooled_by_layer = [[] for _ in range(len(pooled))]
            for layer_index, vector in enumerate(pooled):
                pooled_by_layer[layer_index].append(vector)
        for layer_index, vectors in enumerate(pooled_by_layer or []):
            record_id = len(records)
            arrays[f"record_{record_id}"] = np.stack(vectors).astype(np.float32)
            records.append(
                {
                    "record_id": record_id,
                    "family": "multimodal",
                    "model_id": model_id,
                    "condition": condition,
                    "domain": "sensory",
                    "layer": layer_index,
                    "num_concepts": len(concept_names),
                        "concepts": concept_names,
                    }
                )
        diagnostics.append(
            {
                "model_id": model_id,
                "family": "multimodal",
                "condition": condition,
                "domain": "sensory",
                "attempted_spans": len(concept_names),
                "matched_spans": matched_spans,
                "pooling_target": "concept_span",
            }
        )
    return arrays, records, diagnostics


def build_anchor_records(
    concepts: list[dict[str, str]],
    model_id: str,
    processor: Any,
    model: Any,
    ready_images: dict[str, dict[str, str]],
    torch: Any,
    target_layer_count: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], str]:
    from PIL import Image

    arrays: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    sensory = [row for row in concepts if row["domain"] == "sensory"]
    concept_names = [row["concept"] for row in sensory]
    pooled_by_layer = None
    source = "hidden_states"
    for concept in concept_names:
        image = Image.open(ROOT / ready_images[concept]["matched_image"]).convert("RGB")
        batch = processor(images=image, return_tensors="pt")
        batch = move_batch_to_device(batch, first_model_device(model))
        with torch.no_grad():
            hidden_states, source = extract_anchor_hidden_states(model, batch)
        if source == "last_hidden_state_fallback":
            pooled = [pool_single_vision_state(hidden_states[0]) for _ in range(target_layer_count)]
        else:
            pooled = pool_vision_hidden_states(hidden_states)
            if len(pooled) < target_layer_count:
                pooled.extend([pooled[-1]] * (target_layer_count - len(pooled)))
        if pooled_by_layer is None:
            pooled_by_layer = [[] for _ in range(len(pooled))]
        for layer_index, vector in enumerate(pooled[:target_layer_count]):
            pooled_by_layer[layer_index].append(vector)
    for layer_index, vectors in enumerate(pooled_by_layer or []):
        record_id = len(records)
        arrays[f"record_{record_id}"] = np.stack(vectors).astype(np.float32)
        records.append(
                    {
                        "record_id": record_id,
                        "family": "anchor",
                        "model_id": model_id,
                        "condition": "reference_anchor_image",
                        "domain": "sensory",
                "layer": layer_index,
                "num_concepts": len(concept_names),
                "concepts": concept_names,
            }
        )
    return arrays, records, source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--models")
    parser.add_argument("--concept-subset")
    parser.add_argument("--output-tag")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--multimodal-conditions", default="", help="Comma-separated multimodal conditions to extract; useful for targeted additions.")
    args = parser.parse_args()

    config = load_project_config(args.config)
    if args.multimodal_conditions:
        config["_requested_multimodal_conditions"] = [item.strip() for item in args.multimodal_conditions.split(",") if item.strip()]
    set_global_seed(config["seeds"]["global"])
    configure_hf_cache(config)
    effective_subset = args.concept_subset
    if not effective_subset:
        default_subset = config["analysis"].get("execution", {}).get("default_concept_subset", "")
        if default_subset and (ROOT / default_subset).exists():
            effective_subset = default_subset

    embeddings_npz, metadata_json = embedding_output_paths(args.output_tag)
    require(args.overwrite or not embeddings_npz.exists(), f"{embeddings_npz} already exists. Re-run with --overwrite to replace it.")
    require(args.overwrite or not metadata_json.exists(), f"{metadata_json} already exists. Re-run with --overwrite to replace it.")

    concepts = load_concepts(config, effective_subset)
    ready_images = load_ready_image_map()
    validate_image_coverage(config, concepts, ready_images, effective_subset)
    mismatch_map = remap_mismatch_for_subset(concepts, ready_images, load_mismatch_map())
    selected_models = parse_requested_models(config, args.models)

    import torch
    import transformers

    template_map = pick_text_template_map(config)
    multimodal_prompt_map = pick_multimodal_prompt_map(config)
    precision_name, _ = select_precision(torch, config)
    cache_root = config["analysis"]["runtime"].get("hf_cache_dir", "")

    all_arrays: dict[str, np.ndarray] = {}
    all_records: list[dict[str, Any]] = []
    model_metadata: list[dict[str, Any]] = []
    span_pooling_diagnostics: list[dict[str, Any]] = []
    start = 0
    target_layer_count = 0

    for row in selected_models:
        family = row["family"]
        model_id = row["model_id"]
        model_source = str(resolve_cached_snapshot(model_id, cache_root)) if cache_root else model_id
        if family == "text":
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_source, **tokenizer_load_kwargs(model_source))
            model = transformers.AutoModelForCausalLM.from_pretrained(model_source, **model_load_kwargs(torch, config)).eval()
            arrays, records, diagnostics = build_text_records(concepts, model_id, tokenizer, model, template_map, torch)
            target_layer_count = max(target_layer_count, max(record["layer"] for record in records) + 1)
            adjusted_records, adjusted_arrays = offset_records(records, arrays, start)
            all_records.extend(adjusted_records)
            all_arrays.update(adjusted_arrays)
            span_pooling_diagnostics.extend(diagnostics)
            start += len(records)
            model_metadata.append({"family": family, "model_id": model_id, "device": str(first_model_device(model)), "precision": str(next(model.parameters()).dtype)})
            del model
        elif family == "multimodal":
            processor = transformers.AutoProcessor.from_pretrained(model_source, **tokenizer_load_kwargs(model_source))
            model = load_multimodal_model(transformers, model_source, multimodal_load_kwargs(torch, config)).eval()
            arrays, records, diagnostics = build_multimodal_records(concepts, model_id, config, processor, model, multimodal_prompt_map, ready_images, mismatch_map, torch)
            target_layer_count = max(target_layer_count, max(record["layer"] for record in records) + 1)
            adjusted_records, adjusted_arrays = offset_records(records, arrays, start)
            all_records.extend(adjusted_records)
            all_arrays.update(adjusted_arrays)
            span_pooling_diagnostics.extend(diagnostics)
            start += len(records)
            model_metadata.append({"family": family, "model_id": model_id, "device": str(first_model_device(model)), "precision": str(next(model.parameters()).dtype)})
            del model
        else:
            continue
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    anchor_rows = [row for row in selected_models if row["family"] == "anchor"]
    for row in anchor_rows:
        model_id = row["model_id"]
        model_source = str(resolve_cached_snapshot(model_id, cache_root)) if cache_root else model_id
        processor = transformers.AutoProcessor.from_pretrained(model_source, **tokenizer_load_kwargs(model_source))
        model = transformers.AutoModel.from_pretrained(model_source, **model_load_kwargs(torch, config)).eval()
        arrays, records, source = build_anchor_records(concepts, model_id, processor, model, ready_images, torch, target_layer_count)
        adjusted_records, adjusted_arrays = offset_records(records, arrays, start)
        all_records.extend(adjusted_records)
        all_arrays.update(adjusted_arrays)
        start += len(records)
        model_metadata.append(
            {
                "family": "anchor",
                "model_id": model_id,
                "device": str(first_model_device(model)),
                "precision": str(next(model.parameters()).dtype),
                "vision_source": source,
            }
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    embeddings_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(embeddings_npz, **all_arrays)
    metadata = {
        "mode": "real_model_extraction",
        "output_tag": args.output_tag or "",
        "concept_subset": effective_subset or "",
        "selected_models": model_metadata,
        "precision": precision_name,
        "record_count": len(all_records),
        "span_pooling_diagnostics": span_pooling_diagnostics,
        "records": [{**record, "condition": canonical_condition_name(record["condition"])} for record in all_records],
    }
    write_json(metadata_json, metadata)
    append_run_log(
        "Full Extraction",
        [
            f"Wrote pooled embeddings to {embeddings_npz.relative_to(ROOT)}.",
            f"Wrote extraction metadata to {metadata_json.relative_to(ROOT)}.",
            f"Extracted {len(all_records)} layerwise records across {len(selected_models)} models.",
            f"Concept subset: {effective_subset or 'full_concept_list.csv'}.",
            f"Output tag: {args.output_tag or 'full'}.",
        ],
    )


if __name__ == "__main__":
    main()
