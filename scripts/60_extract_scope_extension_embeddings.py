from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import ROOT, append_run_log, load_project_config, require, set_global_seed, write_json


TEXT_CONDITIONS = ["T_prompt_primary"]
MULTIMODAL_CONDITIONS = ["M_text_only", "M_matched_image", "M_prompt_plus_matched_image", "M_mismatched_image", "M_blank_image"]


def load_stage01_module() -> Any:
    script_path = ROOT / "scripts" / "01_extract_hidden_states.py"
    spec = importlib.util.spec_from_file_location("stage01_extract_hidden_states", script_path)
    require(spec is not None and spec.loader is not None, f"Failed to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prompt_for(dataset: str, condition: str, label: str) -> str:
    if dataset == "imsitu":
        if condition in {"T_prompt_primary", "M_prompt_plus_matched_image"}:
            return f"Imagine vividly perceiving the event: {label}."
        return f"The event is {label}."
    if condition in {"T_prompt_primary", "M_prompt_plus_matched_image"}:
        return f"Imagine vividly perceiving the visual concept: {label}."
    return f"The visual concept is {label}."


def model_ids(config: dict[str, Any]) -> tuple[str, str]:
    execution = config["analysis"]["execution"]
    return execution["sensory_backbone_text_model"], execution["sensory_backbone_multimodal_model"]


def load_manifest(dataset: str, limit: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = ROOT / "outputs" / "scope_extensions"
    label_path = out_dir / f"{dataset}_labels.csv"
    image_path = out_dir / f"{dataset}_images.csv"
    require(label_path.exists(), f"Missing labels manifest: {label_path}")
    require(image_path.exists(), f"Missing image manifest: {image_path}")
    labels = pd.read_csv(label_path)
    images = pd.read_csv(image_path)
    if limit:
        labels = labels.head(limit).copy()
        images = images[images["label"].isin(set(labels["label"]))].copy()
    return labels, images


def image_paths_for(images: pd.DataFrame, label: str, image_limit: int) -> list[Path]:
    rows = images[images["label"].eq(label)].sort_values("image_index")
    if image_limit:
        rows = rows.head(image_limit)
    return [ROOT / str(path) for path in rows["image_path"]]


def average_vectors(vectors: list[list[np.ndarray]]) -> list[np.ndarray]:
    if not vectors:
        raise RuntimeError("No vectors to average.")
    layer_count = len(vectors[0])
    output = []
    for layer_idx in range(layer_count):
        output.append(np.mean(np.stack([item[layer_idx] for item in vectors]), axis=0).astype(np.float32))
    return output


def extract_text_condition(
    *,
    dataset: str,
    labels: pd.DataFrame,
    condition: str,
    model_id: str,
    tokenizer: Any,
    model: Any,
    stage01: Any,
    torch: Any,
) -> tuple[list[list[np.ndarray]], list[dict[str, Any]]]:
    vectors_by_label = []
    diagnostics = []
    for label in labels["label"]:
        prompt = prompt_for(dataset, condition, str(label))
        batch = tokenizer(prompt, return_tensors="pt")
        span_start, span_end = stage01.resolve_text_span(
            tokenizer,
            batch["input_ids"][0].tolist(),
            prompt,
            prompt,
            str(label),
            model_id=model_id,
            condition=condition,
        )
        batch = stage01.move_batch_to_device(batch, stage01.first_model_device(model))
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True)
        vectors_by_label.append(stage01.pool_text_hidden_states(stage01.extract_hidden_states(outputs), span_start, span_end))
        diagnostics.append({"label": label, "condition": condition, "matched_spans": 1, "attempted_spans": 1})
    return vectors_by_label, diagnostics


def extract_multimodal_condition(
    *,
    dataset: str,
    labels: pd.DataFrame,
    images: pd.DataFrame,
    condition: str,
    model_id: str,
    processor: Any,
    model: Any,
    stage01: Any,
    torch: Any,
    image_limit: int,
    max_image_side: int,
) -> tuple[list[list[np.ndarray]], list[dict[str, Any]]]:
    from PIL import Image

    tokenizer = getattr(processor, "tokenizer", None)
    require(tokenizer is not None, f"{model_id} processor does not expose a tokenizer")
    vectors_by_label = []
    diagnostics = []
    for _, row in labels.iterrows():
        label = str(row["label"])
        prompt = prompt_for(dataset, condition, label)
        if condition == "M_text_only":
            paths: list[Path | None] = [None]
        elif condition == "M_mismatched_image":
            paths = image_paths_for(images, str(row["mismatch_label"]), image_limit)
        else:
            paths = image_paths_for(images, label, image_limit)
        require(paths, f"No image paths for {dataset} label={label} condition={condition}")
        exemplar_vectors = []
        for path in paths:
            if path is None:
                image = None
            elif condition == "M_blank_image":
                template = Image.open(path).convert("RGB")
                image = Image.new("RGB", template.size, color=(128, 128, 128))
            else:
                image = Image.open(path).convert("RGB")
            image = stage01.prepare_multimodal_image(image, max_image_side)
            rendered_text = stage01.render_multimodal_text(processor, prompt, image)
            batch = stage01.build_multimodal_inputs(processor, prompt, image)
            span_start, span_end = stage01.resolve_text_span(
                tokenizer,
                batch["input_ids"][0].tolist(),
                rendered_text,
                prompt,
                label,
                model_id=model_id,
                condition=condition,
            )
            batch = stage01.move_batch_to_device(batch, stage01.first_model_device(model))
            with torch.no_grad():
                outputs = model(**batch, output_hidden_states=True)
            exemplar_vectors.append(stage01.pool_text_hidden_states(stage01.extract_hidden_states(outputs), span_start, span_end))
        vectors_by_label.append(average_vectors(exemplar_vectors))
        diagnostics.append({"label": label, "condition": condition, "matched_spans": len(exemplar_vectors), "attempted_spans": len(exemplar_vectors)})
    return vectors_by_label, diagnostics


def add_records(
    arrays: dict[str, np.ndarray],
    records: list[dict[str, Any]],
    *,
    vectors_by_label: list[list[np.ndarray]],
    labels: list[str],
    family: str,
    model_id: str,
    condition: str,
    dataset: str,
) -> None:
    layer_count = len(vectors_by_label[0])
    for layer_idx in range(layer_count):
        record_id = len(records)
        arrays[f"record_{record_id}"] = np.stack([vectors[layer_idx] for vectors in vectors_by_label]).astype(np.float32)
        records.append(
            {
                "record_id": record_id,
                "family": family,
                "model_id": model_id,
                "condition": condition,
                "domain": dataset,
                "layer": layer_idx,
                "num_concepts": len(labels),
                "concepts": labels,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Qwen hidden states for imSitu or MIT-States scope extensions.")
    parser.add_argument("--dataset", required=True, choices=["imsitu", "mitstates"])
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--output-stem", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--image-limit", type=int, default=0)
    parser.add_argument("--conditions", default=",".join(TEXT_CONDITIONS + MULTIMODAL_CONDITIONS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=20260501)
    args = parser.parse_args()

    import torch
    import transformers

    stage01 = load_stage01_module()
    config = load_project_config(args.config)
    stage01.configure_hf_cache(config)
    set_global_seed(args.seed)
    labels_df, images_df = load_manifest(args.dataset, args.limit)
    labels = [str(label) for label in labels_df["label"]]
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    output_stem = args.output_stem or f"{args.dataset}_qwen_scope_embeddings"
    out_dir = ROOT / "outputs" / "scope_extensions"
    npz_path = out_dir / f"{output_stem}.npz"
    json_path = out_dir / f"{output_stem}.json"
    require(args.overwrite or not npz_path.exists(), f"{npz_path} exists; pass --overwrite")
    require(args.overwrite or not json_path.exists(), f"{json_path} exists; pass --overwrite")

    text_model, multimodal_model = model_ids(config)
    cache_root = config["analysis"]["runtime"].get("hf_cache_dir", "")
    text_source = str(stage01.resolve_cached_snapshot(text_model, cache_root)) if cache_root else text_model
    multimodal_source = str(stage01.resolve_cached_snapshot(multimodal_model, cache_root)) if cache_root else multimodal_model
    max_image_side = int(config.get("analysis", {}).get("image_policy", {}).get("multimodal_max_side", 384))

    arrays: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    text_conditions = [condition for condition in conditions if condition.startswith("T_")]
    if text_conditions:
        tokenizer = transformers.AutoTokenizer.from_pretrained(text_source, **stage01.tokenizer_load_kwargs(text_source))
        model = transformers.AutoModelForCausalLM.from_pretrained(text_source, **stage01.model_load_kwargs(torch, config)).eval()
        for condition in text_conditions:
            vectors, diag = extract_text_condition(
                dataset=args.dataset,
                labels=labels_df,
                condition=condition,
                model_id=text_model,
                tokenizer=tokenizer,
                model=model,
                stage01=stage01,
                torch=torch,
            )
            diagnostics.extend(diag)
            add_records(arrays, records, vectors_by_label=vectors, labels=labels, family="text", model_id=text_model, condition=condition, dataset=args.dataset)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    multimodal_conditions = [condition for condition in conditions if condition.startswith("M_")]
    if multimodal_conditions:
        processor = transformers.AutoProcessor.from_pretrained(multimodal_source, **stage01.tokenizer_load_kwargs(multimodal_source))
        model = stage01.load_multimodal_model(transformers, multimodal_source, stage01.multimodal_load_kwargs(torch, config)).eval()
        for condition in multimodal_conditions:
            vectors, diag = extract_multimodal_condition(
                dataset=args.dataset,
                labels=labels_df,
                images=images_df,
                condition=condition,
                model_id=multimodal_model,
                processor=processor,
                model=model,
                stage01=stage01,
                torch=torch,
                image_limit=args.image_limit,
                max_image_side=max_image_side,
            )
            diagnostics.extend(diag)
            add_records(arrays, records, vectors_by_label=vectors, labels=labels, family="multimodal", model_id=multimodal_model, condition=condition, dataset=args.dataset)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_dir.mkdir(parents=True, exist_ok=True)
    # Scope-extension bundles are large; uncompressed npz avoids wasting walltime on compression.
    np.savez(npz_path, **arrays)
    write_json(
        json_path,
        {
            "dataset": args.dataset,
            "output_stem": output_stem,
            "label_count": len(labels),
            "image_limit": args.image_limit,
            "conditions": conditions,
            "text_model": text_model,
            "multimodal_model": multimodal_model,
            "records": records,
            "span_pooling_diagnostics": diagnostics,
        },
    )
    append_run_log("Scope Extension Extraction", [f"Wrote {npz_path.relative_to(ROOT)} with {len(records)} layerwise records."])


if __name__ == "__main__":
    main()
