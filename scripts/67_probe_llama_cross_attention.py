from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import ROOT, append_run_log, load_project_config, read_csv, require, write_csv, write_json
from hardening_common import write_text


LLAMA_FAMILY = "llama"
CONDITIONS = ["M_matched_image", "M_mismatched_image", "M_blank_image"]
PROMPT_TEMPLATE_VERSION = "llama_cross_attention_probe_v1"
PROMPT = 'The text concept is "{concept}". Describe the concept using the available input.'


def load_stage01_module() -> Any:
    script_path = ROOT / "scripts" / "01_extract_hidden_states.py"
    spec = importlib.util.spec_from_file_location("stage01_extract_hidden_states", script_path)
    require(spec is not None and spec.loader is not None, f"Failed to load helper module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def llama_model_id(config: dict[str, Any]) -> str:
    for family in config["analysis"]["analysis"].get("cross_family_families", []):
        if str(family.get("family_name")) == LLAMA_FAMILY:
            return str(family["multimodal_model"])
    raise RuntimeError("Missing llama family in config cross_family_families.")


def image_for(condition: str, item: dict[str, str], ready_images: dict[str, dict[str, str]]) -> Path:
    if condition in {"M_matched_image", "M_blank_image"}:
        return ROOT / ready_images[item["concept"]]["matched_image"]
    if condition == "M_mismatched_image":
        return ROOT / ready_images[item["mismatch_source"]]["matched_image"]
    raise RuntimeError(f"Unsupported condition: {condition}")


def get_cross_attention_layers(model: Any, model_source: str) -> list[int]:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    layers = getattr(text_config, "cross_attention_layers", None)
    if layers:
        return [int(layer) for layer in layers]
    config_path = Path(model_source) / "config.json"
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        return [int(layer) for layer in payload.get("text_config", {}).get("cross_attention_layers", [])]
    return [3, 8, 13, 18, 23, 28, 33, 38]


def output_attention_sequence(outputs: Any) -> tuple[str, Any]:
    for name in ["cross_attentions", "attentions"]:
        value = getattr(outputs, name, None)
        if value is not None:
            return name, value
    nested = getattr(outputs, "language_model_outputs", None)
    if nested is not None:
        for name in ["cross_attentions", "attentions"]:
            value = getattr(nested, name, None)
            if value is not None:
                return f"language_model_outputs.{name}", value
    raise RuntimeError("No attentions or cross_attentions found in model outputs.")


def map_attention_layers(attentions: Any, cross_layers: list[int]) -> list[tuple[int, Any]]:
    tensors = [tensor for tensor in attentions if tensor is not None]
    if len(tensors) == len(cross_layers):
        return list(zip(cross_layers, tensors))
    return list(zip(range(len(tensors)), tensors))


def image_token_positions(input_ids: Any, image_token_index: int) -> list[int]:
    ids = input_ids[0].detach().cpu().tolist()
    return [idx for idx, token_id in enumerate(ids) if int(token_id) == int(image_token_index)]


def attention_stats_for_span(
    tensor: Any,
    *,
    span_start: int,
    span_end: int,
    input_length: int,
    image_positions: list[int],
) -> tuple[float, float, float, float, int, str]:
    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim != 4:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0, f"unsupported_shape_{arr.shape}"
    key_length = arr.shape[-1]
    query_length = arr.shape[-2]
    query_start = max(0, min(span_start, query_length - 1))
    query_end = max(query_start + 1, min(span_end, query_length))
    if key_length == input_length and image_positions:
        key_positions = [idx for idx in image_positions if idx < key_length]
        key_mode = "image_token_positions"
    else:
        key_positions = list(range(key_length))
        key_mode = "cross_attention_visual_keys"
    if not key_positions:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0, key_mode
    sub = arr[0, :, query_start:query_end, :][:, :, key_positions]
    flat = sub.reshape(-1, len(key_positions))
    row_sums = np.maximum(flat.sum(axis=1, keepdims=True), 1e-12)
    probs = np.clip(flat / row_sums, 1e-12, 1.0)
    entropy = -np.sum(probs * np.log(probs), axis=1)
    max_entropy = np.log(len(key_positions)) if len(key_positions) > 1 else 1.0
    normalized_entropy = entropy / max_entropy
    top1 = np.max(probs, axis=1)
    return (
        float(np.nanmean(sub)),
        float(np.nanmean(top1)),
        float(np.nanmean(entropy)),
        float(np.nanmean(normalized_entropy)),
        len(key_positions),
        key_mode,
    )


def summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    df = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    lines = [
        "# Llama Cross-Attention Probe",
        "",
        "This probe measures attention from the concept-token span to image-token keys at Llama cross-attention layers.",
        "",
        "| Condition | Layer | Top-1 attention | Normalized entropy | n | Key mode |",
        "|---|---:|---:|---:|---|",
    ]
    if df.empty:
        return summary_rows, lines + ["", "- No rows produced."]
    grouped = df.groupby(["condition", "attention_layer", "key_mode"], as_index=False).agg(
        n=("concept", "size"),
        mean_attention=("concept_to_image_attention", "mean"),
        median_attention=("concept_to_image_attention", "median"),
        mean_top1_attention=("concept_to_image_top1_attention", "mean"),
        mean_normalized_entropy=("concept_to_image_normalized_entropy", "mean"),
        mean_num_image_keys=("num_image_keys", "mean"),
    )
    cross_grouped = grouped[grouped["key_mode"].eq("cross_attention_visual_keys")].copy()
    display_grouped = cross_grouped if not cross_grouped.empty else grouped
    for _, row in display_grouped.iterrows():
        item = row.to_dict()
        summary_rows.append(item)
        lines.append(
            f"| `{item['condition']}` | {int(item['attention_layer'])} | {item['mean_top1_attention']:.6f} | "
            f"{item['mean_normalized_entropy']:.6f} | {int(item['n'])} | `{item['key_mode']}` |"
        )
    condition_df = df[df["key_mode"].eq("cross_attention_visual_keys")].copy()
    if condition_df.empty:
        condition_df = df
    condition_summary = condition_df.groupby("condition", as_index=False).agg(
        n=("concept", "size"),
        mean_attention=("concept_to_image_attention", "mean"),
        median_attention=("concept_to_image_attention", "median"),
        mean_top1_attention=("concept_to_image_top1_attention", "mean"),
        mean_normalized_entropy=("concept_to_image_normalized_entropy", "mean"),
        valid_attention_rate=("concept_to_image_attention", lambda x: float(np.isfinite(x).mean())),
    )
    lines.extend(["", "## Condition Means", "", "| Condition | Top-1 attention | Normalized entropy | Mean attention | Valid rate | n |", "|---|---:|---:|---:|---:|---:|"])
    for _, row in condition_summary.iterrows():
        lines.append(
            f"| `{row['condition']}` | {row['mean_top1_attention']:.6f} | {row['mean_normalized_entropy']:.6f} | {row['mean_attention']:.6f} | "
            f"{row['valid_attention_rate']:.4f} | {int(row['n'])} |"
        )
    return summary_rows, lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Llama concept-token attention to image tokens at cross-attention layers.")
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--items", default="outputs/metrics/identity_similarity_probe_items.csv")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--output-stem", default="")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    import torch
    import transformers
    from PIL import Image

    if args.smoke:
        args.limit = min(args.limit, 12)
    output_stem = args.output_stem or ("llama_cross_attention_probe_smoke" if args.smoke else "llama_cross_attention_probe")

    stage01 = load_stage01_module()
    config = load_project_config(args.config)
    stage01.configure_hf_cache(config)
    model_id = llama_model_id(config)
    cache_root = config["analysis"]["runtime"].get("hf_cache_dir", "")
    model_source = str(stage01.resolve_cached_snapshot(model_id, cache_root)) if cache_root else model_id
    processor = transformers.AutoProcessor.from_pretrained(model_source, **stage01.tokenizer_load_kwargs(model_source))
    tokenizer = getattr(processor, "tokenizer", None)
    require(tokenizer is not None, f"{model_id} processor does not expose a tokenizer")
    model = stage01.load_multimodal_model(transformers, model_source, stage01.multimodal_load_kwargs(torch, config)).eval()
    device = stage01.first_model_device(model)
    cross_layers = get_cross_attention_layers(model, model_source)
    image_token_index = int(getattr(model.config, "image_token_index", 128256))
    max_side = int(config.get("analysis", {}).get("image_policy", {}).get("multimodal_max_side", 384))
    ready_images = stage01.load_ready_image_map()
    items = read_csv(ROOT / args.items)
    if args.limit:
        items = items[: args.limit]
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]

    rows: list[dict[str, Any]] = []
    failure: str | None = None
    for item in items:
        concept = item["concept"]
        prompt = PROMPT.format(concept=concept)
        for condition in conditions:
            path = image_for(condition, item, ready_images)
            if condition == "M_blank_image":
                template = Image.open(path).convert("RGB")
                image = Image.new("RGB", template.size, color=(128, 128, 128))
            else:
                image = Image.open(path).convert("RGB")
            image = stage01.prepare_multimodal_image(image, max_side)
            rendered_text = stage01.render_multimodal_text(processor, prompt, image)
            batch = stage01.build_multimodal_inputs(processor, prompt, image)
            span_start, span_end = stage01.resolve_text_span(
                tokenizer,
                batch["input_ids"][0].tolist(),
                rendered_text,
                prompt,
                concept,
                model_id=model_id,
                condition=condition,
            )
            image_positions = image_token_positions(batch["input_ids"], image_token_index)
            batch = stage01.move_batch_to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch, output_attentions=True, use_cache=False)
            try:
                attention_source, attentions = output_attention_sequence(outputs)
            except RuntimeError as exc:
                failure = str(exc)
                break
            for attention_layer, tensor in map_attention_layers(attentions, cross_layers):
                mean_attention, top1_attention, entropy, normalized_entropy, num_keys, key_mode = attention_stats_for_span(
                    tensor,
                    span_start=span_start,
                    span_end=span_end,
                    input_length=int(batch["input_ids"].shape[-1]),
                    image_positions=image_positions,
                )
                rows.append(
                    {
                        "concept": concept,
                        "condition": condition,
                        "mismatch_source": item["mismatch_source"],
                        "attention_layer": int(attention_layer),
                        "attention_source": attention_source,
                        "concept_to_image_attention": mean_attention,
                        "concept_to_image_top1_attention": top1_attention,
                        "concept_to_image_entropy": entropy,
                        "concept_to_image_normalized_entropy": normalized_entropy,
                        "num_image_keys": int(num_keys),
                        "key_mode": key_mode,
                        "span_start": int(span_start),
                        "span_end": int(span_end),
                        "input_length": int(batch["input_ids"].shape[-1]),
                        "image_token_count": len(image_positions),
                        "model_id": model_id,
                        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                    }
                )
        if failure:
            break

    out_dir = ROOT / "outputs" / "metrics"
    report_dir = ROOT / "reports" / "main_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{output_stem}.csv"
    json_path = out_dir / f"{output_stem}_summary.json"
    report_path = report_dir / f"{output_stem}_report.md"
    fieldnames = [
        "concept",
        "condition",
        "mismatch_source",
        "attention_layer",
        "attention_source",
        "concept_to_image_attention",
        "concept_to_image_top1_attention",
        "concept_to_image_entropy",
        "concept_to_image_normalized_entropy",
        "num_image_keys",
        "key_mode",
        "span_start",
        "span_end",
        "input_length",
        "image_token_count",
        "model_id",
        "prompt_template_version",
    ]
    write_csv(csv_path, rows, fieldnames)
    summary_rows, lines = summarize(rows)
    payload = {
        "model_id": model_id,
        "cross_attention_layers": cross_layers,
        "image_token_index": image_token_index,
        "num_rows": len(rows),
        "num_concepts": len({row["concept"] for row in rows}),
        "conditions": conditions,
        "failure": failure,
        "summary": summary_rows,
    }
    write_json(json_path, payload)
    if failure:
        lines.extend(["", "## Failure", f"- `{failure}`"])
    write_text(report_path, "\n".join(lines) + "\n")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    append_run_log("Llama Cross-Attention Probe", [f"Wrote {csv_path.relative_to(ROOT)} with {len(rows)} rows. Failure: {failure or 'none'}."])
    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
