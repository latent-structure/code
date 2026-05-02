from __future__ import annotations

import argparse
import gc
import importlib.util
from pathlib import Path
from typing import Any

from common import ROOT, append_run_log, load_project_config, read_csv, require, set_global_seed, write_csv


CONDITIONS = [
    "T_prompt_primary",
    "M_text_only",
    "M_matched_image",
    "M_prompt_plus_matched_image",
    "M_mismatched_image",
    "M_blank_image",
]

CONDITION_SORT_INDEX = {condition: idx for idx, condition in enumerate(CONDITIONS)}

PROMPT_TEMPLATE_VERSION = "behavior_probe_v2_constrained_10_20_words"
BEHAVIOR_TASK = (
    'In 10-20 words, describe the visible appearance, texture, and sensory details of "{concept}". '
    "Do not define it. Do not include reasoning."
)


def load_stage01_module() -> Any:
    script_path = ROOT / "scripts" / "01_extract_hidden_states.py"
    spec = importlib.util.spec_from_file_location("stage01_extract_hidden_states", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load helper module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def balanced_concept_subset(rows: list[dict[str, str]], limit: int, seed: int) -> list[dict[str, str]]:
    import random

    sensory = [row for row in rows if row["domain"] == "sensory"]
    if limit <= 0 or limit >= len(sensory):
        return sorted(sensory, key=lambda row: row["concept"])
    by_subtype: dict[str, list[dict[str, str]]] = {}
    for row in sensory:
        by_subtype.setdefault(row["subtype"], []).append(row)
    rng = random.Random(seed)
    for subtype_rows in by_subtype.values():
        subtype_rows.sort(key=lambda row: row["concept"])
        rng.shuffle(subtype_rows)

    selected: list[dict[str, str]] = []
    subtype_names = sorted(by_subtype)
    while len(selected) < limit:
        progressed = False
        for subtype in subtype_names:
            if by_subtype[subtype]:
                selected.append(by_subtype[subtype].pop())
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
    selected.sort(key=lambda row: row["concept"])
    return selected


def generation_prompt(config: dict[str, Any], condition: str, concept: str) -> str:
    task = BEHAVIOR_TASK.format(concept=concept)
    if condition in {"T_prompt_primary", "M_prompt_plus_matched_image"}:
        sensory = config["prompts"]["text_only"]["sensory_prompt_1"].format(concept=concept)
        return f"{sensory} {task}"
    return task


def normalize_generated_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def generation_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["concept"]).lower(), str(row["condition"])


def load_resume_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in read_csv(path):
        key = generation_key(row)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def sort_generation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (str(row["concept"]).lower(), CONDITION_SORT_INDEX.get(str(row["condition"]), 999)))


def decode_new_tokens(tokenizer: Any, input_ids: Any, generated_ids: Any) -> str:
    prompt_len = int(input_ids.shape[-1])
    new_tokens = generated_ids[0][prompt_len:]
    return normalize_generated_text(tokenizer.decode(new_tokens, skip_special_tokens=True))


def generation_kwargs(tokenizer: Any, max_new_tokens: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
    }
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        kwargs["eos_token_id"] = eos_token_id
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    return kwargs


def image_for_condition(
    stage01: Any,
    condition: str,
    concept: str,
    ready_images: dict[str, dict[str, str]],
    mismatch_map: dict[str, str],
) -> tuple[Any | None, str, str]:
    from PIL import Image

    if condition == "M_text_only":
        return None, "", ""
    if condition == "M_mismatched_image":
        source = mismatch_map[concept]
        path = ROOT / ready_images[source]["matched_image"]
    else:
        source = concept
        path = ROOT / ready_images[concept]["matched_image"]
    image = Image.open(path).convert("RGB")
    if condition == "M_blank_image":
        image = Image.new("RGB", image.size, color=(128, 128, 128))
        return image, str(path.relative_to(ROOT)), source
    return image, str(path.relative_to(ROOT)), source


def generate_text_rows(
    *,
    concepts: list[dict[str, str]],
    config: dict[str, Any],
    model_id: str,
    model_source: str,
    max_new_tokens: int,
    torch: Any,
    transformers: Any,
    stage01: Any,
    completed: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    completed = completed or set()
    pending = [row for row in concepts if (row["concept"].lower(), "T_prompt_primary") not in completed]
    if not pending:
        return []
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_source, **stage01.tokenizer_load_kwargs(model_source))
    model = transformers.AutoModelForCausalLM.from_pretrained(model_source, **stage01.model_load_kwargs(torch, config)).eval()
    rows = []
    kwargs = generation_kwargs(tokenizer, max_new_tokens)
    for row in pending:
        concept = row["concept"]
        prompt = generation_prompt(config, "T_prompt_primary", concept)
        batch = tokenizer(prompt, return_tensors="pt")
        batch = stage01.move_batch_to_device(batch, stage01.first_model_device(model))
        with torch.no_grad():
            generated = model.generate(**batch, **kwargs)
        rows.append(
            {
                "concept": concept,
                "subtype": row["subtype"],
                "condition": "T_prompt_primary",
                "model_id": model_id,
                "prompt": prompt,
                "image_path": "",
                "mismatch_source": "",
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "generated_text": decode_new_tokens(tokenizer, batch["input_ids"], generated),
            }
        )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def generate_multimodal_rows(
    *,
    concepts: list[dict[str, str]],
    config: dict[str, Any],
    model_id: str,
    model_source: str,
    max_new_tokens: int,
    torch: Any,
    transformers: Any,
    stage01: Any,
    ready_images: dict[str, dict[str, str]],
    mismatch_map: dict[str, str],
    completed: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    completed = completed or set()
    pending_by_condition: dict[str, list[dict[str, str]]] = {}
    for condition in CONDITIONS:
        if condition.startswith("T_"):
            continue
        pending = [row for row in concepts if (row["concept"].lower(), condition) not in completed]
        if pending:
            pending_by_condition[condition] = pending
    if not pending_by_condition:
        return []
    processor = transformers.AutoProcessor.from_pretrained(model_source, **stage01.tokenizer_load_kwargs(model_source))
    tokenizer = getattr(processor, "tokenizer", None)
    require(tokenizer is not None, f"{model_id} processor does not expose a tokenizer")
    model = stage01.load_multimodal_model(transformers, model_source, stage01.multimodal_load_kwargs(torch, config)).eval()
    max_side = int(config.get("analysis", {}).get("image_policy", {}).get("multimodal_max_side", 384))
    rows = []
    kwargs = generation_kwargs(tokenizer, max_new_tokens)
    for condition in CONDITIONS:
        if condition.startswith("T_"):
            continue
        for row in pending_by_condition.get(condition, []):
            concept = row["concept"]
            prompt = generation_prompt(config, condition, concept)
            image, image_path, source = image_for_condition(stage01, condition, concept, ready_images, mismatch_map)
            image = stage01.prepare_multimodal_image(image, max_side)
            batch = stage01.build_multimodal_inputs(processor, prompt, image)
            batch = stage01.move_batch_to_device(batch, stage01.first_model_device(model))
            with torch.no_grad():
                generated = model.generate(**batch, **kwargs)
            rows.append(
                {
                    "concept": concept,
                    "subtype": row["subtype"],
                    "condition": condition,
                    "model_id": model_id,
                    "prompt": prompt,
                    "image_path": image_path,
                    "mismatch_source": source if condition == "M_mismatched_image" else "",
                    "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                    "generated_text": decode_new_tokens(tokenizer, batch["input_ids"], generated),
                }
            )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def family_models(config: dict[str, Any], family_name: str) -> tuple[str, str]:
    if family_name == "qwen":
        return (
            config["analysis"]["execution"]["sensory_backbone_text_model"],
            config["analysis"]["execution"]["sensory_backbone_multimodal_model"],
        )
    for family in config["analysis"]["analysis"].get("cross_family_families", []):
        if str(family.get("family_name")) == family_name:
            return str(family["text_model"]), str(family["multimodal_model"])
    raise RuntimeError(f"Unknown family `{family_name}` in config cross_family_families.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic text for the behavior probe.")
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--family", default="qwen", choices=["qwen", "mistral", "llama"])
    parser.add_argument("--vlm-only", action="store_true", help="Skip T_prompt_primary generation and generate only VLM conditions.")
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--output-stem", default="behavior_probe_v2_generations")
    parser.add_argument("--smoke", action="store_true", help="Use a small subset and write _smoke outputs.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output CSV by concept and condition.")
    args = parser.parse_args()

    import torch
    import transformers

    stage01 = load_stage01_module()
    config = load_project_config(args.config)
    stage01.configure_hf_cache(config)
    set_global_seed(args.seed)
    if args.smoke and args.limit == 200:
        args.limit = 12
    suffix = "_smoke" if args.smoke else ""
    output_csv = ROOT / "outputs" / "generations" / f"{args.output_stem}{suffix}.csv"
    concept_rows = stage01.load_concepts(config, None)
    selected = balanced_concept_subset(concept_rows, args.limit, args.seed)
    ready_images = stage01.load_ready_image_map()
    mismatch_map = stage01.remap_mismatch_for_subset(selected, ready_images, stage01.load_mismatch_map())
    text_model, multimodal_model = family_models(config, args.family)
    cache_root = config["analysis"]["runtime"].get("hf_cache_dir", "")
    text_source = str(stage01.resolve_cached_snapshot(text_model, cache_root)) if cache_root else text_model
    multimodal_source = str(stage01.resolve_cached_snapshot(multimodal_model, cache_root)) if cache_root else multimodal_model

    rows = load_resume_rows(output_csv) if args.resume else []
    completed = {generation_key(row) for row in rows}
    if not args.vlm_only:
        rows.extend(
            generate_text_rows(
                concepts=selected,
                config=config,
                model_id=text_model,
                model_source=text_source,
                max_new_tokens=args.max_new_tokens,
                torch=torch,
                transformers=transformers,
                stage01=stage01,
                completed=completed,
            )
        )
    completed = {generation_key(row) for row in rows}
    rows.extend(
        generate_multimodal_rows(
            concepts=selected,
            config=config,
            model_id=multimodal_model,
            model_source=multimodal_source,
            max_new_tokens=args.max_new_tokens,
            torch=torch,
            transformers=transformers,
            stage01=stage01,
            ready_images=ready_images,
            mismatch_map=mismatch_map,
            completed=completed,
        )
    )
    for row in rows:
        row["seed"] = args.seed
        row["max_new_tokens"] = args.max_new_tokens
        row["word_count"] = len(row["generated_text"].split())
    rows = sort_generation_rows(rows)
    write_csv(
        output_csv,
        rows,
        [
            "concept",
            "subtype",
            "condition",
            "model_id",
            "prompt",
            "image_path",
            "mismatch_source",
            "prompt_template_version",
            "generated_text",
            "word_count",
            "seed",
            "max_new_tokens",
        ],
    )
    append_run_log(
        "Behavior Probe Generation",
        [
            f"Generated {len(rows)} rows for {len(selected)} concepts.",
            f"Smoke mode: {args.smoke}.",
            f"Resume mode: {args.resume}.",
            f"Prompt template: {PROMPT_TEMPLATE_VERSION}.",
        ],
    )


if __name__ == "__main__":
    main()
