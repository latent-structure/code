from __future__ import annotations

import argparse
import gc
import importlib.util
from typing import Any

from common import ROOT, append_run_log, load_project_config, read_csv, set_global_seed, write_csv


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


def load_stage30_module() -> Any:
    script_path = ROOT / "scripts" / "30_generate_behavior_probe.py"
    spec = importlib.util.spec_from_file_location("stage30_generate_behavior_probe", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load helper module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generation_prompt(concept: str) -> str:
    return BEHAVIOR_TASK.format(concept=concept)


def decode_new_tokens(tokenizer: Any, input_ids: Any, generated_ids: Any, stage30: Any) -> str:
    prompt_len = int(input_ids.shape[-1])
    new_tokens = generated_ids[0][prompt_len:]
    return stage30.normalize_generated_text(tokenizer.decode(new_tokens, skip_special_tokens=True))


def load_resume_keys(path: str) -> tuple[list[dict[str, str]], set[tuple[str, int]]]:
    out_path = ROOT / path
    if not out_path.exists():
        return [], set()
    rows = read_csv(out_path)
    return rows, {(row["concept"].lower(), int(row["repeat_id"])) for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate stochastic repeated mismatched-image descriptions for behavior reliability ceilings.")
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--output", default="outputs/generations/behavior_repeats_mismatched_200x5.csv")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    import torch
    import transformers

    stage01 = load_stage01_module()
    stage30 = load_stage30_module()
    config = load_project_config(args.config)
    stage01.configure_hf_cache(config)
    set_global_seed(args.seed)
    concept_rows = stage01.load_concepts(config, None)
    selected = stage30.balanced_concept_subset(concept_rows, args.limit, args.seed)
    ready_images = stage01.load_ready_image_map()
    mismatch_map = stage01.remap_mismatch_for_subset(selected, ready_images, stage01.load_mismatch_map())
    model_id = config["analysis"]["execution"]["sensory_backbone_multimodal_model"]
    cache_root = config["analysis"]["runtime"].get("hf_cache_dir", "")
    model_source = str(stage01.resolve_cached_snapshot(model_id, cache_root)) if cache_root else model_id

    rows, completed = load_resume_keys(args.output) if args.resume else ([], set())
    processor = transformers.AutoProcessor.from_pretrained(model_source, **stage01.tokenizer_load_kwargs(model_source))
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError(f"{model_id} processor does not expose a tokenizer")
    model = stage01.load_multimodal_model(transformers, model_source, stage01.multimodal_load_kwargs(torch, config)).eval()
    max_side = int(config.get("analysis", {}).get("image_policy", {}).get("multimodal_max_side", 384))
    generator_device = stage01.first_model_device(model)
    generation_kwargs = {
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }
    if getattr(tokenizer, "eos_token_id", None) is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id

    for repeat_id in range(args.repeats):
        for concept_row in selected:
            concept = concept_row["concept"]
            if (concept.lower(), repeat_id) in completed:
                continue
            repeat_seed = args.seed + repeat_id * 100000 + len(rows)
            torch.manual_seed(repeat_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(repeat_seed)
            prompt = generation_prompt(concept)
            source = mismatch_map[concept]
            image_path = ROOT / ready_images[source]["matched_image"]
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            image = stage01.prepare_multimodal_image(image, max_side)
            batch = stage01.build_multimodal_inputs(processor, prompt, image)
            batch = stage01.move_batch_to_device(batch, generator_device)
            with torch.no_grad():
                generated = model.generate(**batch, **generation_kwargs)
            rows.append(
                {
                    "concept": concept,
                    "subtype": concept_row["subtype"],
                    "condition": "M_mismatched_image",
                    "model_id": model_id,
                    "prompt": prompt,
                    "image_path": str(image_path.relative_to(ROOT)),
                    "mismatch_source": source,
                    "prompt_template_version": "behavior_repeats_mismatched_stochastic",
                    "generated_text": decode_new_tokens(tokenizer, batch["input_ids"], generated, stage30),
                    "word_count": 0,
                    "seed": repeat_seed,
                    "max_new_tokens": args.max_new_tokens,
                    "repeat_id": repeat_id,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                }
            )

    for row in rows:
        row["word_count"] = len(str(row["generated_text"]).split())
    rows = sorted(rows, key=lambda row: (row["concept"].lower(), int(row["repeat_id"])))
    write_csv(
        ROOT / args.output,
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
            "repeat_id",
            "temperature",
            "top_p",
        ],
    )
    append_run_log(
        "Behavior Repeat Generation",
        [
            f"Generated {len(rows)} repeated mismatched rows for {len(selected)} concepts.",
            f"Repeats: {args.repeats}; temperature: {args.temperature}; top_p: {args.top_p}.",
        ],
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
