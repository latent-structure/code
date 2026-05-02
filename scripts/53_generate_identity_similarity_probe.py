from __future__ import annotations

import argparse
import gc
import importlib.util
from pathlib import Path
from typing import Any

from common import ROOT, append_run_log, load_project_config, read_csv, require, set_global_seed, write_csv


CONDITIONS = ["M_text_only", "M_matched_image", "M_mismatched_image", "M_blank_image"]
TASKS = ["similarity"]
PROMPT_TEMPLATE_VERSION = "identity_similarity_probe_v1"

IDENTITY_PROMPT = (
    'The text names the concept "{target}". The image may be unrelated. '
    'Which concept is named by the text: "{target}" or "{source}"? '
    "Answer exactly one option."
)

SIMILARITY_PROMPT = (
    'The text concept is "{target}". The image may be unrelated. '
    'Considering the combined input, which candidate is more similar to the concept you would describe: '
    'A) "{option_a}" or B) "{option_b}"? Answer exactly A or B.'
)


def load_stage01_module() -> Any:
    script_path = ROOT / "scripts" / "01_extract_hidden_states.py"
    spec = importlib.util.spec_from_file_location("stage01_extract_hidden_states", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load helper module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def family_multimodal_model(config: dict[str, Any], family_name: str) -> str:
    if family_name == "qwen":
        return config["analysis"]["execution"]["sensory_backbone_multimodal_model"]
    for family in config["analysis"]["analysis"].get("cross_family_families", []):
        if str(family.get("family_name")) == family_name:
            return str(family["multimodal_model"])
    raise RuntimeError(f"Unknown family `{family_name}` in config cross_family_families.")


def generation_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["concept"]).lower(), str(row["condition"]), str(row["task"])


def normalize_generated_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


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
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or eos_token_id
    if eos_token_id is not None:
        kwargs["eos_token_id"] = eos_token_id
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    return kwargs


def load_resume_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in read_csv(path):
        key = generation_key(row)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def prompt_for(task: str, item: dict[str, str]) -> str:
    if task == "identity":
        return IDENTITY_PROMPT.format(target=item["concept"], source=item["mismatch_source"])
    if task == "similarity":
        return SIMILARITY_PROMPT.format(
            target=item["concept"],
            option_a=item["option_a"],
            option_b=item["option_b"],
        )
    raise RuntimeError(f"Unknown task: {task}")


def image_for(condition: str, item: dict[str, str], ready_images: dict[str, dict[str, str]]) -> Path | None:
    if condition == "M_text_only":
        return None
    if condition in {"M_matched_image", "M_blank_image"}:
        return ROOT / ready_images[item["concept"]]["matched_image"]
    if condition == "M_mismatched_image":
        return ROOT / ready_images[item["mismatch_source"]]["matched_image"]
    raise RuntimeError(f"Unsupported condition: {condition}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate identity/similarity behavioral probe responses.")
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--items", default="outputs/metrics/identity_similarity_probe_items.csv")
    parser.add_argument("--output", default="outputs/generations/identity_similarity_probe_generations.csv")
    parser.add_argument("--family", default="qwen", choices=["qwen", "mistral", "llama"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--tasks", nargs="+", choices=["identity", "similarity"], default=TASKS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    import torch
    import transformers
    from PIL import Image

    stage01 = load_stage01_module()
    config = load_project_config(args.config)
    stage01.configure_hf_cache(config)
    set_global_seed(args.seed)
    items = read_csv(ROOT / args.items)
    if args.smoke and args.limit == 0:
        args.limit = 12
    if args.limit:
        items = items[: args.limit]

    ready_images = stage01.load_ready_image_map()
    output_csv = ROOT / args.output
    rows = load_resume_rows(output_csv) if args.resume else []
    completed = {generation_key(row) for row in rows}
    pending = [
        (item, condition, task)
        for item in items
        for condition in CONDITIONS
        for task in args.tasks
        if (item["concept"].lower(), condition, task) not in completed
    ]
    if not pending:
        print(f"No pending identity/similarity rows; output already has {len(rows)} rows.")
        return

    multimodal_model = family_multimodal_model(config, args.family)
    cache_root = config["analysis"]["runtime"].get("hf_cache_dir", "")
    multimodal_source = str(stage01.resolve_cached_snapshot(multimodal_model, cache_root)) if cache_root else multimodal_model
    processor = transformers.AutoProcessor.from_pretrained(multimodal_source, **stage01.tokenizer_load_kwargs(multimodal_source))
    tokenizer = getattr(processor, "tokenizer", None)
    require(tokenizer is not None, f"{multimodal_model} processor does not expose a tokenizer")
    model = stage01.load_multimodal_model(transformers, multimodal_source, stage01.multimodal_load_kwargs(torch, config)).eval()
    device = stage01.first_model_device(model)
    max_side = int(config.get("analysis", {}).get("image_policy", {}).get("multimodal_max_side", 384))
    kwargs = generation_kwargs(tokenizer, args.max_new_tokens)

    for item, condition, task in pending:
        image_path = image_for(condition, item, ready_images)
        if condition == "M_blank_image":
            template = Image.open(image_path).convert("RGB")
            image = Image.new("RGB", template.size, color=(128, 128, 128))
        elif image_path is None:
            image = None
        else:
            image = Image.open(image_path).convert("RGB")
        image = stage01.prepare_multimodal_image(image, max_side)
        prompt = prompt_for(task, item)
        batch = stage01.build_multimodal_inputs(processor, prompt, image)
        batch = stage01.move_batch_to_device(batch, device)
        with torch.no_grad():
            generated = model.generate(**batch, **kwargs)
        rows.append(
            {
                "concept": item["concept"],
                "subtype": item["subtype"],
                "mismatch_source": item["mismatch_source"],
                "condition": condition,
                "task": task,
                "model_id": multimodal_model,
                "prompt": prompt,
                "image_path": "" if image_path is None else str(image_path.relative_to(ROOT)),
                "target_neighbor": item["target_neighbor"],
                "source_neighbor": item["source_neighbor"],
                "option_a": item["option_a"],
                "option_a_role": item["option_a_role"],
                "option_b": item["option_b"],
                "option_b_role": item["option_b_role"],
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "generated_text": decode_new_tokens(tokenizer, batch["input_ids"], generated),
                "seed": args.seed,
                "max_new_tokens": args.max_new_tokens,
            }
        )

    rows.sort(key=lambda row: (str(row["concept"]).lower(), str(row["condition"]), str(row["task"])))
    write_csv(
        output_csv,
        rows,
        [
            "concept",
            "subtype",
            "mismatch_source",
            "condition",
            "task",
            "model_id",
            "prompt",
            "image_path",
            "target_neighbor",
            "source_neighbor",
            "option_a",
            "option_a_role",
            "option_b",
            "option_b_role",
            "prompt_template_version",
            "generated_text",
            "seed",
            "max_new_tokens",
        ],
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    append_run_log(
        "Identity-Similarity Probe Generation",
        [f"Generated {len(rows)} rows for {len(items)} items. Smoke mode: {args.smoke}."],
    )
    print(f"Wrote {output_csv.relative_to(ROOT)} with {len(rows)} rows.")


if __name__ == "__main__":
    main()
