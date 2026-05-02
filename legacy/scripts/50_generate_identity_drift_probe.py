from __future__ import annotations

import argparse
import gc
import importlib.util
from pathlib import Path
from typing import Any

from common import ROOT, append_run_log, load_project_config, read_csv, require, set_global_seed, write_csv


PROMPT_TEMPLATE_VERSION = "identity_drift_forced_choice_v1"
IDENTITY_PROMPT = (
    'The image may be unrelated to the text. The text concept is "{target}". '
    'Which concept should your answer describe: "{target}" or "{source}"? '
    "Answer with exactly one option."
)


def load_stage01_module() -> Any:
    script_path = ROOT / "scripts" / "01_extract_hidden_states.py"
    spec = importlib.util.spec_from_file_location("stage01_extract_hidden_states", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load helper module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    if eos_token_id is not None:
        kwargs["eos_token_id"] = eos_token_id
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    return kwargs


def selected_concepts(config: dict[str, Any], stage01: Any, limit: int, seed: int) -> list[dict[str, str]]:
    rows = [row for row in stage01.load_concepts(config, None) if row["domain"] == "sensory"]
    rows.sort(key=lambda row: row["concept"].lower())
    if limit <= 0 or limit >= len(rows):
        return rows
    import random

    rng = random.Random(seed)
    sampled = rng.sample(rows, limit)
    sampled.sort(key=lambda row: row["concept"].lower())
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate identity forced-choice responses under mismatched grounding.")
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output", default="outputs/generations/identity_drift_probe_generations.csv")
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
    if args.smoke and args.limit == 0:
        args.limit = 12

    output_csv = ROOT / args.output
    concepts = selected_concepts(config, stage01, args.limit, args.seed)
    ready_images = stage01.load_ready_image_map()
    mismatch_map = stage01.remap_mismatch_for_subset(concepts, ready_images, stage01.load_mismatch_map())
    multimodal_model = config["analysis"]["execution"]["sensory_backbone_multimodal_model"]
    cache_root = config["analysis"]["runtime"].get("hf_cache_dir", "")
    multimodal_source = str(stage01.resolve_cached_snapshot(multimodal_model, cache_root)) if cache_root else multimodal_model

    rows = load_resume_rows(output_csv) if args.resume else []
    completed = {generation_key(row) for row in rows}
    pending = [row for row in concepts if (row["concept"].lower(), "M_mismatched_image") not in completed]
    if not pending:
        print(f"No pending identity-drift rows; output already has {len(rows)} rows.")
        return

    processor = transformers.AutoProcessor.from_pretrained(multimodal_source, **stage01.tokenizer_load_kwargs(multimodal_source))
    tokenizer = getattr(processor, "tokenizer", None)
    require(tokenizer is not None, f"{multimodal_model} processor does not expose a tokenizer")
    model = stage01.load_multimodal_model(transformers, multimodal_source, stage01.multimodal_load_kwargs(torch, config)).eval()
    max_side = int(config.get("analysis", {}).get("image_policy", {}).get("multimodal_max_side", 384))
    kwargs = generation_kwargs(tokenizer, args.max_new_tokens)

    for row in pending:
        target = row["concept"]
        source = mismatch_map[target]
        image_path = ROOT / ready_images[source]["matched_image"]
        image = Image.open(image_path).convert("RGB")
        image = stage01.prepare_multimodal_image(image, max_side)
        prompt = IDENTITY_PROMPT.format(target=target, source=source)
        batch = stage01.build_multimodal_inputs(processor, prompt, image)
        batch = stage01.move_batch_to_device(batch, stage01.first_model_device(model))
        with torch.no_grad():
            generated = model.generate(**batch, **kwargs)
        rows.append(
            {
                "concept": target,
                "subtype": row["subtype"],
                "condition": "M_mismatched_image",
                "model_id": multimodal_model,
                "prompt": prompt,
                "image_path": str(image_path.relative_to(ROOT)),
                "mismatch_source": source,
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "generated_text": decode_new_tokens(tokenizer, batch["input_ids"], generated),
                "seed": args.seed,
                "max_new_tokens": args.max_new_tokens,
            }
        )

    rows.sort(key=lambda item: (str(item["concept"]).lower(), str(item["condition"])))
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
            "seed",
            "max_new_tokens",
        ],
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    append_run_log(
        "Identity-Drift Probe Generation",
        [
            f"Generated {len(rows)} total rows for {len(concepts)} selected concepts.",
            f"Prompt template: {PROMPT_TEMPLATE_VERSION}.",
            f"Smoke mode: {args.smoke}.",
        ],
    )
    print(f"Wrote {output_csv.relative_to(ROOT)} with {len(rows)} rows.")


if __name__ == "__main__":
    main()
