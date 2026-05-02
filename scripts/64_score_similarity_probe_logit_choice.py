from __future__ import annotations

import argparse
import gc
import importlib.util
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from common import ROOT, append_run_log, load_project_config, read_csv, require, write_csv, write_json
from hardening_common import write_text


CONDITIONS = ["M_text_only", "M_matched_image", "M_mismatched_image", "M_blank_image"]
PROMPT_TEMPLATE_VERSION = "identity_similarity_probe_logit_choice_v1"
SIMILARITY_PROMPT = (
    'The text concept is "{target}". The image may be unrelated. '
    'Considering the combined input, which candidate is more similar to the concept you would describe: '
    'A) "{option_a}" or B) "{option_b}"? Answer exactly A or B.'
)


def load_stage01_module() -> Any:
    script_path = ROOT / "scripts" / "01_extract_hidden_states.py"
    spec = importlib.util.spec_from_file_location("stage01_extract_hidden_states", script_path)
    require(spec is not None and spec.loader is not None, f"Failed to load helper module from {script_path}")
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


def prompt_for(item: dict[str, str]) -> str:
    return SIMILARITY_PROMPT.format(
        target=item["concept"],
        option_a=item["option_a"],
        option_b=item["option_b"],
    )


def image_for(condition: str, item: dict[str, str], ready_images: dict[str, dict[str, str]]) -> Path | None:
    if condition == "M_text_only":
        return None
    if condition in {"M_matched_image", "M_blank_image"}:
        return ROOT / ready_images[item["concept"]]["matched_image"]
    if condition == "M_mismatched_image":
        return ROOT / ready_images[item["mismatch_source"]]["matched_image"]
    raise RuntimeError(f"Unsupported condition: {condition}")


def single_token_ids(tokenizer: Any, variants: list[str]) -> list[int]:
    ids: list[int] = []
    for variant in variants:
        token_ids = tokenizer(variant, add_special_tokens=False)["input_ids"]
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        if len(token_ids) == 1:
            token_id = int(token_ids[0])
            if token_id not in ids:
                ids.append(token_id)
    return ids


def choice_token_ids(tokenizer: Any) -> tuple[list[int], list[int]]:
    a_ids = single_token_ids(tokenizer, ["A", " A", "\nA", "A.", " A."])
    b_ids = single_token_ids(tokenizer, ["B", " B", "\nB", "B.", " B."])
    require(a_ids, "Tokenizer produced no single-token variants for answer A.")
    require(b_ids, "Tokenizer produced no single-token variants for answer B.")
    return a_ids, b_ids


def logsumexp(values: list[float]) -> float:
    if not values:
        return float("-inf")
    max_value = max(values)
    return max_value + math.log(sum(math.exp(value - max_value) for value in values))


def extract_logits(outputs: Any) -> Any:
    if getattr(outputs, "logits", None) is not None:
        return outputs.logits
    nested = getattr(outputs, "language_model_outputs", None)
    if nested is not None and getattr(nested, "logits", None) is not None:
        return nested.logits
    raise RuntimeError("Model outputs did not expose logits in a supported location.")


def score_next_token(outputs: Any, torch: Any, a_ids: list[int], b_ids: list[int]) -> tuple[float, float, float]:
    logits = extract_logits(outputs)
    next_logits = logits[0, -1].float()
    log_probs = torch.nn.functional.log_softmax(next_logits, dim=-1)
    logprob_a = logsumexp([float(log_probs[token_id].detach().cpu()) for token_id in a_ids])
    logprob_b = logsumexp([float(log_probs[token_id].detach().cpu()) for token_id in b_ids])
    return logprob_a, logprob_b, logprob_a - logprob_b


def summarize(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        scored.groupby("condition", as_index=False)
        .agg(
            n=("concept", "size"),
            source_neighbor_choice_rate=("source_neighbor_choice", "mean"),
            target_neighbor_choice_rate=("target_neighbor_choice", "mean"),
            mean_source_minus_target_logprob_margin=("source_minus_target_logprob_margin", "mean"),
            median_source_minus_target_logprob_margin=("source_minus_target_logprob_margin", "median"),
            valid_logit_score_rate=("valid_logit_score", "mean"),
        )
    )
    lookup = summary.set_index("condition").to_dict(orient="index")
    contrast_rows = []
    for base in ["M_text_only", "M_matched_image", "M_blank_image"]:
        if base not in lookup or "M_mismatched_image" not in lookup:
            continue
        contrast_rows.append(
            {
                "contrast": f"M_mismatched_image_minus_{base}",
                "source_neighbor_choice_delta": lookup["M_mismatched_image"]["source_neighbor_choice_rate"]
                - lookup[base]["source_neighbor_choice_rate"],
                "source_minus_target_margin_delta": lookup["M_mismatched_image"]["mean_source_minus_target_logprob_margin"]
                - lookup[base]["mean_source_minus_target_logprob_margin"],
            }
        )
    return summary, pd.DataFrame(contrast_rows)


def report_lines(family: str, token_info: dict[str, Any], summary: pd.DataFrame, contrasts: pd.DataFrame) -> list[str]:
    lines = [
        f"# Logit Forced-Choice Similarity Probe: {family}",
        "",
        "This scorer compares next-token probability assigned to A versus B, avoiding free-generation validity failures.",
        "",
        f"- Prompt template: `{PROMPT_TEMPLATE_VERSION}`",
        f"- A token IDs: `{token_info['a_token_ids']}`",
        f"- B token IDs: `{token_info['b_token_ids']}`",
        f"- A decoded variants: `{token_info['a_decoded']}`",
        f"- B decoded variants: `{token_info['b_decoded']}`",
        "",
        "## Condition Summary",
        "",
        "| Condition | Source-neighbor choice | Target-neighbor choice | Mean source-target logprob margin | Valid | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| `{row['condition']}` | {row['source_neighbor_choice_rate']:.4f} | "
            f"{row['target_neighbor_choice_rate']:.4f} | {row['mean_source_minus_target_logprob_margin']:+.4f} | "
            f"{row['valid_logit_score_rate']:.4f} | {int(row['n'])} |"
        )
    if not contrasts.empty:
        lines.extend(
            [
                "",
                "## Mismatched Contrasts",
                "",
                "| Contrast | Source-neighbor delta | Source-target margin delta |",
                "|---|---:|---:|",
            ]
        )
        for _, row in contrasts.iterrows():
            lines.append(
                f"| `{row['contrast']}` | {row['source_neighbor_choice_delta']:+.4f} | "
                f"{row['source_minus_target_margin_delta']:+.4f} |"
            )
    return lines


def copy_to_figures_data(paths: list[Path]) -> None:
    target = ROOT / "figures_data" / "derived"
    target.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists():
            shutil.copy2(path, target / path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score similarity probe by A/B next-token logit choice.")
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--items", default="outputs/metrics/identity_similarity_probe_items.csv")
    parser.add_argument("--family", default="qwen", choices=["qwen", "mistral", "llama"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--output-stem", default="")
    parser.add_argument("--no-copy-figures-data", action="store_true")
    args = parser.parse_args()

    import torch
    import transformers
    from PIL import Image

    stage01 = load_stage01_module()
    config = load_project_config(args.config)
    stage01.configure_hf_cache(config)
    items = read_csv(ROOT / args.items)
    if args.limit:
        items = items[: args.limit]

    ready_images = stage01.load_ready_image_map()
    multimodal_model = family_multimodal_model(config, args.family)
    cache_root = config["analysis"]["runtime"].get("hf_cache_dir", "")
    multimodal_source = str(stage01.resolve_cached_snapshot(multimodal_model, cache_root)) if cache_root else multimodal_model
    processor = transformers.AutoProcessor.from_pretrained(multimodal_source, **stage01.tokenizer_load_kwargs(multimodal_source))
    tokenizer = getattr(processor, "tokenizer", None)
    require(tokenizer is not None, f"{multimodal_model} processor does not expose a tokenizer")
    a_ids, b_ids = choice_token_ids(tokenizer)
    token_info = {
        "a_token_ids": a_ids,
        "b_token_ids": b_ids,
        "a_decoded": [tokenizer.decode([token_id]) for token_id in a_ids],
        "b_decoded": [tokenizer.decode([token_id]) for token_id in b_ids],
    }
    model = stage01.load_multimodal_model(transformers, multimodal_source, stage01.multimodal_load_kwargs(torch, config)).eval()
    device = stage01.first_model_device(model)
    max_side = int(config.get("analysis", {}).get("image_policy", {}).get("multimodal_max_side", 384))

    rows: list[dict[str, Any]] = []
    for item in items:
        for condition in CONDITIONS:
            image_path = image_for(condition, item, ready_images)
            if condition == "M_blank_image":
                template = Image.open(image_path).convert("RGB")
                image = Image.new("RGB", template.size, color=(128, 128, 128))
            elif image_path is None:
                image = None
            else:
                image = Image.open(image_path).convert("RGB")
            image = stage01.prepare_multimodal_image(image, max_side)
            prompt = prompt_for(item)
            batch = stage01.build_multimodal_inputs(processor, prompt, image)
            batch = stage01.move_batch_to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)
            logprob_a, logprob_b, a_minus_b = score_next_token(outputs, torch, a_ids, b_ids)
            choice = "A" if logprob_a >= logprob_b else "B"
            choice_role = item["option_a_role"] if choice == "A" else item["option_b_role"]
            source_minus_target = a_minus_b
            if item["option_a_role"] == "target_neighbor":
                source_minus_target = -source_minus_target
            rows.append(
                {
                    "concept": item["concept"],
                    "subtype": item["subtype"],
                    "mismatch_source": item["mismatch_source"],
                    "condition": condition,
                    "family": args.family,
                    "model_id": multimodal_model,
                    "prompt": prompt,
                    "image_path": "" if image_path is None else str(image_path.relative_to(ROOT)),
                    "target_neighbor": item["target_neighbor"],
                    "source_neighbor": item["source_neighbor"],
                    "option_a": item["option_a"],
                    "option_a_role": item["option_a_role"],
                    "option_b": item["option_b"],
                    "option_b_role": item["option_b_role"],
                    "choice": choice,
                    "choice_role": choice_role,
                    "logprob_A": logprob_a,
                    "logprob_B": logprob_b,
                    "A_minus_B_logprob_margin": a_minus_b,
                    "source_minus_target_logprob_margin": source_minus_target,
                    "source_neighbor_choice": int(choice_role == "source_neighbor"),
                    "target_neighbor_choice": int(choice_role == "target_neighbor"),
                    "valid_logit_score": 1,
                    "target_source_things_similarity": item.get("target_source_things_similarity", ""),
                    "pair_image_similarity": item.get("pair_image_similarity", ""),
                    "pair_difficulty": item.get("pair_difficulty", ""),
                    "source_attraction": item.get("source_attraction", ""),
                    "source_minus_target_margin": item.get("source_minus_target_margin", ""),
                    "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                    "seed": args.seed,
                }
            )

    scored = pd.DataFrame(rows)
    summary, contrasts = summarize(scored)
    stem = args.output_stem or f"identity_similarity_probe_logit_choice_{args.family}_full"
    metrics_dir = ROOT / "outputs" / "metrics"
    report_dir = ROOT / "reports" / "main_results"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    score_path = metrics_dir / f"{stem}.csv"
    summary_path = metrics_dir / f"{stem}_summary.csv"
    contrast_path = metrics_dir / f"{stem}_contrasts.csv"
    json_path = metrics_dir / f"{stem}_summary.json"
    report_path = report_dir / f"{stem}_report.md"
    scored.to_csv(score_path, index=False)
    summary.to_csv(summary_path, index=False)
    contrasts.to_csv(contrast_path, index=False)
    write_json(
        json_path,
        {
            "family": args.family,
            "model_id": multimodal_model,
            "n_rows": int(len(scored)),
            "n_concepts": int(scored["concept"].nunique()),
            "token_info": token_info,
            "condition_summary": summary.to_dict(orient="records"),
            "contrasts": contrasts.to_dict(orient="records"),
        },
    )
    write_text(report_path, "\n".join(report_lines(args.family, token_info, summary, contrasts)))
    if not args.no_copy_figures_data:
        copy_to_figures_data([score_path, summary_path, contrast_path])
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    append_run_log("Logit Similarity Probe", [f"Scored {len(scored)} rows for {args.family}; report={report_path.relative_to(ROOT)}."])
    print(f"Wrote {score_path.relative_to(ROOT)}")
    print(f"Wrote {summary_path.relative_to(ROOT)}")
    print(f"Wrote {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
