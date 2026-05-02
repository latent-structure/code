from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from typing import Any

import numpy as np

from common import ROOT, append_run_log, condensed_cosine_distance, load_project_config, metrics_path, output_path, rankdata, read_csv, spearman_corr, write_csv, write_json
from hardening_common import load_things_reference, selected_layers, write_text


CONDITIONS = ["M_text_only", "M_matched_image", "M_blank_image"]


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
    by_subtype: dict[str, list[dict[str, str]]] = {}
    for row in sensory:
        by_subtype.setdefault(row["subtype"], []).append(row)
    rng = random.Random(seed)
    for subtype_rows in by_subtype.values():
        subtype_rows.sort(key=lambda row: row["concept"])
        rng.shuffle(subtype_rows)
    selected: list[dict[str, str]] = []
    while len(selected) < limit:
        progressed = False
        for subtype in sorted(by_subtype):
            if by_subtype[subtype]:
                selected.append(by_subtype[subtype].pop())
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
    selected.sort(key=lambda row: row["concept"])
    return selected


def control_span(concept_start: int, concept_end: int, sequence_len: int) -> tuple[int, int]:
    length = concept_end - concept_start
    after_start = concept_end
    after_end = after_start + length
    if after_end <= sequence_len:
        return after_start, after_end
    before_end = concept_start
    before_start = max(0, before_end - length)
    if before_end > before_start:
        return before_start, before_end
    raise RuntimeError("Could not select length-matched non-concept control span.")


def image_for_condition(stage01: Any, ready_images: dict[str, dict[str, str]], condition: str, concept: str, max_side: int) -> Any:
    from PIL import Image

    if condition == "M_text_only":
        return None
    image_path = ROOT / ready_images[concept]["matched_image"]
    template = Image.open(image_path).convert("RGB")
    if condition == "M_blank_image":
        template = Image.new("RGB", template.size, color=(128, 128, 128))
    return stage01.prepare_multimodal_image(template, max_side)


def mean_mid_late(pooled_layers: list[np.ndarray], mid_fraction: float) -> np.ndarray:
    layers = selected_layers(list(range(len(pooled_layers))), mid_fraction)
    return np.mean(np.stack([pooled_layers[layer] for layer in layers]), axis=0, dtype=np.float32)


def things_rdm_for_concepts(concepts: list[str]) -> np.ndarray:
    matrix, _things_concepts, index = load_things_reference()
    idx = [index[concept] for concept in concepts]
    sub = matrix[np.ix_(idx, idx)]
    tri = np.triu_indices(len(idx), k=1)
    return sub[tri].astype(float)


def rsa_to_things(embeddings: np.ndarray, concepts: list[str]) -> float:
    return spearman_corr(condensed_cosine_distance(embeddings), things_rdm_for_concepts(concepts))


def extract_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    import transformers

    stage01 = load_stage01_module()
    config = load_project_config(args.config)
    stage01.configure_hf_cache(config)
    concept_rows = balanced_concept_subset(stage01.load_concepts(config, None), args.limit, args.seed)
    concepts = [row["concept"].lower() for row in concept_rows]
    ready_images = stage01.load_ready_image_map()
    prompt_map = stage01.pick_multimodal_prompt_map(config)
    model_id = config["analysis"]["execution"]["sensory_backbone_multimodal_model"]
    cache_root = config["analysis"]["runtime"].get("hf_cache_dir", "")
    model_source = str(stage01.resolve_cached_snapshot(model_id, cache_root)) if cache_root else model_id
    processor = transformers.AutoProcessor.from_pretrained(model_source, **stage01.tokenizer_load_kwargs(model_source))
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError(f"{model_id} processor does not expose a tokenizer")
    model = stage01.load_multimodal_model(transformers, model_source, stage01.multimodal_load_kwargs(torch, config)).eval()
    max_side = int(config.get("analysis", {}).get("image_policy", {}).get("multimodal_max_side", 384))
    mid_fraction = float(config["analysis"]["analysis"]["mid_to_late_fraction"])

    concept_vectors: dict[str, list[np.ndarray]] = {condition: [] for condition in CONDITIONS}
    control_vectors: dict[str, list[np.ndarray]] = {condition: [] for condition in CONDITIONS}
    span_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for condition in CONDITIONS:
            for row in concept_rows:
                concept = row["concept"]
                prompt = prompt_map[condition].format(concept=concept)
                image = image_for_condition(stage01, ready_images, condition, concept, max_side)
                rendered = stage01.render_multimodal_text(processor, prompt, image)
                batch = stage01.build_multimodal_inputs(processor, prompt, image)
                input_ids = batch["input_ids"][0].tolist()
                concept_start, concept_end = stage01.resolve_text_span(
                    tokenizer,
                    input_ids,
                    rendered,
                    prompt,
                    concept,
                    model_id=model_id,
                    condition=condition,
                )
                ctrl_start, ctrl_end = control_span(concept_start, concept_end, len(input_ids))
                batch = stage01.move_batch_to_device(batch, stage01.first_model_device(model))
                outputs = model(**batch, output_hidden_states=True)
                hidden_states = stage01.extract_hidden_states(outputs)
                concept_vectors[condition].append(mean_mid_late(stage01.pool_text_hidden_states(hidden_states, concept_start, concept_end), mid_fraction))
                control_vectors[condition].append(mean_mid_late(stage01.pool_text_hidden_states(hidden_states, ctrl_start, ctrl_end), mid_fraction))
                span_rows.append(
                    {
                        "concept": concept,
                        "subtype": row["subtype"],
                        "condition": condition,
                        "sequence_length": len(input_ids),
                        "concept_span_start": concept_start,
                        "concept_span_end": concept_end,
                        "control_span_start": ctrl_start,
                        "control_span_end": ctrl_end,
                        "span_length": concept_end - concept_start,
                    }
                )

    scores: dict[str, dict[str, float]] = {}
    for condition in CONDITIONS:
        scores[condition] = {
            "concept_span_things_rsa": rsa_to_things(np.stack(concept_vectors[condition]), concepts),
            "control_span_things_rsa": rsa_to_things(np.stack(control_vectors[condition]), concepts),
        }
    concept_gap = scores["M_matched_image"]["concept_span_things_rsa"] - scores["M_text_only"]["concept_span_things_rsa"]
    control_gap = scores["M_matched_image"]["control_span_things_rsa"] - scores["M_text_only"]["control_span_things_rsa"]
    blank_concept_gap = scores["M_matched_image"]["concept_span_things_rsa"] - scores["M_blank_image"]["concept_span_things_rsa"]
    blank_control_gap = scores["M_matched_image"]["control_span_things_rsa"] - scores["M_blank_image"]["control_span_things_rsa"]
    position_flag = abs(control_gap) >= 0.5 * max(abs(concept_gap), 1e-8) or abs(blank_control_gap) >= 0.5 * max(abs(blank_concept_gap), 1e-8)
    summary = {
        "num_concepts": len(concepts),
        "conditions": scores,
        "matched_minus_text_only_concept_span": concept_gap,
        "matched_minus_text_only_control_span": control_gap,
        "matched_minus_blank_concept_span": blank_concept_gap,
        "matched_minus_blank_control_span": blank_control_gap,
        "position_confound_flag": position_flag,
        "preregistered_interpretation": (
            "If length-matched non-concept control spans shift at least half as much as concept spans for matched-minus-text-only "
            "or matched-minus-blank THINGS RSA gaps, the multimodal effect should be qualified as potentially position-confounded. "
            "If control-span gaps are much smaller, the control supports concept-specific pooling rather than a generic position artifact."
        ),
    }
    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return span_rows, summary


def report_lines(summary: dict[str, Any]) -> list[str]:
    lines = [
        "# Concept-Span Position Control",
        "",
        f"- Concepts: `{summary['num_concepts']}`",
        f"- Position-confound flag: `{summary['position_confound_flag']}`",
        f"- Preregistered interpretation: {summary['preregistered_interpretation']}",
        "",
        "## THINGS RSA by Span Type",
        "",
        "| Condition | Concept span | Control span |",
        "|---|---:|---:|",
    ]
    for condition, scores in summary["conditions"].items():
        lines.append(f"| `{condition}` | {scores['concept_span_things_rsa']:.4f} | {scores['control_span_things_rsa']:.4f} |")
    lines.extend(
        [
            "",
            "## Gaps",
            "",
            f"- Matched - text-only concept span: `{summary['matched_minus_text_only_concept_span']:.4f}`",
            f"- Matched - text-only control span: `{summary['matched_minus_text_only_control_span']:.4f}`",
            f"- Matched - blank concept span: `{summary['matched_minus_blank_concept_span']:.4f}`",
            f"- Matched - blank control span: `{summary['matched_minus_blank_control_span']:.4f}`",
        ]
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether multimodal concept-span effects are generic position artifacts.")
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--output-stem", default="position_control")
    args = parser.parse_args()

    span_rows, summary = extract_rows(args)
    suffix = "_smoke" if args.limit < 200 else ""
    write_csv(
        metrics_path(f"{args.output_stem}_spans{suffix}.csv"),
        span_rows,
        [
            "concept",
            "subtype",
            "condition",
            "sequence_length",
            "concept_span_start",
            "concept_span_end",
            "control_span_start",
            "control_span_end",
            "span_length",
        ],
    )
    write_json(metrics_path(f"{args.output_stem}_summary{suffix}.json"), summary)
    write_text(output_path("reports", "main_results", f"{args.output_stem}_report{suffix}.md"), "\n".join(report_lines(summary)))
    append_run_log("Position Control", [f"Wrote position control outputs for {summary['num_concepts']} concepts."])


if __name__ == "__main__":
    main()
