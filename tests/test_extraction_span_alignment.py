from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import csv

from PIL import Image


def load_stage05():
    path = Path("scripts/01_extract_hidden_states.py")
    scripts_dir = str(path.parent.resolve())
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("stage01_extract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_find_subsequence_returns_none_when_missing() -> None:
    stage05 = load_stage05()
    assert stage05.find_subsequence([1, 2, 3], [4]) is None


def test_prompt_anchored_char_span_uses_prompt_occurrence() -> None:
    stage05 = load_stage05()
    rendered = "<system>ignore</system><user>Consider the concept: apple.</user>"
    prompt = "Consider the concept: apple."
    start, end = stage05.prompt_anchored_concept_char_span(rendered, prompt, "apple")
    assert rendered[start:end] == "apple"


def test_token_span_from_offsets_finds_overlap() -> None:
    stage05 = load_stage05()
    offsets = [(0, 0), (0, 8), (9, 12), (13, 20), (20, 21)]
    assert stage05.token_span_from_offsets(offsets, 9, 12) == (2, 3)


def load_ready_image(concept: str) -> Image.Image | None:
    with open("data/manifests/image_manifest.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["concept"] == concept and row["status"] == "ready":
                return Image.open(Path(row["matched_image"])).convert("RGB")
    return None


def cached_snapshot(model_id: str) -> str | None:
    stage05 = load_stage05()
    cache_root = "/tmp/pilot_grounding"
    try:
        return str(stage05.resolve_cached_snapshot(model_id, cache_root))
    except Exception:
        return None


def test_qwen_text_alignment_does_not_pool_full_sequence() -> None:
    transformers = __import__("transformers")
    stage05 = load_stage05()
    snapshot_root = Path("/tmp/pilot_grounding/models--Qwen--Qwen3.5-9B/snapshots")
    if not snapshot_root.exists():
        return
    snapshot = str(sorted(snapshot_root.iterdir())[-1])
    tokenizer = transformers.AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    concept = "apple"
    prompt = f"Consider the concept: {concept}."
    encoded = tokenizer(prompt, add_special_tokens=True)
    start, end = stage05.resolve_text_span(
        tokenizer,
        encoded["input_ids"],
        prompt,
        prompt,
        concept,
        model_id="Qwen/Qwen3.5-9B",
        condition="T_neutral",
    )
    assert end > start
    assert not (start == 0 and end == len(encoded["input_ids"]))


def test_qwen_vl_alignment_does_not_pool_full_sequence() -> None:
    transformers = __import__("transformers")
    stage05 = load_stage05()
    snapshot = cached_snapshot("Qwen/Qwen3-VL-8B-Instruct")
    if snapshot is None:
        return
    processor = transformers.AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    prompt = "Consider the concept: apple."
    rendered = stage05.render_multimodal_text(processor, prompt, image=None)
    batch = processor(text=rendered, return_tensors="pt")
    input_ids = batch["input_ids"][0].tolist()
    start, end = stage05.resolve_text_span(
        processor.tokenizer,
        input_ids,
        rendered,
        prompt,
        "apple",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        condition="M_text_only",
    )
    assert end > start
    assert not (start == 0 and end == len(input_ids))


def check_multimodal_image_present_alignment(model_id: str, concept: str) -> None:
    transformers = __import__("transformers")
    stage05 = load_stage05()
    snapshot = cached_snapshot(model_id)
    if snapshot is None:
        return
    kwargs = {"local_files_only": True}
    if "mistral" in model_id.lower():
        kwargs["fix_mistral_regex"] = True
    processor = transformers.AutoProcessor.from_pretrained(snapshot, **kwargs)
    prompt = f"Consider the concept: {concept}."
    image = load_ready_image(concept)
    if image is None:
        return
    image = stage05.prepare_multimodal_image(image, 448)
    rendered = stage05.render_multimodal_text(processor, prompt, image=image)
    batch = stage05.build_multimodal_inputs(processor, prompt, image)
    input_ids = batch["input_ids"][0].tolist()
    start, end = stage05.resolve_text_span(
        processor.tokenizer,
        input_ids,
        rendered,
        prompt,
        concept,
        model_id=model_id,
        condition="M_matched_image",
    )
    assert end > start
    assert not (start == 0 and end == len(input_ids))
    assert processor.tokenizer.decode(input_ids[start:end]).strip() == concept


def test_qwen_vl_image_present_alignment() -> None:
    check_multimodal_image_present_alignment("Qwen/Qwen3-VL-8B-Instruct", "aardvark")


def test_mistral_vl_image_present_alignment() -> None:
    check_multimodal_image_present_alignment("mistralai/Mistral-Small-3.1-24B-Instruct-2503", "aardvark")


def test_llama_vl_image_present_alignment() -> None:
    check_multimodal_image_present_alignment("meta-llama/Llama-3.2-11B-Vision-Instruct", "aardvark")


def test_qwen_vl_multitoken_image_present_alignment() -> None:
    check_multimodal_image_present_alignment("Qwen/Qwen3-VL-8B-Instruct", "fire truck")
