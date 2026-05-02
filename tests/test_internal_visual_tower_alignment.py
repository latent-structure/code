from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_module():
    path = Path("scripts/22_compute_internal_visual_tower_alignment.py")
    scripts_dir = Path("scripts").resolve()
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location("internal_visual_tower_alignment", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_family_specs_cover_all_requested_models() -> None:
    module = load_module()
    assert module.FAMILY_SPECS["qwen"]["multimodal_model_id"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert module.FAMILY_SPECS["mistral"]["multimodal_model_id"] == "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    assert module.FAMILY_SPECS["llama"]["multimodal_model_id"] == "meta-llama/Llama-3.2-11B-Vision-Instruct"


def test_family_artifact_suffix_rules() -> None:
    module = load_module()
    assert module.family_artifact_suffix("qwen", "qwen", 0) == ""
    assert module.family_artifact_suffix("qwen", "qwen", 5) == "_smoke"
    assert module.family_artifact_suffix("all", "qwen", 0) == "_qwen"
    assert module.family_artifact_suffix("all", "mistral", 3) == "_mistral_smoke"
    assert module.family_artifact_suffix("llama", "llama", 0) == "_llama"
