from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def load_module():
    path = Path("scripts/25_compute_layerwise_global_local_dissociation.py")
    scripts_dir = str(Path("scripts").resolve())
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("global_local", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_families_to_run_expands_all_in_declared_order() -> None:
    module = load_module()
    assert module.families_to_run("all") == ["qwen", "mistral", "llama"]


def test_family_models_are_declared_for_replication_families() -> None:
    module = load_module()
    mistral_text, mistral_vlm = module.family_models("mistral", "config/analysis.yaml")
    llama_text, llama_vlm = module.family_models("llama", "config/analysis.yaml")
    assert mistral_text == "mistralai/Mistral-Small-24B-Instruct-2501"
    assert mistral_vlm == "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    assert llama_text == "meta-llama/Llama-3.1-8B-Instruct"
    assert llama_vlm == "meta-llama/Llama-3.2-11B-Vision-Instruct"
