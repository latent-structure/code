from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from common import ROOT, append_run_log, load_project_config, set_global_seed, write_csv


def flatten_model_entries(config: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for family in ("text", "multimodal", "anchor"):
        for priority in ("primary", "fallback"):
            for model_id in config["models"][family][priority]:
                rows.append({"family": family, "priority": priority, "model_id": model_id})
    return rows


def prefetch_one(hf_api: Any, model_id: str, cache_dir: str) -> tuple[str, str]:
    try:
        hf_api.snapshot_download(
            repo_id=model_id,
            local_dir_use_symlinks=False,
            cache_dir=cache_dir,
            resume_download=True,
        )
        return "cached", "snapshot_download ok"
    except Exception as exc:
        return "download_failed", f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-pattern", action="append", default=["*.json", "*.model", "*.safetensors", "*.bin", "*.txt", "*.py", "*.md"])
    args = parser.parse_args()

    config = load_project_config()
    set_global_seed(config["seeds"]["model_probe"])
    cache_dir = config["analysis"]["runtime"]["hf_cache_dir"]
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(cache_dir) / "transformers"))

    from huggingface_hub import snapshot_download

    rows = []
    for entry in flatten_model_entries(config):
        status, detail = prefetch_one(type("Api", (), {"snapshot_download": staticmethod(snapshot_download)}), entry["model_id"], cache_dir)
        rows.append({**entry, "status": status, "detail": detail})

    csv_path = ROOT / "outputs/logs/model_prefetch_log.csv"
    write_csv(csv_path, rows, ["family", "priority", "model_id", "status", "detail"])
    cached = [row["model_id"] for row in rows if row["status"] == "cached"]
    failed = [row["model_id"] for row in rows if row["status"] != "cached"]
    append_run_log(
        "Model Prefetch",
        [
            f"Wrote model prefetch log to {csv_path.relative_to(ROOT)}.",
            f"Cached models: {json.dumps(cached)}.",
            f"Failed models: {json.dumps(failed)}.",
        ],
    )


if __name__ == "__main__":
    main()
