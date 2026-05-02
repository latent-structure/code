from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload or {}


def config_root_from_arg(config_path: str | Path | None = None) -> Path:
    if config_path is None:
        return ROOT / "config"
    path = Path(config_path)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if path.is_dir():
        return path
    return path.parent


def load_project_config(config_path: str | Path | None = None) -> dict[str, Any]:
    config_dir = config_root_from_arg(config_path)
    config = {
        "models": load_yaml(config_dir / "models.yaml"),
        "prompts": load_yaml(config_dir / "prompts.yaml"),
        "datasets": load_yaml(config_dir / "datasets.yaml"),
        "anchors": load_yaml(config_dir / "anchors.yaml"),
        "analysis": load_yaml(config_dir / "analysis.yaml"),
        "seeds": load_yaml(config_dir / "seeds.yaml"),
    }
    config["_resolved_root"] = ROOT
    config["_config_dir"] = config_dir
    return config


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def append_run_log(section: str, lines: Iterable[str]) -> None:
    log_path = ROOT / "run_log.md"
    ensure_parent(log_path)
    if not log_path.exists():
        log_path.write_text("# Run Log\n", encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {section}\n")
        for line in lines:
            handle.write(f"- {line}\n")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def midpoint_layer_start(num_layers: int, fraction: float) -> int:
    return math.floor(num_layers * (1.0 - fraction))


def contiguous_true_run(mask: list[bool]) -> int:
    best = 0
    current = 0
    for flag in mask:
        if flag:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    sorted_vals = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_vals[end] == sorted_vals[start]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata(np.asarray(x, dtype=float))
    ry = rankdata(np.asarray(y, dtype=float))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    if denom == 0:
        return 0.0
    return float(np.dot(rx, ry) / denom)


def percentile_interval(values: np.ndarray, level: float) -> tuple[float, float]:
    alpha = (1.0 - level) / 2.0
    low = np.quantile(values, alpha)
    high = np.quantile(values, 1.0 - alpha)
    return float(low), float(high)


def condensed_cosine_distance(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = matrix / norms
    cosine = np.clip(normed @ normed.T, -1.0, 1.0)
    distance = 1.0 - cosine
    idx = np.triu_indices(distance.shape[0], k=1)
    return distance[idx]


def output_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def figure_path(name: str) -> Path:
    return output_path("outputs", "figures", name)


def metrics_path(name: str) -> Path:
    return output_path("outputs", "metrics", name)


def embeddings_path(name: str) -> Path:
    return output_path("outputs", "embeddings", name)


def rdm_path(name: str) -> Path:
    return output_path("outputs", "rdms", name)


def report_path(*parts: str) -> Path:
    return output_path("reports", *parts)


CONDITION_ALIASES = {
    "neutral": "T_neutral",
    "sensory_prompt_1": "T_prompt_primary",
    "sensory_prompt_2": "T_prompt_para_1",
    "sensory_prompt_3": "T_prompt_para_2",
    "text_only": "M_text_only",
    "text_prompt_only": "M_prompt_only",
    "text_matched_image": "M_matched_image",
    "text_degraded_image": "M_degraded_image",
    "text_mismatched_image": "M_mismatched_image",
    "blank_image": "M_blank_image",
    "anchor_image": "reference_anchor_image",
}


def canonical_condition_name(name: str) -> str:
    return CONDITION_ALIASES.get(name, name)
