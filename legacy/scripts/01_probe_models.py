from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from common import ROOT, append_run_log, load_project_config, set_global_seed, write_csv, write_json


def flatten_model_entries(config: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for family in ("text", "multimodal", "anchor"):
        for priority in ("primary", "fallback"):
            for model_id in config["models"][family][priority]:
                rows.append({"family": family, "priority": priority, "model_id": model_id})
    return rows


def pick_runtime_python(config: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    return config["analysis"]["runtime"]["pilot_python"]


def run_single_worker(
    runtime_python: str,
    mode: str,
    family: str,
    priority: str,
    model_id: str,
    timeout_seconds: int,
    require_gpu: bool,
) -> dict[str, Any]:
    command = [
        runtime_python,
        str(Path(__file__).resolve()),
        "--worker",
        "--mode",
        mode,
        "--family",
        family,
        "--priority",
        priority,
        "--model-id",
        model_id,
        "--timeout-seconds",
        str(timeout_seconds),
    ]
    if require_gpu:
        command.append("--require-gpu")
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Worker probe failed.\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    if not completed.stdout.strip():
        raise RuntimeError("Worker probe returned no JSON output.")
    return json.loads(completed.stdout)


def choose_locked_models(rows: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, str]]:
    locked: dict[str, str] = {}
    blocked: dict[str, str] = {}
    required_backbone = {
        "text": "Qwen/Qwen3.5-9B",
        "multimodal": "Qwen/Qwen3-VL-8B-Instruct",
    }
    for family in ("text", "multimodal", "anchor"):
        family_rows = [row for row in rows if row["family"] == family]
        required_model = required_backbone.get(family)
        if required_model is not None:
            winner = next((row for row in family_rows if row["model_id"] == required_model and row["status"] == "ok"), None)
            if winner is not None:
                locked[family] = winner["model_id"]
                continue
            required_row = next((row for row in family_rows if row["model_id"] == required_model), None)
            if required_row is not None:
                blocked[family] = required_row["detail"]
                continue
        winner = next((row for row in family_rows if row["status"] == "ok"), None)
        if winner is not None:
            locked[family] = winner["model_id"]
            continue
        informative = [row for row in family_rows if row.get("config_loaded") or row.get("processor_loaded")]
        candidates = informative or [row for row in family_rows if row["status"] != "not_run"]
        reason = candidates[-1]["detail"] if candidates else "no successful probe"
        blocked[family] = reason
    return locked, blocked


def replace_row(rows: list[dict[str, Any]], updated_row: dict[str, Any]) -> None:
    for index, row in enumerate(rows):
        if row["family"] == updated_row["family"] and row["model_id"] == updated_row["model_id"]:
            rows[index] = updated_row
            return


def summarize_results(payload: dict[str, Any], runtime_python: str) -> list[str]:
    locked = payload["locked_models"]
    blocked = payload["blocked_families"]
    lines = [
        f"Wrote model probe log to outputs/logs/model_probe_log.csv.",
        f"Wrote model probe summary to outputs/logs/model_probe_summary.json.",
        f"Probe mode: {payload['mode']}.",
        f"Runtime python: {runtime_python}.",
    ]
    if locked:
        lines.append(f"Locked models: {json.dumps(locked, sort_keys=True)}.")
    if blocked:
        lines.append(f"Blocked families: {json.dumps(blocked, sort_keys=True)}.")
    versions = payload.get("package_versions")
    if versions:
        lines.append(f"Package versions: {json.dumps(versions, sort_keys=True)}.")
    return lines


def load_runtime_dependencies() -> tuple[Any, Any, dict[str, bool], dict[str, str]]:
    result: dict[str, bool] = {}
    versions: dict[str, str] = {}
    torch = None
    transformers = None
    for name in ("torch", "transformers", "PIL"):
        try:
            module = __import__(name)
            result[name] = True
            versions[name] = getattr(module, "__version__", "unknown")
            if name == "torch":
                torch = module
            elif name == "transformers":
                transformers = module
        except Exception:
            result[name] = False
    return torch, transformers, result, versions


def select_precision(torch: Any) -> tuple[str, Any]:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "bf16", torch.bfloat16
        return "fp16", torch.float16
    return "fp32", torch.float32


def model_load_kwargs(torch: Any) -> dict[str, Any]:
    _, dtype = select_precision(torch)
    kwargs: dict[str, Any] = {"dtype": dtype}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    return kwargs


def multimodal_load_kwargs(torch: Any, prefer_cpu: bool = False) -> dict[str, Any]:
    if prefer_cpu or not torch.cuda.is_available():
        return {"dtype": torch.float32}
    _, dtype = select_precision(torch)
    return {"dtype": dtype, "device_map": "auto"}


def first_model_device(model: Any) -> Any:
    return next(model.parameters()).device


def move_batch_to_device(batch: Any, device: Any) -> Any:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def extract_hidden_states(outputs: Any) -> Any:
    if getattr(outputs, "hidden_states", None) is not None:
        return outputs.hidden_states
    nested = getattr(outputs, "language_model_outputs", None)
    if nested is not None and getattr(nested, "hidden_states", None) is not None:
        return nested.hidden_states
    raise RuntimeError("Model outputs did not expose hidden states in a supported location.")


def has_nonempty_hidden_states(hidden_states: Any) -> bool:
    if hidden_states is None:
        return False
    try:
        return len(hidden_states) > 0
    except Exception:
        return bool(hidden_states)


def classify_failure(exc: Exception, local_files_only: bool) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if "couldn't connect to 'https://huggingface.co'" in text and local_files_only:
        return "not_cached"
    if isinstance(exc, subprocess.TimeoutExpired):
        return "timeout"
    if "out of memory" in text.lower() or "cuda out of memory" in text.lower():
        return "oom_or_device_failure"
    if "Temporary failure in name resolution" in text:
        return "download_failed" if not local_files_only else "not_cached"
    return "failed"


def family_timeout_seconds(family: str) -> int:
    if family == "anchor":
        return 120
    if family == "text":
        return 240
    return 420


def load_multimodal_model(transformers: Any, model_id: str, kwargs: dict[str, Any]) -> Any:
    local_kwargs = dict(kwargs)
    local_kwargs.setdefault("attn_implementation", "eager")
    constructors = [
        getattr(transformers, "AutoModelForImageTextToText", None),
        getattr(transformers, "AutoModelForVision2Seq", None),
        getattr(transformers, "AutoModel", None),
    ]
    errors = []
    for constructor in constructors:
        if constructor is None:
            continue
        try:
            return constructor.from_pretrained(model_id, **local_kwargs)
        except Exception as exc:
            errors.append(f"{constructor.__name__}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors))


def build_multimodal_inputs(processor: Any, prompt: str, image: Any | None) -> Any:
    if hasattr(processor, "apply_chat_template"):
        content = []
        if image is not None:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if image is None:
            return processor(text=rendered, return_tensors="pt")
        return processor(text=rendered, images=image, return_tensors="pt")
    if image is None:
        return processor(text=prompt, return_tensors="pt")
    return processor(images=image, text=prompt, return_tensors="pt")


def extract_anchor_hidden_states(model: Any, batch: Any) -> tuple[Any, str]:
    if hasattr(model, "vision_model"):
        outputs = model.vision_model(**batch, output_hidden_states=True)
        hidden_states = getattr(outputs, "hidden_states", None)
        if has_nonempty_hidden_states(hidden_states):
            return hidden_states, "hidden_states"
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is not None:
            return [last_hidden_state], "last_hidden_state_fallback"
        return None, "missing"

    outputs = model(**batch, output_hidden_states=True)
    vision_output = getattr(outputs, "vision_model_output", None)
    hidden_states = getattr(vision_output, "hidden_states", None) if vision_output is not None else None
    if has_nonempty_hidden_states(hidden_states):
        return hidden_states, "hidden_states"
    last_hidden_state = getattr(vision_output, "last_hidden_state", None) if vision_output is not None else None
    if last_hidden_state is not None:
        return [last_hidden_state], "last_hidden_state_fallback"
    top_hidden_states = getattr(outputs, "hidden_states", None)
    if has_nonempty_hidden_states(top_hidden_states):
        return top_hidden_states, "top_level_hidden_states"
    return None, "missing"


def ready_sample_image() -> Path:
    import csv

    with (ROOT / "data/manifests/image_manifest.csv").open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ready = [row for row in rows if row["status"] == "ready"]
    if not ready:
        raise RuntimeError("No ready sensory images in data/manifests/image_manifest.csv")
    return ROOT / ready[0]["matched_image"]


def probe_prefix() -> dict[str, Any]:
    return {
        "config_loaded": False,
        "processor_loaded": False,
        "hidden_state_access": False,
        "forward_pass": False,
        "device": "",
        "dtype": "",
        "detail": "",
        "load_seconds": 0.0,
        "forward_seconds": 0.0,
        "total_seconds": 0.0,
        "stage": "start",
    }


def try_text_probe(transformers: Any, torch: Any, model_id: str, local_files_only: bool) -> dict[str, Any]:
    details = probe_prefix()
    start = time.perf_counter()
    try:
        transformers.AutoConfig.from_pretrained(model_id, local_files_only=local_files_only)
        details["config_loaded"] = True
        details["stage"] = "config_loaded"
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
        details["processor_loaded"] = True
        details["stage"] = "tokenizer_loaded"
        kwargs = model_load_kwargs(torch)
        kwargs["local_files_only"] = local_files_only
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        model = model.eval()
        details["load_seconds"] = round(time.perf_counter() - start, 4)
        details["stage"] = "model_loaded"
        batch = tokenizer("Consider the concept: bell.", return_tensors="pt")
        device = first_model_device(model)
        batch = move_batch_to_device(batch, device)
        forward_start = time.perf_counter()
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True)
        hidden_states = extract_hidden_states(outputs)
        details["hidden_state_access"] = has_nonempty_hidden_states(hidden_states)
        if not details["hidden_state_access"]:
            raise RuntimeError("text model forward pass did not expose non-empty hidden states")
        details["forward_pass"] = True
        details["device"] = str(device)
        details["dtype"] = str(next(model.parameters()).dtype)
        details["forward_seconds"] = round(time.perf_counter() - forward_start, 4)
        details["total_seconds"] = round(time.perf_counter() - start, 4)
        details["stage"] = "forward_pass_ok"
        details["detail"] = "text hidden states accessible"
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"status": "ok", **details}
    except Exception as exc:
        details["total_seconds"] = round(time.perf_counter() - start, 4)
        details["detail"] = f"{type(exc).__name__}: {exc}"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"status": classify_failure(exc, local_files_only), **details}


def try_multimodal_probe(
    transformers: Any,
    torch: Any,
    model_id: str,
    sample_image: Path,
    local_files_only: bool,
    require_gpu: bool,
) -> dict[str, Any]:
    details = probe_prefix()
    start = time.perf_counter()
    from PIL import Image

    image = Image.open(sample_image).convert("RGB")
    try:
        transformers.AutoConfig.from_pretrained(model_id, local_files_only=local_files_only)
        details["config_loaded"] = True
        details["stage"] = "config_loaded"
    except Exception as exc:
        details["total_seconds"] = round(time.perf_counter() - start, 4)
        details["detail"] = f"{type(exc).__name__}: {exc}"
        return {"status": classify_failure(exc, local_files_only), **details}
    try:
        processor = transformers.AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
        details["processor_loaded"] = True
        details["stage"] = "processor_loaded"
    except Exception as exc:
        details["total_seconds"] = round(time.perf_counter() - start, 4)
        details["detail"] = f"{type(exc).__name__}: {exc}"
        return {"status": classify_failure(exc, local_files_only), **details}

    load_attempts = [multimodal_load_kwargs(torch, prefer_cpu=False)]
    if torch.cuda.is_available() and not require_gpu:
        load_attempts.append(multimodal_load_kwargs(torch, prefer_cpu=True))
    for load_kwargs in load_attempts:
        load_kwargs["local_files_only"] = local_files_only
        try:
            model = load_multimodal_model(transformers, model_id, load_kwargs)
            model = model.eval()
            details["load_seconds"] = round(time.perf_counter() - start, 4)
            details["stage"] = "model_loaded"
            batch = build_multimodal_inputs(processor, "Consider the concept: bell.", image)
            device = first_model_device(model)
            if require_gpu and str(device) == "cpu":
                raise RuntimeError("multimodal model did not stay on GPU")
            batch = move_batch_to_device(batch, device)
            forward_start = time.perf_counter()
            with torch.no_grad():
                outputs = model(**batch, output_hidden_states=True)
            hidden_states = extract_hidden_states(outputs)
            details["hidden_state_access"] = has_nonempty_hidden_states(hidden_states)
            if not details["hidden_state_access"]:
                raise RuntimeError("multimodal forward pass did not expose non-empty hidden states")
            details["forward_pass"] = True
            details["device"] = str(device)
            details["dtype"] = str(next(model.parameters()).dtype)
            details["forward_seconds"] = round(time.perf_counter() - forward_start, 4)
            details["total_seconds"] = round(time.perf_counter() - start, 4)
            details["stage"] = "forward_pass_ok"
            details["detail"] = f"multimodal hidden states accessible via {processor.__class__.__name__}"
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {"status": "ok", **details}
        except Exception as exc:
            details["total_seconds"] = round(time.perf_counter() - start, 4)
            details["detail"] = f"{type(exc).__name__}: {exc}"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    return {"status": classify_failure(RuntimeError(details["detail"]), local_files_only), **details}


def try_anchor_probe(
    transformers: Any,
    torch: Any,
    model_id: str,
    sample_image: Path,
    local_files_only: bool,
    require_gpu: bool,
) -> dict[str, Any]:
    details = probe_prefix()
    start = time.perf_counter()
    from PIL import Image

    image = Image.open(sample_image).convert("RGB")
    try:
        transformers.AutoConfig.from_pretrained(model_id, local_files_only=local_files_only)
        details["config_loaded"] = True
        details["stage"] = "config_loaded"
        processor = transformers.AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)
        details["processor_loaded"] = True
        details["stage"] = "processor_loaded"
        kwargs = model_load_kwargs(torch)
        kwargs["local_files_only"] = local_files_only
        model = transformers.AutoModel.from_pretrained(model_id, **kwargs)
        model = model.eval()
        details["load_seconds"] = round(time.perf_counter() - start, 4)
        details["stage"] = "model_loaded"
        batch = processor(images=image, return_tensors="pt")
        device = first_model_device(model)
        if require_gpu and str(device) == "cpu":
            raise RuntimeError("anchor model did not stay on GPU")
        batch = move_batch_to_device(batch, device)
        forward_start = time.perf_counter()
        with torch.no_grad():
            hidden_states, source = extract_anchor_hidden_states(model, batch)
        details["hidden_state_access"] = has_nonempty_hidden_states(hidden_states)
        if not details["hidden_state_access"]:
            raise RuntimeError("anchor forward pass did not expose non-empty hidden states")
        details["forward_pass"] = True
        details["device"] = str(device)
        details["dtype"] = str(next(model.parameters()).dtype)
        details["forward_seconds"] = round(time.perf_counter() - forward_start, 4)
        details["total_seconds"] = round(time.perf_counter() - start, 4)
        details["stage"] = "forward_pass_ok"
        details["detail"] = f"anchor outputs accessible via {source}"
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"status": "ok", **details}
    except Exception as exc:
        details["total_seconds"] = round(time.perf_counter() - start, 4)
        details["detail"] = f"{type(exc).__name__}: {exc}"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"status": classify_failure(exc, local_files_only), **details}


def worker_probe(mode: str) -> dict[str, Any]:
    raise RuntimeError("worker_probe is replaced by single-worker mode.")


def worker_probe_single(mode: str, family: str, priority: str, model_id: str, require_gpu: bool) -> dict[str, Any]:
    config = load_project_config()
    analysis_runtime = config["analysis"]["runtime"]
    os.environ.setdefault("HF_HOME", analysis_runtime["hf_cache_dir"])
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", analysis_runtime["hf_cache_dir"])
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(analysis_runtime["hf_cache_dir"]) / "transformers"))
    if mode == "local_only":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    torch, transformers, deps, versions = load_runtime_dependencies()
    if not deps.get("torch") or not deps.get("transformers"):
        return {
            "mode": mode,
            "dependencies": deps,
            "package_versions": versions,
            "runtime_python": sys.executable,
            "row": {
                "family": family,
                "priority": priority,
                "model_id": model_id,
                "status": "dependency_missing",
                **probe_prefix(),
                "detail": "runtime missing torch or transformers",
            },
        }
    if require_gpu and not torch.cuda.is_available():
        return {
            "mode": mode,
            "dependencies": deps,
            "package_versions": versions,
            "runtime_python": sys.executable,
            "row": {
                "family": family,
                "priority": priority,
                "model_id": model_id,
                "status": "gpu_unavailable",
                **probe_prefix(),
                "detail": "CUDA is not available in this runtime",
            },
        }

    sample_image = ready_sample_image()
    local_files_only = mode == "local_only"
    if family == "text":
        result = try_text_probe(transformers, torch, model_id, local_files_only)
    elif family == "multimodal":
        result = try_multimodal_probe(transformers, torch, model_id, sample_image, local_files_only, require_gpu)
    else:
        result = try_anchor_probe(transformers, torch, model_id, sample_image, local_files_only, require_gpu)
    return {
        "mode": mode,
        "dependencies": deps,
        "package_versions": versions,
        "runtime_python": sys.executable,
        "row": {
            "family": family,
            "priority": priority,
            "model_id": model_id,
            **result,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["imports_only", "local_only"], default="local_only")
    parser.add_argument("--python-executable")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--family")
    parser.add_argument("--priority")
    parser.add_argument("--model-id")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    config = load_project_config()
    set_global_seed(config["seeds"]["model_probe"])

    if args.worker:
        if not (args.family and args.priority and args.model_id):
            raise RuntimeError("worker mode requires --family, --priority, and --model-id")
        payload = worker_probe_single(args.mode, args.family, args.priority, args.model_id, args.require_gpu)
        print(json.dumps(payload))
        return

    runtime_python = pick_runtime_python(config, args.python_executable)
    if args.mode == "imports_only":
        payload = {
            "mode": args.mode,
            "dependencies": {},
            "runtime_python": runtime_python,
            "rows": [
                {
                    **row,
                    "status": "not_run",
                    **probe_prefix(),
                    "detail": "imports_only mode",
                }
                for row in flatten_model_entries(config)
            ],
            "locked_models": {},
            "blocked_families": {},
        }
    else:
        rows = []
        dependencies: dict[str, bool] = {}
        package_versions: dict[str, str] = {}
        for row in flatten_model_entries(config):
            timeout_seconds = max(args.timeout_seconds, family_timeout_seconds(row["family"]))
            try:
                result = run_single_worker(
                    runtime_python,
                    args.mode,
                    row["family"],
                    row["priority"],
                    row["model_id"],
                    timeout_seconds,
                    require_gpu=True,
                )
                rows.append(result["row"])
                dependencies = result["dependencies"]
                package_versions = result.get("package_versions", package_versions)
            except subprocess.TimeoutExpired:
                rows.append(
                    {
                        **row,
                        "status": "timeout",
                        **probe_prefix(),
                        "detail": f"probe exceeded {timeout_seconds}s timeout",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        **row,
                        "status": "failed",
                        **probe_prefix(),
                        "detail": str(exc).splitlines()[0][:500],
                    }
                )
        locked, blocked = choose_locked_models(rows)
        for family, model_id in list(locked.items()):
            original = next(row for row in rows if row["family"] == family and row["model_id"] == model_id)
            confirmed = run_single_worker(
                runtime_python,
                args.mode,
                family,
                original["priority"],
                model_id,
                max(60, min(family_timeout_seconds(family), args.timeout_seconds)),
                require_gpu=True,
            )["row"]
            replace_row(rows, confirmed)
        locked, blocked = choose_locked_models(rows)
        payload = {
            "mode": args.mode,
            "dependencies": dependencies,
            "package_versions": package_versions,
            "runtime_python": runtime_python,
            "rows": rows,
            "locked_models": locked,
            "blocked_families": blocked,
        }

    csv_path = ROOT / "outputs/logs/model_probe_log.csv"
    json_path = ROOT / "outputs/logs/model_probe_summary.json"
    write_csv(
        csv_path,
        payload["rows"],
        [
            "family",
            "priority",
            "model_id",
            "status",
            "config_loaded",
            "processor_loaded",
            "hidden_state_access",
            "forward_pass",
            "device",
            "dtype",
            "stage",
            "load_seconds",
            "forward_seconds",
            "total_seconds",
            "detail",
        ],
    )
    write_json(
        json_path,
        {
            "mode": payload["mode"],
            "dependencies": payload["dependencies"],
            "package_versions": payload.get("package_versions", {}),
            "runtime_python": payload["runtime_python"],
            "locked_models": payload["locked_models"],
            "blocked_families": payload["blocked_families"],
        },
    )
    append_run_log("Model Probe", summarize_results(payload, runtime_python))


if __name__ == "__main__":
    main()
