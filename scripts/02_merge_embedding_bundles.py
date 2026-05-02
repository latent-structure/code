from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from common import ROOT, append_run_log, ensure_parent, require, write_json


def tagged_paths(tag: str) -> tuple[Path, Path]:
    base = ROOT / "outputs" / "embeddings"
    return base / f"pooled_embeddings_{tag}.npz", base / f"embedding_metadata_{tag}.json"


def canonical_paths() -> tuple[Path, Path]:
    base = ROOT / "outputs" / "embeddings"
    return base / "pooled_embeddings_full.npz", base / "embedding_metadata_full.json"


def load_tagged_bundle(tag: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    npz_path, metadata_path = tagged_paths(tag)
    require(npz_path.exists(), f"Missing tagged embedding bundle: {npz_path}")
    require(metadata_path.exists(), f"Missing tagged metadata bundle: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload = np.load(npz_path)
    arrays = {key: np.asarray(payload[key], dtype=np.float32) for key in payload.files}
    return metadata, arrays


def require_clean_span_pooling(tag: str, metadata: dict[str, Any]) -> None:
    diagnostics = metadata.get("span_pooling_diagnostics")
    require(isinstance(diagnostics, list) and diagnostics, f"Missing span pooling diagnostics for tag {tag}")
    failures = []
    for row in diagnostics:
        attempted = int(row.get("attempted_spans", -1))
        matched = int(row.get("matched_spans", -1))
        pooling_target = str(row.get("pooling_target", ""))
        if pooling_target != "concept_span":
            failures.append(
                f"{row.get('model_id')} {row.get('condition')} {row.get('domain')} pooling_target={pooling_target}"
            )
        if attempted <= 0 or matched != attempted:
            failures.append(
                f"{row.get('model_id')} {row.get('condition')} {row.get('domain')} "
                f"matched={matched} attempted={attempted}"
            )
    require(not failures, f"Span pooling diagnostics failed for tag {tag}: {'; '.join(failures)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", required=True, help="Comma-separated extraction tags to merge.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    tags = [item.strip() for item in args.tags.split(",") if item.strip()]
    require(tags, "No extraction tags were provided.")

    canonical_npz, canonical_json = canonical_paths()
    require(args.overwrite or not canonical_npz.exists(), f"{canonical_npz} already exists. Re-run with --overwrite to replace it.")
    require(args.overwrite or not canonical_json.exists(), f"{canonical_json} already exists. Re-run with --overwrite to replace it.")

    merged_arrays: dict[str, np.ndarray] = {}
    merged_records: list[dict[str, Any]] = []
    merged_models: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, int, str]] = set()
    concept_subset: str | None = None
    precision_names: set[str] = set()
    start = 0

    for tag in tags:
        metadata, arrays = load_tagged_bundle(tag)
        require_clean_span_pooling(tag, metadata)
        if concept_subset is None:
            concept_subset = metadata.get("concept_subset", "")
        else:
            require(
                metadata.get("concept_subset", "") == concept_subset,
                f"Concept subset mismatch for tag {tag}: {metadata.get('concept_subset', '')} != {concept_subset}",
            )
        precision_names.add(str(metadata.get("precision", "")))

        for model_row in metadata.get("selected_models", []):
            item = dict(model_row)
            item["output_tag"] = tag
            merged_models.append(item)

        for record in metadata.get("records", []):
            key = (
                str(record["model_id"]),
                str(record["condition"]),
                int(record["layer"]),
                str(record["domain"]),
            )
            require(key not in seen_keys, f"Duplicate merged record key detected: {key}")
            seen_keys.add(key)

            old_id = int(record["record_id"])
            new_id = start
            start += 1
            merged_arrays[f"record_{new_id}"] = arrays[f"record_{old_id}"]
            merged_records.append({**record, "record_id": new_id})

    ensure_parent(canonical_npz)
    tmp_npz = canonical_npz.with_suffix(".tmp.npz")
    if tmp_npz.exists():
        tmp_npz.unlink()
    # The merged bundle is very large; compression is CPU-bound and can exceed
    # SLURM walltime. Write uncompressed to a temp file and publish atomically.
    np.savez(tmp_npz, **merged_arrays)
    tmp_npz.replace(canonical_npz)
    metadata = {
        "mode": "merged_real_model_extraction",
        "source_tags": tags,
        "concept_subset": concept_subset or "",
        "selected_models": merged_models,
        "precision": precision_names.pop() if len(precision_names) == 1 else "mixed",
        "record_count": len(merged_records),
        "span_pooling_verified": True,
        "records": merged_records,
    }
    tmp_json = canonical_json.with_suffix(".tmp.json")
    if tmp_json.exists():
        tmp_json.unlink()
    write_json(tmp_json, metadata)
    tmp_json.replace(canonical_json)
    append_run_log(
        "Merge Embeddings",
        [
            f"Merged tags: {', '.join(tags)}.",
            f"Wrote canonical pooled embeddings to {canonical_npz.relative_to(ROOT)}.",
            f"Wrote canonical extraction metadata to {canonical_json.relative_to(ROOT)}.",
            f"Merged {len(merged_records)} records across {len(merged_models)} selected model rows.",
        ],
    )


if __name__ == "__main__":
    main()
