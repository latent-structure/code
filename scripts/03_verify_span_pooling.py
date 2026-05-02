from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import ROOT, append_run_log, require


def tagged_metadata_path(tag: str) -> Path:
    return ROOT / "outputs" / "embeddings" / f"embedding_metadata_{tag}.json"


def validate_tag(tag: str) -> list[str]:
    metadata_path = tagged_metadata_path(tag)
    require(metadata_path.exists(), f"Missing tagged metadata bundle: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    diagnostics = metadata.get("span_pooling_diagnostics")
    require(isinstance(diagnostics, list) and diagnostics, f"Missing span pooling diagnostics for tag {tag}")

    failures: list[str] = []
    for row in diagnostics:
        attempted = int(row.get("attempted_spans", -1))
        matched = int(row.get("matched_spans", -1))
        pooling_target = str(row.get("pooling_target", ""))
        if pooling_target != "concept_span":
            failures.append(
                f"{tag}: unexpected pooling target for {row.get('model_id')} {row.get('condition')} "
                f"{row.get('domain')}: {pooling_target}"
            )
        if attempted <= 0:
            failures.append(
                f"{tag}: non-positive attempted span count for {row.get('model_id')} "
                f"{row.get('condition')} {row.get('domain')}: {attempted}"
            )
        if matched != attempted:
            failures.append(
                f"{tag}: matched_spans != attempted_spans for {row.get('model_id')} "
                f"{row.get('condition')} {row.get('domain')}: {matched} != {attempted}"
            )
    return failures


def summarize_selected_models(metadata: dict[str, Any]) -> str:
    rows = metadata.get("selected_models", [])
    return ", ".join(str(row.get("model_id", "?")) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", required=True, help="Comma-separated extraction tags to verify.")
    args = parser.parse_args()

    tags = [item.strip() for item in args.tags.split(",") if item.strip()]
    require(tags, "No extraction tags were provided.")

    failures: list[str] = []
    summaries: list[str] = []
    for tag in tags:
        metadata_path = tagged_metadata_path(tag)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        summaries.append(f"{tag}: {summarize_selected_models(metadata)}")
        failures.extend(validate_tag(tag))

    require(not failures, "Span pooling verification failed:\n" + "\n".join(failures))
    append_run_log(
        "Verify Span Pooling",
        [
            f"Verified span pooling diagnostics for tags: {', '.join(tags)}.",
            *summaries,
            "All diagnostics reported concept-span pooling with matched spans equal to attempted spans.",
        ],
    )


if __name__ == "__main__":
    main()
