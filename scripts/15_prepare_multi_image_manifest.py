from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import ROOT, append_run_log, output_path, read_csv, write_csv


THINGS_IMAGE_DIR = ROOT / "THINGS-database" / "osfstorage" / "object_images"
REVERSAL_AUDIT = ROOT / "outputs" / "tables" / "human_anchor_reversal_audit.csv"
DEFAULT_CONTROLS = ["marble", "emerald", "amber", "gold", "ruby", "sandpaper", "moss"]


def preferred_images(concept: str, limit: int) -> list[Path]:
    directory = THINGS_IMAGE_DIR / concept
    if not directory.exists():
        return []
    files = sorted(path for path in directory.iterdir() if path.is_file())
    if not files:
        return []
    lead = [path for path in files if "_01b" in path.name or "_01s" in path.name or "_01n" in path.name]
    ordered: list[Path] = []
    for path in lead + files:
        if path not in ordered:
            ordered.append(path)
    return ordered[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--images-per-concept", type=int, default=5)
    args = parser.parse_args()

    image_manifest = {row["concept"]: row for row in read_csv(ROOT / "data" / "manifests" / "image_manifest.csv") if row["status"] == "ready"}
    reversal_rows = list(csv.DictReader(REVERSAL_AUDIT.open(newline="", encoding="utf-8")))
    reversal_concepts = [row["concept"] for row in reversal_rows]
    selected_concepts = []
    for concept in reversal_concepts + DEFAULT_CONTROLS:
        if concept not in image_manifest:
            continue
        if concept not in selected_concepts:
            selected_concepts.append(concept)

    concept_rows = []
    manifest_rows = []
    for concept in selected_concepts:
        archive_images = preferred_images(concept, max(args.images_per_concept - 1, 2))
        current_matched = ROOT / image_manifest[concept]["matched_image"]
        image_paths = [current_matched]
        for path in archive_images:
            if path.resolve() == current_matched.resolve():
                continue
            image_paths.append(path)
        image_paths = image_paths[: args.images_per_concept]
        concept_rows.append(
            {
                "concept": concept,
                "diagnostic_role": "reversal" if concept in reversal_concepts else "control",
                "selected_image_count": len(image_paths),
            }
        )
        for image_index, path in enumerate(image_paths, start=1):
            relpath = path.relative_to(ROOT)
            manifest_rows.append(
                {
                    "concept": concept,
                    "diagnostic_role": "reversal" if concept in reversal_concepts else "control",
                    "image_index": image_index,
                    "image_id": f"{concept}_{image_index:02d}",
                    "image_role": "current_matched" if path.resolve() == current_matched.resolve() else "things_archive",
                    "image_path": str(relpath),
                }
            )

    write_csv(
        output_path("data", "concepts", "multi_image_diagnostic_concepts.csv"),
        concept_rows,
        ["concept", "diagnostic_role", "selected_image_count"],
    )
    write_csv(
        output_path("data", "manifests", "multi_image_manifest.csv"),
        manifest_rows,
        ["concept", "diagnostic_role", "image_index", "image_id", "image_role", "image_path"],
    )
    output_path("outputs", "metrics", "multi_image_manifest_summary.json").write_text(
        json.dumps(
            {
                "selected_concepts": selected_concepts,
                "reversal_concepts": reversal_concepts,
                "control_concepts": [concept for concept in selected_concepts if concept not in reversal_concepts],
                "images_per_concept_target": args.images_per_concept,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    append_run_log(
        "Multi-Image Manifest",
        [
            f"Wrote diagnostic concept list to {output_path('data', 'concepts', 'multi_image_diagnostic_concepts.csv').relative_to(ROOT)}.",
            f"Wrote multi-image manifest to {output_path('data', 'manifests', 'multi_image_manifest.csv').relative_to(ROOT)}.",
        ],
    )


if __name__ == "__main__":
    main()
