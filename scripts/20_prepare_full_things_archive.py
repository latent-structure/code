from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ROOT, append_run_log, load_project_config, output_path, read_csv, write_csv


THINGS_IMAGE_DIR = ROOT / "THINGS-database" / "osfstorage" / "object_images"


def preferred_archive_images(concept: str) -> list[Path]:
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
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    args = parser.parse_args()
    config = load_project_config(args.config)

    subset_path = config["analysis"].get("execution", {}).get("default_concept_subset", "")
    subset_file = (ROOT / subset_path) if subset_path else (ROOT / "data" / "concepts" / "full_concept_list.csv")
    concept_rows = [row for row in read_csv(subset_file) if row["domain"] == "sensory"]
    image_manifest = {
        row["concept"]: row for row in read_csv(ROOT / "data" / "manifests" / "image_manifest.csv") if row["status"] == "ready"
    }

    concept_summary_rows = []
    manifest_rows = []
    excluded_rows = []

    for row in concept_rows:
        concept = row["concept"]
        current = image_manifest.get(concept)
        archive_images = preferred_archive_images(concept)
        current_rel = current["matched_image"] if current else ""
        if not archive_images:
            excluded_rows.append(
                {
                    "concept": concept,
                    "subtype": row["subtype"],
                    "reason": "missing_archive_directory_or_files",
                    "current_matched_image": current_rel,
                }
            )
            continue

        concept_summary_rows.append(
            {
                "concept": concept,
                "subtype": row["subtype"],
                "source_dataset": row["source_dataset"],
                "current_matched_image": current_rel,
                "archive_image_count": len(archive_images),
                "selected_image_count": len(archive_images),
            }
        )

        for image_index, image_path in enumerate(archive_images, start=1):
            manifest_rows.append(
                {
                    "concept": concept,
                    "subtype": row["subtype"],
                    "image_index": image_index,
                    "image_id": f"{concept}_{image_index:03d}",
                    "image_role": "things_archive",
                    "image_path": str(image_path.relative_to(ROOT)),
                }
            )

    write_csv(
        output_path("data", "concepts", "full_things_archive_concepts.csv"),
        concept_summary_rows,
        ["concept", "subtype", "source_dataset", "current_matched_image", "archive_image_count", "selected_image_count"],
    )
    write_csv(
        output_path("data", "manifests", "full_things_archive_manifest.csv"),
        manifest_rows,
        ["concept", "subtype", "image_index", "image_id", "image_role", "image_path"],
    )
    write_csv(
        output_path("data", "manifests", "full_things_archive_exclusions.csv"),
        excluded_rows,
        ["concept", "subtype", "reason", "current_matched_image"],
    )

    archive_counts = [int(row["archive_image_count"]) for row in concept_summary_rows]
    summary = {
        "available_concept_count": len(concept_summary_rows),
        "excluded_concepts": [row["concept"] for row in excluded_rows],
        "excluded_count": len(excluded_rows),
        "max_archive_images": max(archive_counts) if archive_counts else 0,
        "mean_archive_images": (sum(archive_counts) / len(archive_counts)) if archive_counts else 0.0,
        "min_archive_images": min(archive_counts) if archive_counts else 0,
        "total_archive_images": sum(archive_counts),
    }
    output_path("outputs", "metrics", "full_things_archive_manifest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    append_run_log(
        "Full THINGS Archive Manifest",
        [
            f"Wrote full THINGS concept list to {output_path('data', 'concepts', 'full_things_archive_concepts.csv').relative_to(ROOT)}.",
            f"Wrote full THINGS archive manifest to {output_path('data', 'manifests', 'full_things_archive_manifest.csv').relative_to(ROOT)}.",
            f"Prepared exhaustive archive rows for {len(concept_summary_rows)} concepts and excluded {len(excluded_rows)} concepts with no local THINGS archive directory.",
        ],
    )


if __name__ == "__main__":
    main()
