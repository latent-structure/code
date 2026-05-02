from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import ROOT, append_run_log, ensure_parent, read_csv, write_csv


THINGS_METADATA = ROOT / "data" / "concepts" / "things" / "concepts-metadata_things.tsv"
THINGS_IMAGE_DIR = ROOT / "THINGS-database" / "osfstorage" / "object_images"
THINGS_IMAGES_ZIP = ROOT / "THINGS-database" / "osfstorage" / "images_THINGS.zip"
FULL_CONCEPT_LIST = ROOT / "data" / "concepts" / "full_concept_list.csv"
IMAGE_MANIFEST = ROOT / "data" / "manifests" / "image_manifest.csv"
OUT_DIR = ROOT / "data" / "images" / "sensory"
THINGS_ZIP_PASSWORD = b"things4all"


def normalize_token(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def load_things_metadata() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    with THINGS_METADATA.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    by_word = {row["Word"].strip().lower(): row for row in rows}
    by_uid = {normalize_token(row["uniqueID"]): row for row in rows}
    return by_word, by_uid


def preferred_member(members: list[str]) -> str:
    ordered = sorted(members)
    for suffix in ("_01b.", "_01s.", "_01n.", "_01"):
        for member in ordered:
            if suffix in member:
                return member
    return ordered[0]


def image_member_for(concept: str, metadata: dict[str, str], archive_names: list[str]) -> str | None:
    candidates = [
        f"object_images/{normalize_token(metadata.get('uniqueID', ''))}/",
        f"object_images/{normalize_token(metadata.get('Word', ''))}/",
        f"object_images/{normalize_token(concept)}/",
    ]
    seen = set()
    for prefix in candidates:
        if not prefix or prefix in seen:
            continue
        seen.add(prefix)
        members = [name for name in archive_names if name.startswith(prefix) and not name.endswith("/")]
        if members:
            return preferred_member(members)
    return None


def image_path_for(concept: str, metadata: dict[str, str]) -> Path | None:
    candidates = [
        THINGS_IMAGE_DIR / normalize_token(metadata.get("uniqueID", "")),
        THINGS_IMAGE_DIR / normalize_token(metadata.get("Word", "")),
        THINGS_IMAGE_DIR / normalize_token(concept),
    ]
    seen = set()
    for directory in candidates:
        key = str(directory)
        if not key or key in seen:
            continue
        seen.add(key)
        if not directory.exists() or not directory.is_dir():
            continue
        members = [str(path.relative_to(THINGS_IMAGE_DIR)) for path in directory.iterdir() if path.is_file()]
        if members:
            chosen = preferred_member(members)
            return THINGS_IMAGE_DIR / chosen
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--missing-only", action="store_true")
    parser.add_argument("--replace-covered", action="store_true")
    args = parser.parse_args()

    concepts = [row for row in read_csv(FULL_CONCEPT_LIST) if row["domain"] == "sensory"]
    image_rows = {row["concept"]: row for row in read_csv(IMAGE_MANIFEST)} if IMAGE_MANIFEST.exists() else {}
    by_word, by_uid = load_things_metadata()

    provenance_rows: list[dict[str, str]] = []
    covered_rows: list[dict[str, str]] = []
    missing_coverage: list[str] = []
    extracted_count = 0
    replaced_count = 0

    archive_names: list[str] = []
    archive = None
    if THINGS_IMAGES_ZIP.exists():
        import zipfile

        archive = zipfile.ZipFile(THINGS_IMAGES_ZIP)
        archive_names = archive.namelist()

    try:
        for row in concepts:
            concept = row["concept"]
            current_status = image_rows.get(concept, {}).get("status", "missing")
            metadata = by_word.get(concept.lower()) or by_uid.get(normalize_token(concept))
            if metadata is None:
                provenance_rows.append(
                    {
                        "concept": concept,
                        "status": "uncovered",
                        "things_word": "",
                        "things_unique_id": "",
                        "archive_member": "",
                        "local_image": image_rows.get(concept, {}).get("matched_image", ""),
                        "notes": "No THINGS concept metadata match.",
                    }
                )
                missing_coverage.append(concept)
                continue

            source_path = image_path_for(concept, metadata)
            member = str(source_path.relative_to(ROOT)) if source_path is not None else ""
            if source_path is None and archive_names:
                member = image_member_for(concept, metadata, archive_names) or ""

            if not member:
                provenance_rows.append(
                    {
                        "concept": concept,
                        "status": "uncovered",
                        "things_word": metadata.get("Word", ""),
                        "things_unique_id": metadata.get("uniqueID", ""),
                        "archive_member": "",
                        "local_image": image_rows.get(concept, {}).get("matched_image", ""),
                        "notes": "THINGS concept matched but no local image file was found.",
                    }
                )
                missing_coverage.append(concept)
                continue

            covered_rows.append(row)
            destination = OUT_DIR / f"{concept}{Path(member).suffix.lower()}"
            should_extract = True
            if args.missing_only and current_status == "ready" and not args.replace_covered:
                should_extract = False
            if destination.exists() and not args.replace_covered and current_status == "ready":
                should_extract = False

            note = "selected canonical THINGS image"
            status = "covered"
            if should_extract:
                ensure_parent(destination)
                if source_path is not None:
                    destination.write_bytes(source_path.read_bytes())
                elif archive is not None:
                    try:
                        destination.write_bytes(archive.read(member, pwd=THINGS_ZIP_PASSWORD))
                        note = "selected canonical THINGS image from password-protected archive"
                    except RuntimeError:
                        status = "archive_only_locked"
                        note = "THINGS image appears only in encrypted zip; password extraction failed."
                    except KeyError:
                        status = "uncovered"
                        note = "THINGS archive member was not found during extraction."
                else:
                    status = "uncovered"
                    note = "THINGS archive was not available for fallback extraction."
                if status == "covered":
                    if current_status == "ready":
                        replaced_count += 1
                    else:
                        extracted_count += 1

            if status != "covered":
                if destination.exists() and destination.stat().st_size == 0:
                    destination.unlink()
                missing_coverage.append(concept)

            provenance_rows.append(
                {
                    "concept": concept,
                    "status": status,
                    "things_word": metadata.get("Word", ""),
                    "things_unique_id": metadata.get("uniqueID", ""),
                    "archive_member": member,
                    "local_image": str(destination.relative_to(ROOT)) if destination.exists() else image_rows.get(concept, {}).get("matched_image", ""),
                    "notes": note,
                }
            )
    finally:
        if archive is not None:
            archive.close()

    abstract_rows = [row for row in read_csv(FULL_CONCEPT_LIST) if row["domain"] == "abstract"]
    covered_out = covered_rows + abstract_rows
    write_csv(
        ROOT / "data" / "manifests" / "things_image_provenance.csv",
        provenance_rows,
        ["concept", "status", "things_word", "things_unique_id", "archive_member", "local_image", "notes"],
    )
    write_csv(
        ROOT / "data" / "concepts" / "things_covered_concepts.csv",
        covered_out,
        [
            "concept",
            "domain",
            "subtype",
            "source_dataset",
            "notes",
            "polysemy_risk",
            "image_source",
            "image_quality_flag",
            "human_anchor_available",
        ],
    )
    write_csv(
        ROOT / "data" / "concepts" / "things_missing_sensory_concepts.csv",
        [{"concept": concept} for concept in sorted(missing_coverage)],
        ["concept"],
    )

    append_run_log(
        "THINGS Image Linking",
        [
            f"Extracted {extracted_count} THINGS images and replaced {replaced_count} existing sensory images in data/images/sensory.",
            f"Wrote THINGS provenance manifest to data/manifests/things_image_provenance.csv.",
            f"Wrote THINGS-covered execution subset with {len(covered_rows)} sensory concepts and {len(abstract_rows)} abstract concepts to data/concepts/things_covered_concepts.csv.",
            f"Wrote uncovered sensory list ({len(missing_coverage)} concepts) to data/concepts/things_missing_sensory_concepts.csv.",
        ],
    )


if __name__ == "__main__":
    main()
