from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageFilter

from common import ROOT, append_run_log, ensure_parent, load_project_config, read_csv, write_csv


THINGS_METADATA = ROOT / "data" / "concepts" / "things" / "concepts-metadata_things.tsv"
THINGS_BEHAVIOR_ORDER = ROOT / "THINGS-behavior" / "osfstorage" / "variables" / "unique_id.txt"
THINGS_IMAGE_DIR = ROOT / "THINGS-database" / "osfstorage" / "object_images"
THINGS_IMAGES_ZIP = ROOT / "THINGS-database" / "osfstorage" / "images_THINGS.zip"
THINGS_ZIP_PASSWORD = b"things4all"
MATCHED_IMAGE_DIR = ROOT / "data" / "images" / "things_max"
DEGRADED_IMAGE_DIR = MATCHED_IMAGE_DIR / "degraded"


def normalize_token(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def preferred_member(members: list[str]) -> str:
    ordered = sorted(members)
    for suffix in ("_01b.", "_01s.", "_01n.", "_01"):
        for member in ordered:
            if suffix in member:
                return member
    return ordered[0]


def load_things_metadata() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    with THINGS_METADATA.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    by_word = {row["Word"].strip().lower(): row for row in rows}
    by_uid = {normalize_token(row["uniqueID"]): row for row in rows}
    return by_word, by_uid


def load_archive_names() -> list[str]:
    if not THINGS_IMAGES_ZIP.exists():
        return []
    import zipfile

    with zipfile.ZipFile(THINGS_IMAGES_ZIP) as archive:
        return archive.namelist()


def choose_subtype(metadata: dict[str, str]) -> str:
    for column in (
        "Top-down Category (manual selection)",
        "Top-down Category (WordNet)",
        "Bottom-up Category (Human Raters)",
        "Dominant Part of Speech",
    ):
        value = metadata.get(column, "").strip()
        if value:
            return value
    return "uncategorized"


def derive_polysemy_risk(metadata: dict[str, str]) -> str:
    meanings = metadata.get("Number of word meanings in list", "").strip()
    try:
        count = int(float(meanings))
    except ValueError:
        count = 1
    if count >= 4:
        return "high"
    if count >= 2:
        return "medium"
    return "low"


def image_path_for(concept: str, metadata: dict[str, str], archive_names: list[str]) -> tuple[Path | None, str]:
    candidates = [
        normalize_token(metadata.get("uniqueID", "")),
        normalize_token(metadata.get("Word", "")),
        normalize_token(concept),
    ]
    seen = set()
    for token in candidates:
        if not token or token in seen:
            continue
        seen.add(token)
        directory = THINGS_IMAGE_DIR / token
        if directory.exists() and directory.is_dir():
            files = [path for path in sorted(directory.iterdir()) if path.is_file()]
            if files:
                return files[0], "filesystem"
        if archive_names:
            prefix = f"object_images/{token}/"
            members = [name for name in archive_names if name.startswith(prefix) and not name.endswith("/")]
            if members:
                return Path(preferred_member(members)), "archive"
    return None, ""


def degrade_image(src: Path, dst: Path, fraction: float, grayscale: bool, blur_radius: int) -> None:
    with Image.open(src) as image:
        image = image.convert("RGB")
        small = image.resize(
            (max(1, int(image.width * fraction)), max(1, int(image.height * fraction))),
            Image.Resampling.BILINEAR,
        )
        restored = small.resize((image.width, image.height), Image.Resampling.BILINEAR)
        if grayscale:
            restored = restored.convert("L").convert("RGB")
        restored = restored.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        dst.parent.mkdir(parents=True, exist_ok=True)
        restored.save(dst)


def build_mismatch_map(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    import random

    rng = random.Random(seed)
    by_subtype: dict[str, list[str]] = {}
    for row in rows:
        by_subtype.setdefault(row["subtype"], []).append(row["concept"])
    for concepts in by_subtype.values():
        concepts.sort()
        rng.shuffle(concepts)

    subtype_names = sorted(by_subtype)
    all_concepts = sorted(row["concept"] for row in rows)
    subtype_lookup = {row["concept"]: row["subtype"] for row in rows}
    results: list[dict[str, str]] = []
    subtype_positions = {subtype: 0 for subtype in subtype_names}
    all_position = 0

    for index, row in enumerate(sorted(rows, key=lambda item: item["concept"])):
        same_subtype = index % 2 == 0
        if same_subtype and len(by_subtype[row["subtype"]]) > 1:
            pool = by_subtype[row["subtype"]]
            pos = subtype_positions[row["subtype"]]
            choice = pool[(pos + 1) % len(pool)]
            subtype_positions[row["subtype"]] = (pos + 1) % len(pool)
            if choice == row["concept"]:
                choice = pool[(pos + 2) % len(pool)]
        else:
            for _ in range(len(all_concepts)):
                choice = all_concepts[all_position % len(all_concepts)]
                all_position += 1
                if choice != row["concept"] and choice not in by_subtype[row["subtype"]]:
                    break
            else:
                choice = next(concept for concept in all_concepts if concept != row["concept"])
        results.append(
            {
                "concept": row["concept"],
                "concept_subtype": row["subtype"],
                "mismatch_concept": choice,
                "mismatch_subtype": subtype_lookup[choice],
                "mismatch_mode": "within_subtype" if same_subtype else "cross_subtype",
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    args = parser.parse_args()
    config = load_project_config(args.config)
    seeds = config["seeds"]
    image_policy = config["analysis"]["image_policy"]

    behavior_order = [line.strip() for line in THINGS_BEHAVIOR_ORDER.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_word, by_uid = load_things_metadata()
    archive_names = load_archive_names()

    MATCHED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    DEGRADED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    subset_rows: list[dict[str, str]] = []
    provenance_rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, str]] = []
    missing_rows: list[dict[str, str]] = []

    for index, concept in enumerate(behavior_order, start=1):
        metadata = by_uid.get(normalize_token(concept)) or by_word.get(concept.lower())
        if metadata is None:
            missing_rows.append({"concept": concept, "reason": "missing_things_metadata"})
            continue

        source_path, source_kind = image_path_for(concept, metadata, archive_names)
        if source_path is None:
            missing_rows.append({"concept": concept, "reason": "missing_archive_directory_or_files"})
            continue

        if source_kind == "filesystem":
            matched_source = source_path
        else:
            matched_source = THINGS_IMAGE_DIR.parent / source_path
            if not matched_source.exists():
                import zipfile

                if not THINGS_IMAGES_ZIP.exists():
                    missing_rows.append({"concept": concept, "reason": "missing_zip_fallback"})
                    continue
                with zipfile.ZipFile(THINGS_IMAGES_ZIP) as archive:
                    payload = archive.read(str(source_path), pwd=THINGS_ZIP_PASSWORD)
                matched_source = MATCHED_IMAGE_DIR / f"{normalize_token(concept)}{Path(str(source_path)).suffix.lower()}"
                ensure_parent(matched_source)
                matched_source.write_bytes(payload)

        destination = MATCHED_IMAGE_DIR / f"{normalize_token(concept)}{matched_source.suffix.lower()}"
        if not destination.exists() or destination.stat().st_size == 0:
            ensure_parent(destination)
            destination.write_bytes(matched_source.read_bytes())

        degraded = DEGRADED_IMAGE_DIR / f"{normalize_token(concept)}.png"
        degrade_image(
            destination,
            degraded,
            image_policy["downsample_linear_fraction"],
            image_policy["grayscale"],
            image_policy["gaussian_blur_radius"],
        )

        subtype = choose_subtype(metadata)
        row = {
            "concept": concept,
            "domain": "sensory",
            "subtype": subtype,
            "source_dataset": "things",
            "notes": "THINGS archive exact overlap with local image support",
            "polysemy_risk": derive_polysemy_risk(metadata),
            "image_source": "things_archive",
            "image_quality_flag": "good",
            "human_anchor_available": "yes",
        }
        subset_rows.append(row)
        provenance_rows.append(
            {
                "concept": concept,
                "status": "covered",
                "things_word": metadata.get("Word", ""),
                "things_unique_id": metadata.get("uniqueID", ""),
                "archive_member": str(source_path),
                "local_image": str(destination.relative_to(ROOT)),
                "notes": f"selected THINGS archive image via {source_kind}",
            }
        )
        manifest_rows.append(
            {
                "concept": concept,
                "subtype": subtype,
                "matched_image": str(destination.relative_to(ROOT)),
                "degraded_image": str(degraded.relative_to(ROOT)),
                "status": "ready",
                "source_kind": "things_archive",
                "notes": "THINGS archive image and degraded derivative available",
            }
        )
        if index % 100 == 0:
            print(f"[{index}/{len(behavior_order)}] prepared {len(subset_rows)} concepts", flush=True)

    write_csv(
        ROOT / "data" / "concepts" / "things_max_1854_concepts.csv",
        subset_rows,
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
        ROOT / "data" / "manifests" / "things_max_image_provenance.csv",
        provenance_rows,
        ["concept", "status", "things_word", "things_unique_id", "archive_member", "local_image", "notes"],
    )
    mismatch_rows = build_mismatch_map(manifest_rows, seeds["mismatch_map"])
    write_csv(
        ROOT / "data" / "manifests" / "things_max_image_manifest.csv",
        manifest_rows,
        ["concept", "subtype", "matched_image", "degraded_image", "status", "source_kind", "notes"],
    )
    write_csv(
        ROOT / "data" / "manifests" / "things_max_mismatch_map.csv",
        mismatch_rows,
        ["concept", "concept_subtype", "mismatch_concept", "mismatch_subtype", "mismatch_mode"],
    )
    write_csv(
        ROOT / "data" / "manifests" / "image_manifest.csv",
        manifest_rows,
        ["concept", "subtype", "matched_image", "degraded_image", "status", "source_kind", "notes"],
    )
    write_csv(
        ROOT / "data" / "manifests" / "things_image_provenance.csv",
        provenance_rows,
        ["concept", "status", "things_word", "things_unique_id", "archive_member", "local_image", "notes"],
    )
    write_csv(
        ROOT / "data" / "manifests" / "mismatch_map.csv",
        mismatch_rows,
        ["concept", "concept_subtype", "mismatch_concept", "mismatch_subtype", "mismatch_mode"],
    )
    write_csv(
        ROOT / "data" / "concepts" / "concept_subtypes.csv",
        [{"concept": row["concept"], "domain": row["domain"], "subtype": row["subtype"]} for row in subset_rows],
        ["concept", "domain", "subtype"],
    )
    write_csv(
        ROOT / "data" / "concepts" / "things_max_missing_concepts.csv",
        missing_rows,
        ["concept", "reason"],
    )

    append_run_log(
        "THINGS Max Subset",
        [
            f"Wrote THINGS-max sensory subset with {len(subset_rows)} concepts to data/concepts/things_max_1854_concepts.csv.",
            f"Wrote THINGS-max image manifest to data/manifests/image_manifest.csv and branch-specific copies under data/manifests/things_max_*.csv.",
            f"Wrote degraded images to {DEGRADED_IMAGE_DIR.relative_to(ROOT)}.",
            f"Missing concepts: {len(missing_rows)}.",
        ],
    )


if __name__ == "__main__":
    main()
