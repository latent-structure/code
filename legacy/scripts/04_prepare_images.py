from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageFilter

from common import ROOT, append_run_log, load_project_config, read_csv, set_global_seed, write_csv


def find_image(path: Path, concept: str, extensions: list[str]) -> Path | None:
    for ext in extensions:
        candidate = path / f"{concept}{ext}"
        if candidate.exists():
            return candidate
    return None


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
    rng = random.Random(seed)
    sensory = [row for row in rows if row["domain"] == "sensory"]
    by_subtype: dict[str, list[dict[str, str]]] = {}
    for row in sensory:
        by_subtype.setdefault(row["subtype"], []).append(row)
    subtypes = sorted(by_subtype)
    results: list[dict[str, str]] = []
    used_pairs: set[tuple[str, str]] = set()

    for index, row in enumerate(sorted(sensory, key=lambda item: item["concept"])):
        same_subtype = index % 2 == 0
        if same_subtype:
            pool = [item for item in by_subtype[row["subtype"]] if item["concept"] != row["concept"]]
        else:
            pool = [item for subtype in subtypes if subtype != row["subtype"] for item in by_subtype[subtype]]
        pool = [item for item in pool if (row["concept"], item["concept"]) not in used_pairs]
        if not pool:
            pool = [item for item in sensory if item["concept"] != row["concept"]]
        pool.sort(key=lambda item: item["concept"])
        choice = pool[rng.randrange(len(pool))]
        used_pairs.add((row["concept"], choice["concept"]))
        results.append(
            {
                "concept": row["concept"],
                "concept_subtype": row["subtype"],
                "mismatch_concept": choice["concept"],
                "mismatch_subtype": choice["subtype"],
                "mismatch_mode": "within_subtype" if same_subtype else "cross_subtype",
            }
        )
    return results


def main() -> None:
    config = load_project_config()
    set_global_seed(config["seeds"]["mismatch_map"])
    analysis = config["analysis"]
    rows = read_csv(ROOT / "data/concepts/full_concept_list.csv")

    image_dir = ROOT / analysis["image_policy"]["matched_dir"].replace("./", "")
    degraded_dir = ROOT / analysis["image_policy"]["degraded_dir"].replace("./", "")
    manifest_path = ROOT / "data/manifests/image_manifest.csv"
    mismatch_path = ROOT / "data/manifests/mismatch_map.csv"
    resource_path = ROOT / "data/manifests/resource_manifest.csv"
    allowed_extensions = analysis["image_policy"]["allowed_extensions"]

    manifest_rows: list[dict[str, str]] = []
    for row in rows:
        if row["domain"] != "sensory":
            continue
        matched = find_image(image_dir, row["concept"], allowed_extensions)
        degraded = degraded_dir / f"{row['concept']}.png"
        status = "missing"
        matched_rel = ""
        degraded_rel = ""
        notes = "no local matched image yet"
        if matched is not None:
            matched_rel = str(matched.relative_to(ROOT))
            degrade_image(
                matched,
                degraded,
                analysis["image_policy"]["downsample_linear_fraction"],
                analysis["image_policy"]["grayscale"],
                analysis["image_policy"]["gaussian_blur_radius"],
            )
            degraded_rel = str(degraded.relative_to(ROOT))
            status = "ready"
            notes = "matched image and degraded derivative available"
        manifest_rows.append(
            {
                "concept": row["concept"],
                "subtype": row["subtype"],
                "matched_image": matched_rel,
                "degraded_image": degraded_rel,
                "status": status,
                "source_kind": row["image_source"],
                "notes": notes,
            }
        )

    mismatch_rows = build_mismatch_map(rows, config["seeds"]["mismatch_map"])
    existing_resource_rows = read_csv(resource_path) if resource_path.exists() else []
    resource_rows = existing_resource_rows or [
        {
            "resource_id": "things_concepts",
            "resource_type": "dataset",
            "source_name": "THINGS",
            "license": "see upstream dataset terms",
            "source_url": "https://osf.io/jum2f/",
            "local_path": "",
            "status": "planned",
            "notes": "sensory concept inventory source",
        },
        {
            "resource_id": "thingsplus_metadata",
            "resource_type": "dataset",
            "source_name": "THINGSplus",
            "license": "see upstream dataset terms",
            "source_url": "https://osf.io/3ku9e/",
            "local_path": "",
            "status": "planned",
            "notes": "sensory metadata and norms source",
        },
        {
            "resource_id": "things_behavioral_anchor",
            "resource_type": "dataset",
            "source_name": "THINGS-data behavioral similarity",
            "license": "see upstream dataset terms",
            "source_url": "https://things-initiative.org/",
            "local_path": "",
            "status": "planned",
            "notes": "primary sensory human anchor",
        },
        {
            "resource_id": "simlex999",
            "resource_type": "dataset",
            "source_name": "SimLex-999",
            "license": "research dataset",
            "source_url": "https://fh295.github.io/simlex.html",
            "local_path": "",
            "status": "planned",
            "notes": "abstract control anchor subset",
        },
        {
            "resource_id": "wordnet",
            "resource_type": "lexical",
            "source_name": "WordNet",
            "license": "Princeton WordNet license",
            "source_url": "https://wordnet.princeton.edu/",
            "local_path": "",
            "status": "planned",
            "notes": "sense normalization and ambiguity screening",
        },
        {
            "resource_id": "local_seed_images",
            "resource_type": "image_inventory",
            "source_name": "local seed images",
            "license": "inherited from pilot local assets",
            "source_url": "",
            "local_path": "data/images/sensory",
            "status": "partial",
            "notes": "seeded local sensory images copied from the pilot where available",
        },
    ]

    write_csv(
        manifest_path,
        manifest_rows,
        ["concept", "subtype", "matched_image", "degraded_image", "status", "source_kind", "notes"],
    )
    write_csv(
        mismatch_path,
        mismatch_rows,
        ["concept", "concept_subtype", "mismatch_concept", "mismatch_subtype", "mismatch_mode"],
    )
    write_csv(
        resource_path,
        resource_rows,
        ["resource_id", "resource_type", "source_name", "license", "source_url", "local_path", "status", "notes"],
    )

    ready_count = sum(row["status"] == "ready" for row in manifest_rows)
    append_run_log(
        "Image Preparation",
        [
            f"Wrote image manifest to {manifest_path.relative_to(ROOT)}.",
            f"Wrote mismatch map to {mismatch_path.relative_to(ROOT)}.",
            f"Wrote resource manifest to {resource_path.relative_to(ROOT)}.",
            f"Prepared degraded images for {ready_count} sensory concepts.",
        ],
    )


if __name__ == "__main__":
    main()
