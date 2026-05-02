from __future__ import annotations

import argparse
import csv
import tarfile
import zipfile
from pathlib import Path

from common import ROOT, append_run_log, ensure_parent, output_path, write_csv, write_json


THINGS_ROOT = ROOT / "THINGS-database" / "osfstorage"
THINGS_METADATA = THINGS_ROOT / "02_object-level" / "_concepts-metadata_things.tsv"
THINGS_WORDNET_IDS = THINGS_ROOT / "02_object-level" / "ids-words_single-tables" / "wordnet-id.csv"
THINGS_IMAGES_ZIP = THINGS_ROOT / "images_THINGS.zip"
THINGSPLUS_PARTIAL_DIR = THINGS_ROOT / "02_object-level" / "trial-wise_tables"
THINGSPLUS_IMAGES_ZIP = THINGS_ROOT / "images_THINGSplus-CC0.zip"
THINGSPLUS_IMAGES_DIR = THINGS_ROOT / "object_images_CC0"
THINGS_BEHAVIOR_ROOT = ROOT / "THINGS-behavior" / "osfstorage"
THINGS_BEHAVIOR_MATRIX = THINGS_BEHAVIOR_ROOT / "data" / "spose_similarity.mat"
THINGS_BEHAVIOR_ORDER = THINGS_BEHAVIOR_ROOT / "variables" / "unique_id.txt"
SIMLEX_ZIP = ROOT / "SimLex-999.zip"
WORDNET_TAR = ROOT / "WordNet-3.0.tar.gz"
LANCASTER_DIR = ROOT / "data" / "anchors" / "lancaster"


def copy_text_file(src: Path, dst: Path) -> None:
    ensure_parent(dst)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def validate_things() -> dict[str, object]:
    summary: dict[str, object] = {
        "metadata_tsv": str(THINGS_METADATA.relative_to(ROOT)),
        "images_zip": str(THINGS_IMAGES_ZIP.relative_to(ROOT)),
        "wordnet_id_csv": str(THINGS_WORDNET_IDS.relative_to(ROOT)),
        "present": False,
        "concept_rows": 0,
        "image_members": 0,
    }
    if not (THINGS_METADATA.exists() and THINGS_IMAGES_ZIP.exists()):
        return summary

    with THINGS_METADATA.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    with zipfile.ZipFile(THINGS_IMAGES_ZIP) as archive:
        image_members = [
            name
            for name in archive.namelist()
            if name.startswith("object_images/") and not name.endswith("/")
        ]

    normalized_dir = ROOT / "data" / "concepts" / "things"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    copy_text_file(THINGS_METADATA, normalized_dir / "concepts-metadata_things.tsv")
    if THINGS_WORDNET_IDS.exists():
        copy_text_file(THINGS_WORDNET_IDS, normalized_dir / "wordnet-id.csv")

    summary["present"] = True
    summary["concept_rows"] = len(rows)
    summary["image_members"] = len(image_members)
    summary["normalized_dir"] = str(normalized_dir.relative_to(ROOT))
    return summary


def validate_simlex() -> dict[str, object]:
    summary: dict[str, object] = {
        "archive": str(SIMLEX_ZIP.relative_to(ROOT)),
        "present": False,
        "normalized_file": "",
    }
    if not SIMLEX_ZIP.exists():
        return summary
    out_dir = ROOT / "data" / "anchors" / "simlex999"
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SIMLEX_ZIP) as archive:
        target = "SimLex-999/SimLex-999.txt"
        readme = "SimLex-999/README.txt"
        with archive.open(target) as handle:
            (out_dir / "SimLex-999.txt").write_bytes(handle.read())
        with archive.open(readme) as handle:
            (out_dir / "README.txt").write_bytes(handle.read())
    summary["present"] = True
    summary["normalized_file"] = str((out_dir / "SimLex-999.txt").relative_to(ROOT))
    return summary


def validate_wordnet() -> dict[str, object]:
    summary: dict[str, object] = {
        "archive": str(WORDNET_TAR.relative_to(ROOT)),
        "present": False,
        "normalized_dir": "",
    }
    if not WORDNET_TAR.exists():
        return summary

    out_dir = ROOT / "data" / "lexical" / "wordnet"
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {
        "WordNet-3.0/LICENSE": out_dir / "LICENSE",
        "WordNet-3.0/dict/index.noun": out_dir / "index.noun",
        "WordNet-3.0/dict/index.verb": out_dir / "index.verb",
        "WordNet-3.0/dict/index.adj": out_dir / "index.adj",
        "WordNet-3.0/dict/index.adv": out_dir / "index.adv",
        "WordNet-3.0/dict/data.noun": out_dir / "data.noun",
        "WordNet-3.0/dict/data.verb": out_dir / "data.verb",
        "WordNet-3.0/dict/data.adj": out_dir / "data.adj",
        "WordNet-3.0/dict/data.adv": out_dir / "data.adv",
        "WordNet-3.0/dict/index.sense": out_dir / "index.sense",
        "WordNet-3.0/dict/lexnames": out_dir / "lexnames",
    }
    with tarfile.open(WORDNET_TAR, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
        for member_name, destination in wanted.items():
            if member_name not in members:
                continue
            extracted = archive.extractfile(members[member_name])
            if extracted is None:
                continue
            ensure_parent(destination)
            destination.write_bytes(extracted.read())
    summary["present"] = True
    summary["normalized_dir"] = str(out_dir.relative_to(ROOT))
    return summary


def validate_thingsplus() -> dict[str, object]:
    summary: dict[str, object] = {
        "trialwise_dir": str(THINGSPLUS_PARTIAL_DIR.relative_to(ROOT)),
        "images_zip": str(THINGSPLUS_IMAGES_ZIP.relative_to(ROOT)),
        "images_dir": str(THINGSPLUS_IMAGES_DIR.relative_to(ROOT)),
        "present": False,
        "trialwise_tables": 0,
        "image_dirs": 0,
    }
    trialwise_tables = sorted(THINGSPLUS_PARTIAL_DIR.glob("*thingsplus*.tsv")) if THINGSPLUS_PARTIAL_DIR.exists() else []
    image_dirs = [path for path in THINGSPLUS_IMAGES_DIR.iterdir() if path.is_dir()] if THINGSPLUS_IMAGES_DIR.exists() else []
    summary["trialwise_tables"] = len(trialwise_tables)
    summary["image_dirs"] = len(image_dirs)
    summary["present"] = bool(trialwise_tables or THINGSPLUS_IMAGES_ZIP.exists() or THINGSPLUS_IMAGES_DIR.exists())
    return summary


def validate_things_behavior() -> dict[str, object]:
    summary: dict[str, object] = {
        "matrix": str(THINGS_BEHAVIOR_MATRIX.relative_to(ROOT)),
        "ordering": str(THINGS_BEHAVIOR_ORDER.relative_to(ROOT)),
        "present": False,
        "ordering_count": 0,
    }
    if not (THINGS_BEHAVIOR_MATRIX.exists() and THINGS_BEHAVIOR_ORDER.exists()):
        return summary
    ordering = [line.strip() for line in THINGS_BEHAVIOR_ORDER.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary["present"] = True
    summary["ordering_count"] = len(ordering)
    return summary


def validate_lancaster() -> dict[str, object]:
    summary: dict[str, object] = {
        "directory": str(LANCASTER_DIR.relative_to(ROOT)),
        "present": False,
        "files": [],
    }
    if not LANCASTER_DIR.exists():
        return summary
    files = sorted(str(path.relative_to(ROOT)) for path in LANCASTER_DIR.iterdir() if path.is_file())
    summary["present"] = bool(files)
    summary["files"] = files
    return summary


def write_resource_manifest(summary: dict[str, dict[str, object]]) -> None:
    rows = [
        {
            "resource_id": "things_concepts",
            "resource_type": "dataset",
            "source_name": "THINGS",
            "license": "see upstream dataset terms",
            "source_url": "https://osf.io/jum2f/",
            "local_path": summary["things"]["metadata_tsv"] if summary["things"]["present"] else "",
            "status": "available" if summary["things"]["present"] else "missing",
            "notes": f"images_zip={summary['things']['images_zip']}; normalized_dir={summary['things'].get('normalized_dir', '')}",
        },
        {
            "resource_id": "thingsplus_metadata",
            "resource_type": "dataset",
            "source_name": "THINGSplus",
            "license": "see upstream dataset terms",
            "source_url": "https://osf.io/3ku9e/",
            "local_path": str(THINGSPLUS_IMAGES_DIR.relative_to(ROOT)) if THINGSPLUS_IMAGES_DIR.exists() else str(THINGSPLUS_PARTIAL_DIR.relative_to(ROOT)) if THINGSPLUS_PARTIAL_DIR.exists() else "",
            "status": "available" if summary["thingsplus"]["present"] else "missing_for_now",
            "notes": f"trialwise_tables={summary['thingsplus']['trialwise_tables']}; image_dirs={summary['thingsplus']['image_dirs']}; images_zip={summary['thingsplus']['images_zip']}",
        },
        {
            "resource_id": "things_behavioral_anchor",
            "resource_type": "dataset",
            "source_name": "THINGS-data behavioral similarity",
            "license": "see upstream dataset terms",
            "source_url": "https://things-initiative.org/",
            "local_path": summary["things_behavior"]["matrix"] if summary["things_behavior"]["present"] else "",
            "status": "available" if summary["things_behavior"]["present"] else "missing_for_now",
            "notes": f"ordering={summary['things_behavior']['ordering']}; ordering_count={summary['things_behavior']['ordering_count']}",
        },
        {
            "resource_id": "simlex999",
            "resource_type": "dataset",
            "source_name": "SimLex-999",
            "license": "research dataset",
            "source_url": "https://fh295.github.io/simlex.html",
            "local_path": summary["simlex"]["normalized_file"] if summary["simlex"]["present"] else "",
            "status": "available" if summary["simlex"]["present"] else "missing",
            "notes": "Normalized from local SimLex-999.zip archive.",
        },
        {
            "resource_id": "wordnet",
            "resource_type": "lexical",
            "source_name": "WordNet",
            "license": "Princeton WordNet license",
            "source_url": "https://wordnet.princeton.edu/",
            "local_path": summary["wordnet"]["normalized_dir"] if summary["wordnet"]["present"] else "",
            "status": "available" if summary["wordnet"]["present"] else "missing",
            "notes": "Normalized from local WordNet-3.0.tar.gz archive.",
        },
        {
            "resource_id": "lancaster_sensorimotor",
            "resource_type": "dataset",
            "source_name": "Lancaster Sensorimotor Norms",
            "license": "see upstream dataset terms",
            "source_url": "",
            "local_path": summary["lancaster"]["directory"] if summary["lancaster"]["present"] else "",
            "status": "available" if summary["lancaster"]["present"] else "missing_for_now",
            "notes": f"files={';'.join(summary['lancaster']['files']) if summary['lancaster']['files'] else 'none'}",
        },
        {
            "resource_id": "local_seed_images",
            "resource_type": "image_inventory",
            "source_name": "local seed images",
            "license": "inherited from pilot local assets",
            "source_url": "",
            "local_path": "data/images/sensory",
            "status": "partial",
            "notes": "Existing root image directory contains pilot-seeded files; THINGS-covered concepts will be normalized separately.",
        },
    ]
    write_csv(
        ROOT / "data" / "manifests" / "resource_manifest.csv",
        rows,
        ["resource_id", "resource_type", "source_name", "license", "source_url", "local_path", "status", "notes"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.parse_args()

    summary = {
        "things": validate_things(),
        "thingsplus": validate_thingsplus(),
        "things_behavior": validate_things_behavior(),
        "simlex": validate_simlex(),
        "wordnet": validate_wordnet(),
        "lancaster": validate_lancaster(),
    }
    write_json(output_path("outputs", "logs", "external_resource_validation.json"), summary)
    write_resource_manifest(summary)
    append_run_log(
        "External Resources",
        [
            f"Validated THINGS present={summary['things']['present']} concept_rows={summary['things']['concept_rows']} image_members={summary['things']['image_members']}.",
            f"Validated THINGSplus present={summary['thingsplus']['present']} trialwise_tables={summary['thingsplus']['trialwise_tables']} image_dirs={summary['thingsplus']['image_dirs']}.",
            f"Validated THINGS-behavior present={summary['things_behavior']['present']} ordering_count={summary['things_behavior']['ordering_count']}.",
            f"Validated SimLex-999 present={summary['simlex']['present']} normalized_file={summary['simlex']['normalized_file'] or 'missing'}.",
            f"Validated WordNet present={summary['wordnet']['present']} normalized_dir={summary['wordnet']['normalized_dir'] or 'missing'}.",
            f"Validated Lancaster present={summary['lancaster']['present']} files={len(summary['lancaster']['files'])}.",
            "Updated data/manifests/resource_manifest.csv and outputs/logs/external_resource_validation.json.",
        ],
    )


if __name__ == "__main__":
    main()
