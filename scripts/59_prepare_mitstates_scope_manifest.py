from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import ROOT, append_run_log, write_csv, write_json


MIT_ROOT = ROOT / "datasets" / "mit_states" / "release_dataset"
IMAGE_ROOT = MIT_ROOT / "images"


def read_antonyms() -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    with (MIT_ROOT / "adj_ants.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            adjective = row[0].strip()
            if not adjective or adjective == "adj":
                continue
            mapping[adjective] = [item.strip() for item in row[1:] if item.strip()]
    return mapping


def split_label(name: str) -> tuple[str, str]:
    parts = name.split(" ", 1)
    if len(parts) != 2:
        return "", name
    return parts[0], parts[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MIT-States compositional scope-extension manifests.")
    parser.add_argument("--images-per-composition", type=int, default=10)
    args = parser.parse_args()

    antonyms = read_antonyms()
    dirs = sorted(path for path in IMAGE_ROOT.iterdir() if path.is_dir())
    labels = [path.name for path in dirs]
    label_set = set(labels)
    noun_baselines = {split_label(label)[1]: label for label in labels if split_label(label)[0] == "adj"}

    label_rows = []
    image_rows = []
    for idx, path in enumerate(dirs):
        label = path.name
        adjective, noun = split_label(label)
        files = sorted(item for item in path.iterdir() if item.is_file() and not item.name.startswith("."))
        selected = files if args.images_per_composition <= 0 else files[: args.images_per_composition]
        if not selected:
            continue
        antonym_label = ""
        for antonym in antonyms.get(adjective, []):
            candidate = f"{antonym} {noun}"
            if candidate in label_set:
                antonym_label = candidate
                break
        mismatch_label = labels[(idx + max(1, len(labels) // 2)) % len(labels)]
        label_rows.append(
            {
                "label": label,
                "domain": "mitstates_composition",
                "subtype": adjective,
                "attribute": adjective,
                "object": noun,
                "noun_baseline_label": noun_baselines.get(noun, ""),
                "antonym_label": antonym_label,
                "mismatch_label": mismatch_label,
                "num_available_images": len(files),
                "num_selected_images": len(selected),
            }
        )
        for image_idx, image in enumerate(selected):
            image_rows.append({"label": label, "image_index": image_idx, "image_path": str(image.relative_to(ROOT))})

    out_dir = ROOT / "outputs" / "scope_extensions"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        out_dir / "mitstates_labels.csv",
        label_rows,
        ["label", "domain", "subtype", "attribute", "object", "noun_baseline_label", "antonym_label", "mismatch_label", "num_available_images", "num_selected_images"],
    )
    write_csv(out_dir / "mitstates_images.csv", image_rows, ["label", "image_index", "image_path"])
    write_json(
        out_dir / "mitstates_manifest_summary.json",
        {
            "num_labels": len(label_rows),
            "num_images": len(image_rows),
            "images_per_composition": args.images_per_composition,
            "num_antonym_pairs_available": sum(1 for row in label_rows if row["antonym_label"]),
            "num_noun_baselines_available": sum(1 for row in label_rows if row["noun_baseline_label"]),
        },
    )
    append_run_log("MIT-States Scope Manifest", [f"Prepared {len(label_rows)} compositions and {len(image_rows)} selected images."])


if __name__ == "__main__":
    main()
