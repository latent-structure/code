from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import ROOT, append_run_log, write_csv, write_json


IMSITU_ROOT = ROOT / "datasets" / "imsitu"
ANNOTATION_ROOT = IMSITU_ROOT / "imSitu"
IMAGE_ROOT = IMSITU_ROOT / "of500_images_resized"


def load_split(name: str) -> dict[str, Any]:
    return json.loads((ANNOTATION_ROOT / f"{name}.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare imSitu event/action scope-extension manifests.")
    parser.add_argument("--images-per-verb", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260501)
    args = parser.parse_args()

    space = json.loads((ANNOTATION_ROOT / "imsitu_space.json").read_text(encoding="utf-8"))
    splits = {name: load_split(name) for name in ["train", "dev", "test"]}
    images_by_verb: dict[str, list[dict[str, str]]] = defaultdict(list)
    role_counts: dict[str, Counter[str]] = defaultdict(Counter)
    noun_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for split_name, split_data in splits.items():
        for image_name, payload in split_data.items():
            verb = str(payload["verb"]).lower()
            image_path = IMAGE_ROOT / image_name
            if not image_path.exists():
                continue
            images_by_verb[verb].append({"image_path": str(image_path.relative_to(ROOT)), "split": split_name})
            for frame in payload.get("frames", []):
                for role, noun in frame.items():
                    if noun:
                        role_counts[verb][role] += 1
                        noun_counts[verb][str(noun)] += 1

    verbs = sorted(set(space["verbs"]) & set(images_by_verb))
    label_rows = []
    image_rows = []
    feature_rows = []
    for idx, verb in enumerate(verbs):
        ordered = sorted(images_by_verb[verb], key=lambda row: (row["split"], row["image_path"]))
        selected = ordered if args.images_per_verb <= 0 else ordered[: args.images_per_verb]
        if not selected:
            continue
        mismatch = verbs[(idx + max(1, len(verbs) // 2)) % len(verbs)]
        spec = space["verbs"].get(verb, {})
        label_rows.append(
            {
                "label": verb,
                "domain": "imsitu_event",
                "subtype": str(spec.get("framenet", "")),
                "definition": str(spec.get("def", "")),
                "abstract_frame": str(spec.get("abstract", "")),
                "mismatch_label": mismatch,
                "num_available_images": len(images_by_verb[verb]),
                "num_selected_images": len(selected),
            }
        )
        for image_idx, image in enumerate(selected):
            image_rows.append({"label": verb, "image_index": image_idx, **image})
        roles = role_counts[verb]
        nouns = noun_counts[verb]
        feature_rows.append(
            {
                "label": verb,
                "framenet": str(spec.get("framenet", "")),
                "num_roles": len(roles),
                "num_nouns": len(nouns),
                "total_role_mentions": sum(roles.values()),
                "top_roles": ";".join(f"{k}:{v}" for k, v in roles.most_common(12)),
                "top_nouns": ";".join(f"{k}:{v}" for k, v in nouns.most_common(24)),
            }
        )

    out_dir = ROOT / "outputs" / "scope_extensions"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "imsitu_labels.csv", label_rows, ["label", "domain", "subtype", "definition", "abstract_frame", "mismatch_label", "num_available_images", "num_selected_images"])
    write_csv(out_dir / "imsitu_images.csv", image_rows, ["label", "image_index", "image_path", "split"])
    write_csv(out_dir / "imsitu_label_features.csv", feature_rows, ["label", "framenet", "num_roles", "num_nouns", "total_role_mentions", "top_roles", "top_nouns"])
    write_json(out_dir / "imsitu_manifest_summary.json", {"num_labels": len(label_rows), "num_images": len(image_rows), "images_per_verb": args.images_per_verb})
    append_run_log("imSitu Scope Manifest", [f"Prepared {len(label_rows)} verbs and {len(image_rows)} selected images."])


if __name__ == "__main__":
    main()
