from __future__ import annotations

import argparse
import json

from common import ROOT, append_run_log, output_path, read_csv, write_csv
from hardening_common import LANCASTER_SPACES, lancaster_matrix_for_concepts, normalize_word


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    args = parser.parse_args()

    from common import load_project_config

    config = load_project_config(args.config)
    subset_path = config["analysis"].get("execution", {}).get("default_concept_subset", "")
    subset_file = (ROOT / subset_path) if subset_path else (ROOT / "data" / "concepts" / "full_concept_list.csv")
    sensory_rows = [row for row in read_csv(subset_file) if row["domain"] == "sensory"]
    concepts = [row["concept"] for row in sensory_rows]
    lookup = {}
    for path in [ROOT / "Lancaster_sensorimotor.csv"]:
        for row in read_csv(path):
            lookup[normalize_word(row["Word"])] = row

    mapping_rows = []
    resolved_concepts = []
    unresolved_concepts = []
    for concept in concepts:
        key = normalize_word(concept)
        if key in lookup:
            mapping_rows.append(
                {
                    "concept": concept,
                    "lancaster_word": lookup[key]["Word"],
                    "mapping_method": "exact_normalized",
                    "resolved": "True",
                }
            )
            resolved_concepts.append(concept)
        else:
            mapping_rows.append(
                {
                    "concept": concept,
                    "lancaster_word": "",
                    "mapping_method": "unresolved",
                    "resolved": "False",
                }
            )
            unresolved_concepts.append(concept)

    for space_name, dimensions in LANCASTER_SPACES.items():
        matrix = lancaster_matrix_for_concepts(resolved_concepts, dimensions)
        npy_path = output_path("data", "anchors", f"{space_name}_matrix.npy")
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        import numpy as np

        np.save(npy_path, matrix.astype(np.float32))
        output_path("data", "anchors", f"{space_name}_concepts.json").write_text(json.dumps(resolved_concepts, indent=2), encoding="utf-8")

    write_csv(
        output_path("data", "manifests", "lancaster_mapping_manifest.csv"),
        mapping_rows,
        ["concept", "lancaster_word", "mapping_method", "resolved"],
    )
    output_path("outputs", "metrics", "lancaster_mapping_summary.json").write_text(
        json.dumps(
            {
                "resolved_concepts": resolved_concepts,
                "unresolved_concepts": unresolved_concepts,
                "spaces": {space: dimensions for space, dimensions in LANCASTER_SPACES.items()},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    append_run_log(
        "Lancaster Anchor",
        [
            f"Wrote Lancaster mapping manifest to {output_path('data', 'manifests', 'lancaster_mapping_manifest.csv').relative_to(ROOT)}.",
            f"Resolved {len(resolved_concepts)} sensory concepts against Lancaster.",
        ],
    )


if __name__ == "__main__":
    main()
