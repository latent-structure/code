from __future__ import annotations

import argparse

from common import ROOT, append_run_log, read_csv, write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.parse_args()

    full_rows = read_csv(ROOT / "data" / "concepts" / "full_concept_list.csv")
    image_status = {row["concept"]: row["status"] for row in read_csv(ROOT / "data" / "manifests" / "image_manifest.csv")}

    selected_rows = []
    skipped_sensory = []
    for row in full_rows:
        if row["domain"] == "sensory":
            if image_status.get(row["concept"]) == "ready":
                selected_rows.append(row)
            else:
                skipped_sensory.append(row["concept"])
        else:
            selected_rows.append(row)

    output_path = ROOT / "data" / "concepts" / "max_ready_concepts.csv"
    write_csv(
        output_path,
        selected_rows,
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

    sensory_count = sum(row["domain"] == "sensory" for row in selected_rows)
    abstract_count = sum(row["domain"] == "abstract" for row in selected_rows)
    append_run_log(
        "Max Ready Subset",
        [
            f"Wrote max-ready concept subset with {sensory_count} sensory concepts and {abstract_count} abstract concepts to {output_path.relative_to(ROOT)}.",
            f"Skipped {len(skipped_sensory)} sensory concepts without ready images: {', '.join(skipped_sensory)}.",
        ],
    )


if __name__ == "__main__":
    main()
