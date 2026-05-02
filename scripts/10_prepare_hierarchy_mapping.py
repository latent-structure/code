from __future__ import annotations

import argparse

from common import ROOT, append_run_log, load_project_config, write_csv
from analysis_common import HIERARCHY_MAPPING_PATH, build_hierarchy_mapping_rows, active_concepts_for_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    args = parser.parse_args()

    load_project_config(args.config)
    concept_rows = active_concepts_for_config(args.config)
    rows = build_hierarchy_mapping_rows(concept_rows)
    write_csv(
        HIERARCHY_MAPPING_PATH,
        rows,
        ["concept", "subtype", "coarse_category", "hierarchy_level_available"],
    )
    append_run_log(
        "Hierarchy Mapping",
        [
            f"Wrote hierarchy mapping to {HIERARCHY_MAPPING_PATH.relative_to(ROOT)}.",
            f"Rows written: {len(rows)}.",
        ],
    )


if __name__ == "__main__":
    main()
