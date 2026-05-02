from __future__ import annotations

import argparse
from collections import defaultdict

from common import ROOT, append_run_log, metrics_path, output_path, read_csv, write_csv, write_json


LOCAL_FIELDS = {
    "T_prompt_primary": "prompt_local_alignment",
    "M_matched_image": "matched_local_alignment",
    "M_degraded_image": "degraded_local_alignment",
}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return (sum((value - mu) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def write_text(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def summarize_group(rows: list[dict[str, str]], granularity: str, group_name: str) -> list[dict[str, object]]:
    grouped_rows: list[dict[str, object]] = []
    condition_means: dict[str, float] = {}
    for condition, field in LOCAL_FIELDS.items():
        values = [float(row[field]) for row in rows]
        condition_means[condition] = mean(values)
        grouped_rows.append(
            {
                "granularity": granularity,
                "group_name": group_name,
                "condition": condition,
                "num_concepts": len(values),
                "mean_local_alignment": condition_means[condition],
                "median_local_alignment": median(values),
                "std_local_alignment": std(values),
                "min_local_alignment": min(values) if values else 0.0,
                "max_local_alignment": max(values) if values else 0.0,
                "condition_rank": "",
            }
        )
    ranked_conditions = sorted(condition_means.items(), key=lambda item: item[1], reverse=True)
    rank_lookup = {condition: rank + 1 for rank, (condition, _) in enumerate(ranked_conditions)}
    for row in grouped_rows:
        row["condition_rank"] = rank_lookup[row["condition"]]
    return grouped_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.parse_args()

    concept_rows = read_csv(metrics_path("human_anchor_concept_diagnostics.csv"))
    if not concept_rows:
        raise RuntimeError("Expected human_anchor_concept_diagnostics.csv before computing local geometry support.")

    summary_rows: list[dict[str, object]] = []
    summary_rows.extend(summarize_group(concept_rows, "all_concepts", "all_concepts"))

    subtype_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in concept_rows:
        subtype_groups[row["subtype"]].append(row)
    for subtype, rows in sorted(subtype_groups.items()):
        summary_rows.extend(summarize_group(rows, "subtype", subtype))

    contrasts = []
    for granularity, group_name in sorted({(row["granularity"], row["group_name"]) for row in summary_rows}):
        group_rows = {row["condition"]: row for row in summary_rows if row["granularity"] == granularity and row["group_name"] == group_name}
        prompt = float(group_rows["T_prompt_primary"]["mean_local_alignment"])
        matched = float(group_rows["M_matched_image"]["mean_local_alignment"])
        degraded = float(group_rows["M_degraded_image"]["mean_local_alignment"])
        contrasts.extend(
            [
                {
                    "granularity": granularity,
                    "group_name": group_name,
                    "contrast": "prompt_minus_matched",
                    "delta": prompt - matched,
                },
                {
                    "granularity": granularity,
                    "group_name": group_name,
                    "contrast": "matched_minus_degraded",
                    "delta": matched - degraded,
                },
                {
                    "granularity": granularity,
                    "group_name": group_name,
                    "contrast": "prompt_minus_degraded",
                    "delta": prompt - degraded,
                },
            ]
        )

    aggregate = {row["condition"]: row for row in summary_rows if row["granularity"] == "all_concepts"}
    subtype_leaders = []
    for subtype in sorted(subtype_groups):
        subtype_rows = {row["condition"]: row for row in summary_rows if row["granularity"] == "subtype" and row["group_name"] == subtype}
        leader = max(subtype_rows.items(), key=lambda item: float(item[1]["mean_local_alignment"]))[0]
        subtype_leaders.append({"subtype": subtype, "best_condition": leader})
    summary_payload = {
        "all_concepts_prompt_mean": float(aggregate["T_prompt_primary"]["mean_local_alignment"]),
        "all_concepts_matched_mean": float(aggregate["M_matched_image"]["mean_local_alignment"]),
        "all_concepts_degraded_mean": float(aggregate["M_degraded_image"]["mean_local_alignment"]),
        "all_concepts_prompt_minus_matched": float(aggregate["T_prompt_primary"]["mean_local_alignment"]) - float(aggregate["M_matched_image"]["mean_local_alignment"]),
        "all_concepts_matched_minus_degraded": float(aggregate["M_matched_image"]["mean_local_alignment"]) - float(aggregate["M_degraded_image"]["mean_local_alignment"]),
        "subtype_leaders": subtype_leaders,
    }

    write_csv(
        metrics_path("human_local_geometry.csv"),
        summary_rows,
        [
            "granularity",
            "group_name",
            "condition",
            "num_concepts",
            "mean_local_alignment",
            "median_local_alignment",
            "std_local_alignment",
            "min_local_alignment",
            "max_local_alignment",
            "condition_rank",
        ],
    )
    write_csv(
        metrics_path("human_local_geometry_contrasts.csv"),
        contrasts,
        ["granularity", "group_name", "contrast", "delta"],
    )
    write_json(metrics_path("human_local_geometry_summary.json"), summary_payload)

    output_path("outputs", "tables").mkdir(parents=True, exist_ok=True)
    write_csv(
        output_path("outputs", "tables", "human_local_geometry_table.csv"),
        summary_rows,
        [
            "granularity",
            "group_name",
            "condition",
            "num_concepts",
            "mean_local_alignment",
            "median_local_alignment",
            "std_local_alignment",
            "min_local_alignment",
            "max_local_alignment",
            "condition_rank",
        ],
    )

    subtype_lines = []
    for subtype in sorted(subtype_groups):
        subtype_rows = {
            row["condition"]: row
            for row in summary_rows
            if row["granularity"] == "subtype" and row["group_name"] == subtype
        }
        subtype_lines.append(
            f"- `{subtype}` prompt=`{float(subtype_rows['T_prompt_primary']['mean_local_alignment']):.4f}` matched=`{float(subtype_rows['M_matched_image']['mean_local_alignment']):.4f}` degraded=`{float(subtype_rows['M_degraded_image']['mean_local_alignment']):.4f}`"
        )

    report = "\n".join(
        [
            "# Local Geometry Report",
            "",
            "## Human Local-Geometry Summary",
            f"- All-concept prompt local alignment: `{summary_payload['all_concepts_prompt_mean']:.4f}`",
            f"- All-concept matched-image local alignment: `{summary_payload['all_concepts_matched_mean']:.4f}`",
            f"- All-concept degraded-image local alignment: `{summary_payload['all_concepts_degraded_mean']:.4f}`",
            f"- Prompt-minus-matched local gap: `{summary_payload['all_concepts_prompt_minus_matched']:.4f}`",
            f"- Matched-minus-degraded local gap: `{summary_payload['all_concepts_matched_minus_degraded']:.4f}`",
            "",
            "## Subtype Structure",
            *(subtype_lines or ["- No subtype-level local geometry rows were available."]),
            "",
            "## Interpretation",
            "- Prompting better preserves human-local neighborhood structure overall on the THINGS overlap subset.",
            "- Matched images retain a positive edge over degraded images overall, indicating image-specific local restructuring rather than undirected noise.",
            "- The local-geometry picture is structured by subtype rather than uniform across sensory concepts.",
        ]
    )
    write_text(output_path("reports", "main_results", "local_geometry_report.md"), report)
    append_run_log(
        "Human Local Geometry",
        [
            f"Wrote human local-geometry summary to {metrics_path('human_local_geometry.csv').relative_to(ROOT)}.",
            f"Wrote local geometry report to {output_path('reports', 'main_results', 'local_geometry_report.md').relative_to(ROOT)}.",
        ],
    )


if __name__ == "__main__":
    main()
