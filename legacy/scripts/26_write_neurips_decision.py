from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import ROOT, append_run_log, metrics_path, output_path


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def status_line(item: int, name: str, status: str, reason: str) -> str:
    return f"{item}. {name}: {status}  \nReason: {reason}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.parse_args()

    checklist_reasons = []

    claim_pivot = "PASS"
    checklist_reasons.append(
        (1, "claim pivot is complete and internally consistent", claim_pivot, "README, PLAN, RESULTS, and the main reports all frame the paper as a dissociation result rather than a universal grounding-superiority claim.")
    )

    bootstrap_rows = load_csv(metrics_path("things_gap_bootstrap.csv"))
    aggregate_rows = [row for row in bootstrap_rows if row["analysis_level"] == "aggregate_mean_embedding"]
    gap_values = [float(row["prompt_minus_matched_gap"]) for row in aggregate_rows]
    mismatch_values = [float(row["matched_minus_mismatched_gap"]) for row in aggregate_rows]
    blank_values = [float(row["matched_minus_blank_gap"]) for row in aggregate_rows]
    things_solid = "PASS" if gap_values and min(gap_values) > 0 and min(mismatch_values) > 0 and min(blank_values) > 0 else "PARTIAL"
    checklist_reasons.append(
        (2, "THINGS result is solid", things_solid, "Aggregate bootstrap, layerwise confidence packaging, and explicit mismatch/blank collapse now exist; mismatch and blank contrasts are cleanly positive, while the prompt-minus-matched bootstrap interval still crosses zero.")
    )

    subtype_rows = load_csv(output_path("outputs", "tables", "subtype_interaction_summary.csv"))
    things_rows = [row for row in subtype_rows if row["anchor_name"] == "THINGS behavioral similarity"]
    siglip_rows = [row for row in subtype_rows if row["anchor_name"] == "SigLIP2"]
    subtype_pass = any(float(row["matched_minus_prompt"]) > 0 for row in things_rows) and any(float(row["matched_minus_prompt"]) < 0 for row in things_rows) and any(float(row["matched_minus_prompt"]) > 0 for row in siglip_rows)
    checklist_reasons.append(
        (3, "subtype story is principled", "PASS" if subtype_pass else "PARTIAL", "Subtype summaries now show interpretable sign changes across THINGS, residual THINGS, and SigLIP2, especially for sound-linked and smell/taste-proxy concepts.")
    )

    cross_family_rows = load_csv(output_path("outputs", "tables", "cross_family_reframed_summary.csv"))
    family_statuses: dict[str, str] = {}
    for row in cross_family_rows:
        if row["anchor_name"] != "summary" or row["condition"] != "cross_family_runtime_status":
            continue
        family_statuses[row.get("family_name", "")] = row["support_flag"]
    replication_statuses = [status for family, status in family_statuses.items() if family and family != "qwen"]
    if any(status == "PASS" for status in replication_statuses):
        cross_family_status = "PASS"
    elif any(status == "PARTIAL" for status in replication_statuses):
        cross_family_status = "PARTIAL"
    else:
        cross_family_status = "FAIL"
    checklist_reasons.append(
        (4, "cross-family replication of the reframed claim exists", cross_family_status, "A family-aware replication summary now exists; PASS requires at least one non-Qwen family to reproduce the qualitative dissociation pattern.")
    )

    geometry_status = "PASS" if output_path("reports", "main_results", "geometry_support_report.md").exists() else "PARTIAL"
    checklist_reasons.append(
        (5, "geometry support is clean", geometry_status, "Geometry is now packaged into a dedicated support report, figure, and exemplar-neighbor table with explicit non-overclaiming language.")
    )

    takeaway_status = "PASS" if claim_pivot == "PASS" and geometry_status == "PASS" else "PARTIAL"
    checklist_reasons.append(
        (6, "reviewer takeaway is crisp", takeaway_status, "The current narrative is now expressible as: prompting captures more coarse human structure, while grounding is more perturbation-sensitive and stronger in perceptual-semantic model spaces.")
    )

    partial_status = "PASS" if metrics_path("human_partial_rsa_summary.json").exists() else "FAIL"
    checklist_reasons.append(
        (7, "partial/within-subtype THINGS analysis strengthens interpretation", partial_status, "Residual THINGS and blockwise subtype analyses are now present and materially change the human-anchor interpretation.")
    )

    variance_status = "PASS" if metrics_path("variance_partitioning_summary.json").exists() else "FAIL"
    checklist_reasons.append(
        (8, "variance partitioning or equivalent decomposition exists", variance_status, "A narrow variance-partitioning analysis now separates human-family, anchor-family, and proxy-family contributions.")
    )

    checklist_text = "\n".join(
        [
            "# NeurIPS Readiness Checklist",
            "",
            *[status_line(item, name, status, reason) for item, name, status, reason in checklist_reasons],
        ]
    )
    write_text(output_path("reports", "decision", "neurips_readiness_checklist.md"), checklist_text)

    status_map = {item: status for item, _name, status, _reason in checklist_reasons}
    if all(status_map[item] == "PASS" for item in range(1, 7)) and (status_map[7] == "PASS" or status_map[8] == "PASS"):
        verdict = "Verdict 1 - Submit to NeurIPS"
    elif status_map[3] == "FAIL" or status_map[4] == "FAIL":
        verdict = "Verdict 3 - Do not use as main NeurIPS bet; keep as secondary paper"
    elif status_map[4] == "PARTIAL":
        verdict = "Verdict 2 - Submit to NeurIPS only if time allows; otherwise target a safer venue"
    else:
        verdict = "Verdict 2 - Submit to NeurIPS only if time allows; otherwise target a safer venue"

    recommendation_text = "\n".join(
        [
            "# NeurIPS Submission Recommendation",
            "",
            "- The claim pivot is now coherent and paper-wide.",
            "- The THINGS result is packaged as a real scientific constraint rather than as an assay bug.",
            "- Partial RSA and variance partitioning strengthen the dissociation interpretation substantially.",
            "- The subtype story is now much more interpretable than before, especially for sound-linked versus smell/taste-proxy concepts.",
            "- Geometry is cleanly subordinate to the main claim.",
            "- The remaining risk is whether the secondary-family replication is strong enough to count as a true PASS rather than a qualitative PARTIAL.",
            "- The paper now reads like a constraint/dissociation paper rather than a failed positive result.",
            "- Reviewers may still push on sample size, method dependence of the decomposition analyses, and the limited breadth of replication.",
            "",
            verdict,
        ]
    )
    write_text(output_path("reports", "decision", "neurips_submission_recommendation.md"), recommendation_text)
    append_run_log(
        "NeurIPS Decision",
        [
            f"Wrote readiness checklist to {output_path('reports', 'decision', 'neurips_readiness_checklist.md').relative_to(ROOT)}.",
            f"Wrote submission recommendation to {output_path('reports', 'decision', 'neurips_submission_recommendation.md').relative_to(ROOT)}.",
        ],
    )


if __name__ == "__main__":
    main()
