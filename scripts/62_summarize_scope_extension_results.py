from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from common import ROOT, append_run_log, ensure_parent, write_json


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def add_thingsplus(lines: list[str], payload: dict[str, Any]) -> None:
    if not payload:
        lines.extend(["## THINGSplus", "", "- Not available yet."])
        return
    lines.extend(
        [
            "## THINGSplus object-property moderators",
            "",
            f"- Concepts with joined THINGSplus moderators: {payload.get('n_joined', 'n/a')}",
            f"- Concepts with any THINGSplus moderator: {payload.get('n_with_any_thingsplus', 'n/a')}",
            f"- Moderator columns analyzed: {payload.get('moderator_count', 'n/a')}",
        ]
    )
    top = payload.get("top_absolute_correlations", [])
    if top:
        lines.append("- Strongest absolute moderator associations:")
        for item in top[:8]:
            lines.append(
                f"  - {item['moderator']} -> {item['outcome']}: {item['statistic']}={item['estimate']:+.4f}"
            )


def add_scope_dataset(lines: list[str], dataset: str, payload: dict[str, Any]) -> None:
    title = "imSitu event concepts" if dataset == "imsitu" else "MIT-States compositional concepts"
    if not payload:
        lines.extend([f"## {title}", "", "- Not available yet."])
        return
    lines.extend(
        [
            f"## {title}",
            "",
            f"- Labels: {payload.get('label_count', 'n/a')}",
            f"- Conditions: {', '.join(payload.get('conditions', []))}",
        ]
    )
    contrasts = payload.get("key_contrasts", [])
    if contrasts:
        lines.append("- Reference-space contrasts:")
        for item in contrasts:
            lines.append(
                f"  - {item['reference']}: matched-text={item['matched_minus_text_only']:+.4f}; "
                f"matched-prompt={item['matched_minus_prompt']:+.4f}; "
                f"prompt+image-matched={item['prompt_plus_image_minus_matched']:+.4f}"
            )
    mixture = payload.get("mixture") or {}
    if mixture:
        lines.append(
            f"- Prompt+image mixture: prompt={mixture['prompt_weight']:+.4f}, "
            f"matched={mixture['matched_image_weight']:+.4f}, R2={mixture['r2']:.4f}, "
            f"predictor rho={mixture['prompt_matched_spearman']:+.4f}."
        )
    mismatch = payload.get("mismatch") or {}
    if mismatch:
        lines.append(
            f"- Mismatch local identity: text-retention={mismatch['text_retention_rate']:.4f}, "
            f"source-assignment={mismatch['source_assignment_rate']:.4f}, "
            f"mean source-attraction={mismatch['mean_source_attraction']:+.4f}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize scope-extension analyses for manuscript transfer.")
    parser.parse_args()

    out_dir = ROOT / "outputs" / "scope_extensions"
    report_path = ROOT / "reports" / "main_results" / "scope_extension_report.md"
    thingsplus = read_json(out_dir / "thingsplus_moderator_summary.json")
    imsitu = read_json(out_dir / "imsitu_geometry_summary.json")
    mitstates = read_json(out_dir / "mitstates_geometry_summary.json")
    payload = {"thingsplus": thingsplus, "imsitu": imsitu, "mitstates": mitstates}
    write_json(out_dir / "scope_extension_summary.json", payload)

    lines = [
        "# Scope-extension analyses",
        "",
        "These analyses test whether the THINGS object-noun results are limited to concrete objects.",
        "",
    ]
    add_thingsplus(lines, thingsplus)
    lines.append("")
    add_scope_dataset(lines, "imsitu", imsitu)
    lines.append("")
    add_scope_dataset(lines, "mitstates", mitstates)
    ensure_parent(report_path)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    append_run_log("Scope Extension Summary", [f"Wrote {report_path.relative_to(ROOT)}."])


if __name__ == "__main__":
    main()
