from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

from common import ROOT, append_run_log, read_csv, write_csv, write_json
from hardening_common import LANCASTER_SENSORIMOTOR, load_lancaster_lookup, normalize_word


SIMLEX_PATH = ROOT / "data" / "anchors" / "simlex999" / "SimLex-999.txt"
THINGS_PATH = ROOT / "data" / "concepts" / "things_max_1854_concepts.csv"


def clean_word(word: str) -> str:
    return re.sub(r"\s+", " ", word.strip().lower().replace("_", " "))


def clean_candidate(word: str) -> bool:
    return bool(re.fullmatch(r"[a-z][a-z ]{2,30}", word))


def load_things_rows() -> list[dict[str, str]]:
    return read_csv(THINGS_PATH)


def load_simlex_min_concreteness() -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], float] = {}
    with SIMLEX_PATH.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            pos = row["POS"].strip()
            for side in ("1", "2"):
                word = clean_word(row[f"word{side}"])
                if not clean_candidate(word):
                    continue
                conc = float(row[f"conc(w{side})"])
                key = (word, pos)
                values[key] = min(values.get(key, conc), conc)
    return values


def build_abstract_rows(threshold: float) -> list[dict[str, Any]]:
    things = {clean_word(row["concept"]) for row in load_things_rows()}
    lancaster = load_lancaster_lookup()
    simlex = load_simlex_min_concreteness()
    rows = []
    for (word, pos), concreteness in sorted(simlex.items(), key=lambda item: (item[1], item[0][0])):
        if pos != "N" or concreteness > threshold or word in things or normalize_word(word) not in lancaster:
            continue
        rows.append(
            {
                "concept": word,
                "domain": "abstract",
                "subtype": "abstract_noun",
                "simlex_pos": pos,
                "simlex_concreteness": f"{concreteness:.3f}",
                "selection_rule": f"SimLex POS=N, concreteness<={threshold:g}, Lancaster-covered, non-THINGS",
            }
        )
    return rows


def build_concrete_control_rows(target_count: int) -> list[dict[str, Any]]:
    lancaster = load_lancaster_lookup()
    simlex = load_simlex_min_concreteness()
    things_rows = [row for row in load_things_rows() if normalize_word(row["concept"]) in lancaster]
    selected = []
    seen: set[str] = set()

    def add(row: dict[str, str], criterion: str) -> None:
        concept = clean_word(row["concept"])
        if concept in seen or len(selected) >= target_count:
            return
        seen.add(concept)
        selected.append(
            {
                "concept": concept,
                "domain": "sensory",
                "subtype": row.get("subtype", "sensory"),
                "simlex_pos": "N" if (concept, "N") in simlex else "",
                "simlex_concreteness": f"{simlex[(concept, 'N')]:.3f}" if (concept, "N") in simlex else "",
                "selection_rule": criterion,
            }
        )

    high_concrete = [
        row
        for row in things_rows
        if (clean_word(row["concept"]), "N") in simlex and simlex[(clean_word(row["concept"]), "N")] >= 4.0
    ]
    for row in sorted(high_concrete, key=lambda item: (-simlex[(clean_word(item["concept"]), "N")], clean_word(item["concept"]))):
        add(row, "THINGS, Lancaster-covered, SimLex POS=N, concreteness>=4.0")

    def visual_score(row: dict[str, str]) -> float:
        return float(lancaster[normalize_word(row["concept"])]["Visual.mean"])

    for row in sorted(things_rows, key=lambda item: (-visual_score(item), clean_word(item["concept"]))):
        add(row, "THINGS, Lancaster-covered, visual-rating fill")

    if len(selected) < target_count:
        raise RuntimeError(f"Only selected {len(selected)} concrete control concepts; needed {target_count}.")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Lancaster/SimLex abstract pilot concept lists.")
    parser.add_argument("--primary-threshold", type=float, default=3.5)
    parser.add_argument("--strict-threshold", type=float, default=3.0)
    args = parser.parse_args()

    primary = build_abstract_rows(args.primary_threshold)
    strict = build_abstract_rows(args.strict_threshold)
    concrete = build_concrete_control_rows(len(primary))

    fields = ["concept", "domain", "subtype", "simlex_pos", "simlex_concreteness", "selection_rule"]
    write_csv(ROOT / "data" / "concepts" / f"abstract_lancaster_{len(primary)}_concepts.csv", primary, fields)
    write_csv(ROOT / "data" / "concepts" / f"abstract_lancaster_{len(strict)}_strict_concepts.csv", strict, fields)
    write_csv(ROOT / "data" / "concepts" / f"concrete_lancaster_control_{len(concrete)}_concepts.csv", concrete, fields)
    write_json(
        ROOT / "outputs" / "metrics" / "abstract_lancaster_pilot_concept_summary.json",
        {
            "primary_threshold": args.primary_threshold,
            "strict_threshold": args.strict_threshold,
            "n_primary_abstract": len(primary),
            "n_strict_abstract": len(strict),
            "n_concrete_control": len(concrete),
            "lancaster_source": str(LANCASTER_SENSORIMOTOR.relative_to(ROOT)),
            "simlex_source": str(SIMLEX_PATH.relative_to(ROOT)),
            "things_exclusion_source": str(THINGS_PATH.relative_to(ROOT)),
        },
    )
    append_run_log(
        "Abstract Lancaster Pilot Concepts",
        [
            f"Prepared {len(primary)} primary abstract concepts at concreteness <= {args.primary_threshold:g}.",
            f"Prepared {len(strict)} strict abstract concepts at concreteness <= {args.strict_threshold:g}.",
            f"Prepared {len(concrete)} matched-size concrete control concepts.",
        ],
    )


if __name__ == "__main__":
    main()
