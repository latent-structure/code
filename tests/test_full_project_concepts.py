from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


def test_full_concept_counts_match_plan_targets() -> None:
    rows = list(csv.DictReader(Path("data/concepts/full_concept_list.csv").open(newline="", encoding="utf-8")))
    sensory = [row for row in rows if row["domain"] == "sensory"]
    abstract = [row for row in rows if row["domain"] == "abstract"]
    assert len(sensory) == 60
    assert len(abstract) == 30


def test_sensory_subtypes_are_balanced() -> None:
    rows = list(csv.DictReader(Path("data/concepts/full_concept_list.csv").open(newline="", encoding="utf-8")))
    sensory = [row for row in rows if row["domain"] == "sensory"]
    counts = Counter(row["subtype"] for row in sensory)
    assert counts == {
        "appearance_color": 15,
        "texture_material": 15,
        "sound_linked": 15,
        "smell_taste_proxy": 15,
    }
