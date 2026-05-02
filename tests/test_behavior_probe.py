from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def load_score_module():
    path = Path("scripts/31_score_behavior_probe.py")
    scripts_dir = str(Path("scripts").resolve())
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("behavior_probe_score", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_contains_concept_handles_underscore_and_spaces() -> None:
    module = load_score_module()
    assert module.contains_concept("A bright air conditioner sits on the wall.", "air_conditioner")
    assert module.contains_concept("The fire truck is shiny and red.", "fire_truck")
    assert not module.contains_concept("The truck is shiny and red.", "fire_truck")


def test_lexicon_counts_are_deterministic() -> None:
    module = load_score_module()
    text = "A bright red, smooth object makes a loud sound."
    assert module.count_lexicon_words(text, module.VISUAL_WORDS) == 3
    assert module.count_lexicon_words(text, module.SENSORY_WORDS) == 5


def test_think_blocks_are_removed_before_scoring() -> None:
    module = load_score_module()
    text = "<think>apple bicycle</think> A bright apple sits on a table."
    assert module.strip_think_blocks(text) == "A bright apple sits on a table."
    assert module.contains_concept(text, "apple")
    assert not module.contains_concept(text, "bicycle")


def test_rate_and_lancaster_scores_are_deterministic() -> None:
    module = load_score_module()
    norms = {
        "red": {"Visual.mean": 5.0, "Auditory.mean": 0.0, "Gustatory.mean": 0.0, "Haptic.mean": 1.0, "Interoceptive.mean": 0.0, "Olfactory.mean": 0.0, "Foot_leg.mean": 0.0, "Hand_arm.mean": 0.0, "Head.mean": 0.0, "Mouth.mean": 0.0, "Torso.mean": 0.0},
        "rough": {"Visual.mean": 3.0, "Auditory.mean": 0.0, "Gustatory.mean": 0.0, "Haptic.mean": 4.0, "Interoceptive.mean": 0.0, "Olfactory.mean": 0.0, "Foot_leg.mean": 0.0, "Hand_arm.mean": 0.0, "Head.mean": 0.0, "Mouth.mean": 0.0, "Torso.mean": 0.0},
    }
    row = module.score_row(
        {
            "concept": "apple",
            "subtype": "food",
            "condition": "M_matched_image",
            "model_id": "test",
            "prompt": "",
            "image_path": "",
            "mismatch_source": "",
            "prompt_template_version": "test",
            "generated_text": "A red rough apple.",
            "word_count": "4",
            "seed": "1",
            "max_new_tokens": "8",
        },
        norms,
    )
    assert row["visual_word_rate_per_100"] == 50.0
    assert row["lancaster_token_overlap"] == 2
    assert row["lancaster_visual_mean"] == 4.0


def test_summary_reports_mismatched_leakage_only_for_mismatched_condition() -> None:
    module = load_score_module()
    rows = [
        {
            "condition": "M_mismatched_image",
            "target_retention": 1,
            "mismatched_source_leakage": 1,
            "visual_word_count": 2,
            "sensory_word_count": 3,
            "output_word_count": 8,
        },
        {
            "condition": "M_matched_image",
            "target_retention": 1,
            "mismatched_source_leakage": 0,
            "visual_word_count": 4,
            "sensory_word_count": 5,
            "output_word_count": 9,
        },
    ]
    summary = module.summarize(rows)
    assert summary["conditions"]["M_mismatched_image"]["mean_mismatched_source_leakage"] == 1.0
    assert summary["conditions"]["M_matched_image"]["mean_mismatched_source_leakage"] == 0.0


def test_paired_contrast_statistics_are_concept_paired() -> None:
    module = load_score_module()
    rows = []
    for concept in ["a", "b", "c"]:
        rows.append({"concept": concept, "condition": "M_matched_image", "visual_word_rate_per_100": 4.0})
        rows.append({"concept": concept, "condition": "M_text_only", "visual_word_rate_per_100": 1.0})
    diffs = module.paired_differences(rows, "M_matched_image", "M_text_only", "visual_word_rate_per_100")
    assert diffs == [3.0, 3.0, 3.0]
    ci_low, ci_high = module.bootstrap_ci(diffs, n_bootstrap=20, seed=1)
    assert ci_low == 3.0
    assert ci_high == 3.0
