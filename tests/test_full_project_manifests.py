from __future__ import annotations

import csv
import json
from pathlib import Path


def test_resource_manifest_has_required_sources() -> None:
    rows = list(csv.DictReader(Path("data/manifests/resource_manifest.csv").open(newline="", encoding="utf-8")))
    names = {row["source_name"] for row in rows}
    assert "THINGS" in names
    assert "THINGSplus" in names
    assert "SimLex-999" in names
    assert "WordNet" in names


def test_mismatch_map_covers_all_sensory_concepts() -> None:
    concept_rows = list(csv.DictReader(Path("data/concepts/things_max_1854_concepts.csv").open(newline="", encoding="utf-8")))
    sensory_concepts = {row["concept"] for row in concept_rows if row["domain"] == "sensory"}
    mismatch_rows = list(csv.DictReader(Path("data/manifests/mismatch_map.csv").open(newline="", encoding="utf-8")))
    mapped = {row["concept"] for row in mismatch_rows}
    assert mapped == sensory_concepts


def test_root_pipeline_stage_files_exist() -> None:
    for name in [
        "00_run_results_v5_pipeline.py",
        "01_extract_hidden_states.py",
        "02_merge_embedding_bundles.py",
        "03_verify_span_pooling.py",
        "04_build_rdms.py",
        "05_compute_neighbor_restructuring.py",
        "06_compute_procrustes.py",
        "07_compute_human_partial_rsa.py",
        "08_compute_variance_partitioning.py",
        "09_compute_intrinsic_dimensionality.py",
        "10_prepare_hierarchy_mapping.py",
        "11_compute_modality_interference.py",
        "12_compute_linear_probes.py",
        "13_compute_id_alignment_correlation.py",
        "14_compute_hierarchy_depth_analysis.py",
        "15_prepare_multi_image_manifest.py",
        "16_compute_multi_image_consistency.py",
        "17_prepare_lancaster_anchor.py",
        "18_compute_lancaster_rsa.py",
        "19_compute_layerwise_trajectories.py",
        "20_prepare_full_things_archive.py",
        "21_compute_full_things_archive.py",
        "22_compute_internal_visual_tower_alignment.py",
        "23_compute_residual_interaction_analyses.py",
        "24_compute_mismatched_hijacking.py",
        "25_compute_layerwise_global_local_dissociation.py",
        "26_compute_cross_family_rsa_full.py",
        "27_compute_things_reliability_ceiling.py",
        "28_validate_mixture_decomposition.py",
        "29_validate_mismatched_hijacking.py",
        "30_generate_behavior_probe.py",
        "31_score_behavior_probe.py",
        "32_compute_behavior_geometry_bridge.py",
        "33_compute_position_control.py",
        "34_make_paper_figures.py",
        "35_compute_behavior_bridge_extensions.py",
        "36_prepare_abstract_lancaster_pilot.py",
        "37_compute_abstract_pilot.py",
        "38_compute_clip_forced_choice_behavior.py",
        "39_generate_behavior_repeats.py",
        "40_compute_behavior_reliability_ceiling.py",
        "42_compute_behavior_moderators.py",
    ]:
        assert Path("scripts", name).exists()


def test_things_branch_files_exist() -> None:
    assert Path("data/manifests/things_image_provenance.csv").exists()
    assert Path("data/concepts/things_covered_concepts.csv").exists()
    assert Path("data/concepts/things_missing_sensory_concepts.csv").exists()
    assert Path("outputs/logs/external_resource_validation.json").exists()


def test_external_resource_validation_reflects_local_archives() -> None:
    payload = json.loads(Path("outputs/logs/external_resource_validation.json").read_text(encoding="utf-8"))
    assert payload["things"]["present"] is True
    assert payload["thingsplus"]["present"] is True
    assert payload["things_behavior"]["present"] is True
    assert payload["simlex"]["present"] is True
    assert payload["wordnet"]["present"] is True


def test_things_covered_subset_contains_sensory_and_abstract_rows() -> None:
    rows = list(csv.DictReader(Path("data/concepts/things_covered_concepts.csv").open(newline="", encoding="utf-8")))
    sensory = [row for row in rows if row["domain"] == "sensory"]
    abstract = [row for row in rows if row["domain"] == "abstract"]
    assert len(sensory) == 38
    assert len(abstract) == 30


def test_reproducibility_release_surface_exists() -> None:
    for path in [
        Path("scripts/reproduce_neurips.py"),
        Path("DATA_AVAILABILITY.md"),
        Path("README.md"),
        Path("main.tex"),
    ]:
        assert path.exists()


def test_exploratory_scripts_are_archived_under_legacy() -> None:
    for name in [
        "41_compute_reference_profile_regime_test.py",
        "43_compute_vlm_prompt_only_analysis.py",
        "45_compute_layer_transition_diagnostics.py",
        "47_make_concept_neighborhood_mds_data.py",
        "48_make_pr_vs_rsa_figure.py",
        "49_compute_metric_robustness.py",
        "50_generate_identity_drift_probe.py",
        "51_score_identity_drift_probe.py",
    ]:
        assert Path("legacy", "scripts", name).exists()
