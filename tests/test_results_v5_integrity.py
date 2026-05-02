from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def test_release_markdown_docs_exist() -> None:
    for path in [
        Path("README.md"),
        Path("DATA_AVAILABILITY.md"),
        Path("main.tex"),
    ]:
        assert path.exists()


def test_results_v5_core_sources_exist() -> None:
    for path in [
        "outputs/embeddings/embedding_metadata_full_summary.json",
        "outputs/metrics/cross_family_rsa_full.csv",
        "outputs/metrics/intrinsic_dimensionality.csv",
        "outputs/metrics/lancaster_alignment.csv",
        "outputs/metrics/residual_reference_alignment.csv",
        "outputs/metrics/residual_interaction_summary.json",
        "outputs/metrics/mismatched_hijacking_summary.json",
        "outputs/metrics/layerwise_global_local_summary.json",
        "outputs/metrics/mixture_decomposition_validation_summary.json",
        "outputs/metrics/internal_visual_tower_summary.json",
        "outputs/metrics/things_reliability_ceiling.json",
    ]:
        assert Path(path).exists()


def test_release_scripts_exist() -> None:
    for name in [
        "reproduce_neurips.py",
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
        "44_compute_layerwise_internal_visual_alignment.py",
        "52_build_identity_similarity_probe_items.py",
        "53_generate_identity_similarity_probe.py",
        "54_score_identity_similarity_probe.py",
        "55_compute_clip_reliability_repeats.py",
        "56_compute_behavior_drift_endpoints.py",
        "57_prepare_thingsplus_moderators.py",
        "58_prepare_imsitu_scope_manifest.py",
        "59_prepare_mitstates_scope_manifest.py",
        "60_extract_scope_extension_embeddings.py",
        "61_compute_scope_extension_geometry.py",
        "62_summarize_scope_extension_results.py",
        "64_score_similarity_probe_logit_choice.py",
        "65_compute_thingsplus_joint_moderators.py",
        "66_compute_llama_cross_attention_layer_diagnostics.py",
        "67_probe_llama_cross_attention.py",
        "68_compute_llama_attention_geometry_coupling.py",
    ]:
        assert Path("scripts", name).exists()


def test_exploratory_scripts_are_archived() -> None:
    for name in [
        "41_compute_reference_profile_regime_test.py",
        "43_compute_vlm_prompt_only_analysis.py",
        "45_compute_layer_transition_diagnostics.py",
        "46_compute_cross_family_layerwise_rsa_trajectories.py",
        "47_make_concept_neighborhood_mds_data.py",
        "48_make_pr_vs_rsa_figure.py",
        "49_compute_metric_robustness.py",
        "50_generate_identity_drift_probe.py",
        "51_score_identity_drift_probe.py",
        "submit_abstract_pilot.sh",
        "submit_cross_family_behavior_full.sh",
        "submit_llama_cross_attention_probe.sh",
        "submit_scope_extensions.sh",
        "submit_similarity_logit_choice.sh",
    ]:
        assert Path("legacy", "scripts", name).exists()


def test_pipeline_runner_declares_expected_stage_order() -> None:
    path = Path("scripts/00_run_results_v5_pipeline.py")
    spec = importlib.util.spec_from_file_location("results_v5_pipeline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert [stage.name for stage in module.stages("config/analysis.yaml", "data/concepts/things_max_1854_concepts.csv")] == [
        "extract",
        "merge",
        "verify",
        "core",
        "geometry",
        "stress",
        "mechanism",
        "hardening",
        "behavior",
    ]
