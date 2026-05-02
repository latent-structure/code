from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
PIPELINE = ROOT / "scripts" / "00_run_results_v5_pipeline.py"
FIGURES = ROOT / "scripts" / "34_make_paper_figures.py"

CORE_PIPELINE_SCRIPTS = [
    "scripts/00_run_results_v5_pipeline.py",
    "scripts/01_extract_hidden_states.py",
    "scripts/02_merge_embedding_bundles.py",
    "scripts/03_verify_span_pooling.py",
    "scripts/04_build_rdms.py",
    "scripts/05_compute_neighbor_restructuring.py",
    "scripts/06_compute_procrustes.py",
    "scripts/07_compute_human_partial_rsa.py",
    "scripts/08_compute_variance_partitioning.py",
    "scripts/09_compute_intrinsic_dimensionality.py",
    "scripts/10_prepare_hierarchy_mapping.py",
    "scripts/11_compute_modality_interference.py",
    "scripts/12_compute_linear_probes.py",
    "scripts/13_compute_id_alignment_correlation.py",
    "scripts/14_compute_hierarchy_depth_analysis.py",
    "scripts/15_prepare_multi_image_manifest.py",
    "scripts/16_compute_multi_image_consistency.py",
    "scripts/17_prepare_lancaster_anchor.py",
    "scripts/18_compute_lancaster_rsa.py",
    "scripts/19_compute_layerwise_trajectories.py",
    "scripts/20_prepare_full_things_archive.py",
    "scripts/21_compute_full_things_archive.py",
    "scripts/22_compute_internal_visual_tower_alignment.py",
    "scripts/23_compute_residual_interaction_analyses.py",
    "scripts/24_compute_mismatched_hijacking.py",
    "scripts/25_compute_layerwise_global_local_dissociation.py",
    "scripts/26_compute_cross_family_rsa_full.py",
    "scripts/27_compute_things_reliability_ceiling.py",
    "scripts/28_validate_mixture_decomposition.py",
    "scripts/29_validate_mismatched_hijacking.py",
    "scripts/30_generate_behavior_probe.py",
    "scripts/31_score_behavior_probe.py",
    "scripts/32_compute_behavior_geometry_bridge.py",
    "scripts/33_compute_position_control.py",
]

POST_PIPELINE_SCRIPTS = [
    "scripts/35_compute_behavior_bridge_extensions.py",
    "scripts/36_prepare_abstract_lancaster_pilot.py",
    "scripts/37_compute_abstract_pilot.py",
    "scripts/38_compute_clip_forced_choice_behavior.py",
    "scripts/39_generate_behavior_repeats.py",
    "scripts/40_compute_behavior_reliability_ceiling.py",
    "scripts/42_compute_behavior_moderators.py",
    "scripts/44_compute_layerwise_internal_visual_alignment.py",
    "scripts/52_build_identity_similarity_probe_items.py",
    "scripts/53_generate_identity_similarity_probe.py",
    "scripts/54_score_identity_similarity_probe.py",
    "scripts/55_compute_clip_reliability_repeats.py",
    "scripts/56_compute_behavior_drift_endpoints.py",
    "scripts/57_prepare_thingsplus_moderators.py",
    "scripts/58_prepare_imsitu_scope_manifest.py",
    "scripts/59_prepare_mitstates_scope_manifest.py",
    "scripts/60_extract_scope_extension_embeddings.py",
    "scripts/61_compute_scope_extension_geometry.py",
    "scripts/62_summarize_scope_extension_results.py",
    "scripts/64_score_similarity_probe_logit_choice.py",
    "scripts/65_compute_thingsplus_joint_moderators.py",
    "scripts/66_compute_llama_cross_attention_layer_diagnostics.py",
    "scripts/67_probe_llama_cross_attention.py",
    "scripts/68_compute_llama_attention_geometry_coupling.py",
]

RELEASE_DOCS = [
    "README.md",
    "DATA_AVAILABILITY.md",
    "main.tex",
    "config/analysis.yaml",
]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def verify_release_surface() -> None:
    missing = [path for path in RELEASE_DOCS if not (ROOT / path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required release files: {', '.join(missing)}")

    for relpath in CORE_PIPELINE_SCRIPTS + POST_PIPELINE_SCRIPTS + ["scripts/34_make_paper_figures.py"]:
        if not (ROOT / relpath).exists():
            raise FileNotFoundError(relpath)


def pipeline_command(skip_extraction: bool, dry_run: bool) -> list[str]:
    cmd = [PYTHON, str(PIPELINE), "--config", "config/analysis.yaml"]
    if skip_extraction:
        cmd.append("--skip-extraction")
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def build_commands(mode: str) -> list[list[str]]:
    commands: list[list[str]] = []
    if mode == "smoke":
        commands.append(pipeline_command(skip_extraction=True, dry_run=True))
        commands.extend([[PYTHON, str(ROOT / relpath), "--config", "config/analysis.yaml"] for relpath in POST_PIPELINE_SCRIPTS])
        commands.append([PYTHON, str(FIGURES), "--config", "config/analysis.yaml"])
        return commands
    if mode == "downstream":
        commands.append(pipeline_command(skip_extraction=True, dry_run=False))
    elif mode == "full":
        commands.append(pipeline_command(skip_extraction=False, dry_run=False))
    elif mode == "figures":
        commands.append([PYTHON, str(FIGURES), "--config", "config/analysis.yaml"])
        return commands
    else:
        raise ValueError(mode)

    commands.extend([[PYTHON, str(ROOT / relpath), "--config", "config/analysis.yaml"] for relpath in POST_PIPELINE_SCRIPTS])
    commands.append([PYTHON, str(FIGURES), "--config", "config/analysis.yaml"])
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description="Reviewer-facing NeurIPS reproduction entrypoint.")
    parser.add_argument("--mode", choices=["smoke", "downstream", "full", "figures"], default="smoke")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args()

    verify_release_surface()
    commands = build_commands(args.mode)

    for cmd in commands:
        if args.dry_run or args.mode == "smoke":
            print("+", " ".join(cmd), flush=True)
        else:
            run(cmd)


if __name__ == "__main__":
    main()
