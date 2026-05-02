from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = sys.executable
DEFAULT_CONFIG = "config/analysis.yaml"
DEFAULT_CONCEPT_SUBSET = "data/concepts/things_max_1854_concepts.csv"


@dataclass(frozen=True)
class Stage:
    name: str
    commands: tuple[tuple[str, ...], ...]
    expensive: bool = False


def py(script: str, *args: str) -> tuple[str, ...]:
    return (script, *args)


def stages(config: str, concept_subset: str) -> list[Stage]:
    qwen_models = ",".join(
        [
            "Qwen/Qwen3.5-9B",
            "Qwen/Qwen3-VL-8B-Instruct",
            "google/siglip2-so400m-patch16-384",
            "google/siglip2-base-patch16-224",
        ]
    )
    mistral_models = ",".join(
        [
            "mistralai/Mistral-Small-24B-Instruct-2501",
            "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        ]
    )
    llama_models = ",".join(
        [
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
        ]
    )
    return [
        Stage(
            "extract",
            (
                py("scripts/01_extract_hidden_states.py", "--config", config, "--models", qwen_models, "--concept-subset", concept_subset, "--output-tag", "qwen", "--overwrite"),
                py("scripts/01_extract_hidden_states.py", "--config", config, "--models", mistral_models, "--concept-subset", concept_subset, "--output-tag", "mistral", "--overwrite"),
                py("scripts/01_extract_hidden_states.py", "--config", config, "--models", llama_models, "--concept-subset", concept_subset, "--output-tag", "llama", "--overwrite"),
            ),
            expensive=True,
        ),
        Stage("merge", (py("scripts/02_merge_embedding_bundles.py", "--tags", "qwen,mistral,llama", "--overwrite"),)),
        Stage("verify", (py("scripts/03_verify_span_pooling.py", "--tags", "qwen,mistral,llama"),)),
        Stage(
            "core",
            (
                py("scripts/04_build_rdms.py", "--config", config),
                py("scripts/05_compute_neighbor_restructuring.py", "--config", config),
                py("scripts/06_compute_procrustes.py", "--config", config),
                py("scripts/07_compute_human_partial_rsa.py", "--config", config),
                py("scripts/08_compute_variance_partitioning.py", "--config", config),
            ),
        ),
        Stage(
            "geometry",
            (
                py("scripts/09_compute_intrinsic_dimensionality.py", "--config", config),
                py("scripts/10_prepare_hierarchy_mapping.py", "--config", config),
                py("scripts/11_compute_modality_interference.py", "--config", config),
                py("scripts/12_compute_linear_probes.py", "--config", config),
                py("scripts/13_compute_id_alignment_correlation.py", "--config", config),
                py("scripts/14_compute_hierarchy_depth_analysis.py", "--config", config),
            ),
        ),
        Stage(
            "stress",
            (
                py("scripts/15_prepare_multi_image_manifest.py", "--config", config),
                py("scripts/16_compute_multi_image_consistency.py", "--config", config),
                py("scripts/17_prepare_lancaster_anchor.py", "--config", config),
                py("scripts/18_compute_lancaster_rsa.py", "--config", config),
                py("scripts/19_compute_layerwise_trajectories.py", "--config", config),
                py("scripts/20_prepare_full_things_archive.py", "--config", config),
                py("scripts/21_compute_full_things_archive.py", "--config", config),
            ),
        ),
        Stage(
            "mechanism",
            (
                py("scripts/22_compute_internal_visual_tower_alignment.py", "--config", config),
                py("scripts/23_compute_residual_interaction_analyses.py", "--config", config),
                py("scripts/24_compute_mismatched_hijacking.py", "--config", config),
                py("scripts/25_compute_layerwise_global_local_dissociation.py", "--config", config),
            ),
        ),
        Stage(
            "hardening",
            (
                py("scripts/26_compute_cross_family_rsa_full.py", "--config", config),
                py("scripts/27_compute_things_reliability_ceiling.py", "--config", config),
                py("scripts/28_validate_mixture_decomposition.py", "--config", config, "--permutations", "1000"),
                py("scripts/29_validate_mismatched_hijacking.py", "--config", config),
            ),
        ),
        Stage(
            "behavior",
            (
                py("scripts/30_generate_behavior_probe.py", "--config", config, "--limit", "1854", "--output-stem", "behavior_probe_v2_full_generations", "--resume"),
                py("scripts/31_score_behavior_probe.py", "--input", "outputs/generations/behavior_probe_v2_full_generations.csv", "--output-stem", "behavior_probe_v2_full"),
                py("scripts/32_compute_behavior_geometry_bridge.py", "--config", config),
                py("scripts/33_compute_position_control.py", "--config", config),
            ),
            expensive=True,
        ),
    ]


def selected_stages(all_stages: list[Stage], from_stage: str | None, to_stage: str | None, skip_extraction: bool) -> list[Stage]:
    names = [stage.name for stage in all_stages]
    start = names.index(from_stage) if from_stage else 0
    end = names.index(to_stage) + 1 if to_stage else len(all_stages)
    selected = all_stages[start:end]
    if skip_extraction:
        selected = [stage for stage in selected if stage.name != "extract"]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the corrected RESULTS analysis pipeline in dependency order.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--concept-subset", default=DEFAULT_CONCEPT_SUBSET)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--from-stage", choices=["extract", "merge", "verify", "core", "geometry", "stress", "mechanism", "hardening", "behavior"])
    parser.add_argument("--to-stage", choices=["extract", "merge", "verify", "core", "geometry", "stress", "mechanism", "hardening", "behavior"])
    parser.add_argument("--skip-extraction", action="store_true", help="Run only downstream analyses from existing extracted embeddings.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    args = parser.parse_args()

    plan = selected_stages(stages(args.config, args.concept_subset), args.from_stage, args.to_stage, args.skip_extraction)
    for stage in plan:
        for command in stage.commands:
            full_command = [args.python, *command]
            print("+", " ".join(full_command), flush=True)
            if not args.dry_run:
                subprocess.run(full_command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
