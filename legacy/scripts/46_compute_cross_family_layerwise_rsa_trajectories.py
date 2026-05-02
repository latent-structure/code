from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

import numpy as np

from analysis_common import ordered_embedding_for_concepts
from common import ROOT, append_run_log, condensed_cosine_distance, load_project_config, metrics_path, spearman_corr, write_csv, write_json
from hardening_common import (
    build_proxy_rdms,
    lancaster_matrix_for_concepts,
    load_embedding_bundle,
    load_siglip_reference,
    load_things_reference,
    mean_embedding_for_condition,
    residual_rsa,
    write_text,
)


ANCHORS = ["THINGS", "controlled_THINGS", "SigLIP2", "lancaster_perceptual"]
CONDITIONS = ["T_prompt_primary", "M_text_only", "M_matched_image", "M_mismatched_image", "M_blank_image", "M_degraded_image"]


def family_specs(config: dict[str, Any]) -> list[dict[str, str]]:
    return [dict(row) for row in config["analysis"]["analysis"].get("cross_family_families", [])]


def model_for_condition(family: dict[str, str], condition: str) -> str:
    return family["text_model"] if condition.startswith("T_") else family["multimodal_model"]


def first_positive_gap(layers: list[int], values: list[float], threshold: float = 0.01) -> int | None:
    for layer, value in zip(layers, values):
        if value > threshold:
            return layer
    return None


def split_thirds(values: np.ndarray) -> dict[str, float]:
    n = len(values)
    early = values[: max(1, n // 3)]
    middle = values[max(1, n // 3) : max(2, 2 * n // 3)]
    late = values[max(2, 2 * n // 3) :]
    return {
        "early_mean_gap": float(early.mean()),
        "middle_mean_gap": float(middle.mean()),
        "late_mean_gap": float(late.mean()),
        "late_minus_early_gap": float(late.mean() - early.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute layerwise RSA trajectories for Qwen, Mistral, and Llama.")
    parser.add_argument("--config", default="config/analysis.yaml")
    args = parser.parse_args()

    config = load_project_config(args.config)
    metadata_lookup, pooled, layers_by_model, metadata = load_embedding_bundle()
    things_behavior, things_concepts, _ = load_things_reference()
    target_concepts = things_concepts
    behavior_rdm = np.asarray((1.0 - things_behavior)[np.triu_indices(len(target_concepts), k=1)], dtype=float)
    proxy_rdms = build_proxy_rdms(target_concepts)

    siglip_embedding, siglip_concepts = load_siglip_reference(metadata_lookup, pooled, layers_by_model, metadata)
    siglip_ordered = ordered_embedding_for_concepts(siglip_embedding, siglip_concepts, target_concepts)
    siglip_reference_rdm = condensed_cosine_distance(siglip_ordered)

    lancaster_concepts = [
        concept.lower()
        for concept in json.loads((ROOT / "data" / "anchors" / "lancaster_perceptual_concepts.json").read_text(encoding="utf-8"))
        if concept.lower() in set(target_concepts)
    ]
    lancaster_reference_rdm = condensed_cosine_distance(
        lancaster_matrix_for_concepts(
            lancaster_concepts,
            ["Auditory.mean", "Gustatory.mean", "Haptic.mean", "Interoceptive.mean", "Olfactory.mean", "Visual.mean"],
        )
    )

    rows: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str, str], list[tuple[int, float]]] = defaultdict(list)
    missing: list[dict[str, str]] = []
    for family in family_specs(config):
        family_name = family["family_name"]
        for condition in CONDITIONS:
            model_id = model_for_condition(family, condition)
            layers = layers_by_model.get(model_id, [])
            if not layers:
                missing.append({"family_name": family_name, "condition": condition, "reason": f"missing model {model_id}"})
                continue
            for layer in layers:
                try:
                    embedding, concepts = mean_embedding_for_condition(metadata_lookup, pooled, model_id, condition, [layer])
                except RuntimeError as exc:
                    missing.append({"family_name": family_name, "condition": condition, "reason": str(exc)})
                    continue
                ordered = ordered_embedding_for_concepts(embedding, concepts, target_concepts)
                model_rdm = condensed_cosine_distance(ordered)
                ordered_lancaster = ordered_embedding_for_concepts(embedding, concepts, lancaster_concepts)
                lancaster_model_rdm = condensed_cosine_distance(ordered_lancaster)
                scores = {
                    "THINGS": spearman_corr(model_rdm, behavior_rdm),
                    "controlled_THINGS": residual_rsa(
                        model_rdm,
                        behavior_rdm,
                        [
                            proxy_rdms["subtype_membership"],
                            proxy_rdms["coarse_category_structure"],
                            proxy_rdms["sound_linked_vs_other"],
                            proxy_rdms["lexical_trigram_distance"],
                        ],
                    ),
                    "SigLIP2": spearman_corr(model_rdm, siglip_reference_rdm),
                    "lancaster_perceptual": spearman_corr(lancaster_model_rdm, lancaster_reference_rdm),
                }
                for anchor_name, score in scores.items():
                    rows.append(
                        {
                            "family_name": family_name,
                            "condition": condition,
                            "model_id": model_id,
                            "layer": layer,
                            "anchor_name": anchor_name,
                            "rsa_score": score,
                            "num_concepts": len(lancaster_concepts) if anchor_name == "lancaster_perceptual" else len(target_concepts),
                        }
                    )
                    by_key[(family_name, anchor_name, condition)].append((int(layer), float(score)))

    summary_rows: list[dict[str, Any]] = []
    for family in family_specs(config):
        family_name = family["family_name"]
        for anchor_name in ANCHORS:
            matched = dict(by_key.get((family_name, anchor_name, "M_matched_image"), []))
            prompt = dict(by_key.get((family_name, anchor_name, "T_prompt_primary"), []))
            text_only = dict(by_key.get((family_name, anchor_name, "M_text_only"), []))
            for baseline_name, baseline in [("prompt", prompt), ("vlm_text_only", text_only)]:
                layers = sorted(set(matched) & set(baseline))
                if not layers:
                    continue
                gap = np.asarray([matched[layer] - baseline[layer] for layer in layers], dtype=float)
                steepest_idx = int(np.abs(np.diff(gap)).argmax()) if len(gap) > 1 else 0
                peak_idx = int(gap.argmax())
                summary_rows.append(
                    {
                        "family_name": family_name,
                        "anchor_name": anchor_name,
                        "contrast_name": f"matched_minus_{baseline_name}",
                        "num_layers": len(layers),
                        "first_positive_layer": "" if first_positive_gap(layers, gap.tolist()) is None else first_positive_gap(layers, gap.tolist()),
                        "peak_layer": layers[peak_idx],
                        "peak_gap": float(gap[peak_idx]),
                        "first_layer_gap": float(gap[0]),
                        "last_layer_gap": float(gap[-1]),
                        "steepest_change_from_layer": layers[steepest_idx] if len(gap) > 1 else "",
                        "steepest_change_to_layer": layers[steepest_idx + 1] if len(gap) > 1 else "",
                        "steepest_abs_delta": float(abs(np.diff(gap)[steepest_idx])) if len(gap) > 1 else 0.0,
                        **split_thirds(gap),
                    }
                )

    write_csv(
        metrics_path("cross_family_layerwise_rsa_trajectories.csv"),
        rows,
        ["family_name", "condition", "model_id", "layer", "anchor_name", "rsa_score", "num_concepts"],
    )
    write_csv(
        metrics_path("cross_family_layerwise_rsa_summary.csv"),
        summary_rows,
        [
            "family_name",
            "anchor_name",
            "contrast_name",
            "num_layers",
            "first_positive_layer",
            "peak_layer",
            "peak_gap",
            "first_layer_gap",
            "last_layer_gap",
            "steepest_change_from_layer",
            "steepest_change_to_layer",
            "steepest_abs_delta",
            "early_mean_gap",
            "middle_mean_gap",
            "late_mean_gap",
            "late_minus_early_gap",
        ],
    )
    write_json(metrics_path("cross_family_layerwise_rsa_missing.json"), {"missing": missing})

    lines = ["# Cross-Family Layerwise RSA Trajectories", "", "## Matched-minus-prompt first positive layers", ""]
    for row in summary_rows:
        if row["contrast_name"] != "matched_minus_prompt":
            continue
        lines.append(
            f"- `{row['family_name']}` `{row['anchor_name']}`: first positive layer `{row['first_positive_layer']}`, "
            f"late-minus-early gap `{row['late_minus_early_gap']:.4f}`."
        )
    if missing:
        lines.extend(["", "## Missing"])
        for row in missing[:20]:
            lines.append(f"- `{row['family_name']}` `{row['condition']}`: {row['reason']}")
    write_text(ROOT / "reports" / "main_results" / "cross_family_layerwise_rsa_report.md", "\n".join(lines))
    append_run_log("Cross-Family Layerwise RSA Trajectories", ["Computed cross-family layerwise RSA trajectories."])


if __name__ == "__main__":
    main()
