from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    ROOT,
    append_run_log,
    condensed_cosine_distance,
    load_project_config,
    output_path,
    read_csv,
    set_global_seed,
    spearman_corr,
    write_csv,
)


THINGS_BEHAVIOR_MATRIX = ROOT / "data" / "anchors" / "things_behavioral_similarity.npy"
THINGS_BEHAVIOR_CONCEPTS = ROOT / "data" / "anchors" / "things_behavioral_concepts.json"
FAMILY_CONDITIONS = [
    "T_neutral",
    "T_prompt_primary",
    "M_text_only",
    "M_matched_image",
    "M_degraded_image",
    "M_mismatched_image",
    "M_blank_image",
]


def load_extract_module():
    script_path = ROOT / "scripts" / "05_extract_hidden_states.py"
    spec = importlib.util.spec_from_file_location("stage05_extract_hidden_states", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load helper module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def default_family_specs(config: dict[str, Any]) -> list[dict[str, str]]:
    specs = config["analysis"]["analysis"].get("cross_family_families", [])
    if not specs:
        raise RuntimeError("analysis.cross_family_families is empty")
    return [dict(row) for row in specs]


def select_family_specs(config: dict[str, Any], requested: str | None) -> list[dict[str, str]]:
    specs = default_family_specs(config)
    if not requested:
        return specs
    requested_names = {name.strip() for name in requested.split(",") if name.strip()}
    selected = [spec for spec in specs if spec["family_name"] in requested_names]
    missing = sorted(requested_names - {spec["family_name"] for spec in selected})
    if missing:
        raise RuntimeError(f"Requested families were not configured: {', '.join(missing)}")
    return selected


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve_cached_snapshot(model_id: str, cache_root: str) -> Path:
    org, name = model_id.split("/", 1)
    candidates = [
        Path(cache_root) / f"models--{org}--{name}",
        Path(cache_root) / "hub" / f"models--{org}--{name}",
    ]
    tried_refs = []
    for model_dir in candidates:
        refs_main = model_dir / "refs" / "main"
        tried_refs.append(str(refs_main))
        if not refs_main.exists():
            continue
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot_dir = model_dir / "snapshots" / revision
        if not snapshot_dir.exists():
            raise RuntimeError(f"Snapshot missing for {model_id}: {snapshot_dir}")
        return snapshot_dir
    raise RuntimeError(f"Cache ref missing for {model_id}: {', '.join(tried_refs)}")


def mean_embedding_by_condition(records: list[dict[str, object]], arrays: dict[str, np.ndarray], condition: str) -> tuple[np.ndarray, list[str]]:
    by_layer = [record for record in records if record["condition"] == condition]
    by_layer.sort(key=lambda item: int(item["layer"]))
    if not by_layer:
        raise RuntimeError(f"No records found for condition={condition}")
    start = len(by_layer) // 2
    selected = by_layer[start:]
    concepts = [concept.lower() for concept in selected[0]["concepts"]]
    matrices = [np.asarray(arrays[f"record_{record['record_id']}"], dtype=float) for record in selected]
    return np.mean(np.stack(matrices), axis=0), concepts


def load_existing_main_branch_family(stage05, spec: dict[str, str], subset_path: str) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    if subset_path != "data/concepts/things_max_1854_concepts.csv":
        raise RuntimeError("The existing-main family can only be evaluated on the active 1854 subset.")

    metadata = json.loads((ROOT / "outputs" / "embeddings" / "embedding_metadata_full.json").read_text(encoding="utf-8"))
    pooled_npz = np.load(ROOT / "outputs" / "embeddings" / "pooled_embeddings_full.npz")
    pooled = {key: np.asarray(pooled_npz[key], dtype=float) for key in pooled_npz.files}
    family_models = {spec["text_model"], spec["multimodal_model"]}

    arrays: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    for record in metadata["records"]:
        if record["model_id"] not in family_models or record["domain"] != "sensory":
            continue
        if record["family"] not in {"text", "multimodal"}:
            continue
        item = dict(record)
        records.append(item)
        arrays[f"record_{record['record_id']}"] = np.asarray(pooled[f"record_{record['record_id']}"], dtype=float)
    if not records:
        raise RuntimeError(f"No existing main-branch records found for family {spec['family_name']}")
    return arrays, records


def build_extracted_family(stage05, config: dict[str, Any], spec: dict[str, str], subset_path: str, torch: Any, transformers: Any) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], str, str]:
    concepts = [row for row in stage05.load_concepts(config, subset_path) if row["domain"] == "sensory"]
    ready_images = stage05.load_ready_image_map()
    mismatch_map = stage05.remap_mismatch_for_subset(concepts, ready_images, stage05.load_mismatch_map())

    text_snapshot = resolve_cached_snapshot(spec["text_model"], config["analysis"]["runtime"]["hf_cache_dir"])
    multimodal_snapshot = resolve_cached_snapshot(spec["multimodal_model"], config["analysis"]["runtime"]["hf_cache_dir"])

    text_template_map = {
        "T_neutral": stage05.pick_text_template_map(config)["T_neutral"],
        "T_prompt_primary": stage05.pick_text_template_map(config)["T_prompt_primary"],
    }

    text_tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(text_snapshot),
        **stage05.tokenizer_load_kwargs(str(text_snapshot)),
    )
    text_model = transformers.AutoModelForCausalLM.from_pretrained(str(text_snapshot), **stage05.model_load_kwargs(torch, config)).eval()
    text_arrays, text_records, _ = stage05.build_text_records(
        concepts,
        spec["text_model"],
        text_tokenizer,
        text_model,
        text_template_map,
        torch,
    )
    del text_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    multimodal_processor = transformers.AutoProcessor.from_pretrained(
        str(multimodal_snapshot),
        **stage05.tokenizer_load_kwargs(str(multimodal_snapshot)),
    )
    multimodal_model = stage05.load_multimodal_model(transformers, str(multimodal_snapshot), stage05.multimodal_load_kwargs(torch, config)).eval()
    multimodal_arrays, multimodal_records, _ = stage05.build_multimodal_records(
        concepts,
        spec["multimodal_model"],
        config,
        multimodal_processor,
        multimodal_model,
        stage05.pick_multimodal_prompt_map(config),
        ready_images,
        mismatch_map,
        torch,
    )
    del multimodal_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    all_arrays = {}
    all_arrays.update(text_arrays)
    offset = len(text_records)
    adjusted_multimodal_records = []
    for record in multimodal_records:
        new_record = dict(record)
        new_record["record_id"] = offset + record["record_id"]
        adjusted_multimodal_records.append(new_record)
        all_arrays[f"record_{new_record['record_id']}"] = multimodal_arrays[f"record_{record['record_id']}"]
    for record in text_records:
        all_arrays[f"record_{record['record_id']}"] = text_arrays[f"record_{record['record_id']}"]
    all_records = text_records + adjusted_multimodal_records
    return all_arrays, all_records, str(text_snapshot), str(multimodal_snapshot)


def load_reference_spaces() -> tuple[np.ndarray, dict[str, int], np.ndarray, dict[str, int], list[str]]:
    things_behavior = np.load(THINGS_BEHAVIOR_MATRIX)
    things_concepts = [concept.lower() for concept in json.loads(THINGS_BEHAVIOR_CONCEPTS.read_text(encoding="utf-8"))]
    things_index = {concept: idx for idx, concept in enumerate(things_concepts)}

    metadata = json.loads((ROOT / "outputs" / "embeddings" / "embedding_metadata_full.json").read_text(encoding="utf-8"))
    pooled_npz = np.load(ROOT / "outputs" / "embeddings" / "pooled_embeddings_full.npz")
    pooled = {key: np.asarray(pooled_npz[key], dtype=float) for key in pooled_npz.files}
    siglip_model_ids = sorted({
        record["model_id"]
        for record in metadata["records"]
        if record["family"] == "anchor" and "siglip" in record["model_id"].lower() and record["domain"] == "sensory"
    })
    if not siglip_model_ids:
        raise RuntimeError("No SigLIP2 anchor records were available in the current main-branch embeddings.")
    siglip_records = [
        record for record in metadata["records"]
        if record["model_id"] == siglip_model_ids[0] and record["condition"] == "reference_anchor_image" and record["domain"] == "sensory"
    ]
    siglip_records.sort(key=lambda item: int(item["layer"]))
    siglip_mats = [np.asarray(pooled[f"record_{record['record_id']}"], dtype=float) for record in siglip_records]
    siglip_embedding = np.mean(np.stack(siglip_mats), axis=0)
    siglip_concepts = [concept.lower() for concept in siglip_records[0]["concepts"]]
    siglip_index = {concept: idx for idx, concept in enumerate(siglip_concepts)}
    return things_behavior, things_index, siglip_embedding, siglip_index, siglip_concepts


def compute_family_summary(
    family_name: str,
    family_role: str,
    text_model: str,
    multimodal_model: str,
    arrays: dict[str, np.ndarray],
    records: list[dict[str, Any]],
    things_behavior: np.ndarray,
    things_index: dict[str, int],
    siglip_embedding: np.ndarray,
    siglip_index: dict[str, int],
) -> tuple[list[dict[str, object]], dict[str, dict[str, float]], str]:
    rows: list[dict[str, object]] = []
    summary: dict[str, dict[str, float]] = {}
    for condition in FAMILY_CONDITIONS:
        embedding, concepts_for_condition = mean_embedding_by_condition(records, arrays, condition)
        matched_positions = [idx for idx, concept in enumerate(concepts_for_condition) if concept in things_index]
        matched_concepts = [concepts_for_condition[idx] for idx in matched_positions]
        cond_embedding = np.asarray(embedding[matched_positions], dtype=float)
        model_rdm = condensed_cosine_distance(cond_embedding)

        behavior_idx = [things_index[concept] for concept in matched_concepts]
        behavior_dist = 1.0 - things_behavior[np.ix_(behavior_idx, behavior_idx)]
        things_rdm = np.asarray(behavior_dist[np.triu_indices(len(matched_concepts), k=1)], dtype=float)
        things_score = spearman_corr(model_rdm, things_rdm)

        siglip_positions = [siglip_index[concept] for concept in matched_concepts]
        siglip_rdm = condensed_cosine_distance(np.asarray(siglip_embedding[siglip_positions], dtype=float))
        siglip_score = spearman_corr(model_rdm, siglip_rdm)

        rows.append(
            {
                "family_name": family_name,
                "family_role": family_role,
                "text_model": text_model,
                "multimodal_model": multimodal_model,
                "anchor_name": "THINGS behavioral similarity",
                "condition": condition,
                "rsa_score": things_score,
                "comparison_delta": "",
                "support_flag": "",
            }
        )
        rows.append(
            {
                "family_name": family_name,
                "family_role": family_role,
                "text_model": text_model,
                "multimodal_model": multimodal_model,
                "anchor_name": "SigLIP2",
                "condition": condition,
                "rsa_score": siglip_score,
                "comparison_delta": "",
                "support_flag": "",
            }
        )
        summary[condition] = {"things": things_score, "siglip": siglip_score}

    things_prompt_competitive = summary["T_prompt_primary"]["things"] >= summary["M_matched_image"]["things"]
    siglip_grounding_advantage = summary["M_matched_image"]["siglip"] > summary["T_prompt_primary"]["siglip"]
    perturbation_sensitive = (summary["M_matched_image"]["siglip"] - summary["M_mismatched_image"]["siglip"]) > (
        summary["T_prompt_primary"]["siglip"] - summary["T_neutral"]["siglip"]
    )
    siglip_favors_grounding_more_than_things = (summary["M_matched_image"]["siglip"] - summary["T_prompt_primary"]["siglip"]) > (
        summary["M_matched_image"]["things"] - summary["T_prompt_primary"]["things"]
    )
    checks = {
        "things_prompt_competitive": things_prompt_competitive,
        "siglip_grounding_advantage": siglip_grounding_advantage,
        "perturbation_sensitive": perturbation_sensitive,
        "siglip_favors_grounding_more_than_things": siglip_favors_grounding_more_than_things,
    }
    if all(checks.values()):
        family_status = "PASS"
    elif checks["siglip_grounding_advantage"] and checks["perturbation_sensitive"] and sum(checks.values()) >= 2:
        family_status = "PARTIAL"
    else:
        family_status = "FAIL"

    rows.extend(
        [
            {
                "family_name": family_name,
                "family_role": family_role,
                "text_model": text_model,
                "multimodal_model": multimodal_model,
                "anchor_name": "summary",
                "condition": "cross_family_runtime_status",
                "rsa_score": "",
                "comparison_delta": "",
                "support_flag": family_status,
            },
            {
                "family_name": family_name,
                "family_role": family_role,
                "text_model": text_model,
                "multimodal_model": multimodal_model,
                "anchor_name": "summary",
                "condition": "things_prompt_competitive",
                "rsa_score": "",
                "comparison_delta": summary["T_prompt_primary"]["things"] - summary["M_matched_image"]["things"],
                "support_flag": str(things_prompt_competitive),
            },
            {
                "family_name": family_name,
                "family_role": family_role,
                "text_model": text_model,
                "multimodal_model": multimodal_model,
                "anchor_name": "summary",
                "condition": "siglip_grounding_advantage",
                "rsa_score": "",
                "comparison_delta": summary["M_matched_image"]["siglip"] - summary["T_prompt_primary"]["siglip"],
                "support_flag": str(siglip_grounding_advantage),
            },
            {
                "family_name": family_name,
                "family_role": family_role,
                "text_model": text_model,
                "multimodal_model": multimodal_model,
                "anchor_name": "summary",
                "condition": "perturbation_sensitive",
                "rsa_score": "",
                "comparison_delta": (summary["M_matched_image"]["siglip"] - summary["M_mismatched_image"]["siglip"]) - (summary["T_prompt_primary"]["siglip"] - summary["T_neutral"]["siglip"]),
                "support_flag": str(perturbation_sensitive),
            },
            {
                "family_name": family_name,
                "family_role": family_role,
                "text_model": text_model,
                "multimodal_model": multimodal_model,
                "anchor_name": "summary",
                "condition": "siglip_favors_grounding_more_than_things",
                "rsa_score": "",
                "comparison_delta": (summary["M_matched_image"]["siglip"] - summary["T_prompt_primary"]["siglip"]) - (summary["M_matched_image"]["things"] - summary["T_prompt_primary"]["things"]),
                "support_flag": str(siglip_favors_grounding_more_than_things),
            },
        ]
    )
    return rows, summary, family_status


def blocked_rows(family_spec: dict[str, str], reason: str) -> list[dict[str, str]]:
    return [
        {
            "family_name": family_spec["family_name"],
            "family_role": family_spec.get("family_role", ""),
            "text_model": family_spec["text_model"],
            "multimodal_model": family_spec["multimodal_model"],
            "anchor_name": "summary",
            "condition": "cross_family_runtime_status",
            "rsa_score": "",
            "comparison_delta": "",
            "support_flag": "BLOCKED" if reason == "blocked" else reason,
        },
        {
            "family_name": family_spec["family_name"],
            "family_role": family_spec.get("family_role", ""),
            "text_model": family_spec["text_model"],
            "multimodal_model": family_spec["multimodal_model"],
            "anchor_name": "summary",
            "condition": "things_prompt_competitive",
            "rsa_score": "",
            "comparison_delta": "",
            "support_flag": reason,
        },
        {
            "family_name": family_spec["family_name"],
            "family_role": family_spec.get("family_role", ""),
            "text_model": family_spec["text_model"],
            "multimodal_model": family_spec["multimodal_model"],
            "anchor_name": "summary",
            "condition": "siglip_grounding_advantage",
            "rsa_score": "",
            "comparison_delta": "",
            "support_flag": reason,
        },
        {
            "family_name": family_spec["family_name"],
            "family_role": family_spec.get("family_role", ""),
            "text_model": family_spec["text_model"],
            "multimodal_model": family_spec["multimodal_model"],
            "anchor_name": "summary",
            "condition": "perturbation_sensitive",
            "rsa_score": "",
            "comparison_delta": "",
            "support_flag": reason,
        },
        {
            "family_name": family_spec["family_name"],
            "family_role": family_spec.get("family_role", ""),
            "text_model": family_spec["text_model"],
            "multimodal_model": family_spec["multimodal_model"],
            "anchor_name": "summary",
            "condition": "siglip_favors_grounding_more_than_things",
            "rsa_score": "",
            "comparison_delta": "",
            "support_flag": reason,
        },
    ]


def family_report(
    family_spec: dict[str, str],
    family_status: str,
    summary: dict[str, dict[str, float]] | None,
    text_snapshot: str | None,
    multimodal_snapshot: str | None,
    failure: Exception | None,
) -> str:
    lines = [
        f"# Reframed Cross-Family Report: {family_spec['family_name']}",
        "",
        f"- Family role: `{family_spec.get('family_role', '')}`",
        f"- Text model: `{family_spec['text_model']}`",
        f"- Multimodal model: `{family_spec['multimodal_model']}`",
        f"- Status: `{family_status}`",
    ]
    if text_snapshot:
        lines.append(f"- Resolved text snapshot: `{text_snapshot}`")
    if multimodal_snapshot:
        lines.append(f"- Resolved multimodal snapshot: `{multimodal_snapshot}`")
    if failure is not None:
        lines.extend(
            [
                f"- Blocking error: `{type(failure).__name__}: {failure}`",
                "",
                "## Interpretation",
                "- This family should be treated as blocked on this host rather than silently omitted.",
            ]
        )
        return "\n".join(lines)

    assert summary is not None
    lines.extend(
        [
            "",
            "## Summary",
            f"- THINGS `T_prompt_primary - M_matched_image`: `{summary['T_prompt_primary']['things'] - summary['M_matched_image']['things']:.4f}`",
            f"- SigLIP2 `M_matched_image - T_prompt_primary`: `{summary['M_matched_image']['siglip'] - summary['T_prompt_primary']['siglip']:.4f}`",
            f"- SigLIP2 `M_matched_image - M_mismatched_image`: `{summary['M_matched_image']['siglip'] - summary['M_mismatched_image']['siglip']:.4f}`",
            "",
            "## Interpretation",
            "- The replication should be judged by whether prompting remains competitive on THINGS while grounding remains more perturbation-sensitive and stronger in SigLIP-like space.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--concept-subset", default=None)
    parser.add_argument("--families", default=None)
    args = parser.parse_args()

    config = load_project_config(args.config)
    set_global_seed(config["seeds"]["global"])
    stage05 = load_extract_module()
    stage05.configure_hf_cache(config)
    requested_subset = args.concept_subset or config["analysis"]["execution"]["default_concept_subset"]
    selected_specs = select_family_specs(config, args.families)
    things_behavior, things_index, siglip_embedding, siglip_index, _ = load_reference_spaces()

    if not args.concept_subset and requested_subset != config["analysis"]["execution"]["default_concept_subset"]:
        requested_subset = config["analysis"]["execution"]["default_concept_subset"]

    import torch
    import transformers

    rows: list[dict[str, object]] = []
    report_sections: list[str] = []
    for spec in selected_specs:
        text_snapshot = None
        multimodal_snapshot = None
        family_rows: list[dict[str, object]]
        family_summary: dict[str, dict[str, float]] | None = None
        family_status = "BLOCKED"
        failure: Exception | None = None

        try:
            if spec["mode"] == "existing_main":
                if requested_subset != config["analysis"]["execution"]["default_concept_subset"]:
                    raise RuntimeError("The existing-main family can only be evaluated on the active 1854 subset.")
                family_arrays, family_records = load_existing_main_branch_family(stage05, spec, requested_subset)
                family_rows, family_summary, family_status = compute_family_summary(
                    spec["family_name"],
                    spec.get("family_role", ""),
                    spec["text_model"],
                    spec["multimodal_model"],
                    family_arrays,
                    family_records,
                    things_behavior,
                    things_index,
                    siglip_embedding,
                    siglip_index,
                )
            else:
                family_arrays, family_records, text_snapshot, multimodal_snapshot = build_extracted_family(
                    stage05,
                    config,
                    spec,
                    requested_subset,
                    torch,
                    transformers,
                )
                family_rows, family_summary, family_status = compute_family_summary(
                    spec["family_name"],
                    spec.get("family_role", ""),
                    spec["text_model"],
                    spec["multimodal_model"],
                    family_arrays,
                    family_records,
                    things_behavior,
                    things_index,
                    siglip_embedding,
                    siglip_index,
                )
        except Exception as exc:
            failure = exc
            family_rows = blocked_rows(spec, "blocked")
            family_summary = None
            family_status = "BLOCKED"

        rows.extend(family_rows)
        family_report_text = family_report(spec, family_status, family_summary, text_snapshot, multimodal_snapshot, failure)
        report_sections.append(family_report_text)
        write_text(output_path("reports", "replication", f"reframed_cross_family_{spec['family_name']}.md"), family_report_text)
        append_run_log(
            "Reframed Cross-Family Replication",
            [
                f"Family `{spec['family_name']}` status: {family_status}",
                f"Text model: {spec['text_model']}",
                f"Multimodal model: {spec['multimodal_model']}",
            ],
        )

    write_csv(
        output_path("outputs", "tables", "cross_family_reframed_summary.csv"),
        rows,
        [
            "family_name",
            "family_role",
            "text_model",
            "multimodal_model",
            "anchor_name",
            "condition",
            "rsa_score",
            "comparison_delta",
            "support_flag",
        ],
    )

    consolidated = "\n\n".join(report_sections)
    write_text(output_path("reports", "replication", "reframed_cross_family_report.md"), consolidated)


if __name__ == "__main__":
    main()
