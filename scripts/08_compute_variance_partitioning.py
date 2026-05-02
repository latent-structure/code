from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from common import (
    ROOT,
    append_run_log,
    canonical_condition_name,
    condensed_cosine_distance,
    embeddings_path,
    load_project_config,
    metrics_path,
    output_path,
    rankdata,
    read_csv,
    write_csv,
    write_json,
)
from hardening_common import load_active_concept_rows


THINGS_BEHAVIOR_MATRIX = ROOT / "data" / "anchors" / "things_behavioral_similarity.npy"
THINGS_BEHAVIOR_CONCEPTS = ROOT / "data" / "anchors" / "things_behavioral_concepts.json"
CONDITIONS = ["T_prompt_primary", "M_matched_image", "M_mismatched_image"]
PREDICTORS = [
    "THINGS behavioral similarity",
    "SigLIP2",
    "DINOv2",
    "subtype_membership",
    "lexical_trigram_distance",
]
HUMAN_FAMILY = ["THINGS behavioral similarity"]
ANCHOR_FAMILY = ["SigLIP2", "DINOv2"]
PROXY_FAMILY = ["subtype_membership", "lexical_trigram_distance"]


def mean_embedding_for_condition(
    metadata_lookup: dict[tuple[str, str, int], dict[str, object]],
    pooled: dict[str, np.ndarray],
    model_id: str,
    condition: str,
    layers: list[int],
) -> tuple[np.ndarray, list[str]]:
    matrices = []
    concepts: list[str] | None = None
    for layer in layers:
        record = metadata_lookup.get((model_id, condition, layer))
        if record is None:
            continue
        if concepts is None:
            concepts = [concept.lower() for concept in record["concepts"]]
        matrices.append(np.asarray(pooled[f"record_{record['record_id']}"], dtype=float))
    if not matrices or concepts is None:
        raise RuntimeError(f"Missing embeddings for {model_id} {condition}")
    return np.mean(np.stack(matrices), axis=0), concepts


def condition_model_id(backbone_text: str, backbone_multimodal: str, condition: str) -> str:
    return backbone_text if condition.startswith("T_") else backbone_multimodal


def lexical_distance(a: str, b: str) -> float:
    padded_a = f"__{a.lower()}__"
    padded_b = f"__{b.lower()}__"
    trigrams_a = {padded_a[idx : idx + 3] for idx in range(len(padded_a) - 2)}
    trigrams_b = {padded_b[idx : idx + 3] for idx in range(len(padded_b) - 2)}
    union = trigrams_a | trigrams_b
    if not union:
        return 1.0
    return 1.0 - (len(trigrams_a & trigrams_b) / len(union))


def load_static_anchor(name: str) -> tuple[np.ndarray, list[str]]:
    mapping = {
        "DINOv2": ("data/anchors/dinov2_embeddings.npy", "data/anchors/dinov2_concepts.json"),
    }
    emb_path, concepts_path = mapping[name]
    return np.load(ROOT / emb_path), [concept.lower() for concept in json.loads((ROOT / concepts_path).read_text(encoding="utf-8"))]


def r2_score(target: np.ndarray, predictors: list[np.ndarray]) -> float:
    y = rankdata(np.asarray(target, dtype=float)) - np.mean(rankdata(np.asarray(target, dtype=float)))
    if not predictors:
        return 0.0
    cols = []
    for predictor in predictors:
        ranked = rankdata(np.asarray(predictor, dtype=float))
        cols.append(ranked - np.mean(ranked))
    design = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    y_hat = design @ beta
    denom = float(np.dot(y, y))
    if denom == 0:
        return 0.0
    value = float(np.dot(y_hat, y_hat) / denom)
    return max(0.0, min(1.0, value))


def commonality_3way(r_h: float, r_a: float, r_p: float, r_ha: float, r_hp: float, r_ap: float, r_hap: float) -> dict[str, float]:
    u_h = max(0.0, r_hap - r_ap)
    u_a = max(0.0, r_hap - r_hp)
    u_p = max(0.0, r_hap - r_ha)
    return {
        "unique_human_family": u_h,
        "unique_anchor_family": u_a,
        "unique_proxy_family": u_p,
        "shared_with_human": max(0.0, r_hap - u_h),
        "shared_with_model_anchors": max(0.0, r_hap - u_a),
        "shared_with_proxies": max(0.0, r_hap - u_p),
    }


def write_text(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    args = parser.parse_args()

    config = load_project_config(args.config)
    backbone_text = config["analysis"]["execution"]["sensory_backbone_text_model"]
    backbone_multimodal = config["analysis"]["execution"]["sensory_backbone_multimodal_model"]
    mid_to_late_fraction = float(config["analysis"]["analysis"]["mid_to_late_fraction"])

    metadata = json.loads(embeddings_path("embedding_metadata_full.json").read_text(encoding="utf-8"))
    pooled_npz = np.load(embeddings_path("pooled_embeddings_full.npz"))
    pooled = {key: np.asarray(pooled_npz[key], dtype=float) for key in pooled_npz.files}
    metadata_lookup = {
        (record["model_id"], canonical_condition_name(record["condition"]), int(record["layer"])): record
        for record in metadata["records"]
        if record["domain"] == "sensory"
    }
    layers_by_model: dict[str, list[int]] = defaultdict(list)
    for record in metadata["records"]:
        if record["domain"] == "sensory":
            layers_by_model[record["model_id"]].append(int(record["layer"]))
    text_layers = sorted(set(layers_by_model[backbone_text]))
    multimodal_layers = sorted(set(layers_by_model[backbone_multimodal]))
    selected_text_layers = text_layers[len(text_layers) - int(np.ceil(len(text_layers) * mid_to_late_fraction)) :]
    selected_multimodal_layers = multimodal_layers[len(multimodal_layers) - int(np.ceil(len(multimodal_layers) * mid_to_late_fraction)) :]

    things_behavior = np.load(THINGS_BEHAVIOR_MATRIX)
    things_concepts = [concept.lower() for concept in json.loads(THINGS_BEHAVIOR_CONCEPTS.read_text(encoding="utf-8"))]
    things_index = {concept: idx for idx, concept in enumerate(things_concepts)}
    concept_rows = {row["concept"].lower(): row for row in load_active_concept_rows(args.config)}

    target_rdms: dict[str, np.ndarray] = {}
    overlap_concepts: list[str] | None = None
    for condition in CONDITIONS:
        model_id = condition_model_id(backbone_text, backbone_multimodal, condition)
        layers = selected_text_layers if condition.startswith("T_") else selected_multimodal_layers
        embeddings, concepts = mean_embedding_for_condition(metadata_lookup, pooled, model_id, condition, layers)
        matched_positions = [pos for pos, concept in enumerate(concepts) if concept in things_index]
        matched_concepts = [concepts[pos] for pos in matched_positions]
        matched_embeddings = np.asarray(embeddings[matched_positions], dtype=float)
        if overlap_concepts is None:
            overlap_concepts = matched_concepts
        elif overlap_concepts != matched_concepts:
            raise RuntimeError("Expected aligned THINGS overlap ordering across variance-partitioning conditions.")
        target_rdms[condition] = condensed_cosine_distance(matched_embeddings)
    if overlap_concepts is None:
        raise RuntimeError("Failed to construct overlap concept set for variance partitioning.")

    behavior_idx = [things_index[concept] for concept in overlap_concepts]
    behavior_dist = 1.0 - things_behavior[np.ix_(behavior_idx, behavior_idx)]
    predictor_rdms: dict[str, np.ndarray] = {
        "THINGS behavioral similarity": np.asarray(behavior_dist[np.triu_indices(len(overlap_concepts), k=1)], dtype=float),
    }

    siglip_model_ids = sorted({
        record["model_id"]
        for record in metadata["records"]
        if record["family"] == "anchor" and "siglip" in record["model_id"].lower() and record["domain"] == "sensory"
    })
    if not siglip_model_ids:
        raise RuntimeError("Could not locate SigLIP2 anchor rows in embedding metadata.")
    siglip_embeddings, siglip_concepts = mean_embedding_for_condition(
        metadata_lookup,
        pooled,
        siglip_model_ids[0],
        "reference_anchor_image",
        sorted(set(layers_by_model[siglip_model_ids[0]])),
    )
    siglip_index = {concept: idx for idx, concept in enumerate(siglip_concepts)}
    predictor_rdms["SigLIP2"] = condensed_cosine_distance(np.asarray(siglip_embeddings[[siglip_index[concept] for concept in overlap_concepts]], dtype=float))

    dino_embeddings, dino_concepts = load_static_anchor("DINOv2")
    dino_index = {concept: idx for idx, concept in enumerate(dino_concepts)}
    predictor_rdms["DINOv2"] = condensed_cosine_distance(np.asarray(dino_embeddings[[dino_index[concept] for concept in overlap_concepts]], dtype=float))

    subtype_values = []
    lexical_values = []
    for idx, left in enumerate(overlap_concepts):
        for right in overlap_concepts[idx + 1 :]:
            subtype_values.append(0.0 if concept_rows[left]["subtype"] == concept_rows[right]["subtype"] else 1.0)
            lexical_values.append(lexical_distance(left, right))
    predictor_rdms["subtype_membership"] = np.asarray(subtype_values, dtype=float)
    predictor_rdms["lexical_trigram_distance"] = np.asarray(lexical_values, dtype=float)

    rows = []
    summary_payload: dict[str, object] = {"conditions": {}}
    for condition in CONDITIONS:
        target = target_rdms[condition]
        full_r2 = r2_score(target, [predictor_rdms[name] for name in PREDICTORS])
        for predictor_name in PREDICTORS:
            without_predictor = [predictor_rdms[name] for name in PREDICTORS if name != predictor_name]
            unique_value = max(0.0, full_r2 - r2_score(target, without_predictor))
            rows.append(
                {
                    "condition": condition,
                    "predictor_name": predictor_name,
                    "component_type": "unique",
                    "variance_explained": unique_value,
                    "num_concepts": len(overlap_concepts),
                    "num_pairs": len(target),
                }
            )

        r_h = r2_score(target, [predictor_rdms[name] for name in HUMAN_FAMILY])
        r_a = r2_score(target, [predictor_rdms[name] for name in ANCHOR_FAMILY])
        r_p = r2_score(target, [predictor_rdms[name] for name in PROXY_FAMILY])
        r_ha = r2_score(target, [predictor_rdms[name] for name in HUMAN_FAMILY + ANCHOR_FAMILY])
        r_hp = r2_score(target, [predictor_rdms[name] for name in HUMAN_FAMILY + PROXY_FAMILY])
        r_ap = r2_score(target, [predictor_rdms[name] for name in ANCHOR_FAMILY + PROXY_FAMILY])
        family_components = commonality_3way(r_h, r_a, r_p, r_ha, r_hp, r_ap, full_r2)

        rows.append(
            {
                "condition": condition,
                "predictor_name": "all_predictors",
                "component_type": "total_model_fit",
                "variance_explained": full_r2,
                "num_concepts": len(overlap_concepts),
                "num_pairs": len(target),
            }
        )
        rows.append(
            {
                "condition": condition,
                "predictor_name": "THINGS behavioral similarity",
                "component_type": "shared_with_human",
                "variance_explained": family_components["shared_with_human"],
                "num_concepts": len(overlap_concepts),
                "num_pairs": len(target),
            }
        )
        rows.append(
            {
                "condition": condition,
                "predictor_name": "model_anchor_family",
                "component_type": "shared_with_model_anchors",
                "variance_explained": family_components["shared_with_model_anchors"],
                "num_concepts": len(overlap_concepts),
                "num_pairs": len(target),
            }
        )
        rows.append(
            {
                "condition": condition,
                "predictor_name": "proxy_family",
                "component_type": "shared_with_proxies",
                "variance_explained": family_components["shared_with_proxies"],
                "num_concepts": len(overlap_concepts),
                "num_pairs": len(target),
            }
        )
        summary_payload["conditions"][condition] = {
            "total_model_fit": full_r2,
            "unique_human_family": family_components["unique_human_family"],
            "unique_anchor_family": family_components["unique_anchor_family"],
            "unique_proxy_family": family_components["unique_proxy_family"],
            "shared_with_human": family_components["shared_with_human"],
            "shared_with_model_anchors": family_components["shared_with_model_anchors"],
            "shared_with_proxies": family_components["shared_with_proxies"],
            "unique_predictors": {
                row["predictor_name"]: row["variance_explained"]
                for row in rows
                if row["condition"] == condition and row["component_type"] == "unique"
            },
        }

    summary_payload["highest_unique_human_condition"] = max(
        CONDITIONS,
        key=lambda condition: float(summary_payload["conditions"][condition]["unique_human_family"]),
    )
    summary_payload["highest_unique_anchor_condition"] = max(
        CONDITIONS,
        key=lambda condition: float(summary_payload["conditions"][condition]["unique_anchor_family"]),
    )
    summary_payload["matched_minus_mismatched_anchor_unique"] = float(
        summary_payload["conditions"]["M_matched_image"]["unique_anchor_family"]
        - summary_payload["conditions"]["M_mismatched_image"]["unique_anchor_family"]
    )

    write_csv(
        metrics_path("variance_partitioning.csv"),
        rows,
        ["condition", "predictor_name", "component_type", "variance_explained", "num_concepts", "num_pairs"],
    )
    write_json(metrics_path("variance_partitioning_summary.json"), summary_payload)
    output_path("outputs", "tables").mkdir(parents=True, exist_ok=True)
    write_csv(
        output_path("outputs", "tables", "variance_partitioning_table.csv"),
        rows,
        ["condition", "predictor_name", "component_type", "variance_explained", "num_concepts", "num_pairs"],
    )

    report_lines = []
    for condition in CONDITIONS:
        payload = summary_payload["conditions"][condition]
        report_lines.extend(
            [
                f"### {condition}",
                f"- total_model_fit=`{payload['total_model_fit']:.4f}`",
                f"- unique_human_family=`{payload['unique_human_family']:.4f}`",
                f"- unique_anchor_family=`{payload['unique_anchor_family']:.4f}`",
                f"- unique_proxy_family=`{payload['unique_proxy_family']:.4f}`",
            ]
        )
    report = "\n".join(
        [
            "# Variance Partitioning Report",
            "",
            "## Condition Summary",
            *report_lines,
            "",
            "## Interpretation",
            f"- Highest unique human-family variance appears in `{summary_payload['highest_unique_human_condition']}`.",
            f"- Highest unique anchor-family variance appears in `{summary_payload['highest_unique_anchor_condition']}`.",
            f"- Matched-minus-mismatched anchor-family unique variance: `{summary_payload['matched_minus_mismatched_anchor_unique']:.4f}`.",
            "- This analysis is intended to distinguish human-anchored versus perceptual-semantic variance, not to identify a single ground-truth reference space.",
        ]
    )
    write_text(output_path("reports", "main_results", "variance_partitioning_report.md"), report)
    append_run_log(
        "Variance Partitioning",
        [
            f"Wrote variance-partitioning metrics to {metrics_path('variance_partitioning.csv').relative_to(ROOT)}.",
            f"Wrote variance-partitioning report to {output_path('reports', 'main_results', 'variance_partitioning_report.md').relative_to(ROOT)}.",
        ],
    )


if __name__ == "__main__":
    main()
