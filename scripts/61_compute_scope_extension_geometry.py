from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import ROOT, append_run_log, condensed_cosine_distance, ensure_parent, spearman_corr, write_json


CONDITION_ORDER = [
    "T_prompt_primary",
    "M_text_only",
    "M_matched_image",
    "M_prompt_plus_matched_image",
    "M_mismatched_image",
    "M_blank_image",
]


def participation_ratio(matrix: np.ndarray) -> float:
    x = np.asarray(matrix, dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    eigenvalues = singular_values**2
    denom = float(np.sum(eigenvalues**2))
    if denom <= 0.0:
        return 0.0
    return float((np.sum(eigenvalues) ** 2) / denom)


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = float(values.std())
    if std == 0.0:
        return values - values.mean()
    return (values - values.mean()) / std


def cosine_distance_from_features(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normed = features / norms
    distance = 1.0 - np.clip(normed @ normed.T, -1.0, 1.0)
    return distance[np.triu_indices(distance.shape[0], k=1)]


def binary_same_distance(values: pd.Series) -> np.ndarray:
    arr = values.astype(str).to_numpy()
    same = arr[:, None] == arr[None, :]
    distance = np.where(same, 0.0, 1.0)
    return distance[np.triu_indices(distance.shape[0], k=1)]


def role_feature_matrix(labels: pd.DataFrame) -> np.ndarray:
    roles = sorted(
        {
            role
            for value in labels.get("top_roles", pd.Series(dtype=str)).fillna("")
            for role in str(value).split(";")
            if role
        }
    )
    base_cols = [col for col in ["num_roles", "num_nouns", "total_role_mentions"] if col in labels.columns]
    rows: list[list[float]] = []
    for _, row in labels.iterrows():
        vec = [float(row[col]) for col in base_cols]
        row_roles = set(str(row.get("top_roles", "")).split(";"))
        vec.extend(1.0 if role in row_roles else 0.0 for role in roles)
        rows.append(vec)
    return np.asarray(rows, dtype=float)


def build_reference_rdms(dataset: str, labels: pd.DataFrame) -> dict[str, np.ndarray]:
    if dataset == "imsitu":
        features_path = ROOT / "outputs" / "scope_extensions" / "imsitu_label_features.csv"
        features = pd.read_csv(features_path) if features_path.exists() else labels
        merged = labels[["label"]].merge(features, on="label", how="left")
        frame_col = None
        for candidate in ["framenet_frame", "framenet", "abstract_frame"]:
            if candidate in merged.columns:
                frame_col = candidate
                break
        if frame_col is None:
            raise KeyError("No frame column found for imSitu reference RDMs; expected one of framenet_frame, framenet, abstract_frame")
        refs = {
            "framenet_frame": binary_same_distance(merged[frame_col]),
            "role_signature": cosine_distance_from_features(role_feature_matrix(merged)),
        }
        return refs
    if dataset == "mitstates":
        return {
            "attribute_identity": binary_same_distance(labels["attribute"]),
            "object_identity": binary_same_distance(labels["object"]),
            "composition_identity": binary_same_distance(labels["label"]),
        }
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_bundle(dataset: str, stem: str) -> tuple[dict[str, np.ndarray], dict[str, Any], pd.DataFrame]:
    out_dir = ROOT / "outputs" / "scope_extensions"
    stem = stem or f"{dataset}_qwen_scope_embeddings"
    npz_path = out_dir / f"{stem}.npz"
    json_path = out_dir / f"{stem}.json"
    labels_path = out_dir / f"{dataset}_labels.csv"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)
    arrays = dict(np.load(npz_path))
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    labels = pd.read_csv(labels_path)
    concepts = metadata["records"][0]["concepts"]
    labels = labels[labels["label"].isin(concepts)].copy()
    labels["_order"] = labels["label"].map({label: idx for idx, label in enumerate(concepts)})
    labels = labels.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    return arrays, metadata, labels


def condition_layers(arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> dict[str, dict[int, np.ndarray]]:
    output: dict[str, dict[int, np.ndarray]] = {}
    for record in metadata["records"]:
        condition = str(record["condition"])
        layer = int(record["layer"])
        output.setdefault(condition, {})[layer] = arrays[f"record_{record['record_id']}"]
    return output


def pooled_by_condition(layers_by_condition: dict[str, dict[int, np.ndarray]], fraction: float) -> dict[str, np.ndarray]:
    pooled: dict[str, np.ndarray] = {}
    for condition, layer_map in layers_by_condition.items():
        layers = sorted(layer_map)
        start = int(np.floor(len(layers) * (1.0 - fraction)))
        selected = layers[start:]
        pooled[condition] = np.mean(np.stack([layer_map[layer] for layer in selected]), axis=0).astype(np.float32)
    return pooled


def layerwise_pr(layers_by_condition: dict[str, dict[int, np.ndarray]], fraction: float) -> pd.DataFrame:
    rows = []
    for condition, layer_map in layers_by_condition.items():
        layers = sorted(layer_map)
        start = int(np.floor(len(layers) * (1.0 - fraction)))
        for layer in layers:
            rows.append(
                {
                    "condition": condition,
                    "layer": layer,
                    "in_mid_to_late_band": layer in set(layers[start:]),
                    "participation_ratio": participation_ratio(layer_map[layer]),
                }
            )
    return pd.DataFrame(rows)


def regression_mixture(prompt_rdm: np.ndarray, matched_rdm: np.ndarray, combined_rdm: np.ndarray) -> dict[str, float]:
    y = zscore(combined_rdm)
    x = np.column_stack([zscore(prompt_rdm), zscore(matched_rdm)])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    yhat = x @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    corr = spearman_corr(prompt_rdm, matched_rdm)
    return {
        "prompt_weight": float(coef[0]),
        "matched_image_weight": float(coef[1]),
        "r2": float(r2),
        "prompt_matched_spearman": float(corr),
    }


def mismatch_attraction(labels: pd.DataFrame, pooled: dict[str, np.ndarray]) -> pd.DataFrame:
    required = {"M_mismatched_image", "M_matched_image"}
    if not required.issubset(pooled):
        return pd.DataFrame()
    matched = pooled["M_matched_image"].astype(float)
    mismatch = pooled["M_mismatched_image"].astype(float)
    label_to_idx = {str(label): idx for idx, label in enumerate(labels["label"])}
    norm_matched = matched / np.maximum(np.linalg.norm(matched, axis=1, keepdims=True), 1e-12)
    norm_mismatch = mismatch / np.maximum(np.linalg.norm(mismatch, axis=1, keepdims=True), 1e-12)
    distance = 1.0 - np.clip(norm_mismatch @ norm_matched.T, -1.0, 1.0)
    rows = []
    for idx, row in labels.iterrows():
        label = str(row["label"])
        source = str(row.get("mismatch_label", ""))
        source_idx = label_to_idx.get(source)
        target_distance = float(distance[idx, idx])
        source_distance = float(distance[idx, source_idx]) if source_idx is not None else np.nan
        nearest_idx = int(np.argmin(distance[idx]))
        rows.append(
            {
                "label": label,
                "mismatch_label": source,
                "target_distance": target_distance,
                "source_distance": source_distance,
                "source_attraction": target_distance - source_distance if np.isfinite(source_distance) else np.nan,
                "nearest_label": str(labels.loc[nearest_idx, "label"]),
                "text_retained": nearest_idx == idx,
                "source_assigned": source_idx is not None and nearest_idx == source_idx,
            }
        )
    return pd.DataFrame(rows)


def write_report(path: Path, dataset: str, summaries: dict[str, Any]) -> None:
    ensure_parent(path)
    lines = [
        f"# {dataset} scope-extension geometry",
        "",
        f"- Labels: {summaries['label_count']}",
        f"- Conditions: {', '.join(summaries['conditions'])}",
        f"- Mid-to-late layer fraction: {summaries['mid_to_late_fraction']}",
        "",
        "## Key contrasts",
    ]
    for item in summaries["key_contrasts"]:
        lines.append(
            f"- {item['reference']}: matched-text {item['matched_minus_text_only']:+.4f}; "
            f"matched-prompt {item['matched_minus_prompt']:+.4f}; prompt+image-matched {item['prompt_plus_image_minus_matched']:+.4f}"
        )
    if summaries.get("mixture"):
        mix = summaries["mixture"]
        lines.extend(
            [
                "",
                "## Prompt + image mixture",
                f"- Prompt weight: {mix['prompt_weight']:+.4f}",
                f"- Matched-image weight: {mix['matched_image_weight']:+.4f}",
                f"- R2: {mix['r2']:.4f}",
                f"- Prompt/matched predictor Spearman rho: {mix['prompt_matched_spearman']:+.4f}",
            ]
        )
    if summaries.get("mismatch"):
        mismatch = summaries["mismatch"]
        lines.extend(
            [
                "",
                "## Mismatched-image local identity",
                f"- Text-retention rate: {mismatch['text_retention_rate']:.4f}",
                f"- Source-assignment rate: {mismatch['source_assignment_rate']:.4f}",
                f"- Mean source attraction: {mismatch['mean_source_attraction']:+.4f}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute geometry summaries for scope-extension embeddings.")
    parser.add_argument("--dataset", required=True, choices=["imsitu", "mitstates"])
    parser.add_argument("--stem", default="")
    parser.add_argument("--mid-to-late-fraction", type=float, default=0.5)
    args = parser.parse_args()

    out_dir = ROOT / "outputs" / "scope_extensions"
    arrays, metadata, labels = load_bundle(args.dataset, args.stem)
    layers = condition_layers(arrays, metadata)
    pooled = pooled_by_condition(layers, args.mid_to_late_fraction)
    rdms = {condition: condensed_cosine_distance(matrix) for condition, matrix in pooled.items()}
    refs = build_reference_rdms(args.dataset, labels)

    pr_df = layerwise_pr(layers, args.mid_to_late_fraction)
    pr_summary = (
        pr_df[pr_df["in_mid_to_late_band"]]
        .groupby("condition", as_index=False)["participation_ratio"]
        .mean()
        .rename(columns={"participation_ratio": "mid_to_late_pr"})
    )
    pr_summary.insert(0, "dataset", args.dataset)

    rsa_rows = []
    for condition, rdm in rdms.items():
        for reference, ref_rdm in refs.items():
            rsa_rows.append({"dataset": args.dataset, "condition": condition, "reference": reference, "rsa": spearman_corr(rdm, ref_rdm)})
    rsa_df = pd.DataFrame(rsa_rows)

    contrast_rows = []
    key_contrasts = []
    for reference in refs:
        values = {row.condition: float(row.rsa) for row in rsa_df[rsa_df["reference"].eq(reference)].itertuples()}
        matched = values.get("M_matched_image", np.nan)
        text_only = values.get("M_text_only", np.nan)
        prompt = values.get("T_prompt_primary", np.nan)
        prompt_plus = values.get("M_prompt_plus_matched_image", np.nan)
        row = {
            "dataset": args.dataset,
            "reference": reference,
            "matched_minus_text_only": matched - text_only,
            "matched_minus_prompt": matched - prompt,
            "prompt_plus_image_minus_matched": prompt_plus - matched,
        }
        contrast_rows.append(row)
        key_contrasts.append(row)
    contrast_df = pd.DataFrame(contrast_rows)

    mixture = {}
    if {"T_prompt_primary", "M_matched_image", "M_prompt_plus_matched_image"}.issubset(rdms):
        mixture = regression_mixture(rdms["T_prompt_primary"], rdms["M_matched_image"], rdms["M_prompt_plus_matched_image"])

    mismatch_df = mismatch_attraction(labels, pooled)
    mismatch_summary = {}
    if not mismatch_df.empty:
        mismatch_summary = {
            "text_retention_rate": float(mismatch_df["text_retained"].mean()),
            "source_assignment_rate": float(mismatch_df["source_assigned"].mean()),
            "mean_source_attraction": float(mismatch_df["source_attraction"].mean()),
        }

    stem = args.stem or f"{args.dataset}_qwen_scope_embeddings"
    pr_df.to_csv(out_dir / f"{args.dataset}_geometry_pr_layerwise.csv", index=False)
    pr_summary.to_csv(out_dir / f"{args.dataset}_geometry_condition_summary.csv", index=False)
    rsa_df.to_csv(out_dir / f"{args.dataset}_geometry_reference_rsa.csv", index=False)
    contrast_df.to_csv(out_dir / f"{args.dataset}_geometry_contrasts.csv", index=False)
    if not mismatch_df.empty:
        mismatch_df.to_csv(out_dir / f"{args.dataset}_geometry_mismatch_attraction.csv", index=False)
    summary = {
        "dataset": args.dataset,
        "embedding_stem": stem,
        "label_count": int(len(labels)),
        "conditions": sorted(rdms),
        "mid_to_late_fraction": args.mid_to_late_fraction,
        "key_contrasts": key_contrasts,
        "mixture": mixture,
        "mismatch": mismatch_summary,
    }
    write_json(out_dir / f"{args.dataset}_geometry_summary.json", summary)
    write_report(ROOT / "reports" / "main_results" / f"{args.dataset}_scope_geometry_report.md", args.dataset, summary)
    append_run_log("Scope Extension Geometry", [f"Computed {args.dataset} geometry summaries from {stem}."])


if __name__ == "__main__":
    main()
