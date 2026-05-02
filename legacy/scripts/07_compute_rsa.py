from __future__ import annotations

import argparse
import json

import numpy as np

from common import (
    ROOT,
    append_run_log,
    canonical_condition_name,
    load_project_config,
    metrics_path,
    percentile_interval,
    rdm_path,
    read_csv,
    spearman_corr,
    write_csv,
)


def bootstrap_scores(rdm: np.ndarray, anchor: np.ndarray, resamples: int, seed: int) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    n = len(rdm)
    samples = []
    for _ in range(resamples):
        idx = rng.integers(0, n, size=n)
        samples.append(spearman_corr(rdm[idx], anchor[idx]))
    values = np.asarray(samples, dtype=float)
    return values, spearman_corr(rdm, anchor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    args = parser.parse_args()

    cfg = load_project_config(args.config)
    rdms = np.load(rdm_path("layerwise_rdms_full.npz"))
    index_rows = read_csv(rdm_path("rdm_index_full.csv"))
    probe_summary = json.loads((ROOT / "outputs/logs/model_probe_summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((ROOT / "outputs/embeddings/embedding_metadata_full.json").read_text(encoding="utf-8"))
    bootstrap_resamples = cfg["analysis"]["budgets"]["bootstrap_resamples"]
    ci_level = cfg["analysis"]["analysis"]["ci_level"]
    anchor_model_id = next(
        (row["model_id"] for row in metadata["selected_models"] if row["family"] == "anchor"),
        "",
    )

    by_key = {
        (row["family"], row["model_id"], row["condition"], row["domain"], int(row["layer"])): int(row["record_id"])
        for row in index_rows
    }
    anchor_records = {
        (row["domain"], int(row["layer"])): int(row["record_id"])
        for row in index_rows
        if row["family"] == "anchor"
        and canonical_condition_name(row["condition"]) in {"anchor_image", "reference_anchor_image"}
    }

    rows = []
    for row in index_rows:
        if row["family"] == "anchor":
            continue
        anchor_id = anchor_records.get((row["domain"], int(row["layer"])))
        if anchor_id is None:
            continue
        record_id = by_key[(row["family"], row["model_id"], row["condition"], row["domain"], int(row["layer"]))]
        boot, aggregate = bootstrap_scores(
            rdms[f"record_{record_id}"],
            rdms[f"record_{anchor_id}"],
            bootstrap_resamples,
            20260421 + int(row["layer"]),
        )
        ci_low, ci_high = percentile_interval(boot, ci_level)
        rows.append(
            {
                "family": row["family"],
                "model_id": row["model_id"],
                "condition": row["condition"],
                "domain": row["domain"],
                "layer": int(row["layer"]),
                "anchor_model_id": anchor_model_id,
                "rsa_score": aggregate,
                "bootstrap_id": "aggregate",
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
        for idx, value in enumerate(boot):
            rows.append(
                {
                    "family": row["family"],
                    "model_id": row["model_id"],
                    "condition": row["condition"],
                    "domain": row["domain"],
                    "layer": int(row["layer"]),
                    "anchor_model_id": anchor_model_id,
                    "rsa_score": value,
                    "bootstrap_id": idx,
                    "ci_low": "",
                    "ci_high": "",
                }
            )

    write_csv(
        metrics_path("layerwise_alignment_full.csv"),
        rows,
        ["family", "model_id", "condition", "domain", "layer", "anchor_model_id", "rsa_score", "bootstrap_id", "ci_low", "ci_high"],
    )
    append_run_log(
        "Full RSA",
        [
            f"Wrote alignment metrics to {metrics_path('layerwise_alignment_full.csv').relative_to(ROOT)}.",
            f"Bootstrap resamples per aggregate score: {bootstrap_resamples}.",
            f"Locked model summary source: {probe_summary['runtime_python']}.",
        ],
    )


if __name__ == "__main__":
    main()
