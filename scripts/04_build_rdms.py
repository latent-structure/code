from __future__ import annotations

import argparse
import json

import numpy as np

from common import ROOT, append_run_log, condensed_cosine_distance, embeddings_path, ensure_parent, rdm_path, write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.parse_args()

    metadata = json.loads(embeddings_path("embedding_metadata_full.json").read_text(encoding="utf-8"))
    embeddings = np.load(embeddings_path("pooled_embeddings_full.npz"))

    rdm_vectors = {}
    index_rows = []
    for record in metadata["records"]:
        matrix = embeddings[f"record_{record['record_id']}"]
        rdm_vectors[f"record_{record['record_id']}"] = condensed_cosine_distance(matrix).astype(np.float32)
        index_rows.append(
            {
                "record_id": record["record_id"],
                "family": record["family"],
                "model_id": record["model_id"],
                "condition": record["condition"],
                "domain": record["domain"],
                "layer": record["layer"],
                "num_concepts": record["num_concepts"],
            }
        )

    rdm_npz_path = rdm_path("layerwise_rdms_full.npz")
    ensure_parent(rdm_npz_path)
    tmp_rdm_npz_path = rdm_npz_path.with_suffix(".tmp.npz")
    if tmp_rdm_npz_path.exists():
        tmp_rdm_npz_path.unlink()
    # RDM bundles are large; uncompressed atomic writes avoid slow CPU-bound
    # recompression and prevent corrupt final files after interrupted jobs.
    np.savez(tmp_rdm_npz_path, **rdm_vectors)
    tmp_rdm_npz_path.replace(rdm_npz_path)
    write_csv(
        rdm_path("rdm_index_full.csv"),
        index_rows,
        ["record_id", "family", "model_id", "condition", "domain", "layer", "num_concepts"],
    )
    append_run_log(
        "Full RDMs",
        [
            f"Wrote layerwise RDMs to {rdm_path('layerwise_rdms_full.npz').relative_to(ROOT)}.",
            f"Wrote RDM index to {rdm_path('rdm_index_full.csv').relative_to(ROOT)}.",
            f"Computed {len(index_rows)} RDM vectors.",
        ],
    )


if __name__ == "__main__":
    main()
