from __future__ import annotations

import argparse
import json
from itertools import combinations

import numpy as np

from common import ROOT, append_run_log, condensed_cosine_distance, metrics_path, rdm_path, spearman_corr, write_csv
from hardening_common import load_active_concept_rows


def char_trigram_vector(text: str) -> set[str]:
    padded = f"__{text.lower()}__"
    return {padded[index : index + 3] for index in range(len(padded) - 2)}


def lexical_distance(a: str, b: str) -> float:
    left = char_trigram_vector(a)
    right = char_trigram_vector(b)
    union = left | right
    if not union:
        return 1.0
    return 1.0 - (len(left & right) / len(union))


def proxy_distance(domain: str, concepts: list[str], concept_rows: dict[str, dict[str, str]]) -> np.ndarray:
    values = []
    for left, right in combinations(concepts, 2):
        if domain == "sensory":
            distance = 0.0 if concept_rows[left]["subtype"] == concept_rows[right]["subtype"] else 1.0
        else:
            distance = lexical_distance(left, right)
        values.append(distance)
    return np.asarray(values, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.parse_args()

    concept_rows = {row["concept"]: row for row in load_active_concept_rows(args.config)}
    metadata = json.loads((ROOT / "outputs/embeddings/embedding_metadata_full.json").read_text(encoding="utf-8"))
    rdms = np.load(rdm_path("layerwise_rdms_full.npz"))

    rows = []
    for record in metadata["records"]:
        concepts = record["concepts"]
        human_rdm = proxy_distance(record["domain"], concepts, concept_rows)
        model_rdm = rdms[f"record_{record['record_id']}"]
        rows.append(
            {
                "family": record["family"],
                "model_id": record["model_id"],
                "condition": record["condition"],
                "domain": record["domain"],
                "layer": record["layer"],
                "anchor_source": "proxy_subtype_distance" if record["domain"] == "sensory" else "proxy_lexical_distance",
                "rsa_score": spearman_corr(model_rdm, human_rdm),
                "num_concepts": record["num_concepts"],
            }
        )

    write_csv(
        metrics_path("human_anchor_alignment.csv"),
        rows,
        ["family", "model_id", "condition", "domain", "layer", "anchor_source", "rsa_score", "num_concepts"],
    )
    append_run_log(
        "Human Alignment",
        [
            f"Wrote human-anchor alignment metrics to {metrics_path('human_anchor_alignment.csv').relative_to(ROOT)}.",
            "Used proxy_subtype_distance for sensory rows and proxy_lexical_distance for abstract rows because external human-anchor files are not yet integrated.",
        ],
    )


if __name__ == "__main__":
    main()
