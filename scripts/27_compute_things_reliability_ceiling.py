from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from common import ROOT, append_run_log, condensed_cosine_distance, metrics_path, output_path, spearman_corr, write_csv, write_json
from hardening_common import write_text


TRIPLET_DIR = ROOT / "THINGS-behavior" / "osfstorage" / "data" / "triplet_dataset"
SPLIT_HALF_RDM = ROOT / "THINGS-behavior" / "osfstorage" / "data" / "RDM48_triplet_splithalf.mat"
WORDS48 = ROOT / "THINGS-behavior" / "osfstorage" / "variables" / "words48.mat"


def read_triplets(path: Path, limit: int = 0) -> list[tuple[int, int, int]]:
    rows: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = [int(value) for value in line.strip().split()]
            if len(parts) < 3:
                continue
            rows.append((parts[0], parts[1], parts[2]))
            if limit and len(rows) >= limit:
                break
    return rows


def triplet_choice_key(row: tuple[int, int, int]) -> tuple[tuple[int, int, int], int]:
    sorted_triplet = tuple(sorted(row))
    odd_one_out = row[2]
    choice_position = sorted_triplet.index(odd_one_out) + 1
    return sorted_triplet, choice_position


def triplet_noise_ceiling(paths: list[Path], limit_per_file: int = 0) -> dict[str, Any]:
    grouped: dict[tuple[int, int, int], Counter[int]] = defaultdict(Counter)
    total_rows = 0
    for path in paths:
        for row in read_triplets(path, limit_per_file):
            key, choice = triplet_choice_key(row)
            grouped[key][choice] += 1
            total_rows += 1
    consistencies = []
    repeated = 0
    for counts in grouped.values():
        n = sum(counts.values())
        if n < 2:
            continue
        repeated += 1
        consistencies.append(max(counts.values()) / n)
    values = np.asarray(consistencies, dtype=float)
    return {
        "triplet_files": [str(path.relative_to(ROOT)) for path in paths],
        "num_rows": total_rows,
        "num_unique_triplets": len(grouped),
        "num_repeated_triplets": repeated,
        "mean_consistency": 0.0 if len(values) == 0 else float(values.mean()),
        "ci95": 0.0 if len(values) <= 1 else float(1.96 * values.std(ddof=1) / np.sqrt(len(values))),
    }


def mat_strings(values: np.ndarray) -> list[str]:
    out = []
    for value in values.ravel():
        if isinstance(value, np.ndarray):
            out.append(str(value.item()))
        else:
            out.append(str(value))
    return [item.lower() for item in out]


def split_half_rdm_reliability() -> dict[str, Any]:
    payload = loadmat(SPLIT_HALF_RDM)
    split1 = np.asarray(payload["RDM48_triplet_split1"], dtype=float)
    split2 = np.asarray(payload["RDM48_triplet_split2"], dtype=float)
    idx = np.triu_indices(split1.shape[0], k=1)
    words = mat_strings(loadmat(WORDS48)["words48"])
    return {
        "source": str(SPLIT_HALF_RDM.relative_to(ROOT)),
        "num_concepts": int(split1.shape[0]),
        "num_pairs": int(len(idx[0])),
        "spearman_rho": spearman_corr(split1[idx], split2[idx]),
        "concepts": words,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    parser.add_argument("--limit-per-file", type=int, default=0, help="Optional smoke-test row limit per triplet file.")
    args = parser.parse_args()

    triplet_paths = [
        TRIPLET_DIR / "testset1.txt",
        TRIPLET_DIR / "testset2.txt",
        TRIPLET_DIR / "testset2_repeat.txt",
        TRIPLET_DIR / "testset3.txt",
    ]
    payload = {
        "full_triplet_noise_ceiling": triplet_noise_ceiling(triplet_paths, args.limit_per_file),
        "rdm48_split_half_reliability": split_half_rdm_reliability(),
        "full_1854_split_half_rdm_available": False,
        "interpretation": (
            "The full triplet estimate summarizes repeated odd-one-out consistency. "
            "The available split-half RDM ceiling is limited to the 48-concept THINGS triplet subset; "
            "full 1,854-concept RSA values should therefore be interpreted comparatively rather than ceiling-normalized."
        ),
    }
    suffix = "_smoke" if args.limit_per_file else ""
    write_json(metrics_path(f"things_reliability_ceiling{suffix}.json"), payload)
    rows = [
        {
            "analysis": "full_triplet_noise_ceiling",
            "num_concepts": 1854,
            "num_pairs_or_triplets": payload["full_triplet_noise_ceiling"]["num_repeated_triplets"],
            "estimate": payload["full_triplet_noise_ceiling"]["mean_consistency"],
            "ci95": payload["full_triplet_noise_ceiling"]["ci95"],
            "scope": "repeated odd-one-out triplets",
        },
        {
            "analysis": "rdm48_split_half_reliability",
            "num_concepts": payload["rdm48_split_half_reliability"]["num_concepts"],
            "num_pairs_or_triplets": payload["rdm48_split_half_reliability"]["num_pairs"],
            "estimate": payload["rdm48_split_half_reliability"]["spearman_rho"],
            "ci95": "",
            "scope": "48-concept split-half RDM",
        },
    ]
    write_csv(metrics_path(f"things_reliability_ceiling{suffix}.csv"), rows, ["analysis", "num_concepts", "num_pairs_or_triplets", "estimate", "ci95", "scope"])
    lines = [
        "# THINGS Reliability Ceiling Report",
        "",
        f"- Full triplet repeated-choice consistency: `{payload['full_triplet_noise_ceiling']['mean_consistency']:.4f}` +/- `{payload['full_triplet_noise_ceiling']['ci95']:.4f}`",
        f"- 48-concept split-half RDM Spearman rho: `{payload['rdm48_split_half_reliability']['spearman_rho']:.4f}`",
        "- No full 1,854-concept split-half RDM was available locally; full-set RSA values are comparative rather than ceiling-normalized.",
    ]
    write_text(output_path("reports", "main_results", f"things_reliability_ceiling_report{suffix}.md"), "\n".join(lines))
    append_run_log("THINGS Reliability Ceiling", [f"Wrote reliability ceiling outputs with suffix `{suffix}`."])


if __name__ == "__main__":
    main()
