from __future__ import annotations

import argparse
import csv
import json
import os
from io import BytesIO
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy.io as sio

from common import ROOT, append_run_log, ensure_parent, load_project_config, output_path, read_csv, spearman_corr, write_csv, write_json
from hardening_common import resolve_cached_snapshot


THINGS_BEHAVIOR_MATRIX = ROOT / "THINGS-behavior" / "osfstorage" / "data" / "spose_similarity.mat"
THINGS_BEHAVIOR_ORDER = ROOT / "THINGS-behavior" / "osfstorage" / "variables" / "unique_id.txt"
SIMLEX_PATH = ROOT / "data" / "anchors" / "simlex999" / "SimLex-999.txt"
THINGS_METADATA = ROOT / "data" / "concepts" / "things" / "concepts-metadata_things.tsv"
THINGS_IMAGE_DIR = ROOT / "THINGS-database" / "osfstorage" / "object_images"
THINGS_IMAGES_ZIP = ROOT / "THINGS-database" / "osfstorage" / "images_THINGS.zip"
THINGS_ZIP_PASSWORD = b"things4all"


def resolve_model_source(model_id: str) -> str:
    cache_root = os.environ.get("HF_HOME", ".cache/hf")
    try:
        return str(resolve_cached_snapshot(model_id, cache_root))
    except Exception:
        return model_id


def load_ready_sensory_images(concepts: list[str]) -> list[tuple[str, Path]]:
    rows = read_csv(ROOT / "data" / "manifests" / "image_manifest.csv")
    ready = {row["concept"]: ROOT / row["matched_image"] for row in rows if row["status"] == "ready"}
    return [(concept, ready[concept]) for concept in concepts if concept in ready]


def reduce_vector(value) -> np.ndarray:
    array = value.detach().float().cpu().numpy().astype(np.float32)
    if array.ndim <= 1:
        return array
    return array.reshape(-1, array.shape[-1]).mean(axis=0).astype(np.float32)


def normalize_token(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def preferred_members(members: list[str], limit: int) -> list[str]:
    ordered = sorted(members)
    ranked: list[str] = []
    for suffix in ("_01b.", "_01s.", "_01n.", "_01"):
        ranked.extend([member for member in ordered if suffix in member and member not in ranked])
    ranked.extend([member for member in ordered if member not in ranked])
    return ranked[:limit]


def load_things_metadata() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    with THINGS_METADATA.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    by_word = {row["Word"].strip().lower(): row for row in rows}
    by_uid = {normalize_token(row["uniqueID"]): row for row in rows}
    return by_word, by_uid


def collect_image_candidates(
    concept: str,
    metadata: dict[str, str] | None,
    limit: int,
    archive_names: list[str],
) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    tokens = [concept]
    if metadata:
        tokens = [metadata.get("uniqueID", ""), metadata.get("Word", ""), concept]
    for token in tokens:
        token = normalize_token(token)
        if not token:
            continue
        directory = THINGS_IMAGE_DIR / token
        if directory.exists() and directory.is_dir():
            members = [path for path in sorted(directory.iterdir()) if path.is_file()]
            member_names = preferred_members([str(path.relative_to(THINGS_IMAGE_DIR)) for path in members], limit)
            for member_name in member_names:
                source = THINGS_IMAGE_DIR / member_name
                key = str(source.resolve())
                if key in seen:
                    continue
                seen.add(key)
                candidates.append({"source_kind": "filesystem", "source": str(source.relative_to(ROOT))})
        archive_prefix = f"object_images/{token}/"
        archive_members = [name for name in archive_names if name.startswith(archive_prefix) and not name.endswith("/")]
        for member in preferred_members(archive_members, limit):
            key = f"archive:{member}"
            if key in seen:
                continue
            seen.add(key)
            candidates.append({"source_kind": "archive", "source": member})
        if len(candidates) >= limit:
            break
    matched_local = ROOT / "data" / "images" / "sensory" / f"{concept}.jpg"
    if not candidates and matched_local.exists():
        candidates.append({"source_kind": "local_matched", "source": str(matched_local.relative_to(ROOT))})
    return candidates[:limit]


def load_concept_rows(config: dict[str, object]) -> list[dict[str, str]]:
    subset_path = config["analysis"]["execution"].get("default_concept_subset", "")
    if subset_path:
        subset_file = ROOT / subset_path
        if subset_file.exists():
            return read_csv(subset_file)
    return read_csv(ROOT / "data" / "concepts" / "full_concept_list.csv")


def load_default_subset(config: dict[str, object]) -> list[str]:
    subset_path = config["analysis"]["execution"].get("default_concept_subset", "")
    if subset_path:
        path = ROOT / subset_path
        if path.exists():
            return [row["concept"] for row in read_csv(path)]
    return [row["concept"] for row in load_concept_rows()]


def prepare_things_behavioral(concepts: list[str]) -> dict[str, object]:
    if not (THINGS_BEHAVIOR_MATRIX.exists() and THINGS_BEHAVIOR_ORDER.exists()):
        return {"available": False, "matched_concepts": [], "missing_concepts": concepts}

    order = [line.strip() for line in THINGS_BEHAVIOR_ORDER.read_text(encoding="utf-8").splitlines() if line.strip()]
    matrix = sio.loadmat(THINGS_BEHAVIOR_MATRIX)["spose_sim"]
    index = {name: idx for idx, name in enumerate(order)}
    matched = [concept for concept in concepts if concept in index]
    missing = [concept for concept in concepts if concept not in index]
    if matched:
        idx = [index[concept] for concept in matched]
        subset = np.asarray(matrix[np.ix_(idx, idx)], dtype=np.float32)
        np.save(output_path("data", "anchors", "things_behavioral_similarity.npy"), subset)
        ensure_parent(output_path("data", "anchors", "things_behavioral_concepts.json"))
        output_path("data", "anchors", "things_behavioral_concepts.json").write_text(json.dumps(matched, indent=2), encoding="utf-8")
    return {"available": True, "matched_concepts": matched, "missing_concepts": missing}


def prepare_simlex_anchor(abstract_concepts: list[str]) -> dict[str, object]:
    if not SIMLEX_PATH.exists():
        return {"available": False, "matched_concepts": [], "pair_count": 0}

    rows = []
    with SIMLEX_PATH.open("r", encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            rows.append((parts[0].lower(), parts[1].lower(), float(parts[3])))

    overlap = sorted({concept for concept in abstract_concepts if any(concept == left or concept == right for left, right, _ in rows)})
    pair_rows = []
    score_map = {(left, right): score for left, right, score in rows}
    score_map.update({(right, left): score for left, right, score in rows})
    for left, right in combinations(overlap, 2):
        if (left, right) in score_map:
            pair_rows.append({"left": left, "right": right, "simlex_score": score_map[(left, right)]})
    write_csv(output_path("data", "anchors", "simlex_overlap_pairs.csv"), pair_rows, ["left", "right", "simlex_score"])
    return {"available": True, "matched_concepts": overlap, "pair_count": len(pair_rows)}


def prepare_image_model_anchor(concepts: list[str], model_id: str, anchor_slug: str) -> dict[str, object]:
    image_rows = load_ready_sensory_images(concepts)
    if not image_rows:
        return {"available": False, "matched_concepts": [], "missing_concepts": concepts, "reason": "no_ready_images"}

    try:
        from PIL import Image
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except Exception as exc:
        return {"available": False, "matched_concepts": [], "missing_concepts": concepts, "reason": f"dependency_missing:{type(exc).__name__}"}

    load_id = resolve_model_source(model_id)

    try:
        processor = AutoImageProcessor.from_pretrained(load_id, local_files_only=True)
        model = AutoModel.from_pretrained(load_id, local_files_only=True).eval()
    except Exception as exc:
        return {"available": False, "matched_concepts": [], "missing_concepts": concepts, "reason": f"load_failed:{type(exc).__name__}"}

    by_word, by_uid = load_things_metadata()
    prototype_limit = int(load_project_config()["analysis"]["image_policy"].get("prototype_images_per_concept", 3))
    archive_names: list[str] = []
    archive = None
    if THINGS_IMAGES_ZIP.exists():
        import zipfile

        archive = zipfile.ZipFile(THINGS_IMAGES_ZIP)
        archive_names = archive.namelist()

    device = next(model.parameters()).device
    embeddings = []
    ordered = []
    prototype_rows = []
    with torch.no_grad():
        try:
            for concept, image_path in image_rows:
                metadata = by_word.get(concept.lower()) or by_uid.get(normalize_token(concept))
                candidates = collect_image_candidates(concept, metadata, prototype_limit, archive_names)
                vectors = []
                selected_sources = []
                for candidate in candidates:
                    if candidate["source_kind"] == "archive":
                        try:
                            payload = archive.read(candidate["source"], pwd=THINGS_ZIP_PASSWORD)
                        except Exception:
                            continue
                        image = Image.open(BytesIO(payload)).convert("RGB")
                    else:
                        source_path = ROOT / candidate["source"]
                        if not source_path.exists():
                            continue
                        image = Image.open(source_path).convert("RGB")
                    batch = processor(images=image, return_tensors="pt")
                    batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
                    if hasattr(model, "get_image_features"):
                        vector = reduce_vector(model.get_image_features(**batch)[0])
                    else:
                        outputs = model(**batch)
                        if getattr(outputs, "pooler_output", None) is not None:
                            vector = reduce_vector(outputs.pooler_output[0])
                        elif getattr(outputs, "last_hidden_state", None) is not None:
                            hidden = outputs.last_hidden_state[0]
                            vector = reduce_vector(hidden)
                        elif hasattr(model, "vision_model"):
                            vision_outputs = model.vision_model(**batch)
                            hidden = vision_outputs.last_hidden_state[0]
                            vector = reduce_vector(hidden)
                        else:
                            raise RuntimeError(f"{model_id} did not expose a supported image representation")
                    vectors.append(np.asarray(vector, dtype=np.float32))
                    selected_sources.append(f"{candidate['source_kind']}:{candidate['source']}")
                if not vectors:
                    continue
                embeddings.append(np.mean(np.stack(vectors).astype(np.float32), axis=0).astype(np.float32))
                ordered.append(concept)
                prototype_rows.append(
                    {
                        "anchor_slug": anchor_slug,
                        "concept": concept,
                        "prototype_image_count": len(selected_sources),
                        "selected_sources": " | ".join(selected_sources),
                        "fallback_local_image": str(image_path.relative_to(ROOT)),
                    }
                )
        finally:
            if archive is not None:
                archive.close()

    ensure_parent(output_path("data", "anchors", f"{anchor_slug}_embeddings.npy"))
    np.save(output_path("data", "anchors", f"{anchor_slug}_embeddings.npy"), np.stack(embeddings).astype(np.float32))
    output_path("data", "anchors", f"{anchor_slug}_concepts.json").write_text(json.dumps(ordered, indent=2), encoding="utf-8")
    write_csv(
        output_path("data", "anchors", f"{anchor_slug}_prototype_manifest.csv"),
        prototype_rows,
        ["anchor_slug", "concept", "prototype_image_count", "selected_sources", "fallback_local_image"],
    )
    return {
        "available": True,
        "matched_concepts": ordered,
        "missing_concepts": sorted(set(concepts) - set(ordered)),
        "reason": f"ok:{anchor_slug}:prototype_limit={prototype_limit}",
    }


def build_anchor_inventory(
    config: dict[str, object],
    concept_rows: list[dict[str, str]],
    things_behavior: dict[str, object],
    simlex: dict[str, object],
    dinov2: dict[str, object],
    clip: dict[str, object],
) -> dict[str, object]:
    sensory = [row["concept"] for row in concept_rows if row["domain"] == "sensory"]
    abstract = [row["concept"] for row in concept_rows if row["domain"] == "abstract"]
    return {
        "backbone": {
            "text": config["analysis"]["execution"]["sensory_backbone_text_model"],
            "multimodal": config["analysis"]["execution"]["sensory_backbone_multimodal_model"],
        },
        "anchors": [
            {
                "anchor_name": "THINGS behavioral similarity",
                "anchor_type": "human_behavioral",
                "available": things_behavior["available"],
                "matched_concepts": len(things_behavior["matched_concepts"]),
                "missing_concepts": things_behavior["missing_concepts"],
                "required_for_core": True,
            },
            {
                "anchor_name": "Lancaster Sensorimotor Norms",
                "anchor_type": "human_lexical",
                "available": bool(config["anchors"]["human"]["lancaster_sensorimotor"]["matrix_path"]) and (ROOT / config["anchors"]["human"]["lancaster_sensorimotor"]["matrix_path"]).exists(),
                "matched_concepts": len(sensory) if bool(config["anchors"]["human"]["lancaster_sensorimotor"]["matrix_path"]) and (ROOT / config["anchors"]["human"]["lancaster_sensorimotor"]["matrix_path"]).exists() else 0,
                "missing_concepts": [] if bool(config["anchors"]["human"]["lancaster_sensorimotor"]["matrix_path"]) and (ROOT / config["anchors"]["human"]["lancaster_sensorimotor"]["matrix_path"]).exists() else sensory,
                "required_for_core": False,
            },
            {
                "anchor_name": "SimLex-999",
                "anchor_type": "human_lexical",
                "available": simlex["available"],
                "matched_concepts": len(simlex["matched_concepts"]),
                "missing_concepts": sorted(set(abstract) - set(simlex["matched_concepts"])),
                "required_for_core": False,
            },
            {
                "anchor_name": "DINOv2",
                "anchor_type": "pure_vision",
                "available": dinov2["available"],
                "matched_concepts": len(dinov2["matched_concepts"]),
                "missing_concepts": dinov2["missing_concepts"],
                "notes": dinov2.get("reason", ""),
                "required_for_core": True,
            },
            {
                "anchor_name": "SigLIP2",
                "anchor_type": "vision_language",
                "available": True,
                "matched_concepts": len(sensory),
                "missing_concepts": [],
                "required_for_core": True,
            },
            {
                "anchor_name": "CLIP ViT-L/14",
                "anchor_type": "vision_language",
                "available": clip["available"],
                "matched_concepts": len(clip["matched_concepts"]),
                "missing_concepts": clip["missing_concepts"],
                "notes": clip.get("reason", ""),
                "required_for_core": False,
            },
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/analysis.yaml")
    args = parser.parse_args()

    config = load_project_config(args.config)
    concept_rows = load_concept_rows(config)
    subset = set(load_default_subset(config))
    active_rows = [row for row in concept_rows if row["concept"] in subset]
    sensory = [row["concept"] for row in active_rows if row["domain"] == "sensory"]
    abstract = [row["concept"] for row in active_rows if row["domain"] == "abstract"]

    things_behavior = prepare_things_behavioral(sensory)
    simlex = prepare_simlex_anchor(abstract)
    dinov2 = prepare_image_model_anchor(sensory, "facebook/dinov2-large", "dinov2")
    clip = prepare_image_model_anchor(sensory, "openai/clip-vit-large-patch14", "clip_vitl14")
    inventory = build_anchor_inventory(config, active_rows, things_behavior, simlex, dinov2, clip)

    write_json(output_path("data", "anchors", "anchor_inventory.json"), inventory)
    append_run_log(
        "Prepare Anchors",
        [
            f"Prepared THINGS behavioral anchor for {len(things_behavior['matched_concepts'])} sensory concepts.",
            f"Prepared SimLex overlap pairs for {simlex['pair_count']} abstract pairs across {len(simlex['matched_concepts'])} concepts.",
            f"DINOv2 anchor available={dinov2['available']} matched={len(dinov2['matched_concepts'])} reason={dinov2.get('reason', '')}.",
            f"CLIP anchor available={clip['available']} matched={len(clip['matched_concepts'])} reason={clip.get('reason', '')}.",
            f"Wrote anchor inventory to {output_path('data', 'anchors', 'anchor_inventory.json').relative_to(ROOT)}.",
        ],
    )


if __name__ == "__main__":
    main()
