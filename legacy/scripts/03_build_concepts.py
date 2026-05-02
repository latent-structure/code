from __future__ import annotations

from common import ROOT, append_run_log, load_project_config, set_global_seed, write_csv


SENSORY_CANDIDATES = [
    ("rainbow", "appearance_color", "things", "iconic color arc", "low", "planned_things_image", "good", "yes", True),
    ("sunset", "appearance_color", "things", "strong color-gradient scene", "low", "planned_things_image", "good", "yes", True),
    ("shadow", "appearance_color", "thingsplus", "clear contrast-defined visual cue", "medium", "planned_things_image", "good", "yes", True),
    ("marble", "appearance_color", "things", "distinctive veined appearance", "medium", "planned_things_image", "good", "yes", True),
    ("glitter", "appearance_color", "thingsplus", "reflective sparkle cue", "low", "planned_things_image", "good", "yes", True),
    ("crystal", "appearance_color", "things", "transparent refractive visual cue", "medium", "planned_things_image", "good", "yes", True),
    ("amber", "appearance_color", "thingsplus", "strong warm color identity", "medium", "planned_things_image", "good", "yes", True),
    ("neon", "appearance_color", "thingsplus", "high-salience glow cue", "medium", "planned_things_image", "good", "yes", True),
    ("silver", "appearance_color", "things", "distinctive metallic sheen", "medium", "planned_things_image", "good", "yes", True),
    ("gold", "appearance_color", "things", "stable metallic color cue", "medium", "planned_things_image", "good", "yes", True),
    ("ruby", "appearance_color", "thingsplus", "clear saturated gemstone color", "low", "planned_things_image", "good", "yes", True),
    ("emerald", "appearance_color", "thingsplus", "clear gemstone color identity", "low", "planned_things_image", "good", "yes", True),
    ("sapphire", "appearance_color", "thingsplus", "clear gemstone appearance", "low", "planned_things_image", "good", "yes", True),
    ("coral", "appearance_color", "things", "distinctive textured appearance", "medium", "planned_things_image", "good", "yes", True),
    ("flame", "appearance_color", "things", "highly salient visual phenomenon", "low", "planned_things_image", "good", "yes", True),
    ("mist", "appearance_color", "thingsplus", "too scene-dependent for canonical matching", "medium", "planned_things_image", "fair", "partial", False),
    ("aurora", "appearance_color", "thingsplus", "image scenes are spectacular but too variable", "medium", "planned_things_image", "fair", "partial", False),
    ("mirror", "appearance_color", "things", "same-object-family collision risk", "high", "planned_things_image", "good", "no", False),
    ("sandpaper", "texture_material", "things", "coarse texture cue", "low", "planned_things_image", "good", "yes", True),
    ("fur", "texture_material", "things", "high-salience texture cue", "medium", "planned_things_image", "good", "yes", True),
    ("foam", "texture_material", "things", "distinctive bubbly surface", "low", "planned_things_image", "good", "yes", True),
    ("silk", "texture_material", "things", "smooth reflective textile", "low", "planned_things_image", "good", "yes", True),
    ("mud", "texture_material", "things", "coarse wet material cue", "low", "planned_things_image", "good", "yes", True),
    ("ice", "texture_material", "things", "crisp material and surface cue", "low", "planned_things_image", "good", "yes", True),
    ("velvet", "texture_material", "things", "soft dense textile cue", "low", "planned_things_image", "good", "yes", True),
    ("leather", "texture_material", "things", "clear material identity", "low", "planned_things_image", "good", "yes", True),
    ("wool", "texture_material", "things", "distinctive fabric texture", "low", "planned_things_image", "good", "yes", True),
    ("cotton", "texture_material", "things", "clear tactile material cue", "medium", "planned_things_image", "good", "yes", True),
    ("clay", "texture_material", "things", "stable material cue", "low", "planned_things_image", "good", "yes", True),
    ("gravel", "texture_material", "thingsplus", "rough particulate texture", "low", "planned_things_image", "good", "yes", True),
    ("moss", "texture_material", "things", "visually and tactilely distinctive", "medium", "planned_things_image", "good", "yes", True),
    ("bark", "texture_material", "things", "canonical rough-surface cue", "low", "planned_things_image", "good", "yes", True),
    ("sponge", "texture_material", "things", "canonical porous texture", "low", "planned_things_image", "good", "yes", True),
    ("chalk", "texture_material", "thingsplus", "image quality varies across examples", "medium", "planned_things_image", "fair", "partial", False),
    ("tar", "texture_material", "thingsplus", "too scene-dependent and messy", "medium", "planned_things_image", "fair", "partial", False),
    ("bell", "sound_linked", "things", "canonical ringing object", "low", "planned_things_image", "good", "yes", True),
    ("thunder", "sound_linked", "thingsplus", "canonical sound-linked event", "low", "planned_things_image", "good", "yes", True),
    ("fireworks", "sound_linked", "things", "sound-linked event with clear visuals", "low", "planned_things_image", "good", "yes", True),
    ("whistle", "sound_linked", "things", "clear sound-producing object", "medium", "planned_things_image", "good", "yes", True),
    ("drum", "sound_linked", "things", "canonical sound-producing instrument", "low", "planned_things_image", "good", "yes", True),
    ("siren", "sound_linked", "things", "clear warning-sound object", "low", "planned_things_image", "good", "yes", True),
    ("trumpet", "sound_linked", "things", "canonical sound-producing instrument", "low", "planned_things_image", "good", "yes", True),
    ("gong", "sound_linked", "thingsplus", "clear resonance-linked object", "low", "planned_things_image", "good", "yes", True),
    ("cymbal", "sound_linked", "thingsplus", "clear percussive object", "low", "planned_things_image", "good", "yes", True),
    ("chime", "sound_linked", "thingsplus", "clear ringing object cue", "low", "planned_things_image", "good", "yes", True),
    ("alarm", "sound_linked", "things", "clear sound-linked object", "medium", "planned_things_image", "good", "yes", True),
    ("engine", "sound_linked", "things", "strong auditory association", "medium", "planned_things_image", "good", "yes", True),
    ("waterfall", "sound_linked", "things", "strong event-like sound cue", "low", "planned_things_image", "good", "yes", True),
    ("applause", "sound_linked", "thingsplus", "sound event but image canonicality is weaker", "medium", "planned_things_image", "fair", "partial", True),
    ("rooster", "sound_linked", "things", "iconic sound-linked animal", "low", "planned_things_image", "good", "yes", True),
    ("violin", "sound_linked", "things", "sound-linked instrument overlaps less cleanly than retained set", "low", "planned_things_image", "good", "yes", False),
    ("speaker", "sound_linked", "things", "broad device family and ambiguity risk", "medium", "planned_things_image", "good", "no", False),
    ("coffee", "smell_taste_proxy", "things", "canonical aroma cue", "medium", "planned_things_image", "good", "yes", True),
    ("garlic", "smell_taste_proxy", "things", "clear smell cue", "low", "planned_things_image", "good", "yes", True),
    ("cinnamon", "smell_taste_proxy", "things", "clear spice aroma cue", "low", "planned_things_image", "good", "yes", True),
    ("smoke", "smell_taste_proxy", "thingsplus", "clear smell-linked phenomenon", "medium", "planned_things_image", "fair", "yes", True),
    ("lemon", "smell_taste_proxy", "things", "canonical smell/taste cue", "low", "planned_things_image", "good", "yes", True),
    ("peppermint", "smell_taste_proxy", "things", "clear smell/taste cue", "low", "planned_things_image", "good", "yes", True),
    ("vinegar", "smell_taste_proxy", "thingsplus", "distinctive smell cue", "low", "planned_things_image", "good", "yes", True),
    ("onion", "smell_taste_proxy", "things", "strong smell cue", "low", "planned_things_image", "good", "yes", True),
    ("cocoa", "smell_taste_proxy", "things", "recognizable taste/smell cue", "low", "planned_things_image", "good", "yes", True),
    ("vanilla", "smell_taste_proxy", "thingsplus", "canonical aroma cue", "low", "planned_things_image", "good", "yes", True),
    ("basil", "smell_taste_proxy", "thingsplus", "clear herb aroma cue", "low", "planned_things_image", "good", "yes", True),
    ("ginger", "smell_taste_proxy", "things", "clear spice cue", "low", "planned_things_image", "good", "yes", True),
    ("orange", "smell_taste_proxy", "things", "canonical smell/taste cue", "low", "planned_things_image", "good", "yes", True),
    ("soap", "smell_taste_proxy", "thingsplus", "distinctive smell-linked object", "medium", "planned_things_image", "good", "yes", True),
    ("perfume", "smell_taste_proxy", "things", "brand variance is manageable but present", "medium", "planned_things_image", "good", "yes", True),
    ("honey", "smell_taste_proxy", "things", "same-family overlap makes it redundant with retained set", "medium", "planned_things_image", "good", "yes", False),
    ("spice", "smell_taste_proxy", "thingsplus", "category too broad", "high", "planned_things_image", "good", "no", False),
]

ABSTRACT_CANDIDATES = [
    ("justice", "abstract", "negative_control", "simlex_overlap", "high-level social norm", "medium", "", "", "partial", True),
    ("theory", "abstract", "negative_control", "simlex_overlap", "familiar knowledge term", "low", "", "", "partial", True),
    ("policy", "abstract", "negative_control", "simlex_overlap", "institutional abstraction", "medium", "", "", "partial", True),
    ("contract", "abstract", "negative_control", "simlex_overlap", "normative relation term", "medium", "", "", "partial", True),
    ("virtue", "abstract", "negative_control", "simlex_overlap", "familiar abstract noun", "medium", "", "", "partial", True),
    ("reason", "abstract", "negative_control", "simlex_overlap", "common abstract noun", "medium", "", "", "partial", True),
    ("budget", "abstract", "negative_control", "simlex_overlap", "familiar institutional abstraction", "low", "", "", "partial", True),
    ("logic", "abstract", "negative_control", "simlex_overlap", "familiar abstract noun", "low", "", "", "partial", True),
    ("ethics", "abstract", "negative_control", "simlex_overlap", "familiar abstract noun", "medium", "", "", "partial", True),
    ("tenure", "abstract", "negative_control", "manual_screen", "familiar institutional abstraction", "medium", "", "", "partial", True),
    ("method", "abstract", "negative_control", "simlex_overlap", "common abstraction", "low", "", "", "partial", True),
    ("status", "abstract", "negative_control", "simlex_overlap", "common relational abstraction", "medium", "", "", "partial", True),
    ("truth", "abstract", "negative_control", "simlex_overlap", "common abstract noun", "low", "", "", "partial", True),
    ("system", "abstract", "negative_control", "manual_screen", "broad but common abstraction", "medium", "", "", "partial", True),
    ("culture", "abstract", "negative_control", "manual_screen", "familiar social abstraction", "medium", "", "", "partial", True),
    ("law", "abstract", "negative_control", "simlex_overlap", "institutional abstraction", "medium", "", "", "partial", True),
    ("duty", "abstract", "negative_control", "simlex_overlap", "normative abstraction", "medium", "", "", "partial", True),
    ("value", "abstract", "negative_control", "simlex_overlap", "abstract noun with manageable ambiguity", "medium", "", "", "partial", True),
    ("belief", "abstract", "negative_control", "manual_screen", "common mental-state abstraction", "medium", "", "", "partial", True),
    ("motive", "abstract", "negative_control", "manual_screen", "common abstract noun", "medium", "", "", "partial", True),
    ("process", "abstract", "negative_control", "simlex_overlap", "broad but usable abstraction", "low", "", "", "partial", True),
    ("strategy", "abstract", "negative_control", "manual_screen", "common planning abstraction", "low", "", "", "partial", True),
    ("fairness", "abstract", "negative_control", "manual_screen", "common evaluative abstraction", "low", "", "", "partial", True),
    ("finance", "abstract", "negative_control", "manual_screen", "institutional abstraction", "medium", "", "", "partial", True),
    ("evidence", "abstract", "negative_control", "simlex_overlap", "common epistemic abstraction", "low", "", "", "partial", True),
    ("principle", "abstract", "negative_control", "simlex_overlap", "common abstract noun", "medium", "", "", "partial", True),
    ("doctrine", "abstract", "negative_control", "manual_screen", "familiar but abstract term", "medium", "", "", "partial", True),
    ("concept", "abstract", "negative_control", "manual_screen", "meta-level abstraction", "low", "", "", "partial", True),
    ("signal", "abstract", "negative_control", "manual_screen", "usable abstract control despite broad usage", "medium", "", "", "partial", True),
    ("agency", "abstract", "negative_control", "manual_screen", "social abstraction", "medium", "", "", "partial", True),
    ("freedom", "abstract", "negative_control", "manual_screen", "too slogan-like and culturally variable", "medium", "", "", "partial", False),
    ("energy", "abstract", "negative_control", "manual_screen", "high polysemy across physics and affect", "high", "", "", "partial", False),
    ("memory", "abstract", "negative_control", "manual_screen", "too imageable relative to target control set", "medium", "", "", "partial", False),
    ("fantasy", "abstract", "negative_control", "manual_screen", "too image-inviting", "medium", "", "", "partial", False),
    ("spirit", "abstract", "negative_control", "manual_screen", "high polysemy", "high", "", "", "partial", False),
    ("style", "abstract", "negative_control", "manual_screen", "too visually grounded for the negative-control role", "medium", "", "", "partial", False),
]


def as_row(entry: tuple[str, ...]) -> dict[str, str]:
    return {
        "concept": entry[0],
        "domain": "sensory",
        "subtype": entry[1],
        "source_dataset": entry[2],
        "notes": entry[3],
        "polysemy_risk": entry[4],
        "image_source": entry[5],
        "image_quality_flag": entry[6],
        "human_anchor_available": entry[7],
    }


def build_rows() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    kept: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    subtype_rows: list[dict[str, str]] = []

    for entry in SENSORY_CANDIDATES:
        row = as_row(entry)
        if entry[8]:
            kept.append(row)
            subtype_rows.append({"concept": row["concept"], "domain": row["domain"], "subtype": row["subtype"]})
        else:
            rejected.append({**row, "rejection_reason": row["notes"]})

    for entry in ABSTRACT_CANDIDATES:
        row = {
            "concept": entry[0],
            "domain": entry[1],
            "subtype": entry[2],
            "source_dataset": entry[3],
            "notes": entry[4],
            "polysemy_risk": entry[5],
            "image_source": entry[6],
            "image_quality_flag": entry[7],
            "human_anchor_available": entry[8],
        }
        if entry[9]:
            kept.append(row)
            subtype_rows.append({"concept": row["concept"], "domain": row["domain"], "subtype": row["subtype"]})
        else:
            rejected.append({**row, "rejection_reason": row["notes"]})

    return kept, rejected, subtype_rows


def main() -> None:
    config = load_project_config()
    set_global_seed(config["seeds"]["global"])
    kept, rejected, subtype_rows = build_rows()
    concept_path = ROOT / "data/concepts/full_concept_list.csv"
    reject_path = ROOT / "data/concepts/concept_rejections.csv"
    subtype_path = ROOT / "data/concepts/concept_subtypes.csv"
    write_csv(
        concept_path,
        kept,
        [
            "concept",
            "domain",
            "subtype",
            "source_dataset",
            "notes",
            "polysemy_risk",
            "image_source",
            "image_quality_flag",
            "human_anchor_available",
        ],
    )
    write_csv(
        reject_path,
        rejected,
        [
            "concept",
            "domain",
            "subtype",
            "source_dataset",
            "notes",
            "polysemy_risk",
            "image_source",
            "image_quality_flag",
            "human_anchor_available",
            "rejection_reason",
        ],
    )
    write_csv(subtype_path, subtype_rows, ["concept", "domain", "subtype"])

    sensory_kept = [row for row in kept if row["domain"] == "sensory"]
    abstract_kept = [row for row in kept if row["domain"] == "abstract"]
    append_run_log(
        "Concept Set",
        [
            f"Wrote {len(sensory_kept)} retained sensory concepts to {concept_path.relative_to(ROOT)}.",
            f"Wrote {len(abstract_kept)} retained abstract concepts to {concept_path.relative_to(ROOT)}.",
            f"Wrote {len(rejected)} rejected concepts to {reject_path.relative_to(ROOT)}.",
            f"Wrote subtype labels to {subtype_path.relative_to(ROOT)}.",
        ],
    )


if __name__ == "__main__":
    main()
