"""STEP 3: props.json + the model's hero decisions -> hero_props.json.

Which props are hero props is decided by the model (rag.decide_hero_props), from each prop's scenes, transfers
and retrieved passages; every other prop is "background". Totals of scenes and transfers, a per-scene count, and
per-prop counts are computed here by code.
"""

from collections import Counter

# Transfer endpoints that are places, not subjects (used by the hand-made props.manual.json)
PLACES = {"shelf", "the floor between saba and miri"}

ABOUT = ("Hero props and materials, decided by the RAG method: for each prop, the model weighs its scenes, "
         "transfers and retrieved screenplay passages ('why_hero' / 'why_not' are its reasons). Counts and "
         "totals are computed by code. Written by scripts/extract_hero_props.py.")


def collect_evidence(data):
    """Per prop: the scenes it is touched in and its transfers. Returns (scenes, transfers), both id -> list."""
    catalog = data["catalog"]
    not_subjects = PLACES | {p["name"].lower() for p in catalog.values()}  # props as endpoints: "into the basket"

    def is_subject(endpoint):
        return endpoint.lower() not in not_subjects

    scenes = {pid: [] for pid in catalog}
    transfers = {pid: [] for pid in catalog}
    for n, s in enumerate(data["scenes"], 1):
        for p in s["props"]:
            assert p["id"] in catalog, f"{s['scene']}: unknown prop {p['id']}"
            scenes[p["id"]].append({"scene_number": n, "scene": s["scene"],
                                    **({"state": p["state"]} if p.get("state") else {}),  # only in the manual file
                                    "first_contact": p["first_contact"]})
        for t in s["transfers"]:
            assert t["prop"] in catalog, f"{s['scene']}: unknown prop {t['prop']}"
            transfers[t["prop"]].append({"scene_number": n, "scene": s["scene"],
                                         "between_subjects": is_subject(t["from"]) and is_subject(t["to"]),
                                         **{k: t[k] for k in ("from", "to", "type", "action", "screenplay")}})
    return scenes, transfers


def extract_hero(data, decisions):
    """Build the hero_props.json document from a props.json document.

    decisions: id -> {"is_hero", "reasons", "story_role"} from rag.decide_hero_props.
    """
    catalog = data["catalog"]
    scenes, transfers = collect_evidence(data)

    hero, background = [], []
    for pid, entry in catalog.items():
        handed = [t for t in transfers[pid] if t["type"] == "handed" and t["between_subjects"]]
        d = decisions[pid]
        is_hero, reasons, extra = d["is_hero"], d["reasons"], {"story_role": d["story_role"]}
        why_not = " ".join(reasons)

        summary = {"id": pid, "name": entry["name"], "adjectives": entry["adjectives"],
                   **({"inferred_adjectives": entry["inferred_adjectives"]} if entry.get("inferred_adjectives")
                      else {}),  # only from the RAG method
                   "description": entry["description"]}
        if is_hero:
            hero.append({**summary, "why_hero": reasons, **extra, "scene_count": len(scenes[pid]),
                         "transfer_count": len(transfers[pid]), "handed_between_subjects_count": len(handed),
                         "scenes": scenes[pid], "transfers": transfers[pid]})
        else:
            only = scenes[pid][0] if scenes[pid] else None
            background.append({**summary, "why_not": why_not, **extra,
                               **({"scene": only["scene"], "first_contact": only["first_contact"]} if only else {})})

    hero_ids = {h["id"] for h in hero}
    return {"source": "data/props.json",
            "about": ABOUT,
            "totals": _totals(data, transfers, hero_ids, len(hero), len(background)),
            "per_scene": _per_scene(data, hero_ids),
            "hero_count": len(hero), "hero_props": hero,
            "background_count": len(background), "background": background}


def _totals(data, transfers, hero_ids, hero_count, background_count):
    all_transfers = [t for ts in transfers.values() for t in ts]
    scenes = data["scenes"]
    return {
        "scenes": len(scenes),
        "scenes_with_props": sum(1 for s in scenes if s["props"]),
        "scenes_with_transfers": sum(1 for s in scenes if s["transfers"]),
        "scenes_with_hero_props": sum(1 for s in scenes if any(p["id"] in hero_ids for p in s["props"])),
        "transfers": len(all_transfers),
        "transfers_by_type": dict(Counter(t["type"] for t in all_transfers)),
        "handed_between_subjects": sum(1 for t in all_transfers if t["type"] == "handed" and t["between_subjects"]),
        "hero_prop_transfers": sum(len(transfers[pid]) for pid in hero_ids),
        "props": len(data["catalog"]),
        "hero_props": hero_count,
        "background_props": background_count,
    }


def _per_scene(data, hero_ids):
    return [{"scene_number": n, "scene": s["scene"], "props": len(s["props"]),
             "hero_props": sum(1 for p in s["props"] if p["id"] in hero_ids), "transfers": len(s["transfers"])}
            for n, s in enumerate(data["scenes"], 1)]
