"""Step 3 - check the model's answer against the screenplay text.

Nothing the model says is trusted blindly:
  * a scene is kept only if its evidence quote is really in that scene;
  * a transfer is kept only if its quote is really in that scene;
  * code-checked criteria are recomputed from what survived.
Every dropped item is added to `warnings` with the reason.
"""

from __future__ import annotations

import re

from .config import CRITERIA

_FANCY_CHARS = {"’": "'", "‘": "'", "“": '"', "”": '"', "…": "...", "—": "-", "–": "-"}


def normalize(text: str) -> str:
    """Lower-case, straight quotes, plain dashes, single spaces."""
    for fancy, plain in _FANCY_CHARS.items():
        text = text.replace(fancy, plain)
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_in_scene(quote: str, scene: dict) -> bool:
    q = normalize(quote).strip(" .,!?\"'")
    return bool(q) and q in normalize(scene["text"])


def verified_transfers(prop: dict, appearance: dict, scene: dict, warnings: list[str]) -> list[dict]:
    kept = []
    for t in appearance["transfers"]:
        if quote_in_scene(t["evidence_quote"], scene):
            kept.append({"from": t["from"], "to": t["to"], "evidence": t["evidence_quote"]})
        else:
            warnings.append(f"{prop['name']}: transfer {t['from']}->{t['to']} in scene "
                            f"{scene['number']} dropped, quote not found: {t['evidence_quote']!r}")
    return kept


def verified_appearances(prop: dict, scenes_by_number: dict, warnings: list[str]) -> list[dict]:
    """The prop's appearances whose scene exists and whose quote is in it, by scene."""
    kept: dict[int, dict] = {}
    for app in prop["appearances"]:
        scene = scenes_by_number.get(app["scene_number"])
        if scene is None:
            warnings.append(f"{prop['name']}: unknown scene {app['scene_number']} dropped")
            continue
        if not quote_in_scene(app["evidence_quote"], scene):
            warnings.append(f"{prop['name']}: scene {scene['number']} dropped, "
                            f"quote not found: {app['evidence_quote']!r}")
            continue

        transfers = verified_transfers(prop, app, scene, warnings)
        if scene["number"] in kept:           # same scene listed twice: merge
            kept[scene["number"]]["transfers"].extend(transfers)
            continue
        kept[scene["number"]] = {
            "scene_number": scene["number"],
            "act": scene["act"],
            "heading": scene["heading"],
            "usage": app["usage"],
            "evidence": app["evidence_quote"],
            "transfers": transfers,
        }
    return [kept[n] for n in sorted(kept)]


def alias_scenes(aliases: list[str], prop_scenes: list[int], scenes: list[dict]) -> dict[str, list[int]]:
    """For each alias, the prop's scenes whose text contains it as whole words.

    Only the prop's own (verified) scenes are searched: elsewhere the same word
    can mean a different object (e.g. "bread" in the village vs Yonah's lunch).
    """
    texts = {s["number"]: normalize(s["text"]) for s in scenes}
    result = {}
    for alias in aliases:
        pattern = re.compile(r"(?<!\w)" + re.escape(normalize(alias)) + r"(?!\w)")
        result[alias] = [n for n in prop_scenes if n in texts and pattern.search(texts[n])]
    return result


def final_criteria(prop: dict, appearances: list[dict]) -> dict[str, bool]:
    """The model's flags, with the code-checked ones recomputed from verified data."""
    flags = {c.key: bool(prop["criteria"].get(c.key)) for c in CRITERIA}
    if "high_frequency" in flags:
        flags["high_frequency"] = len(appearances) > 1 or flags["high_frequency"]
    if "transfer" in flags:
        flags["transfer"] = any(a["transfers"] for a in appearances)
    return flags


# ------------------------------------------------------------- fast mode
# In fast mode the model only names the scenes; the code finds the evidence.

def _alias_patterns(aliases: list[str]) -> list[tuple[str, re.Pattern]]:
    """Whole-word patterns for each alias, longest first."""
    names = sorted({a for a in aliases if normalize(a)}, key=lambda a: -len(a))
    return [(a, re.compile(r"(?<!\w)" + re.escape(normalize(a)) + r"(?!\w)")) for a in names]


def find_evidence(scene: dict, patterns: list[tuple[str, re.Pattern]]) -> tuple[str, str] | None:
    """(line, alias) for the first line of the scene naming the prop.

    PDF lines wrap, so an alias may be split over two lines ("honey / cakes");
    each line is also tried joined with the next one.
    """
    lines = scene["text"].splitlines()
    for i, line in enumerate(lines):
        candidates = [line] + ([f"{line} {lines[i + 1]}"] if i + 1 < len(lines) else [])
        for text in candidates:
            for alias, pattern in patterns:
                if pattern.search(normalize(text)):
                    return text.strip(), alias
    return None


def fast_appearances(prop: dict, scenes_by_number: dict, warnings: list[str]) -> list[dict]:
    """Appearances built from the model's scene numbers, verified by alias search.

    A scene is kept only if its text contains one of the prop's names; that line
    becomes the evidence. Transfers are kept only if their quote is in the scene.
    """
    patterns = _alias_patterns([prop["name"], *prop["aliases"]])
    kept: dict[int, dict] = {}

    for number in sorted(set(prop["scenes"])):
        scene = scenes_by_number.get(number)
        if scene is None:
            warnings.append(f"{prop['name']}: unknown scene {number} dropped")
            continue
        found = find_evidence(scene, patterns)
        if found is None:
            warnings.append(f"{prop['name']}: scene {number} dropped, none of its names appear there")
            continue
        line, alias = found
        kept[number] = {
            "scene_number": number,
            "act": scene["act"],
            "heading": scene["heading"],
            "usage": f'Named "{alias}"',
            "evidence": line,
            "transfers": [],
        }

    for t in prop["transfers"]:
        scene = scenes_by_number.get(t["scene_number"])
        if scene is None or not quote_in_scene(t["evidence_quote"], scene):
            warnings.append(f"{prop['name']}: transfer {t['from']}->{t['to']} in scene "
                            f"{t['scene_number']} dropped, quote not found: {t['evidence_quote']!r}")
            continue
        transfer = {"from": t["from"], "to": t["to"], "evidence": t["evidence_quote"]}
        if scene["number"] not in kept:   # a verified transfer proves the prop is in the scene
            kept[scene["number"]] = {
                "scene_number": scene["number"],
                "act": scene["act"],
                "heading": scene["heading"],
                "usage": f"Transferred from {t['from']} to {t['to']}",
                "evidence": t["evidence_quote"],
                "transfers": [],
            }
        kept[scene["number"]]["transfers"].append(transfer)

    return [kept[n] for n in sorted(kept)]
