"""Step 4 - turn the checked answer into the final JSON (data/hero_props.json)."""

from __future__ import annotations

import json
from pathlib import Path

from .config import CRITERIA
from .verify import alias_scenes, fast_appearances, final_criteria, verified_appearances


def build_hero_prop(prop: dict, appearances: list[dict], scenes: list[dict]) -> dict:
    scene_numbers = [a["scene_number"] for a in appearances]
    return {
        "name": prop["name"],
        "aliases": prop["aliases"],
        "alias_scenes": alias_scenes(prop["aliases"], scene_numbers, scenes),
        "description": prop.get("description", prop["reason"]),   # fast mode has no description
        "reason": prop["reason"],
        "criteria": final_criteria(prop, appearances),
        "scene_count": len(appearances),
        "scenes": scene_numbers,
        "acts": list(dict.fromkeys(a["act"] for a in appearances)),
        "transfer_count": sum(len(a["transfers"]) for a in appearances),
        "appearances": appearances,
    }


def build_output(title: str, scenes: list[dict], answer: dict, model: str,
                 source: str = "docs/TSL.pdf") -> tuple[dict, list[str]]:
    """Check the model's answer and assemble the JSON. Returns (output, warnings)."""
    scenes_by_number = {s["number"]: s for s in scenes}
    warnings: list[str] = []
    props = []

    for prop in answer["hero_props"]:
        if "appearances" in prop:          # full mode: the model gave every scene + quote
            appearances = verified_appearances(prop, scenes_by_number, warnings)
        else:                              # fast mode: scene numbers only, code finds evidence
            appearances = fast_appearances(prop, scenes_by_number, warnings)
        if not appearances:
            warnings.append(f"{prop['name']}: no verified scenes, prop dropped")
            continue
        props.append(build_hero_prop(prop, appearances, scenes))

    props.sort(key=lambda p: (-p["scene_count"], -p["transfer_count"], p["name"]))
    output = {
        "screenplay": title,
        "source": source,
        "model": model,
        "hero_prop_criteria": {c.key: c.summary for c in CRITERIA},
        "scenes": [{"scene_number": s["number"], "act": s["act"], "heading": s["heading"]} for s in scenes],
        "hero_prop_count": len(props),
        "hero_props": props,
    }
    return output, warnings


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
