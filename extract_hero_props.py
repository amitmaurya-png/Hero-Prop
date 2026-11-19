"""Extract hero props from docs/TSL.pdf with an LLM and write them to JSON.

A hero prop is a physical object that meets at least one of:
  * center_of_focus  - the story, a shot or a character's attention centres on it
  * high_frequency   - it appears in several scenes / acts
  * transfer         - it passes from one character to another

Steps:
  1. Read the PDF and split it into numbered scenes (by INT./EXT. sluglines),
     tagging each with its act.
  2. Send the whole numbered screenplay to the model, which returns hero props
     and, for each, every scene it appears in with a verbatim evidence quote.
  3. Check every claimed scene: the scene must exist and the quote must appear
     in that scene's text. Unverified appearances are dropped (and reported).
  4. Write data/hero_props.json.

Usage:
  python extract_hero_props.py [--pdf docs/TSL.pdf] [--out data/hero_props.json]
                               [--model gpt-5.4-mini]
Needs OPENAI_API_KEY (read from the environment or ./.env).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent

SLUGLINE = re.compile(r"^(INT\.|EXT\.|INT/EXT\.)\s+.+")
ACT_LINE = re.compile(r"^ACT (ONE|TWO|THREE|FOUR|FIVE|SIX)\s*$")
PAGE_NUMBER = re.compile(r"^\d+\.\s*$")


# ---------------------------------------------------------------- screenplay

def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def read_scenes(pdf: Path) -> tuple[str, list[dict]]:
    """Return (title, scenes). Each scene: number, act, heading, text."""
    lines = []
    for page in PdfReader(pdf).pages:
        for line in (page.extract_text() or "").splitlines():
            line = line.rstrip()
            if line.strip() and not PAGE_NUMBER.match(line.strip()):
                lines.append(line.strip())

    title_lines, scenes, act = [], [], "COLD OPEN"
    current = None
    for line in lines:
        if ACT_LINE.match(line):
            act = line
            continue
        if SLUGLINE.match(line):
            current = {"number": len(scenes) + 1, "act": act, "heading": line, "lines": []}
            scenes.append(current)
        elif current is None:
            title_lines.append(line)
        else:
            current["lines"].append(line)

    for scene in scenes:
        scene["text"] = "\n".join(scene.pop("lines"))
    title = next((l for l in title_lines if "SMALLEST LUNCH" in l.upper()), title_lines[0] if title_lines else pdf.stem)
    return title, scenes


def numbered_screenplay(scenes: list[dict]) -> str:
    return "\n\n".join(
        f"=== SCENE {s['number']} | {s['act']} | {s['heading']} ===\n{s['text']}" for s in scenes
    )


# ----------------------------------------------------------------------- LLM

SYSTEM_PROMPT = """You are a film props master breaking down a screenplay.

Identify the HERO PROPS. A hero prop is a tangible, handleable physical object
(not a character, animal, body part, location, set dressing, food in general,
or abstract idea) that meets AT LEAST ONE of these criteria:
  1. center_of_focus: the story, a shot, or a character's attention centres on it;
     it drives a plot beat or carries the theme.
  2. high_frequency: it appears (shown, handled, or explicitly referred to as
     the same object) in multiple scenes or acts.
  3. transfer: it is handed, given, offered, taken or passed between two
     characters.
  4. It would also be inferred between two characters as well.
  5. It should be a physical object that is tangible and handleable, not a character. It should be actively handled by some character in the scene.

Rules:
- Treat one physical object (or one matched set, e.g. "five barley loaves and two
  fish") as one prop even if it is called different names; list the names used
  in "aliases". If the same object changes form (e.g. bread multiplied), keep it
  as one prop and say so in the description.
- Ignore pure background dressing that is mentioned once and never focused on.
- For each hero prop, list EVERY scene where it is present or explicitly
  referred to. For each scene give a short evidence quote copied VERBATIM
  (exact characters, 3-20 words, from a single line of that scene's text) and
  say how it appears there.
- List transfers only where the screenplay shows the object moving from one
  character to another; use character names as written.
- Use the scene numbers exactly as given in the "=== SCENE n ===" headers.
"""

APPEARANCE = {
    "type": "object",
    "additionalProperties": False,
    "required": ["scene_number", "usage", "evidence_quote", "transfers"],
    "properties": {
        "scene_number": {"type": "integer"},
        "usage": {"type": "string", "description": "How the prop features in this scene."},
        "evidence_quote": {"type": "string"},
        "transfers": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["from", "to", "evidence_quote"],
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "evidence_quote": {"type": "string"},
                },
            },
        },
    },
}

RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["hero_props"],
    "properties": {
        "hero_props": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "aliases", "description", "criteria", "reason", "appearances"],
                "properties": {
                    "name": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "description": {"type": "string"},
                    "criteria": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["center_of_focus", "high_frequency", "transfer"],
                        "properties": {
                            "center_of_focus": {"type": "boolean"},
                            "high_frequency": {"type": "boolean"},
                            "transfer": {"type": "boolean"},
                        },
                    },
                    "reason": {"type": "string", "description": "Why this is a hero prop."},
                    "appearances": {"type": "array", "items": APPEARANCE},
                },
            },
        }
    },
}


def ask_model(model: str, screenplay: str) -> dict:
    response = OpenAI().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Screenplay:\n\n" + screenplay},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "hero_props", "strict": True, "schema": RESPONSE_SCHEMA},
        },
    )
    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------- validation

def _norm(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("…", "...").replace("—", "-").replace("–", "-")
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_in_scene(quote: str, scene: dict) -> bool:
    q = _norm(quote).strip(" .,!?\"'")
    return bool(q) and q in _norm(scene["text"])


def alias_scenes(aliases: list[str], prop_scenes: list[int], scenes: list[dict]) -> dict[str, list[int]]:
    """For each alias, the prop's scenes whose text contains it as whole words.

    Only the prop's own (verified) scenes are searched: elsewhere the same word
    can mean a different object (e.g. "bread" in the village vs Yonah's lunch).
    """
    by_number = {s["number"]: _norm(s["text"]) for s in scenes}
    result = {}
    for alias in aliases:
        pattern = re.compile(r"(?<!\w)" + re.escape(_norm(alias)) + r"(?!\w)")
        result[alias] = [n for n in prop_scenes if n in by_number and pattern.search(by_number[n])]
    return result


def build_output(title: str, scenes: list[dict], answer: dict, model: str) -> tuple[dict, list[str]]:
    by_number = {s["number"]: s for s in scenes}
    warnings, props = [], []

    for prop in answer["hero_props"]:
        kept = {}
        for app in prop["appearances"]:
            scene = by_number.get(app["scene_number"])
            if scene is None:
                warnings.append(f"{prop['name']}: unknown scene {app['scene_number']} dropped")
                continue
            if not quote_in_scene(app["evidence_quote"], scene):
                warnings.append(
                    f"{prop['name']}: scene {scene['number']} dropped, quote not found: {app['evidence_quote']!r}"
                )
                continue
            transfers = []
            for t in app["transfers"]:
                if quote_in_scene(t["evidence_quote"], scene):
                    transfers.append({"from": t["from"], "to": t["to"], "evidence": t["evidence_quote"]})
                else:
                    warnings.append(
                        f"{prop['name']}: transfer {t['from']}->{t['to']} in scene {scene['number']} "
                        f"dropped, quote not found: {t['evidence_quote']!r}"
                    )
            if scene["number"] in kept:  # merge duplicate scene entries
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

        if not kept:
            warnings.append(f"{prop['name']}: no verified scenes, prop dropped")
            continue

        appearances = [kept[n] for n in sorted(kept)]
        acts = list(dict.fromkeys(a["act"] for a in appearances))
        transfer_count = sum(len(a["transfers"]) for a in appearances)
        scene_numbers = [a["scene_number"] for a in appearances]
        props.append({
            "name": prop["name"],
            "aliases": prop["aliases"],
            "alias_scenes": alias_scenes(prop["aliases"], scene_numbers, scenes),
            "description": prop["description"],
            "reason": prop["reason"],
            "criteria": {
                "center_of_focus": prop["criteria"]["center_of_focus"],
                "high_frequency": len(appearances) > 1 or prop["criteria"]["high_frequency"],
                "transfer": transfer_count > 0,
            },
            "scene_count": len(appearances),
            "scenes": scene_numbers,
            "acts": acts,
            "transfer_count": transfer_count,
            "appearances": appearances,
        })

    props.sort(key=lambda p: (-p["scene_count"], -p["transfer_count"], p["name"]))
    output = {
        "screenplay": title,
        "source": "docs/TSL.pdf",
        "model": model,
        "hero_prop_criteria": {
            "center_of_focus": "The story, a shot or a character's attention centres on the object.",
            "high_frequency": "The object is present in more than one scene/act.",
            "transfer": "The object passes between two characters.",
        },
        "scenes": [
            {"scene_number": s["number"], "act": s["act"], "heading": s["heading"]} for s in scenes
        ],
        "hero_prop_count": len(props),
        "hero_props": props,
    }
    return output, warnings


# ---------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", type=Path, default=ROOT / "docs" / "TSL.pdf")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "hero_props.json")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.5"))
    args = parser.parse_args()

    load_env(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set (environment or .env).")

    title, scenes = read_scenes(args.pdf)
    print(f"{title}: {len(scenes)} scenes. Asking {args.model}...", file=sys.stderr)
    answer = ask_model(args.model, numbered_screenplay(scenes))
    output, warnings = build_output(title, scenes, answer, args.model)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")

    for w in warnings:
        print("warning:", w, file=sys.stderr)
    for p in output["hero_props"]:
        flags = ",".join(k for k, v in p["criteria"].items() if v)
        print(f"- {p['name']}: scenes {p['scenes']} [{flags}]", file=sys.stderr)
    print(f"Wrote {len(output['hero_props'])} hero props to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
