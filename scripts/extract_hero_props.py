#!/usr/bin/env python3
"""docs/TSL.pdf -> data/props.json -> data/hero_props.json.

Finds the props each character touches in every scene of the screenplay, every hand-off between characters,
and which props are hero props. Two methods:

  rag    (default) Retrieval-augmented generation for both steps. For each scene, BM25 retrieves related passages
         from the rest of the script and the model extracts props and transfers, each citing a screenplay line;
         then, for each prop, the model decides hero or background from its scenes, transfers and passages.
         Needs OPENAI_API_KEY or ANTHROPIC_API_KEY (read from .env). Answers are cached in data/rag_cache/.
  rules  Hand-written rules (verbs, word lists). No AI, no API.

The logic lives in the props_extractor package; see its __init__.py for a map of the modules.

Usage:
  python3 scripts/extract_hero_props.py                  # RAG: PDF -> props.json -> hero_props.json
  python3 scripts/extract_hero_props.py --dry-run        # RAG: write the scene prompts to data/rag_prompts/
  python3 scripts/extract_hero_props.py --method rules   # rule-based
  python3 scripts/extract_hero_props.py --provider openai --model gpt-5.5
  python3 scripts/extract_hero_props.py --debug          # also print how each scene is read
  python3 scripts/extract_hero_props.py --from-json      # skip the PDF; use the existing data/props.json
  python3 scripts/extract_hero_props.py --pdf other.pdf
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from props_extractor.hero import extract_hero  # noqa: E402
from props_extractor.rag import (HERO_ABOUT, LLMError, decide_hero_props, extract_props_rag,  # noqa: E402
                                 load_dotenv, make_llm)
from props_extractor.rules import extract_props  # noqa: E402
from props_extractor.screenplay import read_screenplay  # noqa: E402

DEFAULT_PDF = ROOT / "docs" / "TSL.pdf"
PROPS_JSON = ROOT / "data" / "props.json"
HERO_JSON = ROOT / "data" / "hero_props.json"
RAG_CACHE = ROOT / "data" / "rag_cache"
RAG_PROMPTS = ROOT / "data" / "rag_prompts"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="screenplay PDF (default docs/TSL.pdf)")
    parser.add_argument("--from-json", action="store_true", help="skip the PDF and use the existing data/props.json")
    parser.add_argument("--method", choices=["rag", "rules"], default="rag", help="how to extract (default rag)")
    parser.add_argument("--provider", choices=["openai", "anthropic"],
                        help="RAG model provider (default: whichever API key is set; Claude if both)")
    parser.add_argument("--model", help="RAG model (default gpt-5.5 for openai, claude-opus-5-5 for anthropic)")
    parser.add_argument("--dry-run", action="store_true",
                        help="RAG: write each scene's prompt to data/rag_prompts/ instead of calling the model")
    parser.add_argument("--debug", action="store_true", help="print how each scene is read")
    args = parser.parse_args()

    try:
        if args.method == "rules":
            run_rules(args)
        else:
            run_rag(args)
    except LLMError as e:
        sys.exit(f"RAG failed: {e}")


def run_rules(args):
    if args.from_json:
        props = _read_json(PROPS_JSON)
    else:
        pdf = args.pdf.resolve()
        acts, characters = read_screenplay(pdf)
        props = extract_props(acts, characters, source=_relative(pdf), debug=args.debug)
        _save_props(props)
    hero = extract_hero(props)
    _write_json(HERO_JSON, hero)
    _print_summary(hero)


def run_rag(args):
    load_dotenv(ROOT / ".env")
    pdf = args.pdf.resolve()
    acts, characters = read_screenplay(pdf, keep_dialogue=True)  # the hero step retrieves passages too
    llm = make_llm(args.provider, args.model, cache_dir=RAG_CACHE)
    print(f"RAG with {llm.name}")

    if args.from_json:
        props = _read_json(PROPS_JSON)
    else:
        props = extract_props_rag(acts, characters, _relative(pdf), llm,
                                  dry_run_dir=RAG_PROMPTS if args.dry_run else None, debug=args.debug)
        if props is None:
            print(f"Dry run: prompts written to {_relative(RAG_PROMPTS)}/ (props.json unchanged)")
            return
        _save_props(props)

    decisions = decide_hero_props(acts, props, llm, debug=args.debug)
    hero = extract_hero(props, decisions, about=HERO_ABOUT)
    _write_json(HERO_JSON, hero)
    _print_summary(hero)


def _save_props(props):
    _write_json(PROPS_JSON, props)
    transfer_count = sum(len(s["transfers"]) for s in props["scenes"])
    print(f"{_relative(PROPS_JSON)}: {len(props['scenes'])} scenes, {len(props['catalog'])} props, "
          f"{transfer_count} transfers, characters: {', '.join(props['characters'])}")


def _print_summary(hero):
    print(f"{_relative(HERO_JSON)}: {hero['hero_count']} hero props, {hero['background_count']} background")
    print(f"  totals: {json.dumps(hero['totals'])}")
    for s in hero["per_scene"]:
        print(f"  {s['scene_number']:2}. {s['scene']:45} props={s['props']} hero={s['hero_props']} "
              f"transfers={s['transfers']}")
    for h in hero["hero_props"]:
        print(f"  HERO  {h['name']} ({', '.join(h['adjectives'])}): {' | '.join(h['why_hero'])}")
    for b in hero["background"]:
        print(f"  bg    {b['name']}")


def _relative(path):
    """Path relative to the project root when it is inside it, else as given."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
