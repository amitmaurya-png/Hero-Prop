"""Make a copy of docs/TSL.pdf with every hero prop highlighted.

Reads data/hero_props.json (written by extract_hero_props.py) and, for each
hero prop, highlights its name, aliases and their main nouns (whole words, outside scene headings and
place names like "Scroll Room") in its own colour - but only inside
the scenes where that prop was found, so a word like "basket" is attributed to
the right prop. Each scene's evidence line also gets a note with the prop's
usage and any transfers, and a colour key is added as the first page.

Usage:
  python highlight_hero_props.py [--pdf docs/TSL.pdf] [--props data/hero_props.json]
                                 [--out docs/TSL_hero_props.pdf]
Needs: pip install pymupdf
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import fitz  # PyMuPDF

from hero_props.screenplay import PAGE_NUMBER, SLUGLINE
from hero_props.verify import normalize

ROOT = Path(__file__).resolve().parent

# Aliases the model sometimes gives that name an idea rather than the object
# itself; highlighting them would mark lines that are not about the prop.
ABSTRACT_ALIASES = {"story", "stories", "food"}

# A match followed by one of these is part of a place name ("Scroll Room").
PLACE_WORDS = {"room"}

COLORS = [  # light, distinct highlight colours (RGB 0-1)
    (1.00, 0.85, 0.30), (0.55, 0.85, 1.00), (0.60, 0.95, 0.55), (1.00, 0.65, 0.75),
    (0.80, 0.70, 1.00), (1.00, 0.70, 0.45), (0.55, 0.95, 0.90), (0.90, 0.90, 0.50),
    (0.85, 0.80, 0.70), (0.75, 0.85, 0.60),
]


# Words that end a noun phrase: "basket holding the bread" -> head "basket".
PHRASE_BREAKS = {"of", "holding", "with", "for", "and", "from", "in", "on"}


def head_noun(phrase: list[str]) -> str | None:
    """Main noun of a multi-word alias ("five barley loaves" -> "loaves")."""
    if len(phrase) < 2:
        return None
    cut = next((i for i, w in enumerate(phrase) if w in PHRASE_BREAKS), len(phrase))
    return phrase[cut - 1] if cut > 0 else None


def token(text: str) -> str:
    return normalize(text).strip(".,!?;:\"'()[]…")


def page_words(doc: fitz.Document) -> list[dict]:
    """All words in reading order, each tagged with its page and scene number."""
    words, scene = [], 0
    for page in doc:
        lines: dict[tuple, list] = {}
        for x0, y0, x1, y1, text, block, line, _ in page.get_text("words", sort=True):
            lines.setdefault((block, line), []).append((fitz.Rect(x0, y0, x1, y1), text))
        for line_words in lines.values():
            line_text = " ".join(t for _, t in line_words)
            if PAGE_NUMBER.match(line_text):   # "6." etc.: not script text; would break phrases across pages
                continue
            heading = bool(SLUGLINE.match(line_text))
            scene += heading
            for rect, text in line_words:
                words.append({"page": page.number, "rect": rect, "tok": token(text),
                              "scene": scene, "heading": heading})
    return words


def matches(words: list[dict], start: int, phrase: list[str], scene: int) -> bool:
    if start + len(phrase) > len(words):
        return False
    for offset, want in enumerate(phrase):
        w = words[start + offset]
        if w["scene"] != scene or w["heading"] or w["tok"] != want:
            return False
    # "Scroll Room" etc. name a place, not the prop.
    end = start + len(phrase)
    return not (end < len(words) and words[end]["tok"] in PLACE_WORDS)


def find_phrase(words: list[dict], phrase: list[str], scene: int) -> list[int]:
    """Start indices of every occurrence of phrase inside the given scene."""
    return [i for i, w in enumerate(words) if w["scene"] == scene and matches(words, i, phrase, scene)]


def line_rects(items: list[dict]) -> list[fitz.Rect]:
    """Merge word boxes on the same line into one box per line."""
    rects: list[fitz.Rect] = []
    for w in items:
        if rects and w["page"] == rects[-1][1] and abs(w["rect"].y0 - rects[-1][0].y0) < 2:
            rects[-1][0] |= w["rect"]
        else:
            rects.append([fitz.Rect(w["rect"]), w["page"]])
    return rects


def highlight(doc, items, color, title, note):
    by_page: dict[int, list] = {}
    for rect, page_no in line_rects(items):
        by_page.setdefault(page_no, []).append(rect)
    for page_no, rects in by_page.items():
        page = doc[page_no]  # keep a reference: annotations need their page alive
        annot = page.add_highlight_annot(rects)
        annot.set_colors(stroke=color)
        annot.set_info(title=title, content=note)
        annot.update()


def plain(text: str) -> str:
    """The built-in PDF font lacks curly quotes and dashes; use ASCII ones."""
    return text.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "—": "-", "–": "-", "…": "..."}))


def add_key_page(doc, title, props, colors):
    page = doc.new_page(0, width=doc[1].rect.width, height=doc[1].rect.height)
    x, y = 54, 60
    page.insert_text((x, y), "Hero props - " + plain(title), fontsize=15, fontname="hebo")
    y += 20
    page.insert_text((x, y), "Highlights mark each prop's mentions in the scenes where it appears. "
                     "Notes on the evidence lines give usage and transfers.", fontsize=8.5, color=(0.3, 0.3, 0.3))
    y += 24
    for prop, color in zip(props, colors):
        flags = ", ".join(k.replace("_", " ") for k, v in prop["criteria"].items() if v)
        lines = [
            f"Scenes {', '.join(map(str, prop['scenes']))}  |  {', '.join(prop['acts'])}",
            f"Criteria: {flags}  |  transfers: {prop['transfer_count']}",
            *textwrap.wrap(plain(prop["reason"]), 105),
        ]
        if y + 13 * (len(lines) + 1) > page.rect.height - 50:
            page = doc.new_page(page.number + 1, width=page.rect.width, height=page.rect.height)
            y = 60
        page.draw_rect(fitz.Rect(x, y - 10, x + 14, y + 2), color=color, fill=color)
        page.insert_text((x + 22, y), plain(prop["name"]), fontsize=11, fontname="hebo")
        for line in lines:
            y += 12
            page.insert_text((x + 22, y), line, fontsize=8.5, color=(0.25, 0.25, 0.25))
        y += 22


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", type=Path, default=ROOT / "docs" / "TSL.pdf")
    parser.add_argument("--props", type=Path, default=ROOT / "data" / "hero_props.json")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "TSL_hero_props.pdf")
    args = parser.parse_args()

    data = json.loads(args.props.read_text())
    props = data["hero_props"]
    colors = [COLORS[i % len(COLORS)] for i in range(len(props))]
    doc = fitz.open(args.pdf)
    words = page_words(doc)

    # Every (alias, prop, scene) candidate, longest alias first, so "linen basket"
    # wins over "basket" and each word is highlighted for one prop only.
    candidates = []
    for p_idx, prop in enumerate(props):
        phrases = set()
        for name in {prop["name"], *prop["aliases"]}:
            phrase = tuple(t for t in (token(x) for x in name.split()) if t)
            if phrase:
                phrases.add(phrase)
                head = head_noun(list(phrase))
                if head:
                    phrases.add((head,))
        for phrase in phrases:
            if len(phrase) == 1 and phrase[0] in ABSTRACT_ALIASES:
                continue
            for scene in prop["scenes"]:
                candidates.append((len(phrase), p_idx, scene, list(phrase)))
    candidates.sort(key=lambda c: (-c[0], c[1]))

    claimed: set[int] = set()
    hits = {i: 0 for i in range(len(props))}
    for _, p_idx, scene, phrase in candidates:
        for start in find_phrase(words, phrase, scene):
            span = range(start, start + len(phrase))
            if claimed.intersection(span):
                continue
            claimed.update(span)
            hits[p_idx] += 1
            highlight(doc, [words[i] for i in span], colors[p_idx], props[p_idx]["name"], f"Scene {scene}")

    # A note on each scene's evidence line: usage + transfers.
    missing = []
    for p_idx, prop in enumerate(props):
        for app in prop["appearances"]:
            phrase = [t for t in (token(x) for x in app["evidence"].split()) if t]
            found = find_phrase(words, phrase, app["scene_number"])
            if not found:
                missing.append(f"{prop['name']} / scene {app['scene_number']}")
                continue
            first = words[found[0]]
            note = f"Scene {app['scene_number']} - {app['heading']}\n{app['usage']}"
            for t in app["transfers"]:
                note += f"\nTransfer: {t['from']} -> {t['to']}"
            page = doc[first["page"]]
            icon = page.add_text_annot(fitz.Point(page.rect.width - 40, first["rect"].y0), note, icon="Note")
            icon.set_colors(stroke=colors[p_idx])
            icon.set_info(title=prop["name"])
            icon.update()

    add_key_page(doc, data.get("screenplay", args.pdf.stem), props, colors)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(args.out, garbage=3, deflate=True)

    for p_idx, prop in enumerate(props):
        print(f"- {prop['name']}: {hits[p_idx]} highlights")
    for m in missing:
        print("warning: evidence line not located:", m)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
