"""Step 1 - read the screenplay PDF and split it into numbered scenes.

A scene is a dict: {"number": int, "act": str, "heading": str, "text": str}.

Debug:  python -m hero_props --show-scenes
"""

from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

SLUGLINE = re.compile(r"^(INT\.|EXT\.|INT/EXT\.)\s+.+")          # scene heading
ACT_LINE = re.compile(r"^ACT (ONE|TWO|THREE|FOUR|FIVE|SIX)\s*$")
PAGE_NUMBER = re.compile(r"^\d+\.\s*$")
FIRST_ACT = "COLD OPEN"   # label for scenes before the first "ACT ..." line


def read_lines(pdf: Path) -> list[str]:
    """Non-empty text lines of the PDF, without page numbers."""
    lines = []
    for page in PdfReader(pdf).pages:
        for line in (page.extract_text() or "").splitlines():
            line = line.strip()
            if line and not PAGE_NUMBER.match(line):
                lines.append(line)
    return lines


def read_scenes(pdf: Path) -> tuple[str, list[dict]]:
    """Return (title, scenes)."""
    title_lines: list[str] = []
    scenes: list[dict] = []
    act = FIRST_ACT

    for line in read_lines(pdf):
        if ACT_LINE.match(line):
            act = line
        elif SLUGLINE.match(line):
            scenes.append({"number": len(scenes) + 1, "act": act, "heading": line, "text": ""})
        elif not scenes:
            title_lines.append(line)          # title page, before the first scene
        else:
            scenes[-1]["text"] += line + "\n"

    for scene in scenes:
        scene["text"] = scene["text"].rstrip()
    return screenplay_title(title_lines, pdf), scenes


def screenplay_title(title_lines: list[str], pdf: Path) -> str:
    """Title-page lines up to "Written by" (falls back to the file name)."""
    title = []
    for line in title_lines:
        if line.lower().startswith("written by"):
            break
        title.append(line)
    return " ".join(title) or pdf.stem


def numbered_screenplay(scenes: list[dict]) -> str:
    """The text sent to the model, with a header in front of each scene."""
    return "\n\n".join(
        f"=== SCENE {s['number']} | {s['act']} | {s['heading']} ===\n{s['text']}" for s in scenes
    )
