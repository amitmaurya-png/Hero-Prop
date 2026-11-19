"""Everything you are likely to change: defaults, hero-prop criteria, prompt rules.

The prompt, the answer schema and the JSON output are all built from the lists
below, so a change here is picked up everywhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ defaults
# Each can be overridden on the command line (see `--help`).

DEFAULT_PDF = ROOT / "docs" / "TSL.pdf"
DEFAULT_OUT = ROOT / "data" / "hero_props.json"
DEBUG_DIR = ROOT / "data" / "debug"     # prompt, raw answer and warnings of the last run

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")
DEFAULT_REASONING = os.environ.get("OPENAI_REASONING") or None   # None = the model's own default
REASONING_LEVELS = ["low", "medium", "high"]

QUOTE_WORDS = (3, 20)   # min / max words in an evidence quote


# ------------------------------------------------------------------ criteria
# A prop is a hero prop if it meets AT LEAST ONE criterion.
# `checked_by_code=True` means verify.final_criteria() recomputes the flag from
# the verified scenes instead of trusting the model.

@dataclass(frozen=True)
class Criterion:
    key: str                # flag name in the JSON
    instruction: str        # what the model is told
    summary: str            # one-line description written to the JSON
    checked_by_code: bool = False


CRITERIA = [
    Criterion(
        key="center_of_focus",
        instruction="the story, a shot, or a character's attention centres on it; "
                    "it drives a plot beat or carries the theme.",
        summary="The story, a shot or a character's attention centres on the object.",
    ),
    Criterion(
        key="high_frequency",
        instruction="it appears (shown, handled, or explicitly referred to as the "
                    "same object) in multiple scenes or acts.",
        summary="The object is present in more than one scene/act.",
        checked_by_code=True,   # true if the prop has more than one verified scene
    ),
    Criterion(
        key="transfer",
        instruction="it is handed, given, offered, taken or passed between two "
                    "characters - shown directly or clearly inferred from the action.",
        summary="The object passes between two characters.",
        checked_by_code=True,   # true only if at least one transfer was verified
    ),
    Criterion(
        key="actively_handled",
        instruction="a character physically interacts with it - holds, uses, repairs, "
                    "opens, carries, wears with intent, eats from, etc. (e.g. Saba "
                    "repairing a sandal). This alone is enough, even in a single scene.",
        summary="A character physically interacts with the object (holds, uses, repairs...).",
    ),
]


# ------------------------------------------------------------------ prompt text

ROLE = "You are a film props master breaking down a screenplay."

WHAT_IS_A_PROP = (
    "A hero prop is a tangible, handleable physical object (not a character, "
    "animal, body part, location, set dressing, food in general, or abstract idea)."
)

RULES = [
    'Treat one physical object (or one matched set, e.g. "five barley loaves and two '
    'fish") as one prop even if it is called different names; list the names used in '
    '"aliases". If the same object changes form (e.g. bread multiplied), keep it as '
    "one prop and say so in the description.",
    "Ignore background dressing that no character handles or interacts with (e.g. "
    "objects only named in a description of the setting).",
    "Keep different objects of the same kind separate (e.g. one character's basket "
    "vs. the crowd's baskets; one character's sandal vs. a crowd's sandals).",
    "For each hero prop, list EVERY scene where it is present or explicitly referred "
    "to. For each scene give a short evidence quote copied VERBATIM (exact characters, "
    f"{QUOTE_WORDS[0]}-{QUOTE_WORDS[1]} words, from a single line of that scene's text) "
    "and say how it appears there.",
    "List transfers only where the screenplay shows the object moving from one "
    "character to another; use character names as written.",
    'Use the scene numbers exactly as given in the "=== SCENE n ===" headers.',
]
