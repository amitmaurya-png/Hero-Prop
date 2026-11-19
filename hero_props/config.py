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
CACHE_DIR = ROOT / "data" / "cache"     # saved answers; identical inputs skip the API call

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")

# Reasoning effort. "low" keeps the model's hidden thinking to ~100 tokens
# (vs ~4,000 at the model default), which is most of the speed-up.
DEFAULT_REASONING = os.environ.get("OPENAI_REASONING", "low")
REASONING_LEVELS = ["none", "low", "medium", "high", "default"]   # "default" = the model's own

# Answer mode (run time is roughly output tokens / ~85 per second):
#   fast - the model lists props, aliases, scene numbers and transfers (~1k tokens);
#          the code finds the evidence line in each scene by searching the aliases.
#   full - the model also writes a usage note and a quote for every scene (~3.5k+ tokens).
MODES = ["fast", "full"]
DEFAULT_MODE = os.environ.get("HERO_PROPS_MODE", "fast")

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

# Rules used in both modes.
RULES = [
    'Treat one physical object (or one matched set, e.g. "five barley loaves and two '
    'fish") as one prop even if it is called different names; list the names used in '
    '"aliases". If the same object changes form (e.g. bread multiplied), keep it as '
    "one prop and say so in the description.",
    "Ignore background dressing that no character handles or interacts with (e.g. "
    "objects only named in a description of the setting).",
    "Keep different objects of the same kind separate (e.g. one character's basket "
    "vs. the crowd's baskets; one character's sandal vs. a crowd's sandals).",
    "Parts, wrappings or attachments of a prop (a wheel, a cord, a seal, a bundle) "
    "belong to that prop; do not list them as separate props.",
    "List transfers only where the screenplay shows the object moving from one "
    "character to another; use character names as written.",
    'Use the scene numbers exactly as given in the "=== SCENE n ===" headers.',
]

# Extra rules for --mode full: a usage note and a quote for every scene.
FULL_RULES = [
    "For each hero prop, list EVERY scene where it is present or explicitly referred "
    "to. For each scene give a short evidence quote copied VERBATIM (exact characters, "
    f"{QUOTE_WORDS[0]}-{QUOTE_WORDS[1]} words, from a single line of that scene's text) "
    "and say how it appears there.",
]

# Extra rules for --mode fast: short answer, the code finds the evidence.
FAST_RULES = [
    'In "scenes", list the number of EVERY scene where the prop is present or '
    "referred to by any name.",
    'In "aliases", list every word or phrase the screenplay uses for the prop, copied '
    'exactly, including short nouns (e.g. "basket", "cart", "coin"). The scenes are '
    "verified by searching for these names, so a scene whose text contains none of "
    "them is dropped.",
    "For each transfer give the scene number, who gives, who receives, and a short "
    "quote copied VERBATIM (3-8 words) from that scene.",
    'Keep "reason" to one short sentence (at most 15 words).',
]
