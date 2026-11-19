"""Step 2a - build the instructions and the answer format sent to the model.

Both come from config.CRITERIA and config.RULES; edit those, not this file.

Debug:  python -m hero_props --show-prompt
"""

from __future__ import annotations

from .config import CRITERIA, FAST_RULES, FULL_RULES, ROLE, RULES, WHAT_IS_A_PROP

STRING = {"type": "string"}
BOOLEAN = {"type": "boolean"}
INTEGER = {"type": "integer"}


def build_prompt(mode: str = "fast") -> str:
    criteria = "\n".join(f"  {i}. {c.key}: {c.instruction}" for i, c in enumerate(CRITERIA, 1))
    extra = FAST_RULES if mode == "fast" else FULL_RULES
    rules = "\n".join(f"- {r}" for r in RULES + extra)
    return (
        f"{ROLE}\n\n"
        f"Identify the HERO PROPS. {WHAT_IS_A_PROP} It must meet AT LEAST ONE of these criteria:\n"
        f"{criteria}\n\n"
        f"Rules:\n{rules}\n"
    )


def _object(**properties) -> dict:
    """JSON-schema object in which every property is required (strict mode)."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def build_schema(mode: str = "fast") -> dict:
    """The exact JSON shape the model must answer in.

    fast: scene numbers + transfers only (short answer, quick).
    full: a usage note and evidence quote for every scene (long answer, slow).
    """
    reason = {"type": "string", "description": "Why this is a hero prop."}
    if mode == "fast":
        # Shorter answer = faster run: no separate description, and only the
        # criteria the code cannot check itself (see Criterion.checked_by_code).
        model_judged = {c.key: BOOLEAN for c in CRITERIA if not c.checked_by_code}
        transfer = _object(**{"scene_number": INTEGER, "from": STRING, "to": STRING, "evidence_quote": STRING})
        hero_prop = _object(
            name=STRING,
            aliases={"type": "array", "items": STRING},
            criteria=_object(**model_judged),
            reason=reason,
            scenes={"type": "array", "items": INTEGER},
            transfers={"type": "array", "items": transfer},
        )
    else:
        common = dict(
            name=STRING,
            aliases={"type": "array", "items": STRING},
            description=STRING,
            criteria=_object(**{c.key: BOOLEAN for c in CRITERIA}),
            reason=reason,
        )
        transfer = _object(**{"from": STRING, "to": STRING, "evidence_quote": STRING})
        appearance = _object(
            scene_number=INTEGER,
            usage={"type": "string", "description": "How the prop features in this scene."},
            evidence_quote=STRING,
            transfers={"type": "array", "items": transfer},
        )
        hero_prop = _object(**common, appearances={"type": "array", "items": appearance})
    return _object(hero_props={"type": "array", "items": hero_prop})
