"""Step 2a - build the instructions and the answer format sent to the model.

Both come from config.CRITERIA and config.RULES; edit those, not this file.

Debug:  python -m hero_props --show-prompt
"""

from __future__ import annotations

from .config import CRITERIA, ROLE, RULES, WHAT_IS_A_PROP

STRING = {"type": "string"}
BOOLEAN = {"type": "boolean"}


def build_prompt() -> str:
    criteria = "\n".join(f"  {i}. {c.key}: {c.instruction}" for i, c in enumerate(CRITERIA, 1))
    rules = "\n".join(f"- {r}" for r in RULES)
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


def build_schema() -> dict:
    """The exact JSON shape the model must answer in."""
    transfer = _object(**{"from": STRING, "to": STRING, "evidence_quote": STRING})
    appearance = _object(
        scene_number={"type": "integer"},
        usage={"type": "string", "description": "How the prop features in this scene."},
        evidence_quote=STRING,
        transfers={"type": "array", "items": transfer},
    )
    hero_prop = _object(
        name=STRING,
        aliases={"type": "array", "items": STRING},
        description=STRING,
        criteria=_object(**{c.key: BOOLEAN for c in CRITERIA}),
        reason={"type": "string", "description": "Why this is a hero prop."},
        appearances={"type": "array", "items": appearance},
    )
    return _object(hero_props={"type": "array", "items": hero_prop})
