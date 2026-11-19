"""Step 2b - send the screenplay to the model and get its answer.

The only module that talks to OpenAI. Everything after this works on the
answer dict, so a saved answer can be re-checked without calling the API
(see `--from-answer` in cli.py).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from openai import OpenAI

from .prompt import build_prompt, build_schema


@dataclass
class ModelReply:
    answer: dict              # parsed JSON, in the shape of prompt.build_schema()
    raw: str                  # the model's text exactly as returned
    seconds: float            # time spent waiting for the API
    usage: dict               # token counts reported by the API


def ask_model(model: str, screenplay: str, reasoning: str | None = None) -> ModelReply:
    """One request containing the whole screenplay."""
    options = {"reasoning_effort": reasoning} if reasoning else {}
    started = time.monotonic()
    response = OpenAI().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": build_prompt()},
            {"role": "user", "content": "Screenplay:\n\n" + screenplay},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "hero_props", "strict": True, "schema": build_schema()},
        },
        **options,
    )
    seconds = time.monotonic() - started

    raw = response.choices[0].message.content
    usage = response.usage.model_dump() if response.usage else {}
    return ModelReply(answer=json.loads(raw), raw=raw, seconds=seconds, usage=usage)
