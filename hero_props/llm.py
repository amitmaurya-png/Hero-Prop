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


def ask_model(model: str, screenplay: str, reasoning: str | None = None, mode: str = "fast") -> ModelReply:
    """One request containing the whole screenplay. reasoning=None: model default."""
    options = {"reasoning_effort": reasoning} if reasoning else {}
    started = time.monotonic()
    response = OpenAI().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": build_prompt(mode)},
            {"role": "user", "content": "Screenplay:\n\n" + screenplay},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "hero_props", "strict": True, "schema": build_schema(mode)},
        },
        **options,
    )
    seconds = time.monotonic() - started

    raw = response.choices[0].message.content
    usage = response.usage.model_dump() if response.usage else {}
    return ModelReply(answer=json.loads(raw), raw=raw, seconds=seconds, usage=usage)


def token_counts(usage: dict, seconds: float | None = None) -> dict:
    """Readable token counts from the API's `usage` block.

    output_tokens = reasoning_tokens (hidden thinking) + answer_tokens (the JSON).
    """
    output = usage.get("completion_tokens") or 0
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    counts = {
        "input_tokens": usage.get("prompt_tokens") or 0,
        "cached_input_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
        "output_tokens": output,
        "reasoning_tokens": reasoning,
        "answer_tokens": output - reasoning,
        "total_tokens": usage.get("total_tokens") or 0,
    }
    if seconds:
        counts["output_tokens_per_second"] = round(output / seconds, 1)
    return counts
