"""The "G" in RAG: one interface over the language models that generate the answers.

    llm = make_llm()                         # picks the provider from the keys in the environment / .env
    answer = llm.generate(system, prompt, schema, name="scene")   # -> dict matching the JSON schema

Two providers:
    openai     OPENAI_API_KEY      default model gpt-5.5
    anthropic  ANTHROPIC_API_KEY   default model claude-opus-5-5
If both keys are set, Claude is used unless --provider says otherwise.

Answers are cached on disk by a hash of the full request (provider, model, prompts, schema), so re-running costs
nothing and gives the same result. Delete the cache folder, or change a prompt, to ask again.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

DEFAULT_MODELS = {"openai": "gpt-5.5", "anthropic": "claude-opus-5-5"}
KEY_NAMES = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
MAX_TOKENS = 16000


class LLMError(RuntimeError):
    pass


def load_dotenv(path):
    """Set KEY=VALUE lines of a .env file as environment variables (existing variables win)."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def make_llm(provider=None, model=None, cache_dir=None):
    """The model to use: the given provider, else Claude if its key is set, else OpenAI if its key is set."""
    if provider is None:
        provider = next((p for p in ("anthropic", "openai") if os.environ.get(KEY_NAMES[p])), None)
        if provider is None:
            raise LLMError("No API key found. Put OPENAI_API_KEY or ANTHROPIC_API_KEY in .env, "
                           "or use --method rules.")
    elif not os.environ.get(KEY_NAMES[provider]):
        raise LLMError(f"{KEY_NAMES[provider]} is not set (in the environment or .env).")
    cls = {"openai": OpenAILLM, "anthropic": ClaudeLLM}[provider]
    return cls(model or DEFAULT_MODELS[provider], cache_dir)


class LLM:
    """Base class: the answer cache. Subclasses implement _call(), the actual API request."""

    provider = ""

    def __init__(self, model, cache_dir=None):
        self.model = model
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._client = None

    @property
    def name(self):
        return f"{self.provider}:{self.model}"

    def generate(self, system, prompt, schema, name):
        """The model's answer as a dict matching schema. Served from the cache when the same request was made."""
        answer = self.cached(system, prompt, schema)
        if answer is None:
            answer = self._call(system, prompt, schema, name)
            self._save(system, prompt, schema, answer)
        return answer

    def cached(self, system, prompt, schema):
        """The cached answer for this request, or None (no API call)."""
        path = self._cache_path(system, prompt, schema)
        return json.loads(path.read_text(encoding="utf-8")) if path and path.exists() else None

    def _save(self, system, prompt, schema, answer):
        path = self._cache_path(system, prompt, schema)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(answer, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _cache_path(self, system, prompt, schema):
        """data/rag_cache/<hash of the full request>.json, or None when caching is off."""
        if not self.cache_dir:
            return None
        request = {"provider": self.provider, "model": self.model, "system": system, "prompt": prompt,
                   "schema": schema}
        key = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
        return self.cache_dir / f"{key}.json"

    def _call(self, system, prompt, schema, name):
        raise NotImplementedError


class OpenAILLM(LLM):
    provider = "openai"

    def _call(self, system, prompt, schema, name):
        try:
            import openai
        except ImportError as e:
            raise LLMError("The openai package is missing: pip install -r requirements.txt") from e
        self._client = self._client or openai.OpenAI()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                max_completion_tokens=MAX_TOKENS,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                response_format={"type": "json_schema",
                                 "json_schema": {"name": name, "strict": True, "schema": schema}},
            )
        except openai.AuthenticationError as e:
            raise LLMError("OpenAI rejected OPENAI_API_KEY.") from e
        except openai.APIConnectionError as e:
            raise LLMError(f"Could not reach the OpenAI API: {e}") from e
        except openai.APIStatusError as e:
            raise LLMError(f"OpenAI API error {e.status_code}: {e.message}") from e
        choice = response.choices[0]
        if getattr(choice.message, "refusal", None):
            raise LLMError(f"The model declined: {choice.message.refusal}")
        if choice.finish_reason == "length":
            raise LLMError("The model's answer was cut off at max_completion_tokens.")
        return json.loads(choice.message.content)


class ClaudeLLM(LLM):
    provider = "anthropic"

    def _call(self, system, prompt, schema, name):
        try:
            import anthropic
        except ImportError as e:
            raise LLMError("The anthropic package is missing: pip install -r requirements.txt") from e
        self._client = self._client or anthropic.Anthropic()
        try:
            response = self._client.beta.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",  # if the model declines, the API retries on Anthropic's recommended fallback
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                output_config={"effort": "high", "format": {"type": "json_schema", "schema": schema}},
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.AuthenticationError as e:
            raise LLMError("Claude rejected ANTHROPIC_API_KEY.") from e
        except anthropic.APIConnectionError as e:
            raise LLMError(f"Could not reach the Claude API: {e}") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Claude API error {e.status_code}: {e.message}") from e
        if response.stop_reason == "refusal":
            raise LLMError(f"Claude declined the request ({getattr(response.stop_details, 'category', None)}).")
        if response.stop_reason == "max_tokens":
            raise LLMError("Claude's answer was cut off at max_tokens.")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise LLMError("Claude returned no text.")
        return json.loads(text)
