"""Tests for the RAG method, with a fake model (no API calls). Run: python3 -m unittest discover tests"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from props_extractor.hero import extract_hero  # noqa: E402
from props_extractor.rag import decide_hero_props, extract_props_rag  # noqa: E402
from props_extractor.rag.llm import LLMError, OpenAILLM, load_dotenv, make_llm  # noqa: E402
from props_extractor.rag.prompts import HERO_SCHEMA, SCENE_SCHEMA  # noqa: E402
from props_extractor.rag.retrieval import BM25Index, build_passages  # noqa: E402

CHARACTERS = ["Miri", "Saba", "Ima", "Tav"]
ACTS = [{"act": "ACT ONE", "scenes": [
    {"scene": "EXT. VILLAGE — MORNING", "events": [
        ("action", "Ima hands Miri a basket of folded linen."),
        ("cue", "Ima"), ("dialogue", "Take these to Saba."),
        ("action", "Miri clutches a cloth bundle of honey cakes.")]},
    {"scene": "INT. SABA'S HOUSE — DAY", "events": [
        ("action", "Miri gives the basket to Saba."),
        ("action", "Saba smiles.")]},
]}]

SCENE_1 = {
    "props": [
        {"id": "linen_basket", "name": "Linen basket", "adjectives": ["folded"], "inferred_adjectives": ["woven"],
         "description": "a basket of folded linen", "first_contact": {"subject": "Ima", "action": "hands", "line": 1}},
        {"id": "cloth_bundle", "name": "cloth bundle", "adjectives": [], "inferred_adjectives": [],
         "description": "a cloth bundle", "first_contact": {"subject": "miri", "action": "clutches", "line": 99}},
    ],
    "transfers": [
        {"prop": "linen_basket", "from": "Ima", "to": "miri", "type": "handed", "action": "hands", "line": 1},
        {"prop": "ghost_prop", "from": "Ima", "to": "Miri", "type": "handed", "action": "hands", "line": 1},
    ],
}
SCENE_2 = {
    "props": [{"id": "linen_basket", "name": "Linen basket", "adjectives": ["heavy"], "inferred_adjectives": [],
               "description": "", "first_contact": {"subject": "Miri", "action": "gives", "line": 1}}],
    "transfers": [{"prop": "linen_basket", "from": "Miri", "to": "Saba", "type": "handed", "action": "gives",
                   "line": 1}],
}
HERO_YES = {"is_hero": True, "reasons": ["Ima -> Miri -> Saba"], "story_role": "the errand"}


class FakeLLM:
    """Stands in for llm.LLM: answers by schema from queued answers, and records every prompt."""

    name = "fake:model"

    def __init__(self, scene_answers, hero_answer=HERO_YES):
        self.scene_answers, self.hero_answer, self.prompts = list(scene_answers), hero_answer, []

    def generate(self, system, prompt, schema, name):
        self.prompts.append(prompt)
        return self.hero_answer if schema is HERO_SCHEMA else self.scene_answers.pop(0)

    def cached(self, system, prompt, schema):
        return None


class RetrievalTests(unittest.TestCase):
    def test_passages_include_dialogue_and_rank_by_shared_words(self):
        index = BM25Index(build_passages(ACTS))
        [(score, best)] = index.search("the linen basket", k=1)
        self.assertEqual(best.scene_number, 1)
        self.assertIn("IMA: Take these to Saba.", best.text)
        self.assertEqual(index.search("the linen basket", k=3, exclude_scene=1)[0][1].scene_number, 2)


class RagTests(unittest.TestCase):
    def test_answers_are_grounded_and_merged(self):
        llm = FakeLLM([SCENE_1, SCENE_2])
        data = extract_props_rag(ACTS, CHARACTERS, "test", llm)
        s1, s2 = data["scenes"]
        # the screenplay text comes from the cited line, not from the model
        self.assertEqual(s1["props"][0]["first_contact"]["screenplay"], "Ima hands Miri a basket of folded linen.")
        # a prop citing a line that doesn't exist, and a transfer of an unknown prop, are dropped
        self.assertEqual([p["id"] for p in s1["props"]], ["linen_basket"])
        self.assertEqual(len(s1["transfers"]), 1)
        self.assertEqual(len(data["warnings"]), 2)
        # names are normalised to the cast list
        self.assertEqual((s1["transfers"][0]["from"], s1["transfers"][0]["to"]), ("Ima", "Miri"))
        # adjectives from later scenes are merged into the catalog entry
        self.assertEqual(data["catalog"]["linen_basket"]["adjectives"], ["folded", "heavy"])
        # the second prompt carried the catalog and the last holder forward
        self.assertIn("linen_basket: Linen basket", llm.prompts[1])
        self.assertIn("last held by Miri", llm.prompts[1])

    def test_hero_decisions_come_from_the_model(self):
        llm = FakeLLM([SCENE_1, SCENE_2])
        data = extract_props_rag(ACTS, CHARACTERS, "test", llm)
        decisions = decide_hero_props(ACTS, data, llm)
        hero_prompt = llm.prompts[-1]
        self.assertIn("handed from Ima to Miri", hero_prompt)  # its transfers are in the prompt
        self.assertIn("<retrieved_passages>", hero_prompt)
        hero = extract_hero(data, decisions)
        [h] = hero["hero_props"]
        self.assertEqual((h["id"], h["why_hero"], h["story_role"]), ("linen_basket", ["Ima -> Miri -> Saba"],
                                                                     "the errand"))
        self.assertEqual(h["handed_between_subjects_count"], 2)  # counts are still computed by code


class LLMTests(unittest.TestCase):
    def test_cache_means_one_call_per_request(self):
        with tempfile.TemporaryDirectory() as cache:
            llm = OpenAILLM("gpt-test", cache)
            with mock.patch.object(OpenAILLM, "_call", return_value={"props": [], "transfers": []}) as call:
                first = llm.generate("system", "prompt", SCENE_SCHEMA, "scene_props")
                second = llm.generate("system", "prompt", SCENE_SCHEMA, "scene_props")
            self.assertEqual(call.call_count, 1)
            self.assertEqual(first, second)
            self.assertEqual(llm.cached("system", "prompt", SCENE_SCHEMA), first)

    def test_openai_request_shape(self):
        llm = OpenAILLM("gpt-test")
        create = mock.Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content=json.dumps(HERO_YES), refusal=None))]))
        llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        self.assertEqual(llm.generate("sys", "prompt", HERO_SCHEMA, "hero_decision"), HERO_YES)
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["response_format"]["json_schema"]["strict"], True)
        self.assertEqual(kwargs["messages"][0], {"role": "system", "content": "sys"})

    def test_provider_is_picked_from_the_keys(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "x"}, clear=True):
            self.assertEqual(make_llm().name, "openai:gpt-5.5")
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(LLMError):
                make_llm()

    def test_load_dotenv_does_not_override(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"A": "kept"}, clear=True):
            env = Path(d) / ".env"
            env.write_text("# comment\nA=new\nexport B='quoted'\n")
            load_dotenv(env)
            self.assertEqual((os.environ["A"], os.environ["B"]), ("kept", "quoted"))


if __name__ == "__main__":
    unittest.main()
