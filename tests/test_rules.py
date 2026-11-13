"""Unit tests for the shared modules and the rules method. Run: python3 -m unittest discover tests"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from props_extractor.cast import Cast  # noqa: E402
from props_extractor.hero import extract_hero  # noqa: E402
from props_extractor.rules import extract_props  # noqa: E402
from props_extractor.rules.noun_phrase import parse_noun_phrase  # noqa: E402
from props_extractor.screenplay import split_sentences  # noqa: E402
from props_extractor.vocabulary import lemma  # noqa: E402

CAST = Cast(["Miri", "Saba", "Nim", "Ima"])


def parse(text):
    return parse_noun_phrase(text, CAST.is_name)


def run_scene(*events):
    """Run STEP 2 on one scene made of ("action", sentence) / ("cue", name) events."""
    acts = [{"act": "ACT ONE", "scenes": [{"scene": "INT. TEST — DAY", "events": list(events)}]}]
    return extract_props(acts, CAST.names, source="test")


class WordTests(unittest.TestCase):
    def test_lemma(self):
        self.assertEqual(lemma("cakes"), "cake")
        self.assertEqual(lemma("berries"), "berry")
        self.assertEqual(lemma("loaves"), "loaf")
        self.assertEqual(lemma("glass"), "glass")

    def test_split_sentences(self):
        self.assertEqual(split_sentences("Miri runs. She stops!  Nim barks."),
                         ["Miri runs.", "She stops!", "Nim barks."])


class NounPhraseTests(unittest.TestCase):
    def test_container_with_contents(self):
        np = parse("a basket of folded linen to Saba.")
        self.assertEqual((np.head, np.dist, np.phrase), ("basket", "linen", "a basket"))
        self.assertEqual(np.content.head, "linen")
        self.assertEqual(np.full, "a basket of folded linen")
        self.assertTrue(np.indefinite)

    def test_adjectives_and_compound(self):
        np = parse("the small cloth bundle close to her chest")
        self.assertEqual((np.head, np.dist, np.adjectives), ("bundle", "cloth", ["small"]))

    def test_pronoun(self):
        self.assertEqual(parse("it to Nim").pronoun, "it")

    def test_not_props(self):
        self.assertIsNone(parse("her hand"))     # body part
        self.assertIsNone(parse("the girl"))     # a person
        self.assertIsNone(parse("Saba the cup"))  # a name ends the phrase before any noun


class CastTests(unittest.TestCase):
    def test_subject_is_not_a_prepositional_object(self):
        self.assertEqual(CAST.first_subject("Beside Miri, Saba sits."), "Saba")

    def test_possessive_with_body_part_is_subject(self):
        sentence = "Miri’s hand takes the cup."
        self.assertEqual(CAST.subject_before(sentence, sentence.index("takes")), "Miri")


class SceneTests(unittest.TestCase):
    def test_hand_off_and_hero(self):
        data = run_scene(("action", "Ima hands a basket of folded linen to Miri."),
                         ("action", "Miri carries the linen basket to Saba."),
                         ("action", "Saba takes the basket."))
        handed = [(t["prop"], t["from"], t["to"]) for t in data["scenes"][0]["transfers"]]
        self.assertEqual(handed, [("linen_basket", "Ima", "Miri"), ("linen_basket", "Miri", "Saba")])
        hero = extract_hero(data)
        self.assertEqual([h["id"] for h in hero["hero_props"]], ["linen_basket"])

    def test_offer_not_taken_is_declined(self):
        data = run_scene(("action", "Nim offers a crumb to Miri."),
                         ("action", "Miri shakes her head."))
        [t] = data["scenes"][0]["transfers"]
        self.assertEqual((t["from"], t["to"], t["type"]), ("Nim", "Miri", "offered-declined"))

    def test_offer_taken_becomes_hand_off(self):
        data = run_scene(("action", "Nim offers a crumb to Miri."),
                         ("action", "Miri takes it."))
        [t] = data["scenes"][0]["transfers"]
        self.assertEqual((t["from"], t["to"], t["type"]), ("Nim", "Miri", "handed"))

    def test_pronoun_subject_after_cue(self):
        data = run_scene(("cue", "Saba"), ("action", "He lifts a wooden peg."))
        [p] = data["scenes"][0]["props"]
        self.assertEqual(p["first_contact"]["subject"], "Saba")


if __name__ == "__main__":
    unittest.main()
