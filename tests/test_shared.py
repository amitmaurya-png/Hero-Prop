"""Unit tests for the shared modules. Run: python3 -m unittest discover tests"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from props_extractor.cast import Cast  # noqa: E402
from props_extractor.screenplay import split_sentences  # noqa: E402
from props_extractor.vocabulary import lemma  # noqa: E402


class WordTests(unittest.TestCase):
    def test_lemma(self):
        self.assertEqual(lemma("cakes"), "cake")
        self.assertEqual(lemma("berries"), "berry")
        self.assertEqual(lemma("loaves"), "loaf")
        self.assertEqual(lemma("glass"), "glass")

    def test_split_sentences(self):
        self.assertEqual(split_sentences("Miri runs. She stops!  Nim barks."),
                         ["Miri runs.", "She stops!", "Nim barks."])


class CastTests(unittest.TestCase):
    def test_canonical(self):
        cast = Cast(["Miri", "Saba"])
        self.assertEqual(cast.canonical("miri"), "Miri")
        self.assertEqual(cast.canonical("the small girl"), "small girl")


if __name__ == "__main__":
    unittest.main()
