"""Word lists used to read noun phrases and sentences, plus two small word helpers.

Edit these lists to tune the extraction for another screenplay: e.g. add a noun to NOT_PROPS if it is wrongly
picked up as a prop, or an adjective to ADJECTIVES if it is wrongly read as a noun.
"""

import re

# One word: letters, apostrophes and hyphens ("honey-date", "Miri’s")
WORD = r"[A-Za-z][A-Za-z’'\-]*"

DEFINITE = {"the", "his", "her", "their", "its", "my", "your", "our", "this", "that", "these", "those"}
NUMBERS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"}
DETERMINERS = DEFINITE | NUMBERS | {"a", "an", "some", "another", "each", "every", "both", "all", "several", "many",
                                    "few", "second", "first", "last"}

# Words that end a noun phrase: prepositions, adverbs, conjunctions, verbs ("the basket to Saba" -> "the basket")
STOP = {"to", "toward", "towards", "into", "in", "inside", "on", "onto", "from", "with", "beside", "under",
        "beneath", "over", "around", "at", "close", "back", "up", "down", "out", "aside", "away", "then", "before",
        "after", "as", "so", "while", "that", "which", "who", "instead", "for", "and", "but", "or", "near", "by",
        "shut", "across", "like", "than", "again", "too", "now", "there", "here", "high", "off", "of", "holding",
        "nearby", "past", "through", "along", "forward", "ahead", "deeper", "together", "apart", "more", "is",
        "are", "was", "were", "has", "have", "remains", "becomes"}
PREPOSITIONS = {"to", "toward", "towards", "at", "beside", "near", "with", "from", "for", "of", "after", "than",
                "behind", "around", "past", "between", "among"}

# Adjectives the -ed rule in is_adjective() does not catch
ADJECTIVES = {"small", "little", "tiny", "large", "big", "great", "huge", "old", "ancient", "new", "young", "warm",
              "cold", "hot", "plain", "simple", "ordinary", "faded", "broken", "empty", "full", "clean", "dirty",
              "soft", "hard", "round", "wooden", "woven", "low", "bright", "dark", "heavy", "thin", "thick", "long",
              "short", "whole", "sticky", "fresh", "dry", "wet", "rough", "smooth", "worn", "open", "closed",
              "golden", "red", "blue", "green", "white", "black", "brown", "purple", "shining", "own", "other",
              "fierce", "hidden", "remaining", "leftover", "sweet", "precious", "humble", "dusty", "wobbly", "torn",
              "ripe", "bare", "faint", "glowing"}
NOT_ADJECTIVES_ENDING_ED = {"bed", "seed", "reed", "thread"}

# Nouns that are containers: "a basket of linen" is a basket prop holding a linen prop
CONTAINERS = {"basket", "bundle", "jar", "cup", "bag", "pouch", "box", "sack", "bowl", "pot"}
# Nouns that are a part of something: "a piece of bread"
PARTITIVES = {"piece", "crumb", "half", "bit", "slice", "handful"}
# Words after the noun that still describe it: "a scroll tied with a cord"
POST_PARTICIPLES = {"tied", "sealed", "tucked", "filled", "patched", "wrapped", "saved", "resting", "lying"}

# Nouns that are never props: body parts, places, abstractions, gestures
NOT_PROPS = {"space", "place", "ground", "way", "side", "edge", "hand", "head", "face", "eye", "nose", "paw", "foot",
             "feet", "mouth", "chest", "arm", "shoulder", "thank", "question", "story", "breath", "moment", "look",
             "attention", "lane", "wall", "doorway", "door", "room", "threshold", "crowd", "grass", "light", "sun",
             "sky", "air", "wind", "movement", "intention", "sleeve", "whisker", "belly", "leg", "ear", "tail",
             "finger", "beetle", "water", "matter", "world", "hill", "floor", "bite", "beat", "group", "lunch-boy",
             "heel", "voice", "veil", "care", "dignity", "shake", "nod", "glance", "smile",
             "hug", "kiss", "posture", "tear"}
# Nouns for people who are not named characters ("the small girl", "a disciple")
PERSON_NOUNS = {"disciple", "disciples", "child", "children", "girl", "boy", "mother", "father", "grandmother",
                "man", "woman", "people", "elders", "fathers", "mothers", "families", "parents", "adults"}
# "Miri’s hands take ...": a possessive followed by one of these is still the subject
BODY = {"hand", "hands", "fingers", "paw", "paws", "arm", "arms"}

# "it", "them" ... refer to the last prop handled
PROP_PRONOUNS = {"it", "them", "one", "this", "these"}
# "him", "her", "them" as a recipient refer to the last other character mentioned
PERSON_PRONOUNS = {"him", "her", "them"}

IRREGULAR_PLURALS = {"loaves": "loaf", "fish": "fish", "children": "child", "halves": "half", "glass": "glass"}


def lemma(word):
    """Singular, lower-case form: "cakes" -> "cake", "berries" -> "berry", "loaves" -> "loaf"."""
    w = word.lower()
    if w in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[w]
    if len(w) > 3 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def is_adjective(word):
    """True for listed adjectives, numbers, "honey-filled"-style compounds, and most words ending in -ed."""
    w = word.lower()
    return (w in ADJECTIVES
            or w in NUMBERS
            or re.fullmatch(r"[a-z]+-(?:filled|made|sized|eyed)", w) is not None
            or (w.endswith("ed") and len(w) > 4 and w not in NOT_ADJECTIVES_ENDING_ED))
