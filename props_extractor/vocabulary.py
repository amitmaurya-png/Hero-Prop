"""Word lists and lemma(), used to turn the screenplay into search words for BM25 retrieval (rag/retrieval.py)."""

# One word: letters, apostrophes and hyphens ("honey-date", "Miri’s")
WORD = r"[A-Za-z][A-Za-z’'\-]*"

DEFINITE = {"the", "his", "her", "their", "its", "my", "your", "our", "this", "that", "these", "those"}
NUMBERS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"}
DETERMINERS = DEFINITE | NUMBERS | {"a", "an", "some", "another", "each", "every", "both", "all", "several", "many",
                                    "few", "second", "first", "last"}

# Prepositions, adverbs, conjunctions and common verbs: too common to help a keyword search
STOP = {"to", "toward", "towards", "into", "in", "inside", "on", "onto", "from", "with", "beside", "under",
        "beneath", "over", "around", "at", "close", "back", "up", "down", "out", "aside", "away", "then", "before",
        "after", "as", "so", "while", "that", "which", "who", "instead", "for", "and", "but", "or", "near", "by",
        "shut", "across", "like", "than", "again", "too", "now", "there", "here", "high", "off", "of", "holding",
        "nearby", "past", "through", "along", "forward", "ahead", "deeper", "together", "apart", "more", "is",
        "are", "was", "were", "has", "have", "remains", "becomes"}

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
