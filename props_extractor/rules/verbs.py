"""Handling verbs: the patterns that say a character touches, gives, takes or puts away an object.

Each pattern has a kind, which decides what happens to the object read after the match:

    passive     "A basket is placed in her hands"    the giver is the next character to speak
    give2       "hands Miri the basket"              subject -> the named recipient
    give        "hands the basket to Saba"           subject -> the "to/toward X" recipient
                ("offers" / "holds out" only become a hand-off if someone takes it soon after)
    take        "takes the bread"                    whoever held it (or offered it) -> subject
    place       "drops the peg into the basket"      subject -> a container (a transfer to an object)
    withhold    "shields the basket"                 kept back from the last other character
    with        "IMA appears with a basket"          first contact only
    carry       "clutches", "unties", "reaches into" first contact only
    instrument  "pokes it with a stick"              first contact only

"{chars}" in a pattern is replaced by the character names.
"""

import re

VERB_PATTERNS = [
    ("passive", r"^(?P<np>.+?)\s+is placed in (?P<r>\w+) hands"),
    ("give2", r"\b(?P<v>hands|gives|offers|tosses|passes)\s+(?P<r>{chars}|him|her|them)\s+"
              r"(?=(?:a|an|the|one|his|her|their|some)\b)"),
    ("give", r"\b(?P<v>hands|gives|offers|tosses|pushes|noses|passing|passes|holding out|holds out)\s+"),
    ("take", r"\b(?P<v>takes(?! out)|take(?! out)|receives|receive|catches|accepts)\s+"),
    ("place", r"\b(?P<v>places|drops|tucks|puts|slips|pass)\s+"),
    ("withhold", r"\b(?P<v>shields|guards|hides)\s+"),
    ("with", r"\b(?P<v>appears|enters|walks|returns|moves?|comes|runs|weaving|wanders|arrives|approaches)\b"
             r"[^,.;]*?\bwith\s+"),
    ("carry", r"\b(?P<v>carrying|carries|carry|clutches|clutching|holds up|holds(?! out)|holding(?! out)|lifts|"
              r"lift|picks up|picks|pulls aside|pulls out|pulls|draws down|draws|unties|unwraps|wraps|opens|"
              r"closes|folds|repairing|repairs|sewing|sews|balance|balances|sniffs|breaks off|breaks|takes out|"
              r"uses|gather|gathers|touches|reaches into|rests? (?:briefly |gently )?on|"
              r"resting (?:briefly |gently )?on|tighten around|tightens around|goes to|returns to|eats|"
              r"playing with|peeks inside|looks into|climbs into|scoops up)\s+"),
    ("instrument", r"\b(?P<v>pokes|taps|points)\b[^,.;]*?\bwith\s+"),
]

GIVE_KINDS = {"give", "give2"}
OFFER_VERBS = {"offers", "holding out", "holds out"}

# "Jesus’s hands receive", "the hands": a possessive or determiner before the match means it is a noun, not a verb
NOUN_BEFORE_VERB = re.compile(r"(?:’s|'s|\b(?:the|her|his|both|their|its|my|your|our))\s+$", re.I)
# Where a placed object goes: "into the basket", "beneath her shawl"
INTO = re.compile(r"\b(?:into|in|inside|beneath|under)\s+(?P<np>(?:the|a|an|her|his|their)\s[^,.;]+)", re.I)

_TARGET = (r"\b(?:to|toward|towards)\s+(?P<r>{chars}|him|her|them|another|the [a-z ]*?(?:girl|child|boy|mother)|"
           r"children(?:, [a-z]+)*)\b")
_FROM = r"\bfrom\s+({chars})\b"


class VerbPatterns:
    """VERB_PATTERNS compiled for one screenplay's characters."""

    def __init__(self, characters):
        names = "|".join(re.escape(c) for c in characters)
        self.verbs = [(kind, re.compile(pattern.replace("{chars}", names), re.I)) for kind, pattern in VERB_PATTERNS]
        self.target = re.compile(_TARGET.replace("{chars}", names), re.I)   # "to Saba", "toward the small girl"
        self.source = re.compile(_FROM.replace("{chars}", names), re.I)     # "from Andrew"

    def find(self, sentence):
        """Every verb match in the sentence, in order: [(start, kind, match)]."""
        hits = []
        for kind, regex in self.verbs:
            for m in regex.finditer(sentence):
                if kind != "passive":
                    if NOUN_BEFORE_VERB.search(sentence[:m.start()]):
                        continue  # "Jesus’s hands receive": "hands" is a noun here
                    if m.start() == 0:
                        continue  # "Hands pass bread": a sentence's first word is its subject, not a verb
                hits.append((m.start(), kind, m))
        hits.sort(key=lambda hit: hit[0])
        return hits
