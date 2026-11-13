"""Finding the characters in a sentence and deciding which one is the subject."""

import re

from .vocabulary import BODY, PERSON_NOUNS, PREPOSITIONS

# "his hand", "her fingers": the subject is the owner named earlier in the sentence
OWNER_BODY_PART = re.compile(r"\b(?:his|her|their)\s+(?:hand|hands|fingers|paw|paws)\b", re.I)
# "the small girl", "another child", "the girl’s mother"
_UNNAMED_PERSON = re.compile(
    r"\b(?:(?:the|a|an|another|her|his|older|small|little|young|giggling)\s+)*(?:girl’s\s+)?("
    + "|".join(sorted(PERSON_NOUNS, key=len, reverse=True)) + r")\b",
    re.I)


class Cast:
    """The screenplay's characters, and how to find them in a sentence."""

    def __init__(self, characters):
        self.names = characters
        self._forms = {form for c in characters for form in (c.lower(), c.lower() + "’s", c.lower() + "'s")}
        self._patterns = [(c, re.compile(rf"\b{re.escape(c)}(’s|'s)?\b", re.I)) for c in characters]

    def is_name(self, word):
        """True for "Miri", "miri’s" ..."""
        return word.lower() in self._forms

    def canonical(self, label):
        """ "miri" -> "Miri"; an unnamed person keeps its words, minus a leading "the"."""
        lower = label.lower()
        return next((c for c in self.names if c.lower() == lower), re.sub(r"^the ", "", lower))

    def mentions(self, text):
        """Every person mentioned in text, in order: [(label, start, end, is_possessive)].

        Named characters keep their name; unnamed people are labelled by their words ("the small girl").
        A mention inside a longer one is dropped ("Girl" inside "Small Girl").
        """
        found = []
        for name, pattern in self._patterns:
            for m in pattern.finditer(text):
                found.append((name, m.start(), m.end(), bool(m.group(1))))
        for m in _UNNAMED_PERSON.finditer(text):
            if not any(start <= m.start() < end for _, start, end, _ in found):
                found.append((m.group(0).lower(), m.start(), m.end(), False))
        found = [m for m in found if not any(_inside(m, other) for other in found if other is not m)]
        return sorted(found, key=lambda m: m[1])

    def first_subject(self, text):
        """The first person mentioned who can be the subject, or None."""
        for label, start, end, possessive in self.mentions(text):
            if _can_be_subject(text, start, end, possessive):
                return label
        return None

    def subject_before(self, text, pos):
        """The last person mentioned before pos who can be the subject of the verb at pos, or None.

        "Miri’s hand reaches into ..." -> Miri. "His hand ..." with no subject before it -> the last possessor.
        """
        mentions = self.mentions(text)
        subject = None
        for label, start, end, possessive in mentions:
            if start >= pos:
                break
            if _can_be_subject(text, start, end, possessive):
                subject = label
        if subject is None and OWNER_BODY_PART.search(text[:pos]):
            owners = [label for label, start, _, possessive in mentions if possessive and start < pos]
            subject = owners[-1] if owners else None
        return subject


def _inside(m, other):
    """True if mention m lies within the strictly longer mention other."""
    return other[1] <= m[1] and m[2] <= other[2] and (other[2] - other[1]) > (m[2] - m[1])


def _can_be_subject(text, start, end, possessive):
    """Not after a preposition ("to Miri"), and not a possessive unless a body part follows ("Miri’s hand")."""
    before = text[:start].split()
    if before and before[-1].lower().strip(",") in PREPOSITIONS:
        return False
    if possessive:
        after = text[end:].split()
        return bool(after) and after[0].lower().strip(",.") in BODY
    return True
