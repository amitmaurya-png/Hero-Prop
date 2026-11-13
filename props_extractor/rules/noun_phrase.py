"""Reading the object of a verb: "a small basket of folded linen, tucked under her arm" ->

    words      ["a", "small", "basket"]      determiners, adjectives and nouns
    head       "basket"                      the noun, singular
    dist       "linen"                       the word that tells this basket from another basket
    adjectives ["small"]
    content    NounPhrase("folded linen")    what is in it ("of X" / "holding X")
    full       "a small basket of folded linen"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Set

from ..vocabulary import (CONTAINERS, DETERMINERS, NOT_PROPS, PARTITIVES, PERSON_NOUNS, POST_PARTICIPLES,
                         PROP_PRONOUNS, STOP, WORD, is_adjective, lemma)

_TOKEN = re.compile(rf"{WORD}|[^\sA-Za-z]")
_PUNCTUATION = {",", ".", ";", "!", "?", "—", ":"}


@dataclass
class NounPhrase:
    phrase: str                           # the words as written ("a small basket")
    full: str                             # phrase plus "of ..." contents or "tied with ..." ("a basket of linen")
    pronoun: Optional[str] = None         # "it" / "them" / "one" ...: refers to the last prop; other fields unset
    words: List[str] = field(default_factory=list)
    head: str = ""                        # singular noun ("cake")
    head_word: str = ""                   # noun as written, lower-case ("cakes")
    dist: Optional[str] = None            # distinguishing noun: "linen" in "linen basket" / "basket of linen"
    adjectives: List[str] = field(default_factory=list)
    content: Optional[NounPhrase] = None  # "folded linen" in "a basket of folded linen"
    post_words: Set[str] = field(default_factory=set)  # lemmas after the head ("of", "linen", "tied", "cord" ...)
    indefinite: bool = False              # "a basket", "baskets": possibly a new prop


def parse_noun_phrase(text: str, is_name: Callable[[str], bool]) -> Optional[NounPhrase]:
    """Read the noun phrase at the start of text, or None if it is not a prop (a person, a place, a gesture...).

    is_name(word) says whether a word is a character's name, which ends the phrase.
    """
    tokens = _TOKEN.findall(text)
    i = 0
    while i < len(tokens) and tokens[i].lower().endswith("ly") and tokens[i].lower() != "only":
        i += 1  # skip adverbs: "hands gently the ..."

    if i < len(tokens) and tokens[i].lower() in PROP_PRONOUNS and _ends_phrase(tokens, i + 1):
        return NounPhrase(phrase=tokens[i], full=tokens[i], pronoun=tokens[i].lower())

    words = []
    while i < len(tokens) and re.fullmatch(WORD, tokens[i]):
        w = tokens[i].lower()
        if w in STOP or w.endswith("ly") or is_name(tokens[i]):
            break
        has_noun = any(x.lower() not in DETERMINERS and not is_adjective(x) for x in words)
        if has_noun and (w in DETERMINERS or w in POST_PARTICIPLES):
            break  # "the basket the ..." / "the scroll tied ...": the phrase has ended
        words.append(tokens[i])
        i += 1

    body = [w for w in words if w.lower() not in DETERMINERS]
    if not body:
        return None
    head = body[-1]
    if lemma(head) in NOT_PROPS or head.lower() in PERSON_NOUNS or len(head) < 3 or is_adjective(head):
        return None

    noun_modifiers = [w.lower() for w in body[:-1] if not is_adjective(w)]
    dist = noun_modifiers[-1] if noun_modifiers else None

    rest = tokens[i:]
    content, post = None, []
    linker = rest[0].lower() if rest else None
    if linker == "holding" or (linker == "of" and lemma(head) in CONTAINERS | PARTITIVES):
        inner = parse_noun_phrase(" ".join(rest[1:]), is_name)
        if inner and not inner.pronoun:
            content, dist = inner, inner.head
            post = [rest[0]] + inner.phrase.split()
    elif linker in POST_PARTICIPLES:
        end = next((j for j, tok in enumerate(rest) if tok in _PUNCTUATION), len(rest))
        post = rest[:end]

    first = words[0].lower()
    is_plural = lemma(head) != head.lower()
    return NounPhrase(
        phrase=" ".join(words),
        full=" ".join(words + post),
        words=words,
        head=lemma(head),
        head_word=head.lower(),
        dist=dist,
        adjectives=[w.lower() for w in words if is_adjective(w)],
        content=content,
        post_words={lemma(w) for w in post},
        indefinite=first in {"a", "an"} or (first not in DETERMINERS and is_plural),
    )


def _ends_phrase(tokens, i):
    """True if nothing noun-like follows a pronoun: "takes it to Saba" (yes) vs "takes one cake" (no)."""
    if i >= len(tokens):
        return True
    nxt = tokens[i]
    return not re.fullmatch(WORD, nxt) or nxt.lower() in STOP or nxt.lower().endswith("ly")
