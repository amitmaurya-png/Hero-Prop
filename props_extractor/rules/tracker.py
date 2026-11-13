"""The props found so far: matching each new mention to a known prop, and who holds each one."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from .noun_phrase import NounPhrase
from ..vocabulary import CONTAINERS, DETERMINERS, is_adjective, lemma


@dataclass
class Prop:
    head: str                  # singular noun: "basket"
    dist: Optional[str]        # distinguishing noun: "linen" for the linen basket
    name: str                  # "linen basket", "basket of pieces"
    adjectives: List[str]
    description: str           # the fullest mention in the script: "Three small HONEY-DATE CAKES"
    description_source: str    # the sentence that mention comes from
    post_words: Set[str]       # words after the noun ("tied", "cord"): "the cord" can mean this scroll
    compound: bool             # name is "<noun> <noun>": "the cloth" can mean the "cloth bundle"
    last_used: int             # when it was last mentioned, for picking the most recent of several matches


class PropTracker:
    def __init__(self):
        self.props: Dict[str, Prop] = {}     # id -> prop, in the order found
        self.holder: Dict[str, str] = {}     # id -> character holding it, or the id of the container it is in
        self.last_prop: Optional[str] = None  # what "it" / "them" refers to
        self._clock = 0

    def _tick(self):
        self._clock += 1
        return self._clock

    # -- matching mentions to props ---------------------------------------------------------------------------

    def resolve(self, np: NounPhrase, scene_props: List[str], sentence: str) -> Optional[str]:
        """The id of the prop np refers to; a new prop is created if it matches none."""
        if np.pronoun:
            if self.last_prop:
                self.props[self.last_prop].last_used = self._tick()
            return self.last_prop
        pid = self._match(np, scene_props)
        now = self._tick()
        if pid is None:
            return self._create(np, sentence)
        self._update(pid, np, now)
        self.last_prop = pid
        return pid

    def _match(self, np: NounPhrase, scene_props: List[str]) -> Optional[str]:
        same_head = [pid for pid, p in self.props.items() if p.head == np.head]

        if np.head in {"half", "bit"} and not np.content:
            return self.last_prop  # "the half" = half of the last thing handled

        if np.dist:
            # "the linen basket" -> the basket whose dist is linen
            pid = self._most_recent([p for p in same_head if _same_dist(self.props[p].dist, np.dist)])
            if not pid and not np.indefinite:
                # "his broken toy cart" -> the plain "cart" already seen
                pid = self._most_recent([p for p in same_head if not self.props[p].dist])
            return pid

        in_scene = [p for p in same_head if p in scene_props]
        if in_scene:
            return self._most_recent(in_scene)

        # "the cloth" can mean the "cloth bundle"; "the cord" can mean the scroll "tied with a cord"
        aliases = [pid for pid, p in self.props.items()
                   if (p.dist == np.head and p.compound) or np.head in p.post_words]
        if np.indefinite:
            # "a piece" is a new prop, unless this scene already has one it could mean
            return self._most_recent([p for p in aliases if p in scene_props])

        exact, alias = self._most_recent(same_head), self._most_recent(aliases)
        # "the cloth": a plain "cloth" if there is one, else the most recent "cloth bundle"; a more specific
        # "hanging cloth" loses to a more recently handled "cloth bundle"
        if exact and (not alias or not self.props[exact].dist
                      or self.props[exact].last_used >= self.props[alias].last_used):
            return exact
        return alias

    def _most_recent(self, pids):
        return max(pids, key=lambda pid: self.props[pid].last_used) if pids else None

    def _update(self, pid: str, np: NounPhrase, now: int):
        """A known prop is mentioned again: learn its new adjectives and fuller name."""
        p = self.props[pid]
        p.last_used = now
        for adjective in np.adjectives:
            if adjective not in p.adjectives:
                p.adjectives.append(adjective)
        if np.dist and not p.dist:
            p.dist = np.dist
        words = [w.lower() for w in np.words]
        if (np.dist and not p.compound and len(words) > 1 and words[-2] == np.dist
                and lemma(words[-1]) == p.head):
            p.name, p.compound = f"{np.dist} {words[-1]}", True  # "basket of linen" -> "linen basket"
        p.post_words |= np.post_words

    def _create(self, np: NounPhrase, sentence: str) -> str:
        now = self._tick()
        pid = self._new_id(f"{np.dist or ''} {np.head}")
        nouns = _nouns(np.words)
        name = " ".join(nouns)
        if np.content:
            name = f"{np.head_word} of {' '.join(_nouns(np.content.words))}"
        self.props[pid] = Prop(head=np.head, dist=np.dist, name=name, adjectives=list(dict.fromkeys(np.adjectives)),
                               description=np.full, description_source=sentence, post_words=set(np.post_words),
                               compound=len(nouns) > 1 and not np.content, last_used=now)
        self.last_prop = pid
        if np.content and np.head in CONTAINERS:  # "a basket of linen": the linen is a prop too, inside the basket
            content_id = self._create(np.content, sentence)
            self.holder[content_id] = pid
            self.last_prop = pid
        return pid

    def _new_id(self, text):
        """ "linen basket" -> linen_basket; a second "bread" -> bread_2."""
        base = re.sub(r"[^a-z0-9]+", "_", text.strip()).strip("_")
        pid, n = base, 2
        while pid in self.props:
            pid, n = f"{base}_{n}", n + 1
        return pid

    # -- who holds what ---------------------------------------------------------------------------------------

    def holder_of(self, pid):
        """The character holding a prop, following containers (the linen is in the basket Miri holds)."""
        seen, holder = set(), self.holder.get(pid)
        while holder in self.props and holder not in seen:
            seen.add(holder)
            holder = self.holder.get(holder)
        return holder

    def note_mentions(self, sentence):
        """For a sentence with no handling verb: "it" in the next sentence means the first prop named here.

        Longer names win ("cloth bundle" over "cloth"), then the most recently used prop.
        """
        hits = []
        for pid, p in self.props.items():
            names = [p.name, f"{p.dist} {p.head}" if p.dist else None, p.head, p.dist if p.compound else None]
            for name in filter(None, names):
                m = re.search(rf"\b{re.escape(name)}\w*", sentence, flags=re.I)
                if m:
                    hits.append((m.start(), -len(name), -p.last_used, pid))
        if hits:
            self.last_prop = min(hits)[3]


def _same_dist(a, b):
    """ "honey-date" matches "honey", "cakes" matches "cake"."""
    return bool(a and b and (a == b or a.split("-")[0] == b.split("-")[0] or lemma(a) == lemma(b)))


def _nouns(words):
    return [w.lower() for w in words if w.lower() not in DETERMINERS and not is_adjective(w)]
