"""STEP 2 (rules): scenes -> props.json.

Only objects a character touches become props. Each action sentence is searched for handling verbs (verbs.py);
the object after the verb is read as a noun phrase (noun_phrase.py) and matched to a known prop (tracker.py).

    subject     the character named nearest before the verb (cast.py); for "He"/"She", the character who just
                spoke, or the last subject
    recipient   "to X" / "toward X" / "hands X ...", where "him"/"her" is the last other character mentioned
    holder      tracked across the whole script, so "Andrew receives the basket" knows who gave it
    offers      an offer nobody takes within OFFER_WINDOW sentences is recorded as "offered-declined"

Adjectives are the script's own words in front of the prop's noun, gathered from every mention. Code cannot
invent adjectives or know what the script leaves unsaid, so there are no "implied" hand-offs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..cast import Cast
from .noun_phrase import parse_noun_phrase
from .tracker import PropTracker
from .verbs import GIVE_KINDS, INTO, OFFER_VERBS, VerbPatterns
from ..vocabulary import DETERMINERS, PERSON_PRONOUNS, WORD, is_adjective, lemma

OFFER_WINDOW = 3     # sentences within which an offer can still be taken
CUE_LOOKAHEAD = 4    # events after "is placed in her hands" in which to look for the giver's cue
EMPTY_SCENE_NOTE = "No character handles an object in this scene's action lines."


def extract_props(acts, characters, source, debug=False):
    """Read every scene; return the props.json document."""
    cast = Cast(characters)
    patterns = VerbPatterns(characters)
    tracker = PropTracker()

    scenes = []
    for scene_no, (act, scene) in enumerate(_all_scenes(acts), 1):
        reader = SceneReader(scene["events"], scene_no, cast, patterns, tracker, debug)
        reader.read()
        scenes.append({"act": act["act"], "scene": scene["scene"],
                       "props": [{"id": pid, "first_contact": reader.first_contact[pid]}
                                 for pid in reader.scene_props if pid in reader.first_contact],
                       "transfers": reader.transfers})

    _add_adjectives_from_mentions(acts, scenes, tracker)
    catalog = _build_catalog(tracker, scenes)
    for scene in scenes:
        if not scene["props"] and not scene["transfers"]:
            scene["note"] = EMPTY_SCENE_NOTE

    return {
        "source": source,
        "method": "scripts/extract_hero_props.py (rule-based: handling verbs in action lines; no AI)",
        "about": "Props a character touches in each scene: the script's own adjectives, who touches each first, and "
                 "every transfer. Transfer types: handed, offered-declined, withheld. 'screenplay' is the action "
                 "sentence, verbatim.",
        "characters": characters,
        "catalog": catalog,
        "scenes": scenes,
    }


def _all_scenes(acts):
    for act in acts:
        for scene in act["scenes"]:
            yield act, scene


@dataclass
class Offer:
    prop: str
    giver: str
    sentence_index: int
    transfer: Optional[dict]  # the "offered-declined" transfer, removed if the offer is taken


class SceneReader:
    """Reads one scene's events, recording which props are touched and every transfer."""

    def __init__(self, events, scene_no, cast, patterns, tracker, debug=False):
        self.events = events
        self.scene_no = scene_no
        self.cast = cast
        self.patterns = patterns
        self.tracker = tracker
        self.debug = debug

        # results
        self.scene_props = []       # prop ids in order of first contact (or of being placed into)
        self.first_contact = {}     # prop id -> {subject, action, screenplay}
        self.transfers = []

        # state across sentences
        self.last_subject = None
        self.recent = []            # characters in order of last mention (in actions and cues)
        self.offers = []            # Offer, for "offers" / "holds out"
        self.last_cue = None
        self.after_cue = False      # the previous event was a cue: "He"/"She" means that speaker

        # state for the current sentence
        self.sentence = ""
        self.sentence_index = -1
        self.first_word = ""
        self.pronoun_subject = None
        self.used_spans = []        # text already read as a verb or its object

    def read(self):
        for event_index, (kind, text) in enumerate(self.events):
            if kind == "cue":
                self.last_cue, self.after_cue = text, True
                self._mentioned(text)
            elif kind == "action":
                self.sentence_index += 1
                self._read_sentence(text, event_index)

    def _mentioned(self, name):
        if name in self.recent:
            self.recent.remove(name)
        self.recent.append(name)

    # -- one sentence -----------------------------------------------------------------------------------------

    def _read_sentence(self, sentence, event_index):
        self.sentence = sentence
        words = sentence.split()
        self.first_word = words[0].lower().strip("—,") if words else ""
        self.pronoun_subject = self.last_cue if self.after_cue and self.last_cue else self.last_subject
        self.after_cue = False
        self.used_spans = []
        for label, *_ in self.cast.mentions(sentence):
            self._mentioned(label)
        if self.debug:
            print(f"  [{self.scene_no}] {sentence}")

        handled = False
        for start, kind, match in self.patterns.find(sentence):
            if any(a <= start < b for a, b in self.used_spans):
                continue  # inside an earlier verb or its object ("takes out" inside "takes", "the bread" ...)
            self.used_spans.append((match.start(), match.end()))
            handled = True
            if kind == "passive":
                self._passive(match, event_index)
            else:
                self._verb(kind, match)

        if not handled:
            subject = self.cast.first_subject(sentence)
            if subject:
                self.last_subject = subject
            elif self.first_word in {"he", "she", "they"}:
                self.last_subject = self.pronoun_subject
            self.tracker.note_mentions(sentence)

    def _verb(self, kind, match):
        """A handling verb: read its object and apply the verb kind."""
        verb = (match.groupdict().get("v") or kind).lower()
        who = self._subject_at(match.start())
        after = self.sentence[match.end():]
        np = parse_noun_phrase(after, self.cast.is_name)
        if not np or who is None:
            return
        self.used_spans.append((match.end(), match.end() + len(np.full)))
        pid = self.tracker.resolve(np, self.scene_props, self.sentence)
        if pid is None:
            return
        how = f"{verb} {np.phrase}"
        rest = after[len(np.phrase):]

        if kind in GIVE_KINDS:
            self._give(kind, match, verb, pid, np, who, how, rest)
        elif kind == "take":
            self._take(pid, who, how, rest)
        elif kind == "place":
            self._place(pid, who, how, rest)
        elif kind == "withhold":
            self._touch(pid, who, how)
            kept_from = self._other_than(who)
            self._transfer(pid, who, kept_from, "withheld", f"{who} {how}, keeping it from {kept_from}")
        else:  # carry, with, instrument: contact only
            self._touch(pid, who, how)
            self.tracker.holder[pid] = who

        self.tracker.last_prop = pid
        self.last_subject = who
        if self.debug:
            print(f"      {kind:10} {verb!r:18} subj={who!r:22} np={np.phrase!r:32} -> {pid}"
                  f"  (held by {self.tracker.holder_of(pid)})")

    # -- verb kinds -------------------------------------------------------------------------------------------

    def _passive(self, match, event_index):
        """ "A basket is placed in her hands": the giver is the next character to speak."""
        np = parse_noun_phrase(match.group("np"), self.cast.is_name)
        if not np:
            return
        owner = match.group("r").lower()
        to = self.last_subject if owner in {"her", "his", "their"} else self._person(owner, None)
        upcoming = self.events[event_index + 1:event_index + 1 + CUE_LOOKAHEAD]
        giver = next((name for kind, name in upcoming if kind == "cue" and name != to), None)
        pid = self.tracker.resolve(np, self.scene_props, self.sentence)
        self._touch(pid, to, f"{np.phrase} is placed in {to}'s hands")
        self._transfer(pid, giver, to, "handed", f"{giver} places {np.phrase} in {to}'s hands")
        if pid:
            self.tracker.holder[pid] = to

    def _give(self, kind, match, verb, pid, np, who, how, rest):
        if kind == "give2":  # "hands Miri the basket"
            to = self._person(match.group("r"), who)
        else:                # "hands the basket to Miri"
            target = self.patterns.target.search(rest.split(".")[0])
            to = self._person(target.group("r"), who) if target else None
        self._touch(pid, who, how + (f" to {to}" if to else ""))
        if verb in OFFER_VERBS:
            t = self._transfer(pid, who, to, "offered-declined", f"{who} offers {np.phrase} to {to}")
            self.offers.append(Offer(pid, who, self.sentence_index, t))
        elif to:
            self._transfer(pid, who, to, "handed", f"{who} {how} to {to}")
            self.tracker.holder[pid] = to

    def _take(self, pid, who, how, rest):
        """Taken "from X", or from someone who just offered it, or from whoever held it."""
        named = self.patterns.source.search(rest)
        offer = next((o for o in reversed(self.offers)
                      if self.sentence_index - o.sentence_index <= OFFER_WINDOW and o.giver != who), None)
        if named:
            giver = self._person(named.group(1), who)
        elif offer:
            giver = offer.giver
        else:
            giver = self.tracker.holder_of(pid)
        if offer and offer.transfer in self.transfers:
            self.transfers.remove(offer.transfer)  # the offer was taken: it becomes this hand-off
            offer.transfer = None
        self._touch(pid, who, how)
        self._transfer(pid, giver, who, "handed", f"{who} {how} from {giver}")
        self.tracker.holder[pid] = who

    def _place(self, pid, who, how, rest):
        """ "drops the peg into the basket": a transfer to the container."""
        into = INTO.search(rest)
        container_np = parse_noun_phrase(into.group("np"), self.cast.is_name) if into else None
        cid = self.tracker.resolve(container_np, self.scene_props, self.sentence) if container_np else None
        self._touch(pid, who, how)
        if cid and cid != pid:
            if cid not in self.scene_props:
                self.scene_props.append(cid)
            # the container is written as {"prop": id} for now; _build_catalog replaces it with the prop's name
            self._transfer(pid, who, {"prop": cid}, "handed", f"{who} {how} into the {container_np.phrase}")
            self.tracker.holder[pid] = cid

    # -- helpers ----------------------------------------------------------------------------------------------

    def _subject_at(self, pos):
        subject = self.cast.subject_before(self.sentence, pos)
        if subject:
            return subject
        if self.first_word in {"one", "another"}:
            return "one of the children"
        return self.pronoun_subject  # "He hands ...", "Then ..."

    def _other_than(self, who):
        """The most recently mentioned character who is not who."""
        return next((name for name in reversed(self.recent) if name != who), None)

    def _person(self, label, who):
        if label is None:
            return None
        if label.lower() in PERSON_PRONOUNS:
            return self._other_than(who)
        return self.cast.canonical(label)

    def _touch(self, pid, who, how):
        """Record first contact with a prop in this scene."""
        if pid and who:
            if pid not in self.scene_props:
                self.scene_props.append(pid)
            self.first_contact.setdefault(pid, {"subject": who, "action": how, "screenplay": self.sentence})

    def _transfer(self, pid, giver, receiver, kind, how):
        if pid is None or giver is None or receiver is None or giver == receiver:
            return None
        t = {"prop": pid, "from": giver, "to": receiver, "type": kind, "action": how, "screenplay": self.sentence}
        self.transfers.append(t)
        return t


# -- after all scenes are read ----------------------------------------------------------------------------------

_MENTION = re.compile(rf"((?:{WORD}\s+){{0,3}})({WORD})\b")  # up to three words, then a noun


def _add_adjectives_from_mentions(acts, scenes, tracker):
    """Collect adjectives from every mention of a prop in a scene where it is handled.

    The mention with the most adjectives becomes the description ("Three small HONEY-DATE CAKES").
    A noun is only credited when exactly one prop handled in the scene has it as its noun.
    """
    for (_, scene), out in zip(_all_scenes(acts), scenes):
        here = [p["id"] for p in out["props"]]
        for kind, sentence in scene["events"]:
            if kind != "action":
                continue
            for m in _MENTION.finditer(sentence):
                candidates = [pid for pid in here if tracker.props[pid].head == lemma(m.group(2))]
                if len(candidates) != 1:
                    continue
                prop = tracker.props[candidates[0]]
                found = [w.lower() for w in m.group(1).split() if is_adjective(w)]
                for adjective in found:
                    if adjective not in prop.adjectives:
                        prop.adjectives.append(adjective)
                if len(found) > sum(1 for w in prop.description.split() if is_adjective(w)):
                    words = m.group(0).split()
                    while words and not is_adjective(words[0]) and words[0].lower() not in DETERMINERS:
                        words.pop(0)  # drop leading verbs etc.: "clutches three small cakes" -> "three small cakes"
                    prop.description, prop.description_source = " ".join(words), sentence


def _build_catalog(tracker, scenes):
    """The catalog of props that appear in some scene; container endpoints in transfers become names."""
    catalog = {pid: {"name": p.name[:1].upper() + p.name[1:], "adjectives": p.adjectives,
                     "description": p.description, "description_source": p.description_source}
               for pid, p in tracker.props.items()}
    for scene in scenes:
        for t in scene["transfers"]:
            if isinstance(t["to"], dict):
                t["to"] = catalog[t["to"]["prop"]]["name"].lower()
    used = {p["id"] for s in scenes for p in s["props"]} | {t["prop"] for s in scenes for t in s["transfers"]}
    return {pid: entry for pid, entry in catalog.items() if pid in used}
