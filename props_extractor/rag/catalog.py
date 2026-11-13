"""Checking the model's answers against the screenplay, and the catalog of props they build up.

Every answer cites a line number of the scene. The line's text is copied into "screenplay" here, not by the
model; answers citing a line that doesn't exist, or a prop that doesn't exist, are dropped with a warning.
"""

import re

# Transfer types after which the receiver holds the prop
CHANGES_HOLDER = {"handed", "placed", "implied"}
EMPTY_SCENE_NOTE = "No character handles an object in this scene."


class PropMemory:
    """The catalog and last known holders, built up scene by scene from checked answers."""

    def __init__(self, cast):
        self.cast = cast
        self.catalog = {}   # id -> {name, adjectives, inferred_adjectives, description, description_source}
        self.holder = {}    # id -> last known holder (a subject or a container prop's name)

    def add_scene(self, act, heading, lines, answer):
        """Check one scene's answer against its lines and merge it in. Returns (props.json scene entry, warnings)."""
        check = _LineCheck(lines)
        props = self._accept_props(answer.get("props", []), check)
        transfers = self._accept_transfers(answer.get("transfers", []), check)
        scene = {"act": act, "scene": heading, "props": props, "transfers": transfers}
        if not props and not transfers:
            scene["note"] = EMPTY_SCENE_NOTE
        return scene, check.warnings

    # -- props ------------------------------------------------------------------------------------------------------

    def _accept_props(self, answers, check):
        props, seen = [], set()
        for p in answers:
            pid = snake_case(p["id"])
            contact = p["first_contact"]
            quote = check.quote(contact["line"], f"prop {pid}")
            if not pid or quote is None or pid in seen or not contact["subject"].strip():
                continue
            seen.add(pid)
            subject = self._endpoint(contact["subject"])
            self._add_to_catalog(pid, p, quote)
            props.append({"id": pid,
                          "first_contact": {"subject": subject, "action": contact["action"], "screenplay": quote}})
            self.holder.setdefault(pid, subject)
        return props

    def _add_to_catalog(self, pid, p, quote):
        """A new prop gets an entry; a known prop gains any new adjectives."""
        if pid not in self.catalog:
            name = p["name"].strip() or pid.replace("_", " ")
            self.catalog[pid] = {"name": name[:1].upper() + name[1:], "adjectives": [], "inferred_adjectives": [],
                                 "description": p["description"].strip() or name, "description_source": quote}
        entry = self.catalog[pid]
        for key in ("adjectives", "inferred_adjectives"):
            for word in p[key]:
                word = word.strip().lower()
                if word and word not in entry[key] and word not in entry["adjectives"]:
                    entry[key].append(word)

    # -- transfers --------------------------------------------------------------------------------------------------

    def _accept_transfers(self, answers, check):
        transfers = []
        for t in answers:
            pid = snake_case(t["prop"])
            quote = check.quote(t["line"], f"transfer of {pid}")
            if quote is None:
                continue
            if pid not in self.catalog:
                check.warn(f"dropped transfer of unknown prop {pid!r}")
                continue
            giver, receiver = self._endpoint(t["from"]), self._endpoint(t["to"])
            if not giver or not receiver or giver == receiver:
                check.warn(f"dropped transfer of {pid}: from {giver!r} to {receiver!r}")
                continue
            entry = {"prop": pid, "from": giver, "to": receiver, "type": t["type"], "action": t["action"],
                     "screenplay": quote}
            if entry not in transfers:
                transfers.append(entry)
            if t["type"] in CHANGES_HOLDER:
                self.holder[pid] = receiver
        return transfers

    def _endpoint(self, value):
        """A character's canonical name, a prop's name (for "placed into"), or an unnamed person's words."""
        value = value.strip()
        pid = snake_case(value)
        if pid in self.catalog:
            return self.catalog[pid]["name"].lower()
        for entry in self.catalog.values():
            if entry["name"].lower() == value.lower():
                return entry["name"].lower()
        return self.cast.canonical(value) if value else ""


class _LineCheck:
    """Looks up cited line numbers in one scene, collecting a warning for each bad citation."""

    def __init__(self, lines):
        self.lines = lines
        self.warnings = []

    def quote(self, number, what):
        """The text of line `number`, or None (with a warning) if the scene has no such line."""
        if isinstance(number, int) and 1 <= number <= len(self.lines):
            return self.lines[number - 1]
        self.warn(f"dropped {what}: cites line {number}, but the scene has {len(self.lines)} lines")
        return None

    def warn(self, message):
        self.warnings.append(message)


def snake_case(text):
    """ "Honey Cakes" -> honey_cakes."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
