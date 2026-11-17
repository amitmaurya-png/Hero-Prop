"""The screenplay's characters, and writing their names one way."""

import re


class Cast:
    """The screenplay's characters, as named in their cues."""

    def __init__(self, characters):
        self.names = characters

    def canonical(self, label):
        """ "miri" -> "Miri"; an unnamed person keeps its words, minus a leading "the"."""
        lower = label.lower()
        return next((c for c in self.names if c.lower() == lower), re.sub(r"^the ", "", lower))
