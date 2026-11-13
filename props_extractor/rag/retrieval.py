"""The "R" in RAG: the screenplay split into small passages, searchable by keyword (BM25).

Each passage is a few consecutive lines of one scene (action and dialogue). Given a query, such as the text of the
scene being extracted, the index returns the passages from other scenes that share the most distinctive words
with it. For the honey-cake scene, that brings back the earlier lines that introduce the cakes and the cloth bundle.

BM25 is a standard keyword-ranking formula: a passage scores higher for each query word it contains, more for rare
words ("linen", "peg") than for common ones ("Miri", "looks"), and long passages are not favoured just for length.
It runs locally: no embeddings API or extra package is needed.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass

from ..screenplay import scene_lines
from ..vocabulary import DETERMINERS, STOP, WORD, lemma

PASSAGE_LINES = 4   # lines per passage
PASSAGE_STEP = 3    # start a new passage every 3 lines, so neighbouring passages share one line

_IGNORED = DETERMINERS | STOP | {"he", "she", "it", "they", "him", "her", "them", "i", "you", "we", "me", "his",
                                 "not", "be", "does", "do", "did", "just", "very", "what", "where", "when"}


@dataclass
class Passage:
    id: str             # "S04.L07-L10": scene 4, lines 7 to 10
    scene_number: int
    scene: str          # the scene heading
    text: str


def build_passages(acts):
    """Split every scene into overlapping passages of PASSAGE_LINES lines."""
    passages, number = [], 0
    for act in acts:
        for scene in act["scenes"]:
            number += 1
            lines = scene_lines(scene)
            for start in range(0, max(len(lines) - PASSAGE_LINES, 0) + 1, PASSAGE_STEP):
                chunk = lines[start:start + PASSAGE_LINES]
                if not chunk:
                    continue
                end = start + len(chunk)
                passages.append(Passage(id=f"S{number:02}.L{start + 1:02}-L{end:02}", scene_number=number,
                                        scene=scene["scene"], text="\n".join(chunk)))
    return passages


def tokenize(text):
    """Content words, singular and lower-case: "the honey cakes" -> ["honey", "cake"]."""
    return [lemma(w) for w in re.findall(WORD, text) if w.lower() not in _IGNORED and len(w) > 1]


class BM25Index:
    """Ranks passages for a query with BM25 (k1 = 1.5, b = 0.75, the usual defaults)."""

    def __init__(self, passages, k1=1.5, b=0.75):
        self.passages = passages
        self.k1, self.b = k1, b
        self._docs = [Counter(tokenize(p.text)) for p in passages]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0
        df = Counter(word for d in self._docs for word in d)
        n = len(self._docs)
        self._idf = {word: math.log(1 + (n - count + 0.5) / (count + 0.5)) for word, count in df.items()}

    def score(self, query_words, i):
        doc, length = self._docs[i], self._lengths[i]
        total = 0.0
        for word in query_words:
            tf = doc.get(word, 0)
            if tf:
                norm = tf + self.k1 * (1 - self.b + self.b * length / self._avg_length)
                total += self._idf[word] * tf * (self.k1 + 1) / norm
        return total

    def search(self, query, k, exclude_scene=None):
        """The k best passages for query, best first, as [(score, passage)]; passages of exclude_scene are skipped."""
        words = set(tokenize(query))
        scored = [(self.score(words, i), i) for i, p in enumerate(self.passages) if p.scene_number != exclude_scene]
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [(round(score, 3), self.passages[i]) for score, i in scored[:k] if score > 0]
