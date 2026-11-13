"""The rules method: hand-written verb patterns and word lists, no AI.

    extractor.py    STEP 2: reads each scene and builds props.json (SceneReader)
    verbs.py        the handling verbs ("hands", "takes", "drops ... into") as regular expressions
    noun_phrase.py  reading the object after a verb ("a basket of folded linen")
    tracker.py      the props found so far, matching a new mention to one of them, and who holds each
"""

from .extractor import extract_props

__all__ = ["extract_props"]
