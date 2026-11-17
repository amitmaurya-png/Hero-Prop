"""STEP 1: screenplay PDF -> acts -> scenes -> events.

pypdf's layout mode keeps each line's indentation, which is how a screenplay marks what a line is:

    INT. MIRI'S HOUSE — DAY          <- scene heading, left margin
    Miri clutches a small bundle.    <- action, left margin
                MIRI                 <- character cue, indented
          It's for later.            <- dialogue, indented

Each scene becomes a list of events in script order:
    ("action", "Miri clutches a small bundle.")   one per sentence
    ("cue", "Miri")                               who speaks next
    ("dialogue", "It's for later.")               only with keep_dialogue=True
"""

import re

SCENE_HEADING = re.compile(r"^(?:INT\.|EXT\.|INT/EXT\.|I/E\.)\s.+|^MAIN TITLES$")
ACT_HEADING = re.compile(r"^(?:TEASER|COLD OPEN|TAG|ACT (?:ONE|TWO|THREE|FOUR|FIVE|SIX|[0-9]+))$")
END_MARK = re.compile(r"^(?:FADE OUT\.|THE END|CUT TO BLACK\.)$")
# An all-caps word that is not inside a parenthetical like (SOFTLY) or (V.O.)
CUE_WORD = re.compile(r"(?<![A-Za-z(’'])[A-Z][A-Z’'\-]+(?![a-z)’',A-Z])")
PAGE_NUMBER = re.compile(r"^\d+\.?$")
SENTENCE_BREAK = re.compile(r"(?<=[.!?…])\s+(?=[A-Z“\"])|:\s+(?=[a-z])")
MIN_INDENT = 6  # dialogue and cue lines are indented at least this far; action lines are not


def read_screenplay(pdf_path, keep_dialogue=False):
    """Return (acts, characters).

    acts: [{"act": "ACT ONE", "scenes": [{"scene": "EXT. VILLAGE — MORNING", "events": [...]}]}]
    characters: every name that appears as a cue, longest first ("Small Girl" before "Girl").
    """
    acts = _split_into_scenes(_pdf_lines(pdf_path), keep_dialogue)
    names = set()
    for act in acts:
        for scene in act["scenes"]:
            events = []
            for kind, text in scene["events"]:
                if kind == "cue":
                    name = _display_name(text)
                    names.add(name)
                    events.append(("cue", name))
                elif kind == "dialogue":
                    if events and events[-1][0] == "dialogue":  # one speech, split over several lines
                        events[-1] = ("dialogue", f"{events[-1][1]} {text}")
                    else:
                        events.append(("dialogue", text))
                else:
                    events += [("action", sentence) for sentence in split_sentences(text)]
            scene["events"] = events
    characters = sorted(names, key=lambda name: (-len(name), name))
    return acts, characters


def split_sentences(text):
    """Split a paragraph into sentences. "She stops: then turns" splits at the colon too."""
    text = re.sub(r"\s+", " ", text).strip()
    return [part.strip() for part in SENTENCE_BREAK.split(text) if part.strip()]


def _pdf_lines(pdf_path):
    from pypdf import PdfReader  # imported here so --from-json works without pypdf installed

    lines = []
    for page in PdfReader(str(pdf_path)).pages:
        lines += page.extract_text(extraction_mode="layout").splitlines()
    return lines


def scene_lines(scene):
    """A scene as readable lines: action sentences as they are, speech as "MIRI: It's for later." """
    lines, speaker = [], None
    for kind, text in scene["events"]:
        if kind == "cue":
            speaker = text
        elif kind == "dialogue":
            lines.append(f"{(speaker or 'SOMEONE').upper()}: {text}")
        else:
            lines.append(text)
    return lines


def _split_into_scenes(lines, keep_dialogue):
    """Group lines into acts and scenes. Action paragraphs are joined; cues are kept in caps for now."""
    acts, act, scene, paragraph = [], None, None, []

    def end_paragraph():
        if paragraph and scene is not None:
            scene["events"].append(("action", " ".join(paragraph)))
        paragraph.clear()

    for raw in lines:
        text = raw.strip()
        indented = len(raw) - len(raw.lstrip()) >= MIN_INDENT

        if not text:
            end_paragraph()
        elif not indented and ACT_HEADING.match(text):
            end_paragraph()
            act = {"act": text, "scenes": []}
            acts.append(act)
        elif not indented and SCENE_HEADING.match(text):
            end_paragraph()
            if act is None:  # a scene before any act heading
                act = {"act": "TEASER", "scenes": []}
                acts.append(act)
            scene = {"scene": text, "events": []}
            act["scenes"].append(scene)
        elif scene is None or END_MARK.match(text):
            end_paragraph()
        elif indented:  # a cue, a line of dialogue, or a page number
            end_paragraph()
            # Layout mode can put the cue on the same line as dialogue ("Blessed be MOTHER the Lord.IN CROWD"),
            # so the cue is every all-caps word on the line, joined.
            cue = " ".join(m.group(0) for m in CUE_WORD.finditer(text))
            if cue:
                scene["events"].append(("cue", cue))
            if keep_dialogue:
                spoken = re.sub(r"\s+", " ", CUE_WORD.sub("", text)).strip()
                if spoken and not PAGE_NUMBER.match(spoken):
                    scene["events"].append(("dialogue", spoken))
        else:
            paragraph.append(text)
    end_paragraph()
    return acts


def _display_name(cue):
    """MOTHER IN CROWD -> Mother In Crowd; SABA’S -> Saba’s."""
    return cue.title().replace("’S", "’s")
