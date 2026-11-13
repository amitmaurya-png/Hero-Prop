"""What the model is asked: the instructions, the JSON formats its answers must match, and the per-request prompts.

Changing any text here changes the requests, so the cached answers in data/rag_cache/ no longer match and the
next run asks the model again.
"""

TRANSFER_TYPES = ["handed", "offered-declined", "withheld", "placed", "implied"]

SYSTEM_PROMPT = """\
You are a prop master breaking down a screenplay, one scene at a time, for the art department.

For the TARGET SCENE, list every prop or material that a character physically touches (holds, carries, gives, \
takes, opens, eats, repairs, uses), and every transfer of a prop between two subjects. A subject is a character \
or an unnamed person (for example "small girl" or "a disciple"). Body parts, places, scenery and clothing that \
is only worn are not props.

You also get the props catalog from earlier scenes and passages retrieved from other scenes. Use them only as \
background: to recognise an object seen before (reuse its id: "the bundle" may be the "cloth_bundle"), to know \
who was holding it, and to collect the script's adjectives for it. Report only what happens in the target scene.

For each prop:
- id: the catalog id if it is the same object; otherwise a new short snake_case id naming the object \
(for example "honey_cakes").
- name: a short name ("Honey cakes").
- adjectives: describing words the script itself uses for this object, in this scene or the passages.
- inferred_adjectives: two to four concrete visual adjectives a prop maker needs (material, size, colour, \
condition), consistent with the script and its setting; leave empty if nothing is reasonable to infer.
- description: a short noun phrase describing the object, preferably the script's own fullest wording.
- first_contact: the first subject in this scene to touch it, what they do ("clutches the bundle"), and the \
number of the line that shows it.

For each transfer, give the prop id, from, to, a type, a short action ("Ima hands the basket to Miri") and the \
line number that shows it. Types:
- handed: it passes from one subject to another (given, tossed, taken from someone, received).
- offered-declined: offered but not accepted.
- withheld: deliberately kept back from a subject who wants it.
- placed: put into or onto another prop (to = that prop's name).
- implied: the scene clearly implies a hand-off without showing it; cite the line that implies it.

Use character names exactly as in the cast list. Cite only line numbers of the target scene. If no one touches \
anything, return empty lists."""

_FIRST_CONTACT = {
    "type": "object", "additionalProperties": False, "required": ["subject", "action", "line"],
    "properties": {"subject": {"type": "string"}, "action": {"type": "string"}, "line": {"type": "integer"}},
}
SCENE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["props", "transfers"],
    "properties": {
        "props": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "name", "adjectives", "inferred_adjectives", "description", "first_contact"],
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                "adjectives": {"type": "array", "items": {"type": "string"}},
                "inferred_adjectives": {"type": "array", "items": {"type": "string"}},
                "description": {"type": "string"},
                "first_contact": _FIRST_CONTACT,
            }}},
        "transfers": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["prop", "from", "to", "type", "action", "line"],
            "properties": {
                "prop": {"type": "string"},
                "from": {"type": "string"},
                "to": {"type": "string"},
                "type": {"type": "string", "enum": TRANSFER_TYPES},
                "action": {"type": "string"},
                "line": {"type": "integer"},
            }}},
    },
}

HERO_SYSTEM_PROMPT = """\
You are a prop master deciding which props in a screenplay are hero props.

A hero prop is an object the story depends on: characters pass it between them, it drives or resolves the plot, \
it carries the theme, it recurs, or the camera will feature it so it must be made with care. A background prop is \
handled only incidentally or is set dressing.

You get one prop: its catalog entry, every scene where a character touches it, every transfer, and passages \
retrieved from the screenplay that mention it. Decide from that evidence.
- is_hero: true or false.
- reasons: one to three short reasons, each grounded in the evidence (name who passes it, or the scene).
- story_role: a short phrase for what the prop does in the story ("the lunch that feeds the crowd")."""

HERO_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["is_hero", "reasons", "story_role"],
    "properties": {
        "is_hero": {"type": "boolean"},
        "reasons": {"type": "array", "items": {"type": "string"}},
        "story_role": {"type": "string"},
    },
}


# -- per-request prompts --------------------------------------------------------------------------------------------

def scene_prompt(number, total, heading, lines, retrieved, characters, memory):
    catalog = "\n".join(
        f"- {pid}: {c['name']} ({', '.join(c['adjectives'] + c['inferred_adjectives']) or 'no adjectives'}); "
        f"last held by {memory.holder.get(pid, 'unknown')}"
        for pid, c in memory.catalog.items()) or "(none yet)"
    numbered = "\n".join(f"L{i}: {line}" for i, line in enumerate(lines, 1)) or "(no lines)"
    return (f"Cast: {', '.join(characters)}\n\n"
            f"<catalog>\n{catalog}\n</catalog>\n\n"
            f"<retrieved_passages>\n{format_passages(retrieved)}\n</retrieved_passages>\n\n"
            f"<target_scene number=\"{number}\" of=\"{total}\" heading=\"{heading}\">\n{numbered}\n</target_scene>")


def format_passages(retrieved):
    return "\n\n".join(f"[{p.id}] {p.scene}\n{p.text}" for _, p in retrieved) or "(none)"


def hero_prompt(pid, entry, scenes, transfers, retrieved):
    adjectives = ", ".join(entry["adjectives"] + entry.get("inferred_adjectives", [])) or "none"
    touched = "\n".join(f"- scene {s['scene_number']} ({s['scene']}): {s['first_contact']['subject']} "
                        f"{s['first_contact']['action']} — \"{s['first_contact']['screenplay']}\""
                        for s in scenes) or "(none)"
    passed = "\n".join(f"- scene {t['scene_number']}: {t['type']} from {t['from']} to {t['to']} — "
                       f"\"{t['screenplay']}\"" for t in transfers) or "(none)"
    return (f"<prop id=\"{pid}\">\nName: {entry['name']}\nAdjectives: {adjectives}\n"
            f"Description: {entry['description']}\n</prop>\n\n"
            f"<scenes_touched count=\"{len(scenes)}\">\n{touched}\n</scenes_touched>\n\n"
            f"<transfers count=\"{len(transfers)}\">\n{passed}\n</transfers>\n\n"
            f"<retrieved_passages>\n{format_passages(retrieved)}\n</retrieved_passages>")
