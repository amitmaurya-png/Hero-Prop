"""STEP 2 (RAG): scenes -> props.json. For each scene, in script order:

  1. Retrieve   BM25 finds the passages in other scenes that best match this scene's text (retrieval.py):
                where a prop was first described, who held it last, what it was called before.
  2. Augment    The prompt holds the scene's numbered lines, the retrieved passages, and the catalog of props
                found in earlier scenes with their last known holder, so a prop keeps one id (prompts.py).
  3. Generate   The model returns the scene's props and transfers as JSON matching SCENE_SCHEMA (llm.py).
  4. Check      Each answer is checked against the line it cites and merged into the catalog (catalog.py).
"""

from pathlib import Path

from ..cast import Cast
from ..screenplay import scene_lines
from .catalog import PropMemory
from .prompts import SCENE_SCHEMA, SYSTEM_PROMPT, TRANSFER_TYPES, scene_prompt
from .retrieval import BM25Index, build_passages

TOP_K = 6  # passages retrieved per scene
NO_PROPS = {"props": [], "transfers": []}


def extract_props_rag(acts, characters, source, llm, *, dry_run_dir=None, debug=False):
    """Run the RAG method over every scene and return the props.json document.

    dry_run_dir: write each scene's prompt there instead of calling the model (cached answers are still used, so
    the catalog in later prompts is as complete as the cache allows). Returns None in that case.
    """
    index = BM25Index(build_passages(acts))
    memory = PropMemory(Cast(characters))
    scenes = [(act, scene) for act in acts for scene in act["scenes"]]
    scenes_out, warnings = [], []
    if dry_run_dir:
        _write(Path(dry_run_dir) / "system_prompt.txt", SYSTEM_PROMPT)

    for number, (act, scene) in enumerate(scenes, 1):
        print(f"  scene {number:2}/{len(scenes)}: {scene['scene']}", flush=True)
        lines = scene_lines(scene)

        retrieved = index.search("\n".join(lines), TOP_K, exclude_scene=number)                        # 1
        prompt = scene_prompt(number, len(scenes), scene["scene"], lines, retrieved, characters, memory)  # 2
        if debug:
            print("      retrieved " + ", ".join(f"{p.id} ({s})" for s, p in retrieved))

        if dry_run_dir:                                                                                # 3
            _write(Path(dry_run_dir) / f"scene_{number:02}.txt", prompt)
            answer = llm.cached(SYSTEM_PROMPT, prompt, SCENE_SCHEMA)
            if answer is None:
                continue
        else:
            answer = llm.generate(SYSTEM_PROMPT, prompt, SCENE_SCHEMA, name="scene_props") if lines else NO_PROPS

        entry, scene_warnings = memory.add_scene(act["act"], scene["scene"], lines, answer)            # 4
        entry["retrieved_passages"] = [p.id for _, p in retrieved]
        scenes_out.append(entry)
        warnings += [f"scene {number}: {w}" for w in scene_warnings]

    for w in warnings:
        print(f"  warning: {w}")
    if dry_run_dir:
        return None
    return _props_document(source, llm, characters, memory, scenes_out, warnings)


def _props_document(source, llm, characters, memory, scenes, warnings):
    return {
        "source": source,
        "method": f"scripts/extract_hero_props.py --method rag (BM25 retrieval + {llm.name}; answers grounded to "
                  f"cited screenplay lines)",
        "about": "Props a character touches in each scene: adjectives from the script (plus inferred visual "
                 "adjectives for prop makers), who touches each first, and every transfer. Transfer types: "
                 + ", ".join(TRANSFER_TYPES) + ". 'screenplay' is the cited screenplay line, verbatim. "
                 "'retrieved_passages' are the passages from other scenes that were given to the model.",
        "characters": characters,
        "catalog": memory.catalog,
        "scenes": scenes,
        "warnings": warnings,
    }


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
