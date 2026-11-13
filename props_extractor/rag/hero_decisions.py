"""STEP 3 (RAG): which props are hero props. For each prop:

  1. Retrieve   BM25 finds the passages anywhere in the script that mention the prop.
  2. Augment    The prompt holds the prop's catalog entry, every scene it is touched in and every transfer.
  3. Generate   The model decides hero or background, with reasons and the prop's role in the story.

hero.py then builds hero_props.json from these decisions, with all counts and totals computed by code.
"""

from concurrent.futures import ThreadPoolExecutor

from ..hero import collect_evidence
from .prompts import HERO_SCHEMA, HERO_SYSTEM_PROMPT, hero_prompt
from .retrieval import BM25Index, build_passages

HERO_TOP_K = 8    # passages retrieved per prop
HERO_WORKERS = 6  # decisions are independent, so they run in parallel

HERO_ABOUT = ("Hero props and materials, decided by the RAG method: for each prop, the model weighs its scenes, "
              "transfers and retrieved screenplay passages ('why_hero' / 'why_not' are its reasons). Counts and "
              "totals are computed by code. Written by scripts/extract_hero_props.py.")


def decide_hero_props(acts, props, llm, *, debug=False):
    """Ask the model, prop by prop, whether it is a hero prop. Returns id -> {is_hero, reasons, story_role}."""
    index = BM25Index(build_passages(acts))
    scenes, transfers = collect_evidence(props)

    def decide(pid):
        entry = props["catalog"][pid]
        query = " ".join([entry["name"], entry["description"], *entry["adjectives"]])
        retrieved = index.search(query, HERO_TOP_K)
        prompt = hero_prompt(pid, entry, scenes[pid], transfers[pid], retrieved)
        if debug:
            print(f"      {pid}: retrieved " + ", ".join(p.id for _, p in retrieved))
        return llm.generate(HERO_SYSTEM_PROMPT, prompt, HERO_SCHEMA, name="hero_decision")

    pids = list(props["catalog"])
    print(f"  deciding {len(pids)} props", flush=True)
    with ThreadPoolExecutor(max_workers=HERO_WORKERS) as pool:
        answers = list(pool.map(decide, pids))
    return {pid: {"is_hero": a["is_hero"], "reasons": [r.strip() for r in a["reasons"] if r.strip()],
                  "story_role": a["story_role"].strip()}
            for pid, a in zip(pids, answers)}
