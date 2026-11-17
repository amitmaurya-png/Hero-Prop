# props_skit

Finds the **hero props** in the screenplay *Little Witnesses, Episode 1: The Smallest Lunch* (`docs/TSL.pdf`).
These are the objects the characters touch and hand to each other, such as the linen basket and the honey cakes.

It uses retrieval-augmented generation (RAG): a language model extracts every prop and transfer, and also decides which props are hero props. It works with OpenAI or Claude.

## Run

```sh
pip install -r requirements.txt
# put OPENAI_API_KEY=... (or ANTHROPIC_API_KEY=...) in .env; the script reads it

python3 scripts/extract_hero_props.py                   # PDF -> data/props.json -> data/hero_props.json
python3 scripts/extract_hero_props.py --dry-run         # write each scene's prompt to data/rag_prompts/, no API calls
python3 scripts/extract_hero_props.py --provider openai --model gpt-5.5
python3 scripts/extract_hero_props.py --from-json       # skip the PDF, redo only the hero step from props.json
python3 -m unittest discover tests                      # run the tests (no API calls)
```

The model is chosen from the API keys:

| Key set | Model used |
| --- | --- |
| `OPENAI_API_KEY` | `gpt-5.5` |
| `ANTHROPIC_API_KEY` | `claude-opus-5-5` |
| Both | Claude, unless you pass `--provider` |

## Files

| Path | What it is |
| --- | --- |
| `docs/TSL.pdf` | The screenplay (input) |
| `data/props.json` | Every prop touched in each scene: who touches it first, adjectives, and every transfer |
| `data/hero_props.json` | Hero props vs background props, with reasons, totals and per-scene counts |
| `data/rag_cache/` | The model's answers, saved so that re-running makes no new API calls. Delete the folder to ask again. |
| `data/props.manual.json` | The earlier hand-curated version of props.json, kept for comparison. The script never writes to it. |
| `.env` | API keys. Keep it private and out of version control. |
| `scripts/extract_hero_props.py` | Command-line entry point |
| `props_extractor/` | The extraction logic (layout below) |
| `tests/` | Unit tests. They use a fake model, so they make no API calls. |

## Code layout

```
props_extractor/
  screenplay.py      STEP 1: PDF -> scenes
  cast.py            character names
  vocabulary.py      word lists and lemma() for the BM25 search
  hero.py            STEP 3: builds hero_props.json, all counts computed by code
  rag/               the RAG method
    scenes.py          STEP 2: retrieve -> prompt -> generate -> check, scene by scene
    hero_decisions.py  STEP 3: the model decides hero or background, prop by prop
    retrieval.py       passages + BM25 search             (the R)
    prompts.py         instructions, JSON formats, prompts (the A)
    llm.py             OpenAI / Claude call + answer cache (the G)
    catalog.py         checks answers against cited lines; the props catalog
```

## How it works

**Step 1: read the PDF** (`screenplay.py`). The PDF is read with its indentation kept. Each scene becomes numbered lines: action sentences, and dialogue such as `MIRI: Maybe later.`

**Step 2: props in each scene** (`rag/scenes.py`). The scenes are processed in script order:

1. **Retrieve.** The whole script is split into passages of 4 lines each. A BM25 keyword search finds the 6 passages from other scenes that best match the scene being read.
2. **Augment.** The model gets three things:
   - the scene's numbered lines;
   - those passages;
   - the props found so far, with their ids and who held each one last, so a prop keeps the same id.
3. **Generate.** The model returns JSON in a fixed format. For each prop:
   - the script's own adjectives;
   - inferred visual adjectives, kept in a separate list;
   - who touches it first.

   Transfers come in five types: *handed, offered-declined, withheld, placed*, or *implied*.
4. **Check.** Every item must cite a line number of the scene. The code copies that line into `screenplay`; the model never writes it. Items citing a line that doesn't exist, or a prop that doesn't exist, are dropped and listed under `warnings`.

**Step 3: hero props** (`rag/hero_decisions.py`, then `hero.py`). For each prop:

1. BM25 retrieves the passages that mention it.
2. The model gets those passages, plus every scene where the prop is touched and every transfer.
3. The model decides **hero** or **background**, gives its reasons, and names the prop's role in the story.

All counts and totals are computed by code, not by the model.

## Tuning

- Edit `SYSTEM_PROMPT` and `HERO_SYSTEM_PROMPT` in `rag/prompts.py` to change what the model looks for.
- Edit `TOP_K` (`rag/scenes.py`), `HERO_TOP_K` (`rag/hero_decisions.py`) or `PASSAGE_LINES` (`rag/retrieval.py`) to change how much context it gets.
- A changed prompt doesn't match the saved answers, so the next run asks the model again.
- Run with `--debug` to see which passages were retrieved for each scene and prop.

## Limits

- A run with an empty cache makes one API call per scene and one per prop.
- Inferred adjectives and implied hand-offs are the model's reading of the script. Check them against the cited line.
