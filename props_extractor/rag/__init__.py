"""Retrieval-augmented generation for STEP 2 (props) and STEP 3 (hero decisions).

    scenes.py          STEP 2: for each scene, retrieve -> prompt -> generate -> check
    hero_decisions.py  STEP 3: for each prop, retrieve -> prompt -> the model decides hero or background
    retrieval.py       the screenplay as small passages, ranked by BM25 keyword search      (the "R")
    prompts.py         the instructions, JSON formats and per-request prompts             (the "A")
    llm.py             the model (OpenAI or Claude) behind one generate() call, cached    (the "G")
    catalog.py         checking answers against the cited screenplay lines; the props catalog
"""

from .hero_decisions import decide_hero_props
from .llm import LLMError, load_dotenv, make_llm
from .scenes import extract_props_rag

__all__ = ["LLMError", "decide_hero_props", "extract_props_rag", "load_dotenv", "make_llm"]
