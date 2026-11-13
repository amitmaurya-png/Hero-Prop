"""Find the hero props in a screenplay PDF.

STEP 1  screenplay.py       PDF -> acts, scenes, action sentences, dialogue and character cues
STEP 2  rag/ or rules/      scenes -> props.json: every prop a character touches, and every transfer
STEP 3  rag/ or rules       which props are hero props (rag/hero_decisions.py, or the rule in hero.py)
        hero.py             props.json (+ decisions) -> hero_props.json, with all counts computed by code

    rag/        the RAG method (default): BM25 retrieval + a language model; see rag/__init__.py
    rules/      the rules method: verb patterns and word lists, no AI; see rules/__init__.py

Shared by both methods:
    screenplay.py   reading the PDF
    cast.py         character names, and which one is the subject of a sentence
    vocabulary.py   word lists and lemma()
    hero.py         building hero_props.json
"""
