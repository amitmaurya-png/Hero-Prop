"""Find the hero props in a screenplay PDF.

STEP 1  screenplay.py           PDF -> acts, scenes, action sentences, dialogue and character cues
STEP 2  rag/scenes.py           scenes -> props.json: every prop a character touches, and every transfer
STEP 3  rag/hero_decisions.py   the model decides which props are hero props
        hero.py                 props.json + decisions -> hero_props.json, with all counts computed by code

    rag/        BM25 retrieval + a language model; see rag/__init__.py

Shared:
    screenplay.py   reading the PDF
    cast.py         character names
    vocabulary.py   word lists and lemma() for the BM25 search
    hero.py         building hero_props.json
"""
