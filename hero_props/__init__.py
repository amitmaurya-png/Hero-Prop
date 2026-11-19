"""Hero-prop extraction from a screenplay PDF.

The pipeline, one module per step:

    config.py      settings, hero-prop criteria and prompt rules  <- edit here
    screenplay.py  1. PDF -> numbered scenes
    prompt.py      2a. instructions + answer format for the model
    llm.py         2b. the OpenAI request
    verify.py      3. check every scene / transfer against the screenplay
    output.py      4. build and write the final JSON
    cli.py         command line, debug files, --from-answer

Run with `python extract_hero_props.py` or `python -m hero_props`.
"""
