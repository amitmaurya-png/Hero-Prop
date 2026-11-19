"""Extract hero props from a screenplay PDF and write data/hero_props.json.

The code lives in the hero_props/ package (one module per step); this file
just runs it. See hero_props/__init__.py for the layout and
`python extract_hero_props.py --help` for options.
"""

from hero_props.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
