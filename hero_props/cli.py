"""Command line: runs the four steps and saves debug files.

    python extract_hero_props.py                       # normal run
    python extract_hero_props.py --reasoning low       # faster, less careful
    python extract_hero_props.py --from-answer data/debug/answer.json
                                                       # re-check a saved answer, no API call
    python extract_hero_props.py --show-scenes         # print how the PDF was split
    python extract_hero_props.py --show-prompt         # print the instructions sent

Every run that calls the model writes data/debug/:
    prompt.txt      the instructions sent
    screenplay.txt  the numbered screenplay sent
    answer.json     the model's raw answer (before any checks)
    run.json        model, reasoning effort, seconds, token usage
    warnings.txt    everything the checks dropped, and why
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .config import DEBUG_DIR, DEFAULT_MODEL, DEFAULT_OUT, DEFAULT_PDF, DEFAULT_REASONING, REASONING_LEVELS, ROOT
from .output import build_output, write_json
from .prompt import build_prompt
from .screenplay import numbered_screenplay, read_scenes


def log(message: str) -> None:
    print(message, file=sys.stderr)


def load_env(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file (existing variables win)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="screenplay PDF")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output JSON file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")
    parser.add_argument("--reasoning", choices=REASONING_LEVELS, default=DEFAULT_REASONING,
                        help="reasoning effort (default: the model's own default)")
    parser.add_argument("--from-answer", type=Path, metavar="ANSWER_JSON",
                        help="skip the API and re-check a saved raw answer (e.g. data/debug/answer.json)")
    parser.add_argument("--debug-dir", type=Path, default=DEBUG_DIR, help="where debug files are written")
    parser.add_argument("--show-scenes", action="store_true", help="print the scenes found in the PDF and exit")
    parser.add_argument("--show-prompt", action="store_true", help="print the instructions sent to the model and exit")
    return parser.parse_args(argv)


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def show_scenes(title: str, scenes: list[dict]) -> None:
    print(title)
    for s in scenes:
        print(f"{s['number']:>3}  {s['act']:<10}  {s['heading']}  ({len(s['text'].split())} words)")


def get_answer(args: argparse.Namespace, screenplay: str) -> tuple[dict, dict]:
    """Return (answer, run_info): from a saved file, or by calling the model."""
    if args.from_answer:
        log(f"Re-checking saved answer {args.from_answer} (no API call)")
        answer = json.loads(args.from_answer.read_text())
        run_file = args.from_answer.with_name("run.json")
        run_info = json.loads(run_file.read_text()) if run_file.exists() else {"model": args.model}
        return answer, run_info

    load_env(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set (environment or .env).")

    from .llm import ask_model   # imported here so --from-answer works without the API set up

    reasoning = args.reasoning or "model default"
    log(f"Asking {args.model} (reasoning: {reasoning})...")
    reply = ask_model(args.model, screenplay, args.reasoning)
    run_info = {"model": args.model, "reasoning_effort": reasoning,
                "seconds": round(reply.seconds, 1), "usage": reply.usage}

    debug = args.debug_dir
    debug.mkdir(parents=True, exist_ok=True)
    (debug / "prompt.txt").write_text(build_prompt())
    (debug / "screenplay.txt").write_text(screenplay)
    write_json(reply.answer, debug / "answer.json")
    write_json(run_info, debug / "run.json")
    return reply.answer, run_info


def report(output: dict, warnings: list[str], out: Path, run_info: dict) -> None:
    for w in warnings:
        log(f"warning: {w}")
    for p in output["hero_props"]:
        flags = ",".join(k for k, v in p["criteria"].items() if v)
        log(f"- {p['name']}: scenes {p['scenes']} [{flags}]")
    timing = f", LLM request {run_info['seconds']}s" if "seconds" in run_info else ""
    log(f"Wrote {output['hero_prop_count']} hero props to {out} "
        f"(reasoning: {run_info.get('reasoning_effort', 'n/a')}{timing})")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Step 1: PDF -> scenes
    title, scenes = read_scenes(args.pdf)
    if args.show_scenes:
        show_scenes(title, scenes)
        return 0
    if args.show_prompt:
        print(build_prompt())
        return 0
    log(f"{title}: {len(scenes)} scenes")

    # Step 2: ask the model (or load a saved answer)
    answer, run_info = get_answer(args, numbered_screenplay(scenes))

    # Steps 3-4: check the answer and build the JSON
    output, warnings = build_output(title, scenes, answer, run_info["model"], relative(args.pdf))
    output["reasoning_effort"] = run_info.get("reasoning_effort", "model default")
    write_json(output, args.out)

    args.debug_dir.mkdir(parents=True, exist_ok=True)
    (args.debug_dir / "warnings.txt").write_text("\n".join(warnings) + ("\n" if warnings else ""))
    report(output, warnings, args.out, run_info)
    return 0
