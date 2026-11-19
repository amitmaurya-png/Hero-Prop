"""Command line: runs the four steps and saves debug files.

    python extract_hero_props.py                       # fast run (~15-20 s; cached re-runs <1 s)
    python extract_hero_props.py --mode full           # usage note + quote per scene (slower)
    python extract_hero_props.py --reasoning default   # more careful, much slower
    python extract_hero_props.py --no-cache            # always call the API
    python extract_hero_props.py --from-answer data/debug/answer.json
                                                       # re-check a saved answer, no API call
    python extract_hero_props.py --show-scenes         # print how the PDF was split
    python extract_hero_props.py --show-prompt         # print the instructions sent

Every run that calls the model writes data/debug/:
    prompt.txt      the instructions sent
    screenplay.txt  the numbered screenplay sent
    answer.json     the model's raw answer (before any checks)
    run.json        model, reasoning effort, seconds, input/output token counts
    warnings.txt    everything the checks dropped, and why
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from .config import (CACHE_DIR, DEBUG_DIR, DEFAULT_MODE, DEFAULT_MODEL, DEFAULT_OUT, DEFAULT_PDF,
                     DEFAULT_REASONING, MODES, REASONING_LEVELS, ROOT)
from .output import build_output, write_json
from .prompt import build_prompt, build_schema
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
                        help=f"reasoning effort; 'default' = the model's own (default: {DEFAULT_REASONING})")
    parser.add_argument("--mode", choices=MODES, default=DEFAULT_MODE,
                        help=f"fast: short answer, code finds evidence; full: quote per scene (default: {DEFAULT_MODE})")
    parser.add_argument("--no-cache", action="store_true", help="always call the API, even for identical inputs")
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


def cache_file(args: argparse.Namespace, screenplay: str) -> Path:
    """Answers are cached under a hash of everything that affects them."""
    key = json.dumps([args.model, args.reasoning, args.mode, build_prompt(args.mode),
                      build_schema(args.mode), screenplay], sort_keys=True)
    return CACHE_DIR / f"{hashlib.sha256(key.encode()).hexdigest()[:20]}.json"


def get_answer(args: argparse.Namespace, screenplay: str) -> tuple[dict, dict]:
    """Return (answer, run_info): from a saved file, the cache, or the model."""
    if args.from_answer:
        log(f"Re-checking saved answer {args.from_answer} (no API call)")
        answer = json.loads(args.from_answer.read_text())
        run_file = args.from_answer.with_name("run.json")
        run_info = json.loads(run_file.read_text()) if run_file.exists() else {"model": args.model}
        return answer, run_info

    cached = cache_file(args, screenplay)
    if cached.exists() and not args.no_cache:
        log(f"Using cached answer {relative(cached)} (no API call; --no-cache to refresh)")
        saved = json.loads(cached.read_text())
        return saved["answer"], {**saved["run_info"], "cached": True}

    load_env(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set (environment or .env).")

    from .llm import ask_model, token_counts   # imported here so --from-answer works without the API set up

    reasoning = None if args.reasoning == "default" else args.reasoning
    log(f"Asking {args.model} (mode: {args.mode}, reasoning: {args.reasoning})...")
    reply = ask_model(args.model, screenplay, reasoning, args.mode)
    run_info = {"model": args.model, "mode": args.mode, "reasoning_effort": args.reasoning,
                "seconds": round(reply.seconds, 1),
                "tokens": token_counts(reply.usage, reply.seconds), "usage": reply.usage}

    debug = args.debug_dir
    debug.mkdir(parents=True, exist_ok=True)
    (debug / "prompt.txt").write_text(build_prompt(args.mode))
    (debug / "screenplay.txt").write_text(screenplay)
    write_json(reply.answer, debug / "answer.json")
    write_json(run_info, debug / "run.json")
    write_json({"answer": reply.answer, "run_info": run_info}, cached)
    return reply.answer, run_info


def report(output: dict, warnings: list[str], out: Path, run_info: dict) -> None:
    for w in warnings:
        log(f"warning: {w}")
    for p in output["hero_props"]:
        flags = ",".join(k for k, v in p["criteria"].items() if v)
        log(f"- {p['name']}: scenes {p['scenes']} [{flags}]")
    timing = f", LLM request {run_info['seconds']}s" if "seconds" in run_info else ""
    if run_info.get("cached"):
        timing += " when first run; this run used the cache"
    log(f"Wrote {output['hero_prop_count']} hero props to {out} "
        f"(reasoning: {run_info.get('reasoning_effort', 'n/a')}{timing})")
    report_tokens(run_info.get("tokens"))


def report_tokens(tokens: dict | None) -> None:
    """Print the input/output token counts of the model request."""
    if not tokens:
        log("Tokens: not available (no API call recorded for this answer)")
        return
    log("")
    log("Tokens processed")
    log(f"  Input tokens   : {tokens['input_tokens']:>7,}   (cached: {tokens['cached_input_tokens']:,})")
    log(f"  Output tokens  : {tokens['output_tokens']:>7,}")
    log(f"    - thinking   : {tokens['reasoning_tokens']:>7,}")
    log(f"    - answer     : {tokens['answer_tokens']:>7,}")
    log(f"  Total tokens   : {tokens['total_tokens']:>7,}")
    if "output_tokens_per_second" in tokens:
        log(f"  Output speed   : {tokens['output_tokens_per_second']:>7} tokens/s")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Step 1: PDF -> scenes
    title, scenes = read_scenes(args.pdf)
    if args.show_scenes:
        show_scenes(title, scenes)
        return 0
    if args.show_prompt:
        print(build_prompt(args.mode))
        return 0
    log(f"{title}: {len(scenes)} scenes")

    # Step 2: ask the model (or load a saved answer)
    answer, run_info = get_answer(args, numbered_screenplay(scenes))

    # Steps 3-4: check the answer and build the JSON
    output, warnings = build_output(title, scenes, answer, run_info["model"], relative(args.pdf))
    output["mode"] = run_info.get("mode", "full")
    output["reasoning_effort"] = run_info.get("reasoning_effort", "model default")
    output["run"] = {k: run_info[k] for k in ("seconds", "tokens") if k in run_info}
    write_json(output, args.out)

    args.debug_dir.mkdir(parents=True, exist_ok=True)
    (args.debug_dir / "warnings.txt").write_text("\n".join(warnings) + ("\n" if warnings else ""))
    report(output, warnings, args.out, run_info)
    return 0
