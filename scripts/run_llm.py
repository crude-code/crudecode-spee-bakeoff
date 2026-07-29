"""Run the LLM arm through the Anthropic API, one run per config file.

Usage:
    python scripts/run_llm.py CONFIG.json [data_dir] [--resume] [--dry-run] [--pads ID,ID]

A run is fully described by its config and lives in one directory:

    benchmark_data/runs/<run_id>/
        config.json    frozen copy of the config that produced the run
        prompts/       generated from the config's skill + the frozen snapshot
        raw/           one .json response (+ .meta.json usage/stop_reason) per pad,
                       or a .err file when the call failed
        forecasts/     validated + promoted by collect_llm_forecasts.py
        manifest.json  per-pad outcomes and token totals

Config (unknown keys are an error — a typo'd knob must not silently no-op):

    {
      "run_id":     "fable5-baseline",
      "model":      "claude-fable-5",
      "effort":     "high",            # low|medium|high|xhigh|max, or null to omit
      "max_tokens": 64000,             # caps thinking + response together
      "thinking":   "adaptive",        # "adaptive" or "none" (omit the param)
      "skill":      "skill/SKILL.md"   # prompt source, relative to repo root
    }

Design decisions, recorded so they don't get relitigated:

- One blind Messages API call per pad, no tools, no system prompt. This
  replaces the earlier `claude -p` runner, whose Claude Code harness put an
  uncontrolled system prompt on top of the skill — results from the two
  runners are not directly comparable.
- **No server-side fallbacks.** A fallback would silently answer with a
  different model than the one the run claims to benchmark. A refusal or
  overlong response is recorded as a loud per-pad failure instead; the
  scorer then reports the pad as forecast_missing.
- **No temperature knob.** Current models reject sampling parameters;
  run-to-run variance is the model's own sampling, which is exactly what
  repeat runs are meant to measure.
- Re-running without --resume refuses to touch an existing run directory,
  and --resume refuses a config that differs from the frozen copy — a run
  directory must never mix outputs from two configurations.

Credentials: ANTHROPIC_API_KEY from the environment, or a line in .env next
to the repo root (same convention as extract_pads.py; .env is gitignored).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]

CONFIG_DEFAULTS = {
    "effort": "high",
    "max_tokens": 64000,
    "thinking": "adaptive",
    "skill": "skill/SKILL.md",
}
REQUIRED_KEYS = {"run_id", "model"}


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    unknown = set(config) - REQUIRED_KEYS - set(CONFIG_DEFAULTS)
    if unknown:
        raise SystemExit(f"unknown config keys {sorted(unknown)} — "
                         f"allowed: {sorted(REQUIRED_KEYS | set(CONFIG_DEFAULTS))}")
    missing = REQUIRED_KEYS - set(config)
    if missing:
        raise SystemExit(f"config missing required keys {sorted(missing)}")
    return {**CONFIG_DEFAULTS, **config}


def load_api_key() -> str | None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    env_file = REPO / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def build_request_kwargs(config: dict, prompt: str) -> dict:
    kwargs = {
        "model": config["model"],
        "max_tokens": config["max_tokens"],
        "messages": [{"role": "user", "content": prompt}],
    }
    # "none" omits the parameter rather than sending {type: "disabled"},
    # which some models (Fable 5) reject outright.
    if config["thinking"] == "adaptive":
        kwargs["thinking"] = {"type": "adaptive"}
    if config["effort"]:
        kwargs["output_config"] = {"effort": config["effort"]}
    return kwargs


def call_one(client, config: dict, prompt: str, anthropic_mod=None) -> dict:
    """One pad, one blind completion. Returns a result dict; never raises on
    API-level failure — the caller records it and moves to the next pad."""
    started = time.monotonic()
    if anthropic_mod is None:
        import anthropic as anthropic_mod  # optional dependency; only needed for live LLM runs
    try:
        with client.messages.stream(**build_request_kwargs(config, prompt)) as stream:
            msg = stream.get_final_message()
    except anthropic_mod.APIConnectionError as e:
        return {"ok": False, "error": f"connection error after SDK retries: {e}"}
    except anthropic_mod.APIStatusError as e:
        return {"ok": False, "error": f"API error {e.status_code}: {e.message}"}

    meta = {
        "model": msg.model,
        "stop_reason": msg.stop_reason,
        "duration_s": round(time.monotonic() - started, 1),
        "usage": {
            "input_tokens": msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
        },
        "request_id": msg._request_id,
    }
    if msg.stop_reason == "refusal":
        detail = getattr(msg, "stop_details", None)
        return {"ok": False, "meta": meta,
                "error": f"model refused (category={getattr(detail, 'category', None)}) — "
                         "recorded, not retried on another model"}
    if msg.stop_reason == "max_tokens":
        return {"ok": False, "meta": meta,
                "error": f"hit max_tokens={config['max_tokens']} (thinking counts "
                         "toward it) — raise max_tokens in the config and --resume"}
    text = "".join(b.text for b in msg.content if b.type == "text")
    return {"ok": True, "meta": meta, "text": text}


def generate_prompts(config: dict, data_dir: Path, run_dir: Path) -> None:
    skill_path = REPO / config["skill"]
    if not skill_path.is_file():
        raise SystemExit(f"no skill file at {skill_path}")
    env = {**os.environ,
           "SKILL_PATH": str(skill_path),
           "PROMPT_SUBDIR": str(run_dir.relative_to(data_dir) / "prompts")}
    subprocess.run([sys.executable, str(REPO / "scripts" / "make_llm_prompts.py"),
                    str(data_dir)], env=env, check=True)


def collect(data_dir: Path, run_dir: Path, pad_ids: list[str] | None = None) -> int:
    rel = run_dir.relative_to(data_dir)
    env = {**os.environ,
           "RAW_SUBDIR": str(rel / "raw"),
           "OUT_SUBDIR": str(rel / "forecasts")}
    if pad_ids:
        env["PAD_IDS"] = ",".join(pad_ids)
    return subprocess.run([sys.executable, str(REPO / "scripts" / "collect_llm_forecasts.py"),
                           str(data_dir)], env=env).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("data_dir", nargs="?", default="benchmark_data", type=Path)
    parser.add_argument("--resume", action="store_true",
                        help="continue an existing run: skip pads with a raw response")
    parser.add_argument("--dry-run", action="store_true",
                        help="generate prompts and report the plan; no API calls")
    parser.add_argument("--pads", default=None,
                        help="comma-separated pad ids to run (default: all)")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = args.data_dir / "runs" / config["run_id"]

    if run_dir.exists() and not (args.resume or args.dry_run):
        raise SystemExit(f"{run_dir} already exists — use --resume to continue it, "
                         "or pick a new run_id (runs are never overwritten)")
    if args.resume and (run_dir / "config.json").exists():
        frozen = json.loads((run_dir / "config.json").read_text())
        if frozen != config:
            raise SystemExit("config differs from the run's frozen config.json — "
                             "a run directory must not mix configurations")

    for sub in ("raw", "forecasts"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=1))

    prompts_dir = run_dir / "prompts"
    if not (prompts_dir.is_dir() and any(prompts_dir.glob("*.md"))):
        generate_prompts(config, args.data_dir, run_dir)

    roster = [p["pad_id"] for p in json.loads((args.data_dir / "pads.json").read_text())]
    if args.pads:
        wanted = args.pads.split(",")
        bad = set(wanted) - set(roster)
        if bad:
            raise SystemExit(f"unknown pad ids: {sorted(bad)}")
        roster = wanted

    if args.dry_run:
        print(f"dry run — would call {config['model']} "
              f"(effort={config['effort']}, max_tokens={config['max_tokens']}, "
              f"thinking={config['thinking']}) on {len(roster)} pads -> {run_dir}")
        return

    api_key = load_api_key()
    if not api_key:
        raise SystemExit("no ANTHROPIC_API_KEY in the environment or .env")
    try:
        import anthropic
    except ImportError as e:
        raise SystemExit("anthropic is not installed; run `pip install -e .[llm]`") from e
    client = anthropic.Anthropic(api_key=api_key)

    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, dict] = (
        json.loads(manifest_path.read_text()) if args.resume and manifest_path.exists() else {}
    )
    failed = []
    for pad_id in roster:
        raw_path = run_dir / "raw" / f"{pad_id}.json"
        if args.resume and raw_path.exists():
            print(f"skip {pad_id} (already answered)")
            if pad_id not in manifest:
                meta_path = run_dir / "raw" / f"{pad_id}.meta.json"
                manifest[pad_id] = {
                    "ok": True,
                    "meta": json.loads(meta_path.read_text()) if meta_path.exists() else None,
                    "error": None,
                }
            continue
        prompt = (prompts_dir / f"{pad_id}.md").read_text()
        result = call_one(client, config, prompt, anthropic)
        if result["ok"]:
            raw_path.write_text(result["text"])
            (run_dir / "raw" / f"{pad_id}.meta.json").write_text(
                json.dumps(result["meta"], indent=1))
            m = result["meta"]
            print(f"done {pad_id}: {m['usage']['output_tokens']} out tokens, "
                  f"{m['duration_s']}s, stop={m['stop_reason']}")
        else:
            (run_dir / "raw" / f"{pad_id}.err").write_text(
                json.dumps(result, indent=1, default=str))
            failed.append(pad_id)
            print(f"FAILED {pad_id}: {result['error']}", file=sys.stderr)
        manifest[pad_id] = {k: result.get(k) for k in ("ok", "meta", "error")}

    manifest_path.write_text(json.dumps(manifest, indent=1))

    collect_rc = collect(args.data_dir, run_dir, roster if args.pads else None)
    print(f"\nscore with:\n  PYTHONPATH=src {sys.executable} examples/run_benchmark.py "
          f"{args.data_dir} --llm-dir runs/{config['run_id']}/forecasts")
    if failed or collect_rc:
        sys.exit(1)


if __name__ == "__main__":
    main()
