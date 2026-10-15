"""Entry point: start a run, resume one, or rebuild its summary."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from openrouter import OpenRouter
from players import build_player
from report import write_report
from runner import now, run_games, write_json

PLAYER_DEFAULTS = {"gpt": "openai/gpt-6-luna", "jev": "typesafe/jev-1.13", "random": None}


def player_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs = []
    for kind in args.players:
        config: dict[str, Any] = {"kind": kind, "model": PLAYER_DEFAULTS[kind]}
        if kind == "gpt":
            config["model"] = args.gpt_model
            config["reasoning_effort"] = args.reasoning_effort
        if kind == "jev":
            config["model"] = args.jev_model
        configs.append(config)
    return configs


async def execute(run_dir: Path, run: dict[str, Any]) -> None:
    async with OpenRouter() as client:
        players = [build_player(config, client) for config in run["players"]]
        names = [player.name for player in players]
        if len(set(names)) != len(names):
            raise SystemExit(f"Player names must be unique: {names}")
        if "models" not in run:
            run["models"] = {}
            for player in players:
                if player.model:
                    try:
                        run["models"][player.model] = await client.model_info(player.model)
                    except Exception as error:
                        run["models"][player.model] = {"error": str(error)}
            run["player_details"] = [player.config() for player in players]
            write_json(run_dir / "run.json", run)
        print(f"Run folder: {run_dir}", flush=True)
        started = time.perf_counter()
        await run_games(players, run["seeds"], run_dir, run["max_moves"])
    write_report(run_dir)
    print(f"\nAll games done in {time.perf_counter() - started:.0f}s\n")
    print((run_dir / "summary.md").read_text(encoding="utf-8"))


def cmd_run(args: argparse.Namespace) -> None:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.runs_dir) / (f"{stamp}-{args.name}" if args.name else stamp)
    run = {
        "run_id": run_dir.name,
        "created_at": now(),
        "players": player_configs(args),
        "seeds": list(range(1, args.games + 1)),
        "max_moves": args.max_moves,
        "python": sys.version,
        "platform": platform.platform(),
    }
    run_dir.mkdir(parents=True)
    write_json(run_dir / "run.json", run)
    asyncio.run(execute(run_dir, run))


def cmd_resume(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if args.games is not None:
        if args.games < len(run["seeds"]):
            raise SystemExit(f"Run already has {len(run['seeds'])} games per player")
        run["seeds"] = list(range(run["seeds"][0], run["seeds"][0] + args.games))
    run.setdefault("resumed_at", []).append(now())
    write_json(run_dir / "run.json", run)
    asyncio.run(execute(run_dir, run))


def cmd_summarize(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    write_report(run_dir)
    print((run_dir / "summary.md").read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="main.py")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Start a new run")
    run.add_argument("--players", nargs="+", choices=list(PLAYER_DEFAULTS), default=["gpt", "jev"])
    run.add_argument("--games", type=int, default=10, help="Games per player")
    run.add_argument("--max-moves", type=int, default=5000)
    run.add_argument("--gpt-model", default=PLAYER_DEFAULTS["gpt"])
    run.add_argument("--jev-model", default=PLAYER_DEFAULTS["jev"])
    run.add_argument("--reasoning-effort", default="low")
    run.add_argument("--runs-dir", default="runs")
    run.add_argument("--name", help="Suffix for the run folder")
    run.set_defaults(func=cmd_run)

    resume = commands.add_parser("resume", help="Finish unfinished or errored games in a run")
    resume.add_argument("run_dir")
    resume.add_argument("--games", type=int, help="Raise games per player (adds seeds)")
    resume.set_defaults(func=cmd_resume)

    summarize = commands.add_parser("summarize", help="Rebuild summary.json and summary.md")
    summarize.add_argument("run_dir")
    summarize.set_defaults(func=cmd_summarize)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
