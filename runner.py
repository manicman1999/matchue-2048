"""Run many games in parallel and log every move, cost and timing to disk."""

from __future__ import annotations

import asyncio
import json
import statistics
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from game import Game
from players import Decision, DecisionFailed, Player

FINISHED = ("game_over", "max_moves")
MILESTONES = [2**power for power in range(5, 18)]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


class Events:
    """events.log gets every line; the console gets the ones marked loud."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("a", encoding="utf-8")

    def __call__(self, message: str, loud: bool = False) -> None:
        line = f"{now()} {message}"
        self.file.write(line + "\n")
        self.file.flush()
        if loud:
            print(line, flush=True)

    def close(self) -> None:
        self.file.close()


def game_paths(run_dir: Path, player: str, seed: int) -> tuple[Path, Path, Path]:
    stem = f"seed-{seed:04d}"
    games = run_dir / "games" / player
    return games / f"{stem}.jsonl", games / f"{stem}.summary.json", run_dir / "raw" / player / f"{stem}.jsonl"


def load_records(path: Path) -> list[dict[str, Any]]:
    """Read a move log, dropping a half-written last line left by a crash."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            break
    if len(records) != len(lines):
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return records


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))]


def summarize_game(
    player: dict[str, Any],
    game: Game,
    records: list[dict[str, Any]],
    end_reason: str,
    error: str | None,
    failed_cost: float,
) -> dict[str, Any]:
    decisions = [r for r in records if not r["forced"]]
    latencies = [r["latency_s"] for r in decisions]
    milestones: dict[str, Any] = {}
    for record in records:
        for tile in MILESTONES:
            if record["max_tile"] >= tile and str(tile) not in milestones:
                milestones[str(tile)] = {
                    "move_number": record["move_number"],
                    "score": record["score_after"],
                    "total_cost_usd": record["total_cost_usd"],
                    "game_elapsed_s": record["game_elapsed_s"],
                }
    last = records[-1] if records else None
    final_board = game.board
    return {
        "player": player["name"],
        "kind": player["kind"],
        "model": player["model"],
        "seed": game.seed,
        "end_reason": end_reason,
        "error": error,
        "moves": len(records),
        "decisions": len(decisions),
        "forced_moves": len(records) - len(decisions),
        "final_score": game.score,
        "max_tile": max(max(row) for row in final_board),
        "initial_board": game.initial_board,
        "final_board": final_board,
        "cost_usd": (last["total_cost_usd"] if last else 0.0) + failed_cost,
        "failed_decision_cost_usd": failed_cost,
        "cost_fully_reported": all(r["cost_reported"] for r in records),
        "cost_per_decision_usd": (last["total_cost_usd"] / len(decisions)) if decisions else None,
        "model_time_s": sum(latencies),
        "game_elapsed_s": last["game_elapsed_s"] if last else 0.0,
        "latency_s": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": percentile(latencies, 0.5),
            "p90": percentile(latencies, 0.9),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies) if latencies else None,
        },
        "input_tokens": sum(r["input_tokens"] for r in records),
        "output_tokens": sum(r["output_tokens"] for r in records),
        "reasoning_tokens": sum(r["reasoning_tokens"] for r in records),
        "retried_decisions": sum(1 for r in records if r["attempts"] > 1),
        "milestones": milestones,
        "started_at": records[0]["timestamp"] if records else None,
        "finished_at": now(),
    }


async def play_game(
    player: Player, seed: int, run_dir: Path, max_moves: int, events: Events
) -> dict[str, Any]:
    moves_path, summary_path, raw_path = game_paths(run_dir, player.name, seed)
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["end_reason"] in FINISHED:
            return summary

    records = load_records(moves_path)
    game = Game(seed)
    for record in records:
        game.play(record["move"])
        if game.board != record["board_after"]:
            raise RuntimeError(f"Replay of {moves_path} diverged at move {record['move_number']}")
    total_cost = records[-1]["total_cost_usd"] if records else 0.0
    model_time = records[-1]["total_model_time_s"] if records else 0.0
    elapsed_before = records[-1]["game_elapsed_s"] if records else 0.0
    session_start = perf_counter()
    tag = f"{player.name} seed {seed}"
    events(f"{tag} {'resumed at move ' + str(game.moves) if records else 'started'}", loud=True)

    end_reason, error, failed_cost = "game_over", None, 0.0
    moves_path.parent.mkdir(parents=True, exist_ok=True)
    with moves_path.open("a", encoding="utf-8") as moves_file:
        while True:
            legal = game.legal_moves()
            if not legal:
                break
            if game.moves >= max_moves:
                end_reason = "max_moves"
                break
            state = game.state(legal)
            forced = len(legal) == 1
            try:
                decision = Decision(move=legal[0]) if forced else await player.choose(state)
            except DecisionFailed as failure:
                write_raw(raw_path, state["move_number"], failure.decision.calls)
                end_reason, error, failed_cost = "error", str(failure), failure.decision.cost_usd
                break
            except Exception as failure:  # keep the other games running
                end_reason, error = "error", "".join(traceback.format_exception(failure))
                break
            write_raw(raw_path, state["move_number"], decision.calls)
            step = game.play(decision.move)
            total_cost += decision.cost_usd
            model_time += decision.latency_s
            record = {
                "move_number": state["move_number"],
                "timestamp": now(),
                "legal_moves": legal,
                "forced": forced,
                **step,
                "details": decision.details,
                "latency_s": decision.latency_s,
                "wall_s": decision.wall_s,
                "cost_usd": decision.cost_usd,
                "cost_reported": decision.cost_reported,
                "input_tokens": decision.input_tokens,
                "output_tokens": decision.output_tokens,
                "reasoning_tokens": decision.reasoning_tokens,
                "attempts": decision.attempts,
                "errors": decision.errors,
                "generation_ids": decision.generation_ids,
                "total_cost_usd": total_cost,
                "total_model_time_s": model_time,
                "game_elapsed_s": elapsed_before + perf_counter() - session_start,
            }
            moves_file.write(json.dumps(record) + "\n")
            moves_file.flush()
            records.append(record)
            events(
                f"{tag} move {game.moves} {decision.move} score {game.score} "
                f"max {step['max_tile']} ${total_cost:.5f} {decision.latency_s:.2f}s",
                loud=game.moves % 50 == 0,
            )

    if error:
        events(f"{tag} ERROR {error}", loud=True)
    summary = summarize_game(player.config(), game, records, end_reason, error, failed_cost)
    write_json(summary_path, summary)
    events(
        f"{tag} finished ({end_reason}): score {summary['final_score']}, max tile "
        f"{summary['max_tile']}, {summary['moves']} moves, ${summary['cost_usd']:.4f}, "
        f"{summary['game_elapsed_s']:.0f}s",
        loud=True,
    )
    return summary


def write_raw(path: Path, move_number: int, calls: list[dict[str, Any]]) -> None:
    if not calls:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for call in calls:
            file.write(json.dumps({"move_number": move_number, "timestamp": now(), **call}) + "\n")


async def run_games(
    players: list[Player], seeds: list[int], run_dir: Path, max_moves: int
) -> list[dict[str, Any]]:
    """Every player plays every seed, all at once."""
    events = Events(run_dir / "events.log")
    try:
        return await asyncio.gather(
            *(play_game(player, seed, run_dir, max_moves, events) for seed in seeds for player in players)
        )
    finally:
        events.close()
