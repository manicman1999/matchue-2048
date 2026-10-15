"""Aggregate per-game summaries into summary.json and summary.md."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from runner import percentile, write_json


def load_summaries(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run_dir / "games").glob("*/seed-*.summary.json"))
    ]


def load_latencies(run_dir: Path, player: str) -> list[float]:
    latencies = []
    for path in sorted((run_dir / "games" / player).glob("seed-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if not record["forced"]:
                latencies.append(record["latency_s"])
    return latencies


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def aggregate(run_dir: Path, games: list[dict[str, Any]]) -> dict[str, Any]:
    order = list(dict.fromkeys(game["player"] for game in games))
    players: dict[str, Any] = {}
    for name in order:
        mine = [game for game in games if game["player"] == name]
        scores = [game["final_score"] for game in mine]
        decisions = sum(game["decisions"] for game in mine)
        cost = sum(game["cost_usd"] for game in mine)
        latencies = load_latencies(run_dir, name)
        tiles = Counter(game["max_tile"] for game in mine)
        players[name] = {
            "kind": mine[0]["kind"],
            "model": mine[0]["model"],
            "games": len(mine),
            "end_reasons": dict(Counter(game["end_reason"] for game in mine)),
            "score": {
                "mean": mean(scores),
                "median": statistics.median(scores),
                "min": min(scores),
                "max": max(scores),
            },
            "max_tile": {
                "best": max(tiles),
                "median": statistics.median(game["max_tile"] for game in mine),
                "counts": {str(tile): tiles[tile] for tile in sorted(tiles)},
                "reached_rate": {
                    str(tile): sum(game["max_tile"] >= tile for game in mine) / len(mine)
                    for tile in (128, 256, 512, 1024, 2048, 4096)
                },
            },
            "moves": {
                "mean": mean([game["moves"] for game in mine]),
                "total": sum(game["moves"] for game in mine),
                "decisions": decisions,
            },
            "cost_usd": {
                "total": cost,
                "per_game": cost / len(mine),
                "per_decision": cost / decisions if decisions else None,
                "per_1000_points": cost / sum(scores) * 1000 if sum(scores) else None,
                "fully_reported": all(game["cost_fully_reported"] for game in mine),
            },
            "latency_s": {
                "mean": mean(latencies),
                "p50": percentile(latencies, 0.5),
                "p90": percentile(latencies, 0.9),
                "p99": percentile(latencies, 0.99),
                "max": max(latencies) if latencies else None,
            },
            "model_time_s": sum(game["model_time_s"] for game in mine),
            "game_elapsed_s": {
                "mean": mean([game["game_elapsed_s"] for game in mine]),
                "max": max(game["game_elapsed_s"] for game in mine),
            },
            "tokens": {
                "input": sum(game["input_tokens"] for game in mine),
                "output": sum(game["output_tokens"] for game in mine),
                "reasoning": sum(game["reasoning_tokens"] for game in mine),
            },
            "retried_decisions": sum(game["retried_decisions"] for game in mine),
        }

    head_to_head: dict[str, Any] = {}
    for i, first in enumerate(order):
        for second in order[i + 1 :]:
            a = {g["seed"]: g for g in games if g["player"] == first}
            b = {g["seed"]: g for g in games if g["player"] == second}
            shared = sorted(set(a) & set(b))
            pa, pb = players[first], players[second]
            head_to_head[f"{first} vs {second}"] = {
                "seeds": len(shared),
                f"{first}_wins": sum(a[s]["final_score"] > b[s]["final_score"] for s in shared),
                f"{second}_wins": sum(b[s]["final_score"] > a[s]["final_score"] for s in shared),
                "ties": sum(a[s]["final_score"] == b[s]["final_score"] for s in shared),
                "mean_score_ratio": ratio(pa["score"]["mean"], pb["score"]["mean"]),
                "cost_per_decision_ratio": ratio(
                    pa["cost_usd"]["per_decision"], pb["cost_usd"]["per_decision"]
                ),
                "median_latency_ratio": ratio(pa["latency_s"]["p50"], pb["latency_s"]["p50"]),
            }
    return {"run": run_dir.name, "players": players, "head_to_head": head_to_head}


def ratio(a: float | None, b: float | None) -> float | None:
    return a / b if a is not None and b else None


def money(value: float | None) -> str:
    if value is None:
        return "-"
    return f"${value:.6f}" if value < 0.01 else f"${value:.4f}"


def seconds(value: float | None) -> str:
    if value is None:
        return "-"
    if value < 60:
        return f"{value:.2f}s"
    return f"{value / 60:.1f}m"


def markdown(summary: dict[str, Any], games: list[dict[str, Any]]) -> str:
    lines = [
        f"# Run {summary['run']}",
        "",
        "| Player | Games | Mean Score | Median Score | Best Score | Median Max Tile | Best Tile "
        "| Moves/Game | Cost/Decision | Cost/Game | Total Cost | Latency p50 | Latency p90 "
        "| Mean Game Time |",
        "|" + "---|" * 14,
    ]
    for name, p in summary["players"].items():
        lines.append(
            f"| {name} | {p['games']} | {p['score']['mean']:.0f} | {p['score']['median']:.0f} "
            f"| {p['score']['max']} | {p['max_tile']['median']:.0f} | {p['max_tile']['best']} "
            f"| {p['moves']['mean']:.0f} | {money(p['cost_usd']['per_decision'])} "
            f"| {money(p['cost_usd']['per_game'])} | {money(p['cost_usd']['total'])} "
            f"| {seconds(p['latency_s']['p50'])} | {seconds(p['latency_s']['p90'])} "
            f"| {seconds(p['game_elapsed_s']['mean'])} |"
        )
    lines += ["", "## Max Tile Reached", ""]
    tiles = ["128", "256", "512", "1024", "2048", "4096"]
    lines += ["| Player | " + " | ".join(tiles) + " |", "|" + "---|" * (len(tiles) + 1)]
    for name, p in summary["players"].items():
        rates = p["max_tile"]["reached_rate"]
        lines.append(f"| {name} | " + " | ".join(f"{rates[t]:.0%}" for t in tiles) + " |")
    lines += ["", "## Head to Head (Same Seeds)", ""]
    for pair, h in summary["head_to_head"].items():
        first, second = pair.split(" vs ")
        lines.append(
            f"- **{pair}**: {h[first + '_wins']}-{h[second + '_wins']} "
            f"({h['ties']} ties) over {h['seeds']} seeds. Mean score ratio "
            f"{fmt_ratio(h['mean_score_ratio'])}, cost per decision ratio "
            f"{fmt_ratio(h['cost_per_decision_ratio'])}, median latency ratio "
            f"{fmt_ratio(h['median_latency_ratio'])}."
        )
    lines += ["", "## Games", ""]
    lines += ["| Player | Seed | End | Score | Max Tile | Moves | Cost | Time |", "|" + "---|" * 8]
    for game in sorted(games, key=lambda g: (g["player"], g["seed"])):
        lines.append(
            f"| {game['player']} | {game['seed']} | {game['end_reason']} | {game['final_score']} "
            f"| {game['max_tile']} | {game['moves']} | {money(game['cost_usd'])} "
            f"| {seconds(game['game_elapsed_s'])} |"
        )
    return "\n".join(lines) + "\n"


def fmt_ratio(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}x"


def write_report(run_dir: Path) -> dict[str, Any]:
    games = load_summaries(run_dir)
    if not games:
        raise ValueError(f"No finished games in {run_dir}")
    summary = aggregate(run_dir, games)
    write_json(run_dir / "summary.json", summary)
    (run_dir / "summary.md").write_text(markdown(summary, games), encoding="utf-8")
    return summary
