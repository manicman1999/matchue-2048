"""2048 with the original rules, seeded tile spawns, and per-tile movement tracking."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

SIZE = 4
MOVES = ("up", "down", "left", "right")
Board = list[list[int]]
Cell = tuple[int, int]


def empty_board() -> Board:
    return [[0] * SIZE for _ in range(SIZE)]


def copy_board(board: Board) -> Board:
    return [row[:] for row in board]


def _lines(move: str) -> list[list[Cell]]:
    """Every row or column, listed from the edge the tiles slide toward."""
    forward = list(range(SIZE))
    backward = forward[::-1]
    if move == "left":
        return [[(r, c) for c in forward] for r in forward]
    if move == "right":
        return [[(r, c) for c in backward] for r in forward]
    if move == "up":
        return [[(r, c) for r in forward] for c in forward]
    if move == "down":
        return [[(r, c) for r in backward] for c in forward]
    raise ValueError(f"Unknown move: {move!r}")


@dataclass
class Slide:
    board: Board
    score_gain: int
    changed: bool
    # Where every tile went: {"from": [r, c], "to": [r, c], "value": value before merging}.
    tiles: list[dict[str, Any]]
    # Tiles created by merging: {"at": [r, c], "value": new value}.
    merges: list[dict[str, Any]]


def slide(board: Board, move: str) -> Slide:
    result = empty_board()
    gain = 0
    tiles: list[dict[str, Any]] = []
    merges: list[dict[str, Any]] = []
    for line in _lines(move):
        # Each entry is [value, merged this move, source cells with their original values].
        placed: list[list[Any]] = []
        for r, c in line:
            value = board[r][c]
            if not value:
                continue
            if placed and placed[-1][0] == value and not placed[-1][1]:
                placed[-1][0] = value * 2
                placed[-1][1] = True
                placed[-1][2].append((r, c, value))
                gain += value * 2
            else:
                placed.append([value, False, [(r, c, value)]])
        for (tr, tc), (value, merged, sources) in zip(line, placed):
            result[tr][tc] = value
            for sr, sc, source_value in sources:
                tiles.append({"from": [sr, sc], "to": [tr, tc], "value": source_value})
            if merged:
                merges.append({"at": [tr, tc], "value": value})
    return Slide(result, gain, result != board, tiles, merges)


def max_tile(board: Board) -> int:
    return max(max(row) for row in board)


def empty_cells(board: Board) -> list[Cell]:
    return [(r, c) for r in range(SIZE) for c in range(SIZE) if not board[r][c]]


class Game:
    """One game. The same seed always produces the same spawns for the same moves."""

    def __init__(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed)
        self.board = empty_board()
        self.score = 0
        self.moves = 0
        self.initial_spawns = [self._spawn(), self._spawn()]
        self.initial_board = copy_board(self.board)

    def _spawn(self) -> dict[str, Any]:
        r, c = self.rng.choice(empty_cells(self.board))
        value = 4 if self.rng.random() < 0.1 else 2
        self.board[r][c] = value
        return {"at": [r, c], "value": value}

    def legal_moves(self) -> list[str]:
        return [move for move in MOVES if slide(self.board, move).changed]

    def state(self, legal_moves: list[str] | None = None) -> dict[str, Any]:
        """What a player sees before choosing a move."""
        return {
            "board": copy_board(self.board),
            "score": self.score,
            "move_number": self.moves + 1,
            "empty_cells": len(empty_cells(self.board)),
            "legal_moves": legal_moves if legal_moves is not None else self.legal_moves(),
        }

    def play(self, move: str) -> dict[str, Any]:
        """Apply a move, spawn a tile, and return everything that happened."""
        board_before = copy_board(self.board)
        score_before = self.score
        result = slide(self.board, move)
        if not result.changed:
            raise ValueError(f"Illegal move: {move}")
        self.board = result.board
        self.score += result.score_gain
        self.moves += 1
        board_after_slide = copy_board(self.board)
        spawn = self._spawn()
        return {
            "board_before": board_before,
            "score_before": score_before,
            "move": move,
            "tiles": result.tiles,
            "merges": result.merges,
            "score_gain": result.score_gain,
            "board_after_slide": board_after_slide,
            "spawn": spawn,
            "board_after": copy_board(self.board),
            "score_after": self.score,
            "max_tile": max_tile(self.board),
            "empty_cells_after": len(empty_cells(self.board)),
        }
