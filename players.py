"""Players that pick 2048 moves. Both models get the same rules and the same state."""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from openrouter import OpenRouter, OpenRouterError, usage_of

RULES = (
    "You are playing 2048 on a 4x4 grid. The board is listed as rows from top to bottom, "
    "and 0 marks an empty cell. A move slides every tile as far as it can go in one "
    "direction. When two tiles with the same number collide, they merge into one tile "
    "worth their sum, and that sum is added to the score. A tile can merge only once per "
    "move. After every move, a new tile appears in a random empty cell: a 2 nine times "
    "out of ten, otherwise a 4. The game is lost when the board is full and no move can "
    "merge anything.\n\n"
    "The goal is to build a 2048 tile, then keep going for bigger tiles. Every move adds "
    "a tile, so the game only lasts as long as merges keep making room, and the score is "
    "the sum of every merge across the whole game. Choose the move that gives the best "
    "chance of reaching the biggest tile possible."
)

MOVE_GUIDE = {
    "up": "Slide every tile toward the top row.",
    "down": "Slide every tile toward the bottom row.",
    "left": "Slide every tile toward the left column.",
    "right": "Slide every tile toward the right column.",
}

GPT_SYSTEM = (
    RULES
    + "\n\nMoves:\n"
    + "\n".join(f"- {move}: {guide}" for move, guide in MOVE_GUIDE.items())
    + "\n\nAnswer with a one-sentence reason and the move you choose from the legal moves."
)


def render_state(state: dict[str, Any]) -> str:
    """The text GPT sees. It carries exactly the fields Jev receives as its state object."""
    rows = "\n".join(json.dumps(row) for row in state["board"])
    return (
        f"Board (rows from top to bottom, 0 = empty):\n{rows}\n"
        f"Score: {state['score']}\n"
        f"Move number: {state['move_number']}\n"
        f"Empty cells: {state['empty_cells']}\n"
        f"Legal moves: {', '.join(state['legal_moves'])}"
    )


class InvalidReply(Exception):
    """The model answered, but not with a usable move."""


@dataclass
class Decision:
    move: str
    details: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0  # the request that produced the move
    wall_s: float = 0.0  # every attempt, including retries and backoff
    cost_usd: float = 0.0  # every attempt that reported a cost
    cost_reported: bool = True
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    attempts: int = 0
    errors: list[str] = field(default_factory=list)
    generation_ids: list[str] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)  # raw requests and responses

    def add_usage(self, usage: dict[str, Any]) -> None:
        self.input_tokens += usage["input_tokens"]
        self.output_tokens += usage["output_tokens"]
        self.reasoning_tokens += usage["reasoning_tokens"]
        if usage["cost_usd"] is None:
            self.cost_reported = False
        else:
            self.cost_usd += usage["cost_usd"]


class DecisionFailed(Exception):
    """Every attempt failed. Carries the partial decision so its cost is still counted."""

    def __init__(self, message: str, decision: Decision):
        super().__init__(message)
        self.decision = decision


def _strip_encrypted(data: dict[str, Any]) -> dict[str, Any]:
    """Drop encrypted reasoning blobs from raw logs; they are large and unreadable."""
    for choice in data.get("choices") or []:
        for detail in (choice.get("message") or {}).get("reasoning_details") or []:
            if detail.get("type") == "reasoning.encrypted":
                detail.pop("data", None)
    return data


class Player:
    name: str
    kind: str
    model: str | None = None
    max_attempts = 6

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "model": self.model}

    async def choose(self, state: dict[str, Any]) -> Decision:
        raise NotImplementedError


class ModelPlayer(Player):
    def __init__(self, client: OpenRouter):
        self.client = client

    def request(self, state: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    async def send(self, request: dict[str, Any]) -> tuple[dict[str, Any], float]:
        raise NotImplementedError

    def parse(self, data: dict[str, Any], legal: list[str]) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError

    async def choose(self, state: dict[str, Any]) -> Decision:
        decision = Decision(move="")
        started = perf_counter()
        for attempt in range(1, self.max_attempts + 1):
            decision.attempts = attempt
            request = self.request(state)
            call: dict[str, Any] = {"attempt": attempt, "request": request}
            decision.calls.append(call)
            try:
                data, latency = await self.send(request)
                call["latency_s"] = latency
                call["response"] = _strip_encrypted(data)
                decision.add_usage(usage_of(data))
                if data.get("id"):
                    decision.generation_ids.append(data["id"])
                decision.move, decision.details = self.parse(data, state["legal_moves"])
            # A malformed response body surfaces as KeyError/IndexError/TypeError while parsing.
            except (OpenRouterError, InvalidReply, KeyError, IndexError, TypeError) as error:
                message = f"{type(error).__name__}: {error}"
                call["error"] = message
                decision.errors.append(f"attempt {attempt}: {message}")
                retryable = not isinstance(error, OpenRouterError) or error.retryable
                if not retryable or attempt == self.max_attempts:
                    decision.wall_s = perf_counter() - started
                    raise DecisionFailed(f"{self.name}: {message}", decision) from error
                await asyncio.sleep(min(60.0, 2.0**attempt) * random.uniform(0.75, 1.25))
                continue
            decision.latency_s = latency
            decision.wall_s = perf_counter() - started
            return decision
        raise AssertionError("unreachable")


class GptPlayer(ModelPlayer):
    kind = "gpt"

    def __init__(
        self,
        client: OpenRouter,
        model: str = "openai/gpt-6-luna",
        reasoning_effort: str = "low",
    ):
        super().__init__(client)
        self.model = model
        self.name = model.split("/")[-1]
        self.reasoning_effort = reasoning_effort

    def config(self) -> dict[str, Any]:
        return {
            **super().config(),
            "reasoning_effort": self.reasoning_effort,
            "system_prompt": GPT_SYSTEM,
        }

    def request(self, state: dict[str, Any]) -> dict[str, Any]:
        reasoning = {"effort": self.reasoning_effort}
        if self.reasoning_effort != "none":
            reasoning["summary"] = "auto"
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": GPT_SYSTEM},
                {"role": "user", "content": render_state(state)},
            ],
            "reasoning": reasoning,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "move",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "reason": {"type": "string"},
                            "move": {"type": "string", "enum": state["legal_moves"]},
                        },
                        "required": ["reason", "move"],
                        "additionalProperties": False,
                    },
                },
            },
        }

    async def send(self, request: dict[str, Any]) -> tuple[dict[str, Any], float]:
        options = {key: value for key, value in request.items() if key not in ("model", "messages")}
        return await self.client.chat(request["model"], request["messages"], **options)

    def parse(self, data: dict[str, Any], legal: list[str]) -> tuple[str, dict[str, Any]]:
        choice = data["choices"][0]
        message = choice["message"]
        if choice.get("finish_reason") != "stop":
            raise InvalidReply(f"finish_reason was {choice.get('finish_reason')!r}")
        try:
            reply = json.loads(message.get("content") or "")
        except json.JSONDecodeError as error:
            raise InvalidReply(f"content was not JSON: {message.get('content')!r}") from error
        move = reply.get("move")
        if move not in legal:
            raise InvalidReply(f"move {move!r} is not one of {legal}")
        summaries = [
            detail["summary"]
            for detail in message.get("reasoning_details") or []
            if detail.get("type") == "reasoning.summary" and detail.get("summary")
        ]
        return move, {
            "reason": reply.get("reason"),
            "reasoning_summary": "\n\n".join(summaries) or message.get("reasoning"),
        }


class JevPlayer(ModelPlayer):
    kind = "jev"

    def __init__(self, client: OpenRouter, model: str = "typesafe/jev-1.13"):
        super().__init__(client)
        self.model = model
        self.name = model.split("/")[-1]

    def config(self) -> dict[str, Any]:
        return {**super().config(), "instructions": RULES, "criteria": MOVE_GUIDE}

    def request(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": self.model,
            "state": state,
            "questions": {
                "move": {
                    "type": "choice",
                    "instructions": RULES,
                    "criteria": {move: MOVE_GUIDE[move] for move in state["legal_moves"]},
                }
            },
        }

    async def send(self, request: dict[str, Any]) -> tuple[dict[str, Any], float]:
        return await self.client.decide(request["model"], request["state"], request["questions"])

    def parse(self, data: dict[str, Any], legal: list[str]) -> tuple[str, dict[str, Any]]:
        answer = (data.get("answers") or {}).get("move") or {}
        move = answer.get("choice")
        if move not in legal:
            raise InvalidReply(f"choice {move!r} is not one of {legal}")
        return move, {
            "probabilities": answer.get("probabilities"),
            "confidence": answer.get("confidence"),
        }


class RandomPlayer(Player):
    """A free baseline. Seeded by the state, so reruns make the same moves."""

    kind = "random"
    name = "random"

    async def choose(self, state: dict[str, Any]) -> Decision:
        rng = random.Random(json.dumps(state, sort_keys=True))
        return Decision(move=rng.choice(state["legal_moves"]))


def build_player(config: dict[str, Any], client: OpenRouter) -> Player:
    if config["kind"] == "gpt":
        return GptPlayer(client, config["model"], config.get("reasoning_effort", "low"))
    if config["kind"] == "jev":
        return JevPlayer(client, config["model"])
    if config["kind"] == "random":
        return RandomPlayer()
    raise ValueError(f"Unknown player kind: {config['kind']}")
