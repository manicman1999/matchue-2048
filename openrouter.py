"""A small OpenRouter client for chat models and decision models such as Jev."""

from __future__ import annotations

import os
from time import perf_counter
from typing import Any

import httpx

BASE_URL = "https://openrouter.ai/api"
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}


class OpenRouterError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class OpenRouter:
    def __init__(self, api_key: str | None = None, *, timeout: float = 600):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.http = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
        )

    async def __aenter__(self) -> OpenRouter:
        await self.http.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.http.__aexit__(exc_type, exc_value, traceback)

    async def post(self, path: str, body: dict[str, Any]) -> tuple[dict[str, Any], float]:
        """Send one request and return the JSON body with its latency in seconds."""
        if not self.api_key:
            raise ValueError("Set OPENROUTER_API_KEY or pass api_key")
        started = perf_counter()
        try:
            response = await self.http.post(
                BASE_URL + path,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
            )
        except httpx.TransportError as error:
            raise OpenRouterError(f"{type(error).__name__}: {error}", retryable=True) from error
        latency = perf_counter() - started
        if response.status_code >= 400:
            raise OpenRouterError(
                f"HTTP {response.status_code}: {response.text[:1000]}",
                response.status_code,
                response.status_code in RETRYABLE_STATUS,
            )
        try:
            data = response.json()
        except ValueError as error:
            raise OpenRouterError(
                f"Response was not JSON: {response.text[:500]}", response.status_code, True
            ) from error
        if data.get("error"):
            # OpenRouter reports some upstream failures inside a 200 response.
            raise OpenRouterError(f"Upstream error: {data['error']}", response.status_code, True)
        return data, latency

    async def chat(
        self, model: str, messages: list[dict[str, Any]], **options: Any
    ) -> tuple[dict[str, Any], float]:
        """Chat completion. Options such as reasoning or response_format go straight into the body."""
        return await self.post(
            "/v1/chat/completions", {**options, "model": model, "messages": messages}
        )

    async def decide(
        self, model: str, state: Any, questions: dict[str, dict[str, Any]], **options: Any
    ) -> tuple[dict[str, Any], float]:
        """Decisions API: typed answers (choice, noul, score) about a state, e.g. from Jev."""
        return await self.post(
            "/alpha/decisions",
            {**options, "model": model, "state": state, "questions": questions},
        )

    async def model_info(self, model: str) -> dict[str, Any]:
        """Public model metadata, including per-token pricing."""
        response = await self.http.get(f"{BASE_URL}/v1/model/{model}")
        response.raise_for_status()
        return response.json()["data"]


def usage_of(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize usage from either API. Chat reports prompt/completion tokens; decisions report input/output."""
    usage = data.get("usage") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    cost = usage.get("cost")
    return {
        "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")) or 0,
        "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")) or 0,
        "reasoning_tokens": completion_details.get("reasoning_tokens") or 0,
        "cached_tokens": prompt_details.get("cached_tokens") or 0,
        "cost_usd": float(cost) if cost is not None else None,
    }
