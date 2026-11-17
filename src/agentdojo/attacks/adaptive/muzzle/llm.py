"""The two pieces of SuperRed's core that MUZZLE's judge and PAIR use, on top of the OpenAI SDK.

Upstream imports `LLMClient` from `superred.core.llm` (a litellm wrapper) and `BudgetExhaustedError` from
`superred.core.types.llm`. The vendored files only ever call `await llm.complete(messages, **kwargs)` and read
`result.choices[0].message.content`, so this is all they need. Calls go to OpenRouter by default (DESIGN.md:
"LLM calls go through OpenRouter"); `base_url` points them at any other OpenAI-compatible server.
"""

import os
from typing import Any, Protocol

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"


class BudgetExhaustedError(RuntimeError):
    """Raised before a call once the client's call budget is spent. The vendored code always lets it propagate."""


class LLMClient(Protocol):
    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> ChatCompletion: ...


class AttackerModelConfig(BaseModel):
    """The one attacker model that plays MUZZLE's judge and all three PAIR roles (as upstream: one `self.llm`)."""

    model: str
    base_url: str = OPENROUTER_BASE_URL
    api_key_env: str = DEFAULT_API_KEY_ENV
    """Name of the environment variable holding the key; the key itself is never stored."""
    provider_order: list[str] | None = None
    """OpenRouter providers in order of preference; see `DeepAgentsConfig.provider_order`."""
    allow_fallbacks: bool = True
    seed: int | None = None
    """Sampling seed sent with every request. Not every provider honours it."""
    max_retries: int = 5
    request_timeout: float = 120.0
    max_calls_per_rewrite: int | None = None
    """Raise `BudgetExhaustedError` past this many calls in one rewrite. None: no cap (PAIR's own caps bound it)."""
    reasoning: dict[str, Any] | None = None
    """OpenRouter's `reasoning` request field, e.g. `{"enabled": False}`. Set it for a reasoning model: upstream's
    token caps (PAIR judge 10, attacker and stand-in victim 500) leave no room for reasoning tokens, which OpenRouter
    counts against `max_tokens`, so the replies come back empty (PAIR then scores every proposal 1)."""

    def extra_body(self) -> dict[str, Any] | None:
        body: dict[str, Any] = {}
        if self.provider_order is not None:
            body["provider"] = {"order": list(self.provider_order), "allow_fallbacks": self.allow_fallbacks}
        if self.reasoning is not None:
            body["reasoning"] = dict(self.reasoning)
        return body or None


class OpenAICompatibleLLM:
    """One rewrite's client. Counts calls and tokens, and must be closed with `aclose` (it owns an HTTP pool)."""

    def __init__(self, config: AttackerModelConfig, client: AsyncOpenAI | None = None) -> None:
        self.config = config
        if client is None:
            api_key = os.environ.get(config.api_key_env)
            if not api_key:
                raise ValueError(f"{config.api_key_env} is not set (the MUZZLE attacker's API key)")
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=config.base_url,
                max_retries=config.max_retries,
                timeout=config.request_timeout,
            )
        self._client = client
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.empty_replies = 0
        """Replies with no text, e.g. a reasoning model that spent `max_tokens` before answering."""
        self.truncated_replies = 0
        """Replies cut off by `max_tokens` (finish reason "length")."""
        self.served_models: set[str] = set()

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> ChatCompletion:
        cap = self.config.max_calls_per_rewrite
        if cap is not None and self.calls >= cap:
            raise BudgetExhaustedError(f"MUZZLE attacker call budget of {cap} per rewrite is spent")
        self.calls += 1
        if self.config.seed is not None:
            kwargs.setdefault("seed", self.config.seed)
        extra_body = self.config.extra_body()
        if extra_body is not None:
            kwargs.setdefault("extra_body", extra_body)
        result = await self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,  # type: ignore[arg-type]  # plain role/content dicts
            **kwargs,
        )
        if result.usage is not None:
            self.prompt_tokens += result.usage.prompt_tokens
            self.completion_tokens += result.usage.completion_tokens
        if result.model:
            self.served_models.add(result.model)
        if result.choices:
            choice = result.choices[0]
            if not (choice.message.content or "").strip():
                self.empty_replies += 1
            if choice.finish_reason == "length":
                self.truncated_replies += 1
        return result

    def usage(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "empty_replies": self.empty_replies,
            "truncated_replies": self.truncated_replies,
            "served_models": sorted(self.served_models),
        }

    async def aclose(self) -> None:
        await self._client.close()
