"""The OpenRouter judge without the network: what it sends, what it reads back, and the runner's preflight check."""

from types import SimpleNamespace
from typing import Any

import pytest

from agentdojo import experiment_models
from agentdojo.skillfile.judge import DEFAULT_JUDGE_MODEL, openrouter_client, openrouter_judge_fn


class FakeCompletions:
    def __init__(self, content: str | None, choices: bool = True) -> None:
        self.content = content
        self.choices = choices
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)] if self.choices else [],
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30),
        )


def fake_client(completions: FakeCompletions) -> Any:
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_default_judge_is_deepseek_on_openrouter():
    assert DEFAULT_JUDGE_MODEL == "deepseek/deepseek-v4.1-flash"
    assert experiment_models.JUDGE_BASE_URL == "https://openrouter.ai/api/v1"
    assert experiment_models.JUDGE_API_KEY_ENV == "OPENROUTER_API_KEY"


def test_judge_sends_system_and_user_and_reads_usage():
    completions = FakeCompletions('{"diagnosis": [], "edits": []}')
    judge = openrouter_judge_fn(seed=3, client=fake_client(completions))
    call = judge("SYSTEM", "USER")
    assert call.text == '{"diagnosis": [], "edits": []}'
    assert (call.input_tokens, call.output_tokens) == (120, 30)
    (request,) = completions.requests
    assert request["model"] == DEFAULT_JUDGE_MODEL and request["seed"] == 3
    assert request["messages"] == [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "USER"}]


def test_empty_reply_is_empty_text_and_no_choices_raises():
    assert openrouter_judge_fn(client=fake_client(FakeCompletions(None)))("s", "u").text == ""
    with pytest.raises(RuntimeError, match="no choices"):
        openrouter_judge_fn(client=fake_client(FakeCompletions("x", choices=False)))("s", "u")


def test_client_needs_the_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        openrouter_client()
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    client = openrouter_client()
    assert str(client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"
    assert client.api_key == "sk-or-test"
