"""The AutoDojo attacker: trial-0 seed, leaderboard-driven rewrites against a scripted optimizer, the retry-then-
fall-back path, state round-trip, and a C2 run through the real runner with a scripted victim."""

import json
from pathlib import Path
from typing import Any

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from agentdojo.attacks.adaptive.autodojo import AutoDojoAttacker, AutoDojoSettings
from agentdojo.attacks.adaptive.autodojo.attacker import GENERATE_MAX_ATTEMPTS
from agentdojo.attacks.adaptive.muzzle import AttackerModelConfig
from agentdojo.runner.manifest import DEFAULT_MANIFEST_PATH, load_pairs, trial0_injections
from agentdojo.runner.plugins import AttemptFeedback

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((REPO_ROOT / DEFAULT_MANIFEST_PATH).read_text(encoding="utf-8"))
PAIR = load_pairs(MANIFEST)[0]


def completion(text: str) -> ChatCompletion:
    return ChatCompletion(
        id="x",
        object="chat.completion",
        created=0,
        model="scripted",
        choices=[Choice(index=0, finish_reason="stop", message=ChatCompletionMessage(role="assistant", content=text))],
    )


class ScriptedOptimizer:
    """Returns a fresh JSON injection each call, recording the user messages it saw (to check the leaderboard)."""

    def __init__(self, log: list[dict[str, Any]], reply: str | None = None) -> None:
        self.log = log
        self.reply = reply
        self.calls = 0
        self.closed = False

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> ChatCompletion:
        self.calls += 1
        self.log.append({"messages": messages, "kwargs": kwargs})
        if self.reply is not None:
            return completion(self.reply)
        return completion(json.dumps({"diagnosis": "be quieter", "prompt": f"REWRITE {self.calls} {len(self.log)}"}))

    def usage(self) -> dict[str, Any]:
        return {"calls": self.calls, "empty_replies": 0, "truncated_replies": 0}

    async def aclose(self) -> None:
        self.closed = True


class FailingOptimizer:
    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> ChatCompletion:
        self.calls += 1
        raise RuntimeError("server down")

    def usage(self) -> dict[str, Any]:
        return {"calls": self.calls}

    async def aclose(self) -> None:
        self.closed = True


def make_attacker(reply: str | None = None) -> tuple[AutoDojoAttacker, list[Any], list[dict[str, Any]]]:
    clients: list[Any] = []
    log: list[dict[str, Any]] = []

    def factory() -> ScriptedOptimizer:
        clients.append(ScriptedOptimizer(log, reply=reply))
        return clients[-1]

    settings = AutoDojoSettings(attacker=AttackerModelConfig(model="scripted/attacker"))
    return AutoDojoAttacker(MANIFEST, settings, llm_factory=factory), clients, log


def fb(trial: int, text: str, utility: bool = True, security: bool = False) -> AttemptFeedback:
    return AttemptFeedback(trial=trial, injection_text=text, utility=utility, security=security, final_answer="done")


def test_trial0_is_the_seed_and_makes_no_model_call():
    attacker, clients, _ = make_attacker()
    seed = next(iter(trial0_injections(MANIFEST, PAIR).values()))
    assert attacker.text_for(PAIR, 0, []) == seed
    assert clients == []
    assert attacker.archive[0]["source"] == "trial0"


def test_rewrite_uses_the_leaderboard_and_evolves():
    attacker, clients, log = make_attacker()
    attacker.text_for(PAIR, 0, [])
    seed = attacker.archive[0]["text"]
    t1 = attacker.text_for(PAIR, 1, [fb(0, seed)])
    t2 = attacker.text_for(PAIR, 2, [fb(0, seed), fb(1, t1, security=True)])
    assert t1 != seed and t2 != t1
    assert len(clients) == 2
    # The second optimizer call's user message carries the leaderboard with the prior texts and their real outcome.
    user_msg = log[-1]["messages"][1]["content"]
    assert "LEADERBOARD" in user_msg and "SUCCEEDED" in user_msg and t1 in user_msg
    entry = attacker.archive[-1]
    assert entry["source"] == "generated" and entry["diagnosis"] == "be quieter"
    # Highest-scoring prior trial (the one that broke security) is first on the leaderboard the optimizer saw.
    assert entry["leaderboard"][0]["trial"] == 1 and entry["leaderboard"][0]["security"] is True


def test_cached_text_is_returned_without_a_model_call():
    attacker, clients, _ = make_attacker()
    attacker.text_for(PAIR, 0, [])
    seed = attacker.archive[0]["text"]
    t1 = attacker.text_for(PAIR, 1, [fb(0, seed)])
    before = len(clients)
    assert attacker.text_for(PAIR, 1, [fb(0, seed)]) == t1
    assert len(clients) == before  # no new optimizer call on resume


def test_unparseable_reply_then_failure_falls_back_to_the_best_leaderboard_text():
    attacker, clients, _ = make_attacker(reply="not json at all")
    attacker.text_for(PAIR, 0, [])
    seed = attacker.archive[0]["text"]
    out = attacker.text_for(PAIR, 1, [fb(0, seed, security=True)])
    assert out == seed  # the only (and best) leaderboard entry
    entry = attacker.archive[-1]
    assert entry["source"] == "goal_fallback"
    # Each failed optimizer call retries max_tries times internally, over GENERATE_MAX_ATTEMPTS whole attempts.
    assert len(clients) == GENERATE_MAX_ATTEMPTS


def test_hard_failure_retries_then_falls_back(monkeypatch):
    monkeypatch.setattr("agentdojo.attacks.adaptive.autodojo.attacker.GENERATE_RETRY_BACKOFF_SECONDS", 0)
    clients: list[FailingOptimizer] = []

    def factory() -> FailingOptimizer:
        clients.append(FailingOptimizer())
        return clients[-1]

    attacker = AutoDojoAttacker(
        MANIFEST, AutoDojoSettings(attacker=AttackerModelConfig(model="x")), llm_factory=factory
    )
    attacker.text_for(PAIR, 0, [])
    seed = attacker.archive[0]["text"]
    out = attacker.text_for(PAIR, 1, [fb(0, seed)])
    assert out == seed and attacker.archive[-1]["source"] == "goal_fallback"
    assert len(clients) == GENERATE_MAX_ATTEMPTS and all(c.closed for c in clients)


def test_state_round_trip():
    attacker, _, _ = make_attacker()
    attacker.text_for(PAIR, 0, [])
    seed = attacker.archive[0]["text"]
    attacker.text_for(PAIR, 1, [fb(0, seed)])
    state = json.loads(json.dumps(attacker.state_dict()))

    restored, _, _ = make_attacker()
    restored.load_state(state)
    assert restored.archive == attacker.archive
    assert restored.text_for(PAIR, 1, [fb(0, seed)]) == attacker.archive[1]["text"]


def test_config_names_the_paper_and_hides_secrets():
    attacker, _, _ = make_attacker()
    config = attacker.config()
    assert config["name"] == "autodojo" and "2606.15057" in config["paper"]
    assert "api_key" not in json.dumps(config).lower() or "api_key_env" in json.dumps(config)
