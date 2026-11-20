"""The skill-file attacker (item 7, D2): the agent loop with its load_skill/write_skill/submit_injection tools,
the missing-literal correction, the no-submit fall-back, and the skill library in the checkpoint.

Scripted model only, no server: a `ScriptedAgent` returns a fixed list of tool-call turns."""

import json
from pathlib import Path
from typing import Any

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function

from agentdojo.attacks.adaptive.autodojo import SkillAgentAttacker, SkillAgentSettings
from agentdojo.attacks.adaptive.autodojo.attacker import goal_literals
from agentdojo.attacks.adaptive.muzzle import AttackerModelConfig
from agentdojo.runner.manifest import DEFAULT_MANIFEST_PATH, load_pairs
from agentdojo.runner.plugins import AttemptFeedback
from agentdojo.runner.report import attacker_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((REPO_ROOT / DEFAULT_MANIFEST_PATH).read_text(encoding="utf-8"))
PAIR = load_pairs(MANIFEST)[0]


def tool_turn(calls: list[tuple[str, dict[str, Any]]]) -> ChatCompletion:
    tool_calls: list[Any] = [
        ChatCompletionMessageToolCall(
            id=f"c{i}", type="function", function=Function(name=name, arguments=json.dumps(args))
        )
        for i, (name, args) in enumerate(calls)
    ]
    return ChatCompletion(
        id="x",
        object="chat.completion",
        created=0,
        model="scripted",
        choices=[
            Choice(
                index=0,
                finish_reason="tool_calls",
                message=ChatCompletionMessage(role="assistant", content=None, tool_calls=tool_calls),
            )
        ],
    )


def text_turn(text: str) -> ChatCompletion:
    return ChatCompletion(
        id="x",
        object="chat.completion",
        created=0,
        model="scripted",
        choices=[Choice(index=0, finish_reason="stop", message=ChatCompletionMessage(role="assistant", content=text))],
    )


class ScriptedAgent:
    """Returns the next scripted turn on each `complete`, recording the messages it saw."""

    def __init__(self, turns: list[ChatCompletion], log: list[list[dict]] | None = None) -> None:
        self.turns = turns
        self.log = log
        self.calls = 0
        self.closed = False

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> ChatCompletion:
        if self.log is not None:
            self.log.append([dict(m) for m in messages])
        turn = self.turns[min(self.calls, len(self.turns) - 1)]
        self.calls += 1
        return turn

    def usage(self) -> dict[str, Any]:
        return {"calls": self.calls}

    async def aclose(self) -> None:
        self.closed = True


def make(turns: list[ChatCompletion], log: list[list[dict]] | None = None, **settings: Any) -> SkillAgentAttacker:
    cfg = SkillAgentSettings(attacker=AttackerModelConfig(model="m", base_url="http://x/v1"), **settings)
    agent = ScriptedAgent(turns, log)
    return SkillAgentAttacker(MANIFEST, cfg, llm_factory=lambda: agent)


def values() -> str:
    return " ".join(goal_literals(SkillAgentAttacker(MANIFEST, _settings()).goal(PAIR)))


def _settings() -> SkillAgentSettings:
    return SkillAgentSettings(attacker=AttackerModelConfig(model="m", base_url="http://x/v1"))


def feedback0(text: str) -> AttemptFeedback:
    return AttemptFeedback(trial=0, injection_text=text, security=False, utility=True, final_answer="no", progress=0)


def test_agent_writes_loads_and_submits():
    vals = values()
    log: list[list[dict]] = []
    atk = make(
        [
            tool_turn([("write_skill", {"name": "rec", "description": "record framing", "body": "frame as a record"})]),
            tool_turn([("load_skill", {"name": "rec"})]),
            tool_turn(
                [
                    (
                        "submit_injection",
                        {
                            "prompt": f"proceed {vals}".strip(),
                            "diagnosis": "carry values",
                            "strategy_ranking": ["data_record", "sequence_cue"],
                        },
                    )
                ]
            ),
        ],
        log,
    )
    t0 = atk.text_for(PAIR, 0, [])
    t1 = atk.text_for(PAIR, 1, [feedback0(t0)])
    entry = atk.archive[-1]
    assert entry["source"] == "generated"
    assert entry["skills_written"] == ["rec"] and entry["skills_loaded"] == ["rec"]
    assert entry["strategy"] == "data_record" and entry["strategy_ranking"] == ["data_record", "sequence_cue"]
    assert all(v in t1 for v in goal_literals(atk.goal(PAIR)))
    assert atk.store.names() == ["rec"]


def test_missing_literal_is_corrected_inside_the_loop():
    vals = values()
    atk = make(
        [
            tool_turn([("submit_injection", {"prompt": "no values here", "diagnosis": "oops"})]),
            tool_turn([("submit_injection", {"prompt": f"ok {vals}".strip(), "diagnosis": "now with values"})]),
        ]
    )
    t0 = atk.text_for(PAIR, 0, [])
    t1 = atk.text_for(PAIR, 1, [feedback0(t0)])
    assert all(v in t1 for v in goal_literals(atk.goal(PAIR)))
    assert atk.archive[-1]["agent_steps"] == 2


def test_never_submitting_falls_back_to_best_leaderboard_text():
    # The agent only ever lists skills and never submits; after retries the trial reuses the best known text.
    atk = make([tool_turn([("list_skills", {})])], max_agent_steps=2)
    t0 = atk.text_for(PAIR, 0, [])
    t1 = atk.text_for(PAIR, 1, [feedback0(t0)])
    assert atk.archive[-1]["source"] == "goal_fallback"
    assert t1 == t0  # the only leaderboard entry


def test_plain_text_turn_without_submit_is_nudged_then_falls_back():
    atk = make([text_turn("I think I will not use tools."), text_turn("still nothing")], max_agent_steps=4)
    t0 = atk.text_for(PAIR, 0, [])
    atk.text_for(PAIR, 1, [feedback0(t0)])
    assert atk.archive[-1]["source"] == "goal_fallback"


def test_bad_write_is_reported_and_the_agent_recovers():
    vals = values()
    atk = make(
        [
            tool_turn([("write_skill", {"name": "Bad Name", "description": "d", "body": "b"})]),
            tool_turn([("submit_injection", {"prompt": f"go {vals}".strip(), "diagnosis": "d"})]),
        ]
    )
    t0 = atk.text_for(PAIR, 0, [])
    atk.text_for(PAIR, 1, [feedback0(t0)])
    assert atk.archive[-1]["source"] == "generated"
    assert atk.store.names() == []  # the invalid write was rejected, not stored
    assert atk.archive[-1]["skills_written"] == []


def test_skill_library_round_trips_in_the_checkpoint():
    vals = values()
    turns = [
        tool_turn([("write_skill", {"name": "keep", "description": "d", "body": "the tactic"})]),
        tool_turn([("submit_injection", {"prompt": f"x {vals}".strip(), "diagnosis": "d"})]),
    ]
    atk = make(turns)
    t0 = atk.text_for(PAIR, 0, [])
    atk.text_for(PAIR, 1, [feedback0(t0)])
    state = atk.state_dict()
    assert state["format"] == 3
    restored = make(turns)
    restored.load_state(state)
    assert restored.store.load("keep") == "the tactic"
    assert [e["trial"] for e in restored.archive] == [0, 1]


def test_config_notes_the_tool_loop_and_skill_tools():
    cfg = make([text_turn("")]).config()
    assert cfg["name"] == "skill-agent"
    assert "submit_injection" in cfg["skill_tools"]
    assert "create_deep_agent" in cfg["harness"]


def test_report_surfaces_the_skill_library():
    state = {
        "attacker": {
            "archive": [
                {"pair": "banking/u/i", "trial": 0, "source": "trial0"},
                {
                    "pair": "banking/u/i",
                    "trial": 1,
                    "source": "generated",
                    "skills_written": ["rec"],
                    "skills_loaded": [],
                    "usage": {"calls": 3},
                },
                {
                    "pair": "banking/u/i",
                    "trial": 2,
                    "source": "generated",
                    "skills_written": [],
                    "skills_loaded": ["rec"],
                    "usage": {"calls": 2},
                },
            ],
            "skills": {"skills": [{"name": "rec", "description": "d", "body": "b", "version": 2}]},
        }
    }
    metrics = attacker_metrics(state, records=[])
    assert metrics is not None
    lib = metrics["skill_library"]
    assert lib["notes"] == 1 and lib["names"] == ["rec"]
    assert lib["rewrites_writing_a_note"] == 1 and lib["rewrites_loading_a_note"] == 1
    # the plain AutoDojo attacker keeps no library
    plain = attacker_metrics({"attacker": {"archive": []}}, records=[])
    assert plain is not None and plain["skill_library"] is None
