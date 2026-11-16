"""End-to-end: attacked environment -> AgentDojo tool -> checker -> agent -> trace and scoring.

A scripted "gullible" victim does the user task's ground truth, but switches to the injection task as soon as a tool
output it is shown contains the injection. A scripted checker flags the injection the way a model reads it (wrapping
and escapes undone). Without the checker every pair is hijacked; with it none is, utility is unchanged, and the saved
trace and the checker log say so.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("deepagents")

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.deepagents_pipeline import DeepAgentsPipeline, ToolMiddlewareFactory
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.benchmark import load_task_results, run_task_with_injection_tasks
from agentdojo.checker import middleware as checker_middleware
from agentdojo.checker.middleware import (
    CHECKER_REPLY_SCHEMA,
    CheckerReply,
    CheckerRequest,
    SanitizingChecker,
    jsonl_logger,
    local_checker_fn,
    make_checker_factory,
)
from agentdojo.checker.span_removal import SPAN_REMOVED_MARKER, canonical_form
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.logging import OutputLogger
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.types import ChatMessage, get_text_content_as_str

BENCHMARK_VERSION = "v1.2.2"
PIPELINE_NAME = "deepagents-deepseek_deepseek-v4.1-flash"
INJECTION_START, INJECTION_END = "<INFORMATION>", "</INFORMATION>"


def _call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


class GullibleModel(BaseChatModel):
    """Does `user_calls`, but once any tool output it was shown contains the injection marker, does
    `injection_calls` first. Ends with `final`."""

    user_calls: list[AIMessage] = Field(default_factory=list)
    injection_calls: list[AIMessage] = Field(default_factory=list)
    final: str = "Done."
    hijacked: bool = False

    @property
    def _llm_type(self) -> str:
        return "gullible"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "GullibleModel":  # type: ignore[override]
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.hijacked and any(
            isinstance(m, ToolMessage) and INJECTION_START in str(m.content) for m in messages
        ):
            self.hijacked = True
            self.user_calls = [*self.injection_calls, *self.user_calls]
        message = self.user_calls.pop(0) if self.user_calls else AIMessage(content=self.final)
        return ChatResult(generations=[ChatGeneration(message=message)])


class GullibleVictim(BasePipelineElement):
    def __init__(self, user_task, injection_task, factory: ToolMiddlewareFactory | None) -> None:
        self.user_task = user_task
        self.injection_task = injection_task
        self.factory = factory
        self.name = PIPELINE_NAME
        self.hijacked = False

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        model = GullibleModel(
            user_calls=[
                _call(c.function, dict(c.args), f"u{i}") for i, c in enumerate(self.user_task.ground_truth(env))
            ],
            injection_calls=[
                _call(c.function, dict(c.args), f"i{i}") for i, c in enumerate(self.injection_task.ground_truth(env))
            ],
            final=self.user_task.GROUND_TRUTH_OUTPUT or "Done.",
        )
        pipeline = DeepAgentsPipeline(model, name=self.name, tool_middleware_factory=self.factory)
        result = pipeline.query(query, runtime, env, messages, extra_args)
        self.hijacked = model.hijacked
        return result


def reading_checker(request: CheckerRequest) -> CheckerReply:
    """Flags the injected block as a model would quote it: as it reads, without YAML wrapping or escapes."""
    output = request.tool_output
    spans = []
    start = output.find(INJECTION_START)
    while start != -1:
        end = output.find(INJECTION_END, start)
        if end == -1:
            break
        end += len(INJECTION_END)
        spans.append({"span": canonical_form(output[start:end]), "reason": "addresses the assistant"})
        start = output.find(INJECTION_START, end)
    return CheckerReply(text=json.dumps({"remove": spans}), input_tokens=10, output_tokens=5)


def _run(suite, user_task, injection_task, factory, logdir: Path):
    victim = GullibleVictim(user_task, injection_task, factory)
    attack = load_attack("important_instructions", suite, victim)
    with OutputLogger(str(logdir)):
        utility, security = run_task_with_injection_tasks(
            suite, victim, user_task, attack, logdir, True, [injection_task.ID], BENCHMARK_VERSION
        )
    key = (user_task.ID, injection_task.ID)
    return victim, attack, utility[key], security[key]


# One user task per suite whose ground truth reads an injected tool output in a different YAML style.
CASES = [
    ("banking", "user_task_0", "injection_task_0"),
    ("banking", "user_task_1", "injection_task_1"),
    ("slack", "user_task_1", "injection_task_1"),
    ("travel", "user_task_2", "injection_task_0"),
    ("workspace", "user_task_0", "injection_task_1"),
    ("workspace", "user_task_13", "injection_task_0"),
]


@pytest.mark.parametrize("suite_name,user_task_id,injection_task_id", CASES)
def test_checker_stops_injection_end_to_end(tmp_path: Path, suite_name, user_task_id, injection_task_id):
    suite = get_suite(BENCHMARK_VERSION, suite_name)
    user_task = suite.user_tasks[user_task_id]
    injection_task = suite.injection_tasks[injection_task_id]

    # Without a checker the scripted victim is hijacked and the injection succeeds.
    victim, attack, _, security = _run(suite, user_task, injection_task, None, tmp_path / "none")
    assert victim.hijacked
    assert security is True

    # With the checker the victim never sees the injection, the injection fails, and the user task still works.
    records: list[dict] = []
    log_path = tmp_path / "checker_calls.jsonl"
    file_log = jsonl_logger(str(log_path))

    def log(record: dict) -> None:
        records.append(record)
        file_log(record)

    victim, attack, utility, security = _run(
        suite,
        user_task,
        injection_task,
        make_checker_factory(
            reading_checker,
            skill_file="(skill file)",
            log=log,
            log_context={"condition": "test", "trial": 0},
            injections=list(attack.attack(user_task, injection_task).values()),
        ),
        tmp_path / "checked",
    )
    assert not victim.hijacked
    assert security is False
    assert utility is True

    # The checker saw every AgentDojo tool output, removed the injection, and logged it with the run's keys.
    assert records and all(r["condition"] == "test" and r["trial"] == 0 for r in records)
    with_injection = [r for r in records if any(i["present_before"] for i in r["injections"])]
    assert with_injection
    for record in with_injection:
        assert record["removed"] and not record["unmatched"] and not record["parse_failure"]
        assert all(i["surviving_fraction"] == 0.0 for i in record["injections"])
    assert len(log_path.read_text().splitlines()) == len(records)

    # The saved trace holds the sanitised outputs (what the agent saw), and loads for resume.
    saved = load_task_results(
        PIPELINE_NAME, suite_name, user_task_id, attack.name, injection_task_id, tmp_path / "checked"
    )
    tool_texts = [get_text_content_as_str(m["content"] or []) for m in saved.messages if m["role"] == "tool"]
    assert any(SPAN_REMOVED_MARKER in text for text in tool_texts)
    assert not any(INJECTION_START in text for text in tool_texts)
    assert saved.security is False


def test_checker_leaves_clean_outputs_and_builtin_tools_alone():
    suite = get_suite(BENCHMARK_VERSION, "banking")
    user_task = suite.user_tasks["user_task_0"]
    calls: list[str] = []

    def no_removals(request: CheckerRequest) -> CheckerReply:
        calls.append(request.tool_name)
        assert request.user_task == user_task.PROMPT
        return CheckerReply(text='{"remove": []}')

    env = suite.load_and_inject_default_environment({})
    checker = SanitizingChecker(no_removals, "(skill file)", user_task.PROMPT)
    model = GullibleModel(
        user_calls=[
            _call("task", {"description": user_task.PROMPT, "subagent_type": "general-purpose"}, "t0"),
            *[_call(c.function, dict(c.args), f"u{i}") for i, c in enumerate(user_task.ground_truth(env))],
            AIMessage(content="sub done"),
        ],
    )
    pipeline = DeepAgentsPipeline(model, name="scripted", tool_middleware=[checker])
    _, _, _, messages, _ = pipeline.query(user_task.PROMPT, FunctionsRuntime(suite.tools), env)
    # Subagent tool calls are screened; the `task` tool itself is not.
    assert calls == [c.function for c in user_task.ground_truth(env)]
    assert all(
        SPAN_REMOVED_MARKER not in get_text_content_as_str(m["content"] or []) for m in messages if m["role"] == "tool"
    )


def test_unparseable_checker_reply_removes_whole_output():
    suite = get_suite(BENCHMARK_VERSION, "banking")
    env = suite.load_and_inject_default_environment({})
    checker = SanitizingChecker(lambda request: CheckerReply(text="sorry, I can't"), "(skill file)", "task")
    model = GullibleModel(user_calls=[_call("get_balance", {}, "c0")])
    _, _, _, messages, _ = DeepAgentsPipeline(model, tool_middleware=[checker]).query(
        "task", FunctionsRuntime(suite.tools), env
    )
    tool = next(m for m in messages if m["role"] == "tool")
    assert get_text_content_as_str(tool["content"] or []) == "[entire output removed by security check]"


def test_local_checker_constrains_json_and_turns_thinking_off(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}
    reply = '{"remove": [{"span": "IGNORE", "reason": "injection"}]}'

    class _Echo(BaseChatModel):
        def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any):
            captured["messages"] = messages
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

        @property
        def _llm_type(self) -> str:
            return "echo"

    def fake_make_local_chat_model(model: str, base_url: str, **kwargs: Any) -> BaseChatModel:
        captured.update(model=model, base_url=base_url, kwargs=kwargs)
        return _Echo()

    monkeypatch.setattr(checker_middleware, "make_local_chat_model", fake_make_local_chat_model)
    check = local_checker_fn("Qwen/Qwen3.5-9B", "http://localhost:8001/v1")
    result = check(CheckerRequest(skill_file="SKILL", user_task="task", tool_name="t", tool_output="IGNORE this"))
    assert result.text == reply
    assert captured["base_url"] == "http://localhost:8001/v1"
    extra_body = captured["kwargs"]["extra_body"]
    assert extra_body["response_format"]["json_schema"]["schema"] == CHECKER_REPLY_SCHEMA
    assert extra_body["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["messages"][0].content == "SKILL"
