import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("deepagents")

from langchain.agents.middleware import AgentMiddleware
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from pydantic import Field

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.deepagents_pipeline import (
    DEEPAGENTS_BUILTIN_TOOL_NAMES,
    DeepAgentsConfig,
    DeepAgentsPipeline,
    ToolMiddlewareFactory,
)
from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
from agentdojo.attacks.base_attacks import get_model_name_from_pipeline
from agentdojo.benchmark import benchmark_suite_without_injections, load_task_results
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.logging import OutputLogger
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.types import ChatMessage

BENCHMARK_VERSION = "v1.2.2"


class ScriptedChatModel(BaseChatModel):
    """Returns pre-scripted AI messages in order, regardless of input."""

    script: list[AIMessage] = Field(default_factory=list)
    position: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedChatModel":  # type: ignore[override]
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.position >= len(self.script):
            raise RuntimeError("Scripted model ran out of responses")
        message = self.script[self.position]
        self.position += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


class ToolCallSpy(AgentMiddleware):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[str] = []

    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]
    ) -> ToolMessage | Command:
        self.seen.append(request.tool_call["name"])
        return handler(request)


def _tool_call_message(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


class GroundTruthReplay(BasePipelineElement):
    """Scripts the deep agent to replay a task's ground-truth tool calls, either directly or via the subagent."""

    def __init__(
        self,
        task,
        delegate: bool,
        tool_middleware: Sequence[AgentMiddleware] = (),
        tool_middleware_factory: ToolMiddlewareFactory | None = None,
    ) -> None:
        self.task = task
        self.delegate = delegate
        self.model = ScriptedChatModel()
        self.pipeline = DeepAgentsPipeline(
            self.model,
            name="scripted",
            tool_middleware=tool_middleware,
            tool_middleware_factory=tool_middleware_factory,
        )
        self.name = self.pipeline.name

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        calls = [
            _tool_call_message(call.function, dict(call.args), f"call_{i}")
            for i, call in enumerate(self.task.ground_truth(env))
        ]
        final = AIMessage(content=self.task.GROUND_TRUTH_OUTPUT or "Done.")
        if self.delegate:
            task_call = _tool_call_message(
                "task", {"description": query, "subagent_type": "general-purpose"}, "call_task"
            )
            script = [task_call, *calls, AIMessage(content="Subagent finished."), final]
        else:
            script = [*calls, final]
        self.model.script = script
        self.model.position = 0
        return self.pipeline.query(query, runtime, env, messages, extra_args)


SUITE_NAMES = ["banking", "slack", "travel", "workspace"]


@pytest.mark.parametrize("suite_name", SUITE_NAMES)
@pytest.mark.parametrize("delegate", [False, True], ids=["main", "subagent"])
def test_ground_truth_replay_matches_native(suite_name: str, delegate: bool):
    suite = get_suite(BENCHMARK_VERSION, suite_name)
    mismatches = []
    for task_id, task in suite.user_tasks.items():
        native, _ = suite.run_task_with_pipeline(GroundTruthPipeline(task), task, None, {})
        deep, _ = suite.run_task_with_pipeline(GroundTruthReplay(task, delegate), task, None, {})
        if native != deep:
            mismatches.append((task_id, native, deep))
    assert mismatches == []


def test_subagent_tool_calls_are_traced_and_intercepted():
    suite = get_suite(BENCHMARK_VERSION, "banking")
    task = suite.user_tasks["user_task_0"]
    spy = ToolCallSpy()
    pipeline = GroundTruthReplay(task, delegate=True, tool_middleware=[spy])
    env = task.init_environment(suite.load_and_inject_default_environment({}))
    expected = [call.function for call in task.ground_truth(env.model_copy(deep=True))]
    _, _, _, messages, extra_args = pipeline.query(task.PROMPT, FunctionsRuntime(suite.tools), env)

    traced = [call.function for m in messages if m["role"] == "assistant" for call in m["tool_calls"] or []]
    assert traced == ["task", *expected]
    # The spy runs in the main agent (sees `task`) and in the subagent (sees the AgentDojo calls).
    assert spy.seen == ["task", *expected]
    assert extra_args["deepagents"]["n_subagent_events"] > 0
    assert messages[-1]["role"] == "assistant"


class ToolListCapture(ScriptedChatModel):
    bound: list[list[str]] = Field(default_factory=list)

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ToolListCapture":  # type: ignore[override]
        self.bound.append([tool.name if hasattr(tool, "name") else tool["name"] for tool in tools])
        return self


def test_builtin_tool_names_are_up_to_date():
    model = ToolListCapture(script=[AIMessage(content="ok")], bound=[])
    pipeline = DeepAgentsPipeline(model, hide_filesystem_tools=False)
    _, _, _, _, extra_args = pipeline.query("hi", FunctionsRuntime([]), EmptyEnv())
    assert set(model.bound[0]) == DEEPAGENTS_BUILTIN_TOOL_NAMES
    assert set(extra_args["deepagents"]["agent_setup"]["main"]["tools"]) == DEEPAGENTS_BUILTIN_TOOL_NAMES


def test_agentdojo_tool_shadows_builtin_with_same_name():
    suite = get_suite(BENCHMARK_VERSION, "banking")
    env = suite.load_and_inject_default_environment({})
    read_call = _tool_call_message("read_file", {"file_path": "bill-december-2023.txt"}, "call_0")
    model = ToolListCapture(script=[read_call, AIMessage(content="ok")], bound=[])
    pipeline = DeepAgentsPipeline(model, hide_filesystem_tools=False)
    _, _, _, messages, extra_args = pipeline.query("hi", FunctionsRuntime(suite.tools), env)

    assert model.bound[0].count("read_file") == 1
    assert extra_args["deepagents"]["shadowed_builtin_tools"] == ["read_file"]
    tool_result = next(m for m in messages if m["role"] == "tool")
    assert tool_result["error"] is None
    assert "Bill for the month of December 2023" in tool_result["content"][0]["content"]


def test_filesystem_tools_hidden_by_default_in_main_agent_and_subagent():
    suite = get_suite(BENCHMARK_VERSION, "banking")
    env = suite.load_and_inject_default_environment({})
    task_call = _tool_call_message("task", {"description": "hi", "subagent_type": "general-purpose"}, "call_task")
    script = [task_call, AIMessage(content="sub done"), AIMessage(content="ok")]
    model = ToolListCapture(script=script, bound=[])
    _, _, _, _, extra_args = DeepAgentsPipeline(model).query("hi", FunctionsRuntime(suite.tools), env)

    agentdojo_tools = {function.name for function in suite.tools}
    setup = extra_args["deepagents"]["agent_setup"]
    assert set(setup["main"]["tools"]) == agentdojo_tools | {"task"}
    assert set(setup["general-purpose"]["tools"]) <= agentdojo_tools | {"task"}
    assert agentdojo_tools <= set(setup["general-purpose"]["tools"])
    # What the model was actually bound with matches what was recorded (main agent, then subagent).
    assert set(model.bound[0]) == set(setup["main"]["tools"])
    assert set(model.bound[1]) == set(setup["general-purpose"]["tools"])
    # Banking's own `read_file` is an AgentDojo tool and stays visible.
    assert "read_file" in setup["main"]["tools"]


def test_call_to_hidden_tool_is_rejected_not_executed():
    suite = get_suite(BENCHMARK_VERSION, "banking")
    env = suite.load_and_inject_default_environment({})
    write_call = _tool_call_message("write_file", {"file_path": "/notes.txt", "content": "x"}, "call_0")
    ls_call = _tool_call_message("ls", {"path": "/"}, "call_1")
    model = ScriptedChatModel(script=[write_call, ls_call, AIMessage(content="ok")])
    _, _, _, messages, extra_args = DeepAgentsPipeline(model).query("hi", FunctionsRuntime(suite.tools), env)

    tool_results = [m for m in messages if m["role"] == "tool"]
    assert [m["error"] for m in tool_results] == ["Invalid tool write_file provided.", "Invalid tool ls provided."]
    assert extra_args["deepagents"]["rejected_hidden_tool_calls"] == 2


def test_tool_middleware_factory_gets_user_task_in_main_agent_and_subagent():
    suite = get_suite(BENCHMARK_VERSION, "banking")
    task = suite.user_tasks["user_task_0"]
    prompts: list[str] = []
    spies: list[ToolCallSpy] = []

    def factory(prompt: str) -> Sequence[AgentMiddleware]:
        prompts.append(prompt)
        spies.append(ToolCallSpy())
        return spies[-1:]

    pipeline = GroundTruthReplay(task, delegate=True, tool_middleware_factory=factory)
    env = task.init_environment(suite.load_and_inject_default_environment({}))
    expected = [call.function for call in task.ground_truth(env.model_copy(deep=True))]
    pipeline.query(task.PROMPT, FunctionsRuntime(suite.tools), env)

    # Built once per task from the user's prompt, and shared by the main agent and the subagent.
    assert prompts == [task.PROMPT]
    assert spies[0].seen == ["task", *expected]


def test_benchmark_saves_trace_and_diagnostics(tmp_path: Path):
    suite = get_suite(BENCHMARK_VERSION, "banking")
    task = suite.user_tasks["user_task_0"]
    pipeline = GroundTruthReplay(task, delegate=True)
    with OutputLogger(str(tmp_path)):
        results = benchmark_suite_without_injections(
            pipeline,
            suite,
            user_tasks=["user_task_0"],
            logdir=tmp_path,
            force_rerun=False,
            benchmark_version=BENCHMARK_VERSION,
        )
    assert results["utility_results"][("user_task_0", "")] is True

    # Loading succeeds only with the messages logged, which is what lets an interrupted run resume.
    saved = load_task_results("scripted", "banking", "user_task_0", "none", "none", tmp_path)
    assert [m["role"] for m in saved.messages[:2]] == ["system", "user"]
    assert saved.messages[-1]["role"] == "assistant"
    raw = json.loads((tmp_path / "scripted" / "banking" / "user_task_0" / "none" / "none.json").read_text())
    assert raw["deepagents"]["n_subagent_events"] > 0


def test_from_config_name_lets_attacks_address_the_model():
    config = DeepAgentsConfig(model_id="deepseek/deepseek-v4.1-flash", max_model_calls=7)
    pipeline = DeepAgentsPipeline.from_config(config, model=ScriptedChatModel())
    assert pipeline.name == "deepagents-deepseek_deepseek-v4.1-flash"
    assert "/" not in pipeline.name
    assert get_model_name_from_pipeline(pipeline) == "DeepSeek"
    assert pipeline.max_model_calls == 7
