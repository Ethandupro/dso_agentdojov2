import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import openai
import pytest
from click.testing import CliRunner

pytest.importorskip("deepagents")

from langchain.agents.middleware import AgentMiddleware
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from pydantic import Field

from agentdojo import experiment_models
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.deepagents_pipeline import (
    CONTEXT_OVERFLOW_FINAL_MESSAGE,
    DEEPAGENTS_BUILTIN_TOOL_NAMES,
    REASONING_KWARG,
    DeepAgentsConfig,
    DeepAgentsPipeline,
    ReasoningChatOpenAI,
    ThinkingPromptChatOpenAI,
    ToolMiddlewareFactory,
    is_context_overflow_error,
    make_local_chat_model,
    thinking_prompt_chunks,
)
from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.attacks.base_attacks import get_model_name_from_pipeline
from agentdojo.benchmark import (
    benchmark_suite_without_injections,
    load_task_results,
    run_task_with_injection_tasks,
)
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.logging import OutputLogger
from agentdojo.scripts import benchmark as benchmark_script
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.types import ChatMessage
from tests.conftest import FAKE_REASONING_PROMPT

BENCHMARK_VERSION = "v1.2.2"


class ScriptedChatModel(BaseChatModel):
    """Returns pre-scripted AI messages in order, regardless of input. An exception in the script is raised."""

    script: Sequence[AIMessage | Exception] = Field(default_factory=list)
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
        if isinstance(message, Exception):
            raise message
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


class SlowToolSpy(AgentMiddleware):
    """Records when each tool call starts and ends, and makes earlier calls slower, so concurrent calls would
    finish out of order."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, str]] = []

    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]
    ) -> ToolMessage | Command:
        call_id = str(request.tool_call["id"])
        self.events.append(("start", call_id))
        time.sleep(0.05 * (3 - int(call_id.removeprefix("call_"))))
        result = handler(request)
        self.events.append(("end", call_id))
        return result


def test_parallel_tool_calls_run_in_order_one_at_a_time():
    suite = get_suite(BENCHMARK_VERSION, "banking")
    env = suite.load_and_inject_default_environment({})
    n_before = len(env.bank_account.transactions)
    calls = [
        {
            "name": "send_money",
            "args": {"recipient": f"R{i}", "amount": 1.0, "subject": "x", "date": "2022-01-01"},
            "id": f"call_{i}",
            "type": "tool_call",
        }
        for i in range(3)
    ]
    model = ScriptedChatModel(script=[AIMessage(content="", tool_calls=calls), AIMessage(content="ok")])
    spy = SlowToolSpy()
    DeepAgentsPipeline(model, tool_middleware=[spy]).query("pay", FunctionsRuntime(suite.tools), env)

    assert spy.events == [(event, f"call_{i}") for i in range(3) for event in ("start", "end")]
    new_transactions = env.bank_account.transactions[n_before:]
    assert [t.recipient for t in new_transactions] == ["R0", "R1", "R2"]


def test_from_config_name_lets_attacks_address_the_model():
    config = DeepAgentsConfig(model_id="deepseek/deepseek-v4.1-flash", max_model_calls=7)
    pipeline = DeepAgentsPipeline.from_config(config, model=ScriptedChatModel())
    assert pipeline.name == "deepagents-deepseek_deepseek-v4.1-flash"
    assert "/" not in pipeline.name
    assert get_model_name_from_pipeline(pipeline) == "DeepSeek"
    assert pipeline.max_model_calls == 7
    assert pipeline.config == config


def test_from_config_sets_openrouter_retries_and_timeout(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    config = DeepAgentsConfig(model_id="deepseek/deepseek-v4.1-flash", max_retries=4, request_timeout=60.0)
    model = DeepAgentsPipeline.from_config(config).model
    assert model.max_retries == 4  # type: ignore[attr-defined]
    assert model.request_timeout == 60.0  # type: ignore[attr-defined]


def test_from_config_pins_openrouter_provider_and_seed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    unpinned = DeepAgentsPipeline.from_config(DeepAgentsConfig(model_id="deepseek/deepseek-v4.1-flash")).model
    assert unpinned.extra_body is None  # type: ignore[attr-defined]
    assert unpinned.seed is None  # type: ignore[attr-defined]
    config = DeepAgentsConfig(
        model_id="deepseek/deepseek-v4.1-flash", provider_order=["deepinfra"], allow_fallbacks=False, seed=3
    )
    model = DeepAgentsPipeline.from_config(config).model
    assert model.extra_body == {"provider": {"order": ["deepinfra"], "allow_fallbacks": False}}  # type: ignore[attr-defined]
    assert model.seed == 3  # type: ignore[attr-defined]


def test_from_config_uses_a_local_server(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)
    config = DeepAgentsConfig(
        model_id="Qwen/Qwen3.5-27B", base_url="http://localhost:8000/v1", enable_thinking=False, seed=1
    )
    pipeline = DeepAgentsPipeline.from_config(config)
    model = pipeline.model
    assert str(model.openai_api_base) == "http://localhost:8000/v1"  # type: ignore[attr-defined]
    assert model.model_name == "Qwen/Qwen3.5-27B"  # type: ignore[attr-defined]
    assert model.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}  # type: ignore[attr-defined]
    assert model.seed == 1  # type: ignore[attr-defined]
    # Separate trace directory from OpenRouter runs of the same model, and attacks can address the model.
    assert pipeline.name == "deepagents-local-Qwen_Qwen3.5-27B"
    assert get_model_name_from_pipeline(pipeline) == "Qwen"
    no_thinking_flag = DeepAgentsPipeline.from_config(
        DeepAgentsConfig(model_id="Qwen/Qwen3-32B", base_url="http://localhost:8000/v1")
    ).model
    assert no_thinking_flag.extra_body is None  # type: ignore[attr-defined]
    gemma = DeepAgentsPipeline.from_config(
        DeepAgentsConfig(model_id="google/gemma-4-26B-A4B-it", base_url="http://localhost:8000/v1")
    )
    assert get_model_name_from_pipeline(gemma) == "Gemma"


def test_config_rejects_openrouter_options_with_a_local_server():
    with pytest.raises(ValueError, match="provider_order"):
        DeepAgentsConfig(model_id="m", base_url="http://localhost:8000/v1", provider_order=["deepinfra"])
    with pytest.raises(ValueError, match="enable_thinking"):
        DeepAgentsConfig(model_id="m", enable_thinking=False)


def test_token_usage_is_recorded_per_agent():
    suite = get_suite(BENCHMARK_VERSION, "banking")
    call = {"name": "get_balance", "args": {}, "id": "call_0", "type": "tool_call"}
    usage = {"input_tokens": 100, "output_tokens": 7, "total_tokens": 107}
    model = ScriptedChatModel(
        script=[
            AIMessage(content="", tool_calls=[call], usage_metadata=usage, response_metadata={"model_name": "m"}),
            AIMessage(content="done", usage_metadata=usage, response_metadata={"model_name": "m"}),
        ]
    )
    _, _, _, _, extra = DeepAgentsPipeline(model).query(
        "balance?", FunctionsRuntime(suite.tools), suite.load_and_inject_default_environment({})
    )
    assert extra["deepagents"]["usage"] == {"main": {"model_calls": 2, "input_tokens": 200, "output_tokens": 14}}
    assert extra["deepagents"]["served_models"] == ["m"]


def test_json_tool_output_format_handles_datetimes():
    suite = get_suite(BENCHMARK_VERSION, "workspace")
    env = suite.load_and_inject_default_environment({})
    config = DeepAgentsConfig(model_id="deepseek/deepseek-v4.1-flash", tool_output_format="json")
    pipeline = DeepAgentsPipeline.from_config(config, model=ScriptedChatModel())
    result, error = FunctionsRuntime(suite.tools).run_function(env, "search_calendar_events", {"query": "Networking"})
    assert error is None
    assert json.loads(pipeline.tool_output_formatter(result))


def _bad_request(message: str) -> openai.BadRequestError:
    """A 400 error as the OpenAI client raises it (built without an HTTP response)."""
    error = openai.BadRequestError.__new__(openai.BadRequestError)
    Exception.__init__(error, message)
    error.message = message
    error.status_code = 400
    error.body = None
    error.code = None
    error.param = None
    return error


OPENROUTER_CONTEXT_ERROR = (
    "Error code: 400 - {'error': {'message': \"This endpoint's maximum context length is 163840 tokens. However, "
    "you requested about 170000 tokens. Please reduce the length of either one.\", 'code': 400}}"
)


def test_context_overflow_detection():
    assert is_context_overflow_error(_bad_request(OPENROUTER_CONTEXT_ERROR))
    assert is_context_overflow_error(_bad_request("Error code: 400 - {'code': 'context_length_exceeded'}"))
    assert not is_context_overflow_error(_bad_request("Error code: 400 - invalid tool schema"))
    assert not is_context_overflow_error(RuntimeError("maximum context length"))


class HijackedReplay(BasePipelineElement):
    """Scripts the deep agent to do the injection task's ground truth (as a hijacked agent would), then to end
    with the given final step: an answer, or an exception raised by the model."""

    def __init__(self, injection_task, final: AIMessage | Exception, name: str = "scripted") -> None:
        self.injection_task = injection_task
        self.final = final
        self.model = ScriptedChatModel()
        self.pipeline = DeepAgentsPipeline(self.model, name=name)
        self.name = name

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
            for i, call in enumerate(self.injection_task.ground_truth(env))
        ]
        self.model.script = [*calls, self.final]
        self.model.position = 0
        return self.pipeline.query(query, runtime, env, messages, extra_args)


def _run_banking_pair(pipeline: BasePipelineElement, logdir: Path):
    suite = get_suite(BENCHMARK_VERSION, "banking")
    user_task = suite.user_tasks["user_task_0"]
    attack = load_attack("direct", suite, pipeline)
    with OutputLogger(str(logdir)):
        return run_task_with_injection_tasks(
            suite, pipeline, user_task, attack, logdir, False, ["injection_task_0"], BENCHMARK_VERSION
        )


def test_context_overflow_ends_run_and_scores_actions_already_taken(tmp_path: Path):
    suite = get_suite(BENCHMARK_VERSION, "banking")
    injection_task = suite.injection_tasks["injection_task_0"]
    pipeline = HijackedReplay(injection_task, final=_bad_request(OPENROUTER_CONTEXT_ERROR))
    utility, security = _run_banking_pair(pipeline, tmp_path)

    # The injected transfer happened before the overflow, so the attack counts as successful.
    assert security[("user_task_0", "injection_task_0")] is True
    assert utility[("user_task_0", "injection_task_0")] is False
    saved = load_task_results("scripted", "banking", "user_task_0", "direct", "injection_task_0", tmp_path)
    # DeepAgents' summarization middleware first tries to summarize and retry, then gives up.
    assert saved.error is not None and "maximum context length" in saved.error
    assert saved.messages[-1]["content"] == [{"type": "text", "content": CONTEXT_OVERFLOW_FINAL_MESSAGE}]
    assert any(m["role"] == "tool" and m["tool_call"].function == "send_money" for m in saved.messages)
    raw = json.loads(
        (tmp_path / "scripted" / "banking" / "user_task_0" / "direct" / "injection_task_0.json").read_text()
    )
    assert "maximum context length" in raw["deepagents"]["error"]


def test_other_model_errors_stop_the_run_and_the_pair_reruns_on_resume(tmp_path: Path):
    suite = get_suite(BENCHMARK_VERSION, "banking")
    injection_task = suite.injection_tasks["injection_task_0"]
    failing = HijackedReplay(injection_task, final=_bad_request("Error code: 400 - provider returned an error"))
    with pytest.raises(openai.BadRequestError):
        _run_banking_pair(failing, tmp_path)

    # The partial transcript is saved for debugging, but the pair has no result, so a resumed run redoes it.
    raw = json.loads(
        (tmp_path / "scripted" / "banking" / "user_task_0" / "direct" / "injection_task_0.json").read_text()
    )
    assert any(m["role"] == "tool" for m in raw["messages"])
    assert "utility" not in raw
    resumed = HijackedReplay(injection_task, final=AIMessage(content="Done."))
    _, security = _run_banking_pair(resumed, tmp_path)
    assert security[("user_task_0", "injection_task_0")] is True
    assert resumed.model.position > 0


def test_empty_final_answer_is_scored_without_rerunning_the_task():
    suite = get_suite(BENCHMARK_VERSION, "banking")
    injection_task = suite.injection_tasks["injection_task_0"]
    user_task = suite.user_tasks["user_task_0"]
    env = suite.load_and_inject_default_environment({})
    n_before = len(env.bank_account.transactions)
    pipeline = HijackedReplay(injection_task, final=AIMessage(content=""))
    _, security = suite.run_task_with_pipeline(pipeline, user_task, injection_task, {}, environment=env)

    assert security is True
    # One run only: AgentDojo reruns a task whose final answer has no content, which would repeat the transfer.
    assert len(env.bank_account.transactions) == n_before + 1


class _CombinedTask:
    """A user task whose ground truth is followed by an injection task's ground truth (what a hijacked agent does).
    Duck-types the task interface that `GroundTruthPipeline` and `GroundTruthReplay` use."""

    def __init__(self, user_task, injection_task) -> None:
        self.user_task = user_task
        self.injection_task = injection_task
        self.GROUND_TRUTH_OUTPUT = user_task.GROUND_TRUTH_OUTPUT

    def ground_truth(self, env):
        calls = list(self.user_task.ground_truth(env))
        if self.injection_task is not None:
            calls += self.injection_task.ground_truth(env)
        return calls


def _pair_replay(user_task, injection_task, pipeline_class) -> BasePipelineElement:
    combined = _CombinedTask(user_task, injection_task)
    if pipeline_class is GroundTruthReplay:
        return GroundTruthReplay(combined, delegate=False)
    return GroundTruthPipeline(combined)  # type: ignore[arg-type]


@pytest.mark.parametrize("suite_name", SUITE_NAMES)
def test_every_injection_task_scores_like_native_through_the_harness(suite_name: str):
    """For every injection task, a hijacked run (user task ground truth, then the injection task's ground truth)
    and a benign run through the deep agent get the same utility and security as through AgentDojo's own loop.
    Injections come from the `important_instructions` attack, placed in the environment as in a real run."""
    suite = get_suite(BENCHMARK_VERSION, suite_name)
    name = "deepagents-deepseek_deepseek-v4.1-flash"
    attack = load_attack("important_instructions", suite, HijackedReplay(None, AIMessage(content=""), name=name))
    mismatches = []
    # Three user tasks per suite keep this fast; all 949 pairs of v1.2.2 were checked the same way (no mismatches).
    for user_task in list(suite.user_tasks.values())[:3]:
        for injection_task in suite.injection_tasks.values():
            injections = attack.attack(user_task, injection_task)
            for hijacked in (False, True):
                results = []
                for pipeline in (GroundTruthPipeline, GroundTruthReplay):
                    replay = _pair_replay(user_task, injection_task if hijacked else None, pipeline)
                    results.append(suite.run_task_with_pipeline(replay, user_task, injection_task, injections))
                if results[0] != results[1]:
                    mismatches.append((user_task.ID, injection_task.ID, hijacked, results))
    assert mismatches == []


def test_summarization_is_recorded_in_diagnostics():
    suite = get_suite(BENCHMARK_VERSION, "workspace")
    env = suite.load_and_inject_default_environment({})
    # Each `list_files` result is ~7k tokens, so a 20k-token model triggers DeepAgents' summarization.
    # The scripted model also answers the summarization call, with whichever script entry comes next.
    calls = [_tool_call_message("list_files", {}, f"call_{i}") for i in range(3)]
    model = ScriptedChatModel(script=[*calls, AIMessage(content="Done."), AIMessage(content="Done.")])
    model.profile = {"max_input_tokens": 20_000}
    _, _, _, _, extra_args = DeepAgentsPipeline(model).query("hi", FunctionsRuntime(suite.tools), env)

    summaries = extra_args["deepagents"]["summaries"]
    assert len(summaries) == 1
    assert summaries[0]["agent"] == "main"
    assert "conversation that has been summarized" in summaries[0]["summary"]


def test_cli_passes_provider_pinning_to_the_victim_config(monkeypatch: pytest.MonkeyPatch):
    captured: list[DeepAgentsConfig] = []

    def fake_from_config(config: DeepAgentsConfig, **kwargs: Any) -> DeepAgentsPipeline:
        captured.append(config)
        raise SystemExit(0)

    monkeypatch.setattr(DeepAgentsPipeline, "from_config", staticmethod(fake_from_config))
    args = [
        "-s",
        "banking",
        "-ut",
        "user_task_0",
        "--harness",
        "deepagents",
        "--model-id",
        "deepseek/deepseek-v4.1-flash",
    ]
    CliRunner().invoke(benchmark_script.main, [*args, "--provider", "deepinfra", "--no-provider-fallbacks"])
    assert captured[-1].provider_order == ["deepinfra"] and captured[-1].allow_fallbacks is False
    CliRunner().invoke(benchmark_script.main, args)
    assert captured[-1].provider_order is None


def test_thinking_prompt_is_split_into_mistral_content_chunks():
    assert thinking_prompt_chunks(FAKE_REASONING_PROMPT) == [
        {"type": "text", "text": "Think first."},
        {"type": "thinking", "thinking": "Draft your thoughts here.", "closed": True},
        {"type": "text", "text": "Then answer."},
    ]
    with pytest.raises(ValueError, match="THINK"):
        thinking_prompt_chunks("no think tags")
    with pytest.raises(ValueError, match="THINK"):
        DeepAgentsConfig(model_id="m", base_url="http://localhost:8000/v1", reasoning_system_prompt="no tags")
    with pytest.raises(ValueError, match="local server"):
        DeepAgentsConfig(model_id="m", reasoning_system_prompt=FAKE_REASONING_PROMPT)


def test_magistral_requests_carry_the_reasoning_prompt_ahead_of_the_agents():
    config = DeepAgentsConfig(
        model_id="mistralai/Magistral-Small-2509",
        base_url="http://localhost:8000/v1",
        reasoning_system_prompt=FAKE_REASONING_PROMPT,
        temperature=0.7,
        top_p=0.95,
    )
    pipeline = DeepAgentsPipeline.from_config(config)
    model = pipeline.model
    assert isinstance(model, ThinkingPromptChatOpenAI)
    payload = model._get_request_payload([SystemMessage("You are the agent."), HumanMessage("hi")])
    assert payload["top_p"] == 0.95 and payload["temperature"] == 0.7
    system, user = payload["messages"]
    assert system["role"] == "system" and user == {"role": "user", "content": "hi"}
    assert system["content"] == [
        {"type": "text", "text": "Think first."},
        {"type": "thinking", "thinking": "Draft your thoughts here.", "closed": True},
        {"type": "text", "text": "Then answer.\n\nYou are the agent."},
    ]
    no_system = model._get_request_payload([HumanMessage("hi")])
    assert no_system["messages"][0]["content"][1]["type"] == "thinking"
    assert get_model_name_from_pipeline(pipeline) == "Magistral"


def test_cli_passes_local_server_to_the_victim_config(monkeypatch: pytest.MonkeyPatch):
    captured: list[DeepAgentsConfig] = []

    def fake_from_config(config: DeepAgentsConfig, **kwargs: Any) -> DeepAgentsPipeline:
        captured.append(config)
        raise SystemExit(0)

    monkeypatch.setattr(DeepAgentsPipeline, "from_config", staticmethod(fake_from_config))
    args = ["-s", "banking", "-ut", "user_task_0", "--harness", "deepagents", "--model-id", "Qwen/Qwen3.5-27B"]
    CliRunner().invoke(benchmark_script.main, [*args, "--base-url", "http://localhost:8000/v1", "--no-thinking"])
    assert captured[-1].base_url == "http://localhost:8000/v1" and captured[-1].enable_thinking is False
    result = CliRunner().invoke(benchmark_script.main, [*args, "--no-thinking"])
    assert isinstance(result.exception, ValueError)


def test_cli_defaults_to_the_project_victim_and_checker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    captured: list[tuple[DeepAgentsConfig, Any]] = []

    def fake_from_config(config: DeepAgentsConfig, **kwargs: Any) -> DeepAgentsPipeline:
        captured.append((config, kwargs.get("tool_middleware_factory")))
        raise SystemExit(0)

    monkeypatch.setattr(DeepAgentsPipeline, "from_config", staticmethod(fake_from_config))
    args = ["-s", "banking", "-ut", "user_task_0", "--harness", "deepagents", "--logdir", str(tmp_path)]
    CliRunner().invoke(benchmark_script.main, args)
    config, factory = captured[-1]
    assert config.model_id == experiment_models.VICTIM_MODEL_ID == "mistralai/Magistral-Small-2509"
    assert config.base_url == experiment_models.VICTIM_BASE_URL
    assert config.enable_thinking is None  # Magistral thinks through its reasoning system prompt, not a switch
    assert config.reasoning_system_prompt == FAKE_REASONING_PROMPT
    assert (config.temperature, config.top_p) == (0.7, 0.95)
    assert factory is None

    CliRunner().invoke(benchmark_script.main, [*args, "--checker"])
    config, factory = captured[-1]
    assert config.model_id == experiment_models.VICTIM_MODEL_ID
    assert factory is not None
    (checker,) = factory("the user's task")
    assert checker.skill_file == experiment_models.CHECKER_SKILL_FILE.read_text()


def test_checker_runs_get_their_own_trace_directory():
    s0 = experiment_models.CHECKER_SKILL_FILE.read_text()
    suffix = benchmark_script.checker_pipeline_suffix("Qwen/Qwen3.5-9B", s0)
    assert suffix.startswith("-checker-Qwen_Qwen3.5-9B-skill-")
    assert suffix != benchmark_script.checker_pipeline_suffix("Qwen/Qwen3.5-9B", s0 + "\nnew rule")


def test_local_model_keeps_the_servers_reasoning_out_of_band():
    model = make_local_chat_model("m", "http://localhost:8000/v1")
    assert isinstance(model, ReasoningChatOpenAI)
    response = {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Paid.", "reasoning_content": "The bill asks for more."},
            }
        ],
    }
    message = model._create_chat_result(response).generations[0].message
    assert message.content == "Paid."
    assert message.additional_kwargs[REASONING_KWARG] == "The bill asks for more."
    # Newer vLLM builds name the field `reasoning`; a reply without reasoning gets no key.
    response["choices"][0]["message"] = {"role": "assistant", "content": "Paid.", "reasoning": "Hmm."}
    assert model._create_chat_result(response).generations[0].message.additional_kwargs[REASONING_KWARG] == "Hmm."
    response["choices"][0]["message"] = {"role": "assistant", "content": "Paid."}
    assert REASONING_KWARG not in model._create_chat_result(response).generations[0].message.additional_kwargs
    # The reasoning is never sent back to the model.
    payload = model._get_request_payload([HumanMessage("hi"), message, HumanMessage("again")])
    assert all("reasoning" not in json.dumps(m) for m in payload["messages"])


def test_reasoning_is_recorded_per_message_and_kept_out_of_the_content():
    suite = get_suite(BENCHMARK_VERSION, "banking")
    call = {"name": "get_balance", "args": {}, "id": "call_0", "type": "tool_call"}
    model = ScriptedChatModel(
        script=[
            AIMessage(content="", tool_calls=[call], additional_kwargs={REASONING_KWARG: "Check the balance first."}),
            AIMessage(content="done", additional_kwargs={REASONING_KWARG: "Now answer."}),
        ]
    )
    _, _, _, messages, extra = DeepAgentsPipeline(model).query(
        "balance?", FunctionsRuntime(suite.tools), suite.load_and_inject_default_environment({})
    )
    reasoning = extra["deepagents"]["reasoning"]
    assert [r["text"] for r in reasoning] == ["Check the balance first.", "Now answer."]
    for r in reasoning:
        assert messages[r["message"]]["role"] == "assistant" and r["agent"] == "main"
    assert messages[reasoning[-1]["message"]] is messages[-1]
    # AgentDojo scores the content as the model's output, so the reasoning must not be in it.
    assert all("Now answer" not in json.dumps(m["content"]) for m in messages if m["content"])
