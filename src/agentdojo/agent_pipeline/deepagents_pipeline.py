"""Runs the victim agent with LangChain DeepAgents (`create_deep_agent`) inside an AgentDojo pipeline.

Requires the `deepagents` extra (`uv sync --extra deepagents`, Python >= 3.11).

AgentDojo's tools are exposed as LangChain tools bound to the task environment, so the benchmark's utility and
security checks see the same environment mutations as with the native pipeline. By default, the agent sees
AgentDojo's tools plus DeepAgents' general-purpose subagent (`task`); DeepAgents' in-memory virtual filesystem tools
are hidden (see `DeepAgentsPipeline`).

Every model response and tool result, at every agent level (main agent and subagents), is recorded by
`_TranscriptRecorder` and flattened in chronological order into AgentDojo messages. This is needed because a
subagent's tool calls never appear in the main agent's message list, but AgentDojo's checks read the full
function-call trace.

Experiment runs (C0-C5 in `skillfile_evolution/DESIGN.md`) build the victim with
`DeepAgentsPipeline.from_config(DeepAgentsConfig(...), tool_middleware_factory=...)`, where the factory builds the
checker for each task. The benchmark CLI selects it with `--harness deepagents --model-id <OpenRouter model id>`.
"""

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal

from deepagents import create_deep_agent
from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware, ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from pydantic import BaseModel

from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import (
    EmptyEnv,
    Env,
    Function,
    FunctionCall,
    FunctionReturnType,
    FunctionsRuntime,
    TaskEnvironment,
)
from agentdojo.logging import Logger, TraceLogger
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    ChatSystemMessage,
    ChatToolResultMessage,
    ChatUserMessage,
    text_content_block_from_string,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEEPAGENTS_BUILTIN_TOOL_NAMES = frozenset(
    {"ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "task"}
)
"""Tools added by `create_deep_agent` with its default middleware and no model-specific harness profile
(deepagents 0.7.25). There is no `write_todos` planning tool and no `execute` tool in this configuration, and
DeepAgents adds no system-prompt text of its own. Some harness profiles (selected by model) change this; the tools and
system prompt each agent actually received are recorded in `extra_args["deepagents"]["agent_setup"]`.

If an AgentDojo tool has the same name (banking's `read_file`), LangChain registers the AgentDojo tool last, so it
replaces the built-in one for both the model's tool list and execution. Shadowed names are reported in
`extra_args["deepagents"]["shadowed_builtin_tools"]`.
"""

DEEPAGENTS_FILESYSTEM_TOOL_NAMES = DEEPAGENTS_BUILTIN_TOOL_NAMES - {"task"}

DEEPAGENTS_EVICTION_THRESHOLD_CHARS = 4 * 20_000
"""DeepAgents' `FilesystemMiddleware` replaces tool results longer than this (20k tokens at 4 chars/token, deepagents
0.7.25 defaults) with a pointer to a file in the virtual filesystem. With the filesystem tools hidden, the agent could
not read it back, so such outputs are counted in `extra_args["deepagents"]["oversized_tool_outputs"]`. The largest
ground-truth tool output in AgentDojo v1.2.2 is ~28k chars (workspace `list_files`)."""

_AGENTDOJO_TOOL_METADATA_KEY = "agentdojo_function"

ToolMiddlewareFactory = Callable[[str], Sequence[AgentMiddleware]]
"""Builds tool middleware (e.g. the checker) for one task, given the user's task prompt. The middleware it returns
is attached to both the main agent and the subagent, so the subagent's checker also sees the user's task rather
than the main agent's delegation message."""


class DeepAgentsConfig(BaseModel):
    """Everything that defines the DeepAgents victim of a run, apart from the tool middleware.

    Experiment runs save it in their `config.json`, so a run only resumes with an identical victim.
    """

    model_id: str
    """OpenRouter model id, e.g. `deepseek/deepseek-v4.1-flash`."""
    temperature: float | None = 0.0
    max_model_calls: int = 30
    """Per-agent limit on model calls in one run (main agent and each subagent run separately)."""
    hide_filesystem_tools: bool = True
    system_message_name: str | None = None
    """The name of the system message to use. If not provided, AgentDojo's default system message is used."""
    system_message: str | None = None
    """The system message to use. If provided, it overrides `system_message_name`."""
    tool_output_format: Literal["yaml", "json"] | None = None
    """Format to use for tool outputs. If not provided, the default format is yaml."""

    @property
    def pipeline_name(self) -> str:
        """Contains the model id, so attacks can address the model (see `agentdojo.models.MODEL_NAMES`). `/` is
        replaced because AgentDojo saves traces under `<logdir>/<pipeline name with / replaced>/` but looks
        them up under the raw name, so a `/` would make finished tasks look unfinished."""
        return f"deepagents-{self.model_id.replace('/', '_')}"


def make_openrouter_chat_model(model: str, temperature: float | None = 0.0, **kwargs: Any) -> ChatOpenAI:
    """Builds a LangChain chat model that talks to OpenRouter's OpenAI-compatible API.

    Reads the API key from the `OPENROUTER_API_KEY` environment variable.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set")
    return ChatOpenAI(model=model, base_url=OPENROUTER_BASE_URL, api_key=api_key, temperature=temperature, **kwargs)  # type: ignore[arg-type]


@dataclass
class AgentDojoToolArtifact:
    """Attached to the `ToolMessage` of every AgentDojo tool call, so middleware and the transcript can tell the raw
    tool result and error apart from what the agent was shown."""

    raw_output: str
    error: str | None


def agentdojo_function_to_langchain_tool(
    function: Function,
    runtime: FunctionsRuntime,
    env: TaskEnvironment,
    tool_output_formatter: Callable[[FunctionReturnType], str] = tool_result_to_str,
) -> StructuredTool:
    """Wraps an AgentDojo function as a LangChain tool that runs through `runtime` against `env`.

    Like the native OpenAI pipeline, the agent is shown the error message if the call failed, and the formatted
    result otherwise.
    """

    def _run(**kwargs: Any) -> tuple[str, AgentDojoToolArtifact]:
        result, error = runtime.run_function(env, function.name, kwargs)
        raw_output = tool_output_formatter(result)
        shown = error if error is not None else raw_output
        return shown, AgentDojoToolArtifact(raw_output=raw_output, error=error)

    return StructuredTool.from_function(
        func=_run,
        name=function.name,
        description=function.description,
        args_schema=function.parameters,
        response_format="content_and_artifact",
        metadata={_AGENTDOJO_TOOL_METADATA_KEY: True},
    )


def _is_agentdojo_tool(tool: BaseTool | dict | None) -> bool:
    return isinstance(tool, BaseTool) and bool((tool.metadata or {}).get(_AGENTDOJO_TOOL_METADATA_KEY))


@dataclass
class _TranscriptEvent:
    agent: str
    message: BaseMessage
    tool_call: dict | None = None


@dataclass
class _Transcript:
    events: list[_TranscriptEvent] = field(default_factory=list)
    # First model call of each agent: the tool names and system prompt it was given
    agent_setup: dict[str, dict[str, Any]] = field(default_factory=dict)
    oversized_tool_outputs: int = 0
    rejected_hidden_tool_calls: int = 0


class _TranscriptRecorder(AgentMiddleware):
    """Records model responses and tool results (as the agent saw them) into a shared transcript.

    It must be the first (outermost) middleware so that it records tool outputs after other middleware, such as a
    checker, has modified them.
    """

    def __init__(self, transcript: _Transcript, agent: str) -> None:
        super().__init__()
        self.transcript = transcript
        self.agent = agent

    @property
    def name(self) -> str:
        return f"AgentDojoTranscriptRecorder[{self.agent}]"

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        response = handler(request)
        for message in response.result:
            if isinstance(message, AIMessage):
                self.transcript.events.append(_TranscriptEvent(agent=self.agent, message=message))
        return response

    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]
    ) -> ToolMessage | Command:
        result = handler(request)
        tool_message = _tool_message_from_result(result)
        if tool_message is not None:
            if (
                _is_agentdojo_tool(request.tool)
                and len(_message_text(tool_message)) > DEEPAGENTS_EVICTION_THRESHOLD_CHARS
            ):
                self.transcript.oversized_tool_outputs += 1
            self.transcript.events.append(
                _TranscriptEvent(agent=self.agent, message=tool_message, tool_call=dict(request.tool_call))
            )
        return result


def _tool_name(tool: BaseTool | dict) -> str:
    return tool.name if isinstance(tool, BaseTool) else str(tool.get("name"))


class _ToolVisibility(AgentMiddleware):
    """Optionally hides DeepAgents' virtual filesystem tools, and records what each agent was given.

    Hidden tools are removed from the model's tool list, and calls to them (e.g. a hallucinated `ls`) are answered
    with an error instead of being executed, as the native AgentDojo pipeline does for unknown tools. AgentDojo tools
    are never hidden, even when their name matches a filesystem tool (banking's `read_file`).

    Placed right inside the transcript recorder, so the recorder logs the error messages the agent was shown.
    """

    def __init__(self, transcript: _Transcript, agent: str, hide_filesystem_tools: bool) -> None:
        super().__init__()
        self.transcript = transcript
        self.agent = agent
        self.hide_filesystem_tools = hide_filesystem_tools

    @property
    def name(self) -> str:
        return f"AgentDojoToolVisibility[{self.agent}]"

    def _is_hidden(self, tool: BaseTool | dict) -> bool:
        return (
            self.hide_filesystem_tools
            and not _is_agentdojo_tool(tool)
            and _tool_name(tool) in DEEPAGENTS_FILESYSTEM_TOOL_NAMES
        )

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        visible_tools = [tool for tool in request.tools if not self._is_hidden(tool)]
        if len(visible_tools) != len(request.tools):
            request = request.override(tools=visible_tools)
        if self.agent not in self.transcript.agent_setup:
            self.transcript.agent_setup[self.agent] = {
                "tools": sorted(_tool_name(tool) for tool in visible_tools),
                "system_prompt": request.system_message.text if request.system_message is not None else None,
            }
        return handler(request)

    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]
    ) -> ToolMessage | Command:
        if request.tool is not None and self._is_hidden(request.tool):
            self.transcript.rejected_hidden_tool_calls += 1
            name = request.tool_call["name"]
            return ToolMessage(
                content=f"Invalid tool {name} provided.",
                tool_call_id=request.tool_call["id"],
                name=name,
                status="error",
            )
        return handler(request)


def _tool_message_from_result(result: ToolMessage | Command) -> ToolMessage | None:
    if isinstance(result, ToolMessage):
        return result
    update = result.update
    if isinstance(update, dict):
        for message in reversed(update.get("messages", [])):
            if isinstance(message, ToolMessage):
                return message
    return None


def _message_text(message: BaseMessage) -> str:
    return message.text if isinstance(message.text, str) else message.text()  # type: ignore[operator]


def _to_function_call(tool_call: dict) -> FunctionCall:
    return FunctionCall(function=tool_call["name"], args=dict(tool_call.get("args") or {}), id=tool_call.get("id"))


def _event_to_agentdojo(event: _TranscriptEvent) -> ChatMessage:
    message = event.message
    if isinstance(message, AIMessage):
        text = _message_text(message)
        tool_calls = [_to_function_call(dict(tc)) for tc in message.tool_calls]
        return ChatAssistantMessage(
            role="assistant",
            content=[text_content_block_from_string(text)] if text else None,
            tool_calls=tool_calls or None,
        )
    assert isinstance(message, ToolMessage) and event.tool_call is not None
    artifact = message.artifact
    if isinstance(artifact, AgentDojoToolArtifact):
        error = artifact.error
    else:
        error = _message_text(message) if message.status == "error" else None
    return ChatToolResultMessage(
        role="tool",
        content=[text_content_block_from_string(_message_text(message))],
        tool_call_id=message.tool_call_id,
        tool_call=_to_function_call(event.tool_call),
        error=error,
    )


class DeepAgentsPipeline(BasePipelineElement):
    """A complete AgentDojo agent pipeline whose agent is a LangChain deep agent.

    Use it on its own (not inside an `AgentPipeline` with other LLM elements): it adds the system and user
    messages itself and runs the whole tool-calling loop.

    Args:
        model: the LangChain chat model used by the main agent and the general-purpose subagent.
        name: the pipeline name. AgentDojo attacks derive the model name from it, so it should contain the model
            name (see `agentdojo.models.MODEL_NAMES`).
        system_prompt: passed to `create_deep_agent`. Defaults to AgentDojo's default system message.
        tool_middleware: extra middleware applied to both the main agent and the general-purpose subagent, inside
            the transcript recorder.
        tool_middleware_factory: builds more such middleware (e.g. the checker) for each task, from the user's task
            prompt. Its middleware runs after `tool_middleware`.
        hide_filesystem_tools: hide DeepAgents' in-memory virtual filesystem tools from the main agent and the
            subagent. On by default: with them visible, agents look for the user's files in the (empty) virtual
            filesystem instead of using AgentDojo's tools, which lowers both utility and ASR for reasons unrelated
            to security.
        max_model_calls: per-agent limit on model calls in one run (main agent and each subagent run separately).
            When it is hit, the agent ends gracefully instead of raising.
    """

    def __init__(
        self,
        model: BaseChatModel,
        name: str | None = None,
        system_prompt: str | None = None,
        tool_middleware: Sequence[AgentMiddleware] = (),
        max_model_calls: int = 30,
        tool_output_formatter: Callable[[FunctionReturnType], str] = tool_result_to_str,
        hide_filesystem_tools: bool = True,
        tool_middleware_factory: ToolMiddlewareFactory | None = None,
    ) -> None:
        self.model = model
        self.name = name
        self.system_prompt = system_prompt if system_prompt is not None else load_system_message(None)
        self.tool_middleware = list(tool_middleware)
        self.tool_middleware_factory = tool_middleware_factory
        self.max_model_calls = max_model_calls
        self.tool_output_formatter = tool_output_formatter
        self.hide_filesystem_tools = hide_filesystem_tools

    @classmethod
    def from_config(
        cls,
        config: DeepAgentsConfig,
        tool_middleware: Sequence[AgentMiddleware] = (),
        tool_middleware_factory: ToolMiddlewareFactory | None = None,
        model: BaseChatModel | None = None,
    ) -> "DeepAgentsPipeline":
        """Builds the pipeline for a run. `model` replaces the OpenRouter model built from `config` (for tests)."""
        if model is None:
            model = make_openrouter_chat_model(config.model_id, temperature=config.temperature)
        if config.tool_output_format == "json":
            tool_output_formatter = partial(tool_result_to_str, dump_fn=json.dumps)
        else:
            tool_output_formatter = tool_result_to_str
        system_prompt = (
            config.system_message
            if config.system_message is not None
            else load_system_message(config.system_message_name)
        )
        return cls(
            model,
            name=config.pipeline_name,
            system_prompt=system_prompt,
            tool_middleware=tool_middleware,
            max_model_calls=config.max_model_calls,
            tool_output_formatter=tool_output_formatter,
            hide_filesystem_tools=config.hide_filesystem_tools,
            tool_middleware_factory=tool_middleware_factory,
        )

    def _build_agent(self, tools: list[BaseTool], transcript: _Transcript, tool_middleware: list[AgentMiddleware]):
        # The default general-purpose subagent is "isolated" and does not inherit the main agent's custom
        # middleware (deepagents 0.7.25, graph.py). Declaring it explicitly, with the same name, replaces the
        # default and lets us attach the recorder and the tool middleware to it.
        general_purpose: Any = {
            **GENERAL_PURPOSE_SUBAGENT,
            "middleware": [
                _TranscriptRecorder(transcript, agent="general-purpose"),
                _ToolVisibility(transcript, agent="general-purpose", hide_filesystem_tools=self.hide_filesystem_tools),
                *tool_middleware,
                ModelCallLimitMiddleware(run_limit=self.max_model_calls, exit_behavior="end"),
            ],
        }
        main_middleware: list[AgentMiddleware[Any, Any, Any]] = [
            _TranscriptRecorder(transcript, agent="main"),
            _ToolVisibility(transcript, agent="main", hide_filesystem_tools=self.hide_filesystem_tools),
            *tool_middleware,
            ModelCallLimitMiddleware(run_limit=self.max_model_calls, exit_behavior="end"),
        ]
        return create_deep_agent(
            model=self.model,
            tools=tools,
            system_prompt=self.system_prompt,
            middleware=main_middleware,
            subagents=[general_purpose],
        )

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if len(messages) > 0:
            raise ValueError("DeepAgentsPipeline runs the whole agent and must receive an empty message list")

        tools: list[BaseTool] = [
            agentdojo_function_to_langchain_tool(function, runtime, env, self.tool_output_formatter)
            for function in runtime.functions.values()
        ]
        shadowed = sorted({tool.name for tool in tools} & DEEPAGENTS_BUILTIN_TOOL_NAMES)
        tool_middleware = list(self.tool_middleware)
        if self.tool_middleware_factory is not None:
            tool_middleware.extend(self.tool_middleware_factory(query))
        transcript = _Transcript()
        agent = self._build_agent(tools, transcript, tool_middleware)
        # LangGraph runs the tool calls of one model turn concurrently, in threads. AgentDojo's own loop runs them
        # one after another, in the order the model gave them, and environment changes (e.g. banking transaction
        # ids) depend on that order. `max_concurrency=1` restores it, for the main agent and the subagent.
        final_state = agent.invoke({"messages": [HumanMessage(content=query)]}, config={"max_concurrency": 1})

        new_messages: list[ChatMessage] = [
            ChatSystemMessage(role="system", content=[text_content_block_from_string(self.system_prompt)]),
            ChatUserMessage(role="user", content=[text_content_block_from_string(query)]),
            *(_event_to_agentdojo(event) for event in transcript.events),
        ]

        # A message that ends the run without going through a model call (e.g. the model-call limit's exit
        # message) is not seen by the recorder; append it so the last message is the agent's final answer.
        final_message = final_state["messages"][-1]
        recorded_ids = {event.message.id for event in transcript.events}
        hit_limit = False
        if isinstance(final_message, AIMessage) and final_message.id not in recorded_ids:
            new_messages.append(_event_to_agentdojo(_TranscriptEvent(agent="main", message=final_message)))
            hit_limit = True

        diagnostics = {
            "n_events": len(transcript.events),
            "n_subagent_events": sum(event.agent != "main" for event in transcript.events),
            "final_message_unrecorded": hit_limit,
            "shadowed_builtin_tools": shadowed,
            "agent_setup": transcript.agent_setup,
            "hide_filesystem_tools": self.hide_filesystem_tools,
            "rejected_hidden_tool_calls": transcript.rejected_hidden_tool_calls,
            "oversized_tool_outputs": transcript.oversized_tool_outputs,
        }

        # The benchmark runs this element on its own, not inside an `AgentPipeline`, and drops `extra_args`. So it
        # logs the trace itself (without messages, saved results fail to load and resumed runs redo every task),
        # and saves the diagnostics in the trace file under "deepagents".
        logger = Logger.get()
        logger.log(new_messages)
        if isinstance(logger, TraceLogger):
            logger.set_contextarg("deepagents", diagnostics)

        extra_args = {**extra_args, "deepagents": diagnostics}
        return query, runtime, env, new_messages, extra_args
