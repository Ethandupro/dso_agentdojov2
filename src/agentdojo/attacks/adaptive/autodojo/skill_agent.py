"""`SkillAgentAttacker`: the evolving attacker run as a tool-using agent with its own skill files (item 7, D2).

This is the "DeepAgents attacker" the user asked for (decision 2026-11-20), implemented as a self-contained
tool-calling agent loop on the attacker's own OpenAI-compatible endpoint rather than wrapping LangChain's
`create_deep_agent`. The behaviour is the one item 7 asks for: the optimizer runs as an agent that reads and writes
its own strategy skill files (`load_skill` / `write_skill`, mirroring the victim's `load_skill`) across the run's
pairs, then submits one injection per trial with `submit_injection`. Keeping it on the attacker's plain vLLM endpoint
(the same `OpenAICompatibleLLM` the single-call optimizer uses) means no victim-side LangChain stack on the attacker
box, and the whole loop is testable with a scripted model and no server. See `config()` for the recorded deviation.

Everything else is inherited from `AutoDojoAttacker`: the per-pair leaderboard of real outcomes, the cross-pair
playbook memory, the goal-literal guard, the retry-then-fall-back path, and the archive/checkpoint. Only the single
optimizer call is replaced by the agent loop, and the skill library is added to the checkpoint. The attacker stays
black-box: it sees only success/failure (plus the S0-redacted victim trace, as the base attacker already does), never
the skill file or the victim's raw reasoning.
"""

import json
import logging
from typing import Any

from agentdojo.attacks.adaptive.autodojo import prompts
from agentdojo.attacks.adaptive.autodojo.attacker import (
    GENERATE_MAX_ATTEMPTS,
    AutoDojoAttacker,
    AutoDojoSettings,
    _valid_strategies,
    goal_literals,
)
from agentdojo.attacks.adaptive.autodojo.skillstore import (
    DEFAULT_MAX_BODY_CHARS,
    DEFAULT_MAX_SKILLS,
    SkillStore,
    SkillStoreError,
)
from agentdojo.attacks.adaptive.autodojo.strategies import strategies_for
from agentdojo.attacks.adaptive.muzzle.attacker import sha256_text, upstream_pin
from agentdojo.runner.manifest import Pair

logger = logging.getLogger(__name__)

SKILL_AGENT_STATE_FORMAT = 3
"""Adds the attacker's own skill library to format 2 (archive + playbook). Formats 1/2 load with an empty library."""


class SkillAgentSettings(AutoDojoSettings):
    """`AutoDojoSettings` plus the agent loop's knobs. Pinned in config.json and identical in C2 and C4."""

    max_agent_steps: int = 8
    """Model calls allowed in one trial's agent loop before it must have submitted (else the trial falls back to the
    best leaderboard text, like a failed optimizer call)."""
    skill_body_chars: int = DEFAULT_MAX_BODY_CHARS
    """Cap on each skill note's body, so the agent cannot blow up its own context or the checkpoint."""
    max_skills: int = DEFAULT_MAX_SKILLS
    """Cap on how many notes the library holds; past it the agent must overwrite an existing note."""


class SkillAgentAttacker(AutoDojoAttacker):
    """The runner `Attacker` for C2/C4 with `--attacker skill-agent`. See the module docstring."""

    name = "skill-agent"
    settings: SkillAgentSettings

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self.settings, SkillAgentSettings):
            raise TypeError("SkillAgentAttacker needs SkillAgentSettings")
        self.store = SkillStore(max_body_chars=self.settings.skill_body_chars, max_skills=self.settings.max_skills)
        """The attacker's own strategy notes, shared across the run's pairs and saved in the checkpoint."""

    def _run_tool(
        self,
        name: str,
        args: dict[str, Any],
        literals: list[str],
        written: list[str],
        loaded: list[str],
    ) -> tuple[str, dict[str, Any] | None]:
        """Run one tool call. Returns (text shown back to the agent, submission dict or None). A submission is
        returned only when the injection carries every goal literal; otherwise the agent is told what is missing and
        asked to submit again."""
        s = self.settings
        if name == prompts.LIST_TOOL_NAME:
            return self.store.listing(), None
        if name == prompts.LOAD_TOOL_NAME:
            body = self.store.load(str(args.get("name", "")))
            if body is None:
                return (
                    f"No note named {args.get('name')!r}. Your notes: {', '.join(self.store.names()) or '(none)'}.",
                    None,
                )
            loaded.append(str(args.get("name", "")).strip())
            return body, None
        if name == prompts.WRITE_TOOL_NAME:
            try:
                skill = self.store.write(
                    str(args.get("name", "")), str(args.get("description", "")), str(args.get("body", ""))
                )
            except SkillStoreError as exc:
                return f"Could not save the note: {exc}", None
            written.append(skill.name)
            return f"Saved note {skill.name!r} (version {skill.version}).", None
        if name == prompts.SUBMIT_TOOL_NAME:
            candidate = str(args.get("prompt", "")).strip()
            if not candidate:
                return "submit_injection needs a non-empty 'prompt'.", None
            missing = [v for v in literals if v not in candidate]
            if missing:
                return prompts.missing_literals_message(missing), None
            raw_diag = args.get("diagnosis")
            diagnosis = raw_diag.strip() if isinstance(raw_diag, str) and raw_diag.strip() else None
            strategy_ranking = _valid_strategies(args.get("strategy_ranking"), s.max_strategies)
            strategy = strategy_ranking[0] if strategy_ranking else None
            submission = {
                "text": candidate,
                "diagnosis": diagnosis,
                "strategy": strategy,
                "strategy_ranking": strategy_ranking,
            }
            return "Submitted. You are done for this item.", submission
        return f"Unknown tool {name!r}.", None

    @staticmethod
    def _function_calls(msg: Any) -> list[tuple[str, str, str]]:
        """The (id, name, arguments) of each function tool call on a reply, skipping any non-function tool calls."""
        out: list[tuple[str, str, str]] = []
        for tc in msg.tool_calls or []:
            if getattr(tc, "type", None) == "function":
                out.append((tc.id, tc.function.name, tc.function.arguments))
        return out

    @staticmethod
    def _assistant_tool_message(msg: Any, calls: list[tuple[str, str, str]]) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
                for call_id, name, arguments in calls
            ],
        }

    @staticmethod
    def _parse_args(raw: str | None) -> dict[str, Any]:
        try:
            parsed = json.loads(raw or "{}")
        except (ValueError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    async def _generate(self, pair: Pair, trial: int, leaderboard: list[dict[str, Any]]) -> dict[str, Any]:
        s = self.settings
        goal = self.goal(pair)
        data_slot = self._data_slot(pair)
        winners, lessons = self.memory_for(pair)
        memory = prompts.memory_block(winners, lessons)
        bank = strategies_for(s.target) if s.strategy_bank else ()
        strategies_block = prompts.strategy_bank_block(bank, s.max_strategies)
        strategy_names = [st.name for st in bank]
        tools = prompts.skill_agent_tools(strategy_names)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompts.SKILL_AGENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": prompts.skill_agent_task(
                    goal,
                    data_slot,
                    prompts.leaderboard_block(leaderboard),
                    self.store.listing(),
                    memory,
                    strategies_block,
                ),
            },
        ]
        literals = goal_literals(goal)
        written: list[str] = []
        loaded: list[str] = []
        submission: dict[str, Any] | None = None
        nudged = False
        steps = 0
        llm = self.llm_factory()
        try:
            for step in range(s.max_agent_steps):
                steps += 1
                result = await llm.complete(
                    messages,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=s.max_tokens,
                    temperature=s.temperature,
                    top_p=s.top_p,
                    seed=(s.attacker.seed or 0) + trial + 1000 * step,
                )
                msg = result.choices[0].message
                calls = self._function_calls(msg)
                if not calls:
                    if submission is not None or nudged:
                        break
                    messages.append({"role": "assistant", "content": msg.content or ""})
                    messages.append(
                        {"role": "user", "content": f"Call {prompts.SUBMIT_TOOL_NAME} now with the final injection."}
                    )
                    nudged = True
                    continue
                messages.append(self._assistant_tool_message(msg, calls))
                for call_id, name, arguments in calls:
                    out, sub = self._run_tool(name, self._parse_args(arguments), literals, written, loaded)
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": out})
                    if sub is not None:
                        submission = sub
                if submission is not None:
                    break
            usage = llm.usage()
        finally:
            await llm.aclose()
        if submission is None:
            raise ValueError(f"skill-agent produced no valid injection for {pair.id} trial {trial} in {steps} steps")
        return {
            "pair": pair.id,
            "trial": trial,
            "text": submission["text"],
            "source": "generated",
            "diagnosis": submission["diagnosis"],
            "strategy": submission["strategy"],
            "strategy_ranking": submission["strategy_ranking"],
            "skills_written": written,
            "skills_loaded": loaded,
            "agent_steps": steps,
            "leaderboard": [
                {"trial": e["trial"], "security": e["security"], "utility": e["utility"]} for e in leaderboard
            ],
            "usage": usage,
            "memory_shown": {"winners": len(winners), "lessons": len(lessons)},
            "errors": [],
        }

    def config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "settings": self.settings.model_dump(),
            "paper": "arXiv:2606.15057 (AutoDojo), MIT; item 7 skill-file agent",
            "upstream": upstream_pin(),
            "harness": (
                "a self-contained tool-calling agent loop on the attacker's OpenAI-compatible endpoint "
                "(load_skill/write_skill/list_skills/submit_injection), not LangChain create_deep_agent; same "
                "read/write-skill behaviour, testable with a scripted model, no victim-side LangChain stack on the "
                "attacker box"
            ),
            "prompt_sha256": {
                "system": sha256_text(prompts.SKILL_AGENT_SYSTEM_PROMPT),
                "memory_header": sha256_text(prompts.MEMORY_HEADER),
            },
            "skill_tools": [
                prompts.LIST_TOOL_NAME,
                prompts.LOAD_TOOL_NAME,
                prompts.WRITE_TOOL_NAME,
                prompts.SUBMIT_TOOL_NAME,
            ],
            "deviations": [
                "The optimizer is an agent that reads and writes its own strategy skill files across the run's pairs "
                "(item 7, D2); vanilla AutoDojo has one optimizer call per trial and no attacker memory of its own.",
                "The skill library starts empty in every run, is identical in C2 and C4, is saved in the checkpoint, "
                "and never carries across runs; each note's body and the library size are capped.",
                "Implemented as a tool-calling loop on the attacker's own endpoint, not LangChain create_deep_agent "
                "(same behaviour; keeps the attacker black-box and the loop testable without a server).",
            ],
            "on_api_failure": (
                f"an agent loop that errors is retried {GENERATE_MAX_ATTEMPTS}x, then falls back to the best "
                f"leaderboard text (archive source 'goal_fallback'), so the run never aborts; BudgetExhaustedError "
                f"still stops the run (resumable)"
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        base = super().state_dict()
        base["format"] = SKILL_AGENT_STATE_FORMAT
        base["skills"] = self.store.state()
        return base

    def load_state(self, state: dict[str, Any]) -> None:
        fmt = state.get("format")
        if fmt not in (1, 2, SKILL_AGENT_STATE_FORMAT):
            raise ValueError(f"unknown skill-agent attacker state format {fmt!r}")
        self.archive = list(state["archive"])
        self.playbook = list(state.get("playbook", []))
        self.store.load_from_state(state.get("skills") or {})
