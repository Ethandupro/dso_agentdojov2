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
import re
from typing import Any

from agentdojo.attacks.adaptive.autodojo import prompts
from agentdojo.attacks.adaptive.autodojo.attacker import (
    GENERATE_MAX_ATTEMPTS,
    AutoDojoAttacker,
    AutoDojoSettings,
    _clip,
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
from agentdojo.attacks.adaptive.muzzle.attacker import run_sync, sha256_text, upstream_pin
from agentdojo.attacks.adaptive.muzzle.ipi import extract_json_object
from agentdojo.attacks.adaptive.muzzle.llm import BudgetExhaustedError
from agentdojo.runner.manifest import Pair
from agentdojo.runner.plugins import AttemptFeedback

logger = logging.getLogger(__name__)

SKILL_AGENT_STATE_FORMAT = 3
"""Adds the attacker's own skill library to format 2 (archive + playbook). Formats 1/2 load with an empty library.
Distillation state (`trajectories`, `distillations`, `distilled_pairs`) is optional within format 3."""

COPIED_WORDS = 8
"""A distilled note is rejected if it repeats this many consecutive words of one of the attempts it was distilled
from: a pattern must be general, not a stored payload."""
_WORD = re.compile(r"[a-z0-9]+")


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = _WORD.findall(text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


class SkillAgentSettings(AutoDojoSettings):
    """`AutoDojoSettings` plus the agent loop's knobs. Pinned in config.json and identical in C2 and C4."""

    max_agent_steps: int = 8
    """Model calls allowed in one trial's agent loop before it must have submitted (else the trial falls back to the
    best leaderboard text, like a failed optimizer call)."""
    skill_body_chars: int = DEFAULT_MAX_BODY_CHARS
    """Cap on each skill note's body, so the agent cannot blow up its own context or the checkpoint."""
    max_skills: int = DEFAULT_MAX_SKILLS
    """Cap on how many notes the library holds; past it the agent must overwrite an existing note."""
    distill_every: int = 0
    """Distillation (`run_condition --distill-every K`, user request 2026-11-23): every K finished pairs, one extra
    attacker call reads those pairs' attempts, wins and failures, and writes general pattern notes into the library.
    0 = off (the agent only writes notes itself during a trial, as in PR #49)."""
    distill_max_attempts: int = 40
    """Most recent attempts shown to one distillation call."""
    distill_max_notes: int = 3
    """Most notes one distillation call may write or update."""
    distill_text_chars: int = 500
    """Each attempt's text and the agent's response are clipped to this many characters in the distillation prompt."""
    note_exploration: float = 0.5
    """With distillation on, `list_skills` ranks notes by a UCB score (mean 0-4 score / 4 + this weight * sqrt(ln(N+1)
    / uses)) and shows each note's record, so the attacker leans on notes that worked but still tries new ones."""
    prune_after_uses: int = 4
    """With distillation on, a note loaded this many times that never won and averages at most `prune_max_score` is
    dropped from the library (a cheap version of JailbreakSkill's keep-only-if-it-helps gate). 0 = never prune."""
    prune_max_score: float = 1.0


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
        self.trajectories: list[dict[str, Any]] = []
        """Every finished evolving trial (wins and failures), kept only when distillation is on."""
        self.distillations: list[dict[str, Any]] = []
        """One entry per distillation call: which pairs it read, what it wrote and rejected, usage or error."""
        self.distilled_pairs = 0
        """How many finished pairs (in run order) earlier distillation calls have covered."""
        self.pruned: list[dict[str, Any]] = []
        """Notes dropped for never helping, with their record and the trial that triggered it."""

    def _listing(self) -> str:
        """The note menu: ranked with each note's record when distillation is on, plain otherwise (as in PR #49)."""
        if self.settings.distill_every > 0:
            return self.store.ranked_listing(self.settings.note_exploration)
        return self.store.listing()

    # --- distillation ---------------------------------------------------------------------------------------------

    def observe(self, pair: Pair, feedback: AttemptFeedback, skill_version: str | None) -> None:
        super().observe(pair, feedback, skill_version)
        if self.settings.distill_every <= 0:
            return
        if any(t["pair"] == pair.id and t["trial"] == feedback.trial for t in self.trajectories):
            return
        # Credit the notes the agent loaded before writing this trial's text, then drop notes that never help.
        entry = next((e for e in self.archive if e["pair"] == pair.id and e["trial"] == feedback.trial), {})
        self.store.record_use(entry.get("skills_loaded") or [], feedback.security, feedback.progress)
        if self.settings.prune_after_uses > 0:
            for dropped in self.store.prune(self.settings.prune_after_uses, self.settings.prune_max_score):
                self.pruned.append({**dropped, "pair": pair.id, "trial": feedback.trial})
        n = self.settings.distill_text_chars
        self.trajectories.append(
            {
                "pair": pair.id,
                "suite": pair.suite,
                "trial": feedback.trial,
                "text": feedback.injection_text,
                "security": feedback.security,
                "utility": feedback.utility,
                "progress": feedback.progress,
                "response": _clip(feedback.victim_trace or feedback.final_answer, n),
                "literals": goal_literals(self.goal(pair)),
            }
        )

    def _finished_pairs(self, current: str) -> list[str]:
        """Pairs with recorded trials, in run order, other than the one about to start (pairs run one after another)."""
        return [p for p in dict.fromkeys(t["pair"] for t in self.trajectories) if p != current]

    def text_for(self, pair: Pair, trial: int, history: list[AttemptFeedback]) -> str:
        if trial == 0 and self.settings.distill_every > 0 and self._cached(pair, trial) is None:
            finished = self._finished_pairs(pair.id)
            if len(finished) - self.distilled_pairs >= self.settings.distill_every:
                self._distill(finished[self.distilled_pairs :])
                self.distilled_pairs = len(finished)
        return super().text_for(pair, trial, history)

    def _attempts_block(self, attempts: list[dict[str, Any]]) -> str:
        n = self.settings.distill_text_chars
        lines = []
        for i, t in enumerate(attempts, 1):
            outcome = "SUCCEEDED" if t["security"] else "failed"
            score = f", progress {t['progress']}/4" if t.get("progress") is not None else ""
            lines.append(
                f"[{i}] item {t['suite']} #{t['pair'].split('/', 1)[-1]}, trial {t['trial']}: {outcome}{score}"
                f"\nTEXT: {_clip(t['text'], n)}\nAGENT: {t['response'] or '(nothing recorded)'}"
            )
        return "\n\n".join(lines)

    def _notes_block(self) -> str:
        if not self.store.skills:
            return "(none yet)"
        return "\n\n".join(
            f"- {s.name}: {s.description}\n{s.body}" for s in sorted(self.store.skills.values(), key=lambda s: s.name)
        )

    @staticmethod
    def _too_specific(note: str, attempts: list[dict[str, Any]]) -> str | None:
        """Why a distilled note is not a general pattern (None if it is): it carries a goal value of one of the
        attempts (email, URL, IBAN-like number), or repeats `COPIED_WORDS` consecutive words of an attempt's text."""
        lowered = note.lower()
        grams = _ngrams(note, COPIED_WORDS)
        for t in attempts:
            for value in t["literals"]:
                if value.lower() in lowered:
                    return f"contains the item value {value!r}"
            if grams & _ngrams(t["text"], COPIED_WORDS):
                return f"copies {COPIED_WORDS}+ consecutive words of an attempt"
        return None

    def _distill(self, pairs: list[str]) -> None:
        """One distillation call over the attempts of `pairs`. Never stops the run: an error is recorded and skipped
        (`BudgetExhaustedError` still propagates)."""
        s = self.settings
        members = set(pairs)
        attempts = [t for t in self.trajectories if t["pair"] in members][-s.distill_max_attempts :]
        event: dict[str, Any] = {"pairs": list(pairs), "attempts": len(attempts), "written": [], "rejected": []}
        try:
            reply, usage = run_sync(lambda: self._distill_call(attempts))
            event["usage"] = usage
            parsed = extract_json_object(reply) or {}
            raw_notes = parsed.get("skills")
            notes: list[Any] = raw_notes if isinstance(raw_notes, list) else []
            for note in notes[: s.distill_max_notes]:
                if not isinstance(note, dict):
                    continue
                name, desc, body = (str(note.get(k, "")) for k in ("name", "description", "body"))
                reason = self._too_specific(f"{name}\n{desc}\n{body}", attempts)
                if reason is not None:
                    event["rejected"].append({"name": name, "reason": reason})
                    continue
                try:
                    event["written"].append(self.store.write(name, desc, body).name)
                except SkillStoreError as exc:
                    event["rejected"].append({"name": name, "reason": str(exc)})
        except BudgetExhaustedError:
            raise
        except Exception as exc:  # a failed distillation must not abort the run
            logger.warning("attacker distillation over %d pairs failed: %s", len(pairs), exc)
            event["error"] = f"{type(exc).__name__}: {exc}"
        self.distillations.append(event)

    async def _distill_call(self, attempts: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        s = self.settings
        llm = self.llm_factory()
        try:
            result = await llm.complete(
                [
                    {"role": "system", "content": prompts.DISTILL_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": prompts.distill_task(
                            self._notes_block(), self._attempts_block(attempts), s.distill_max_notes
                        ),
                    },
                ],
                max_tokens=max(s.max_tokens, 1200),
                temperature=s.temperature,
                top_p=s.top_p,
                seed=(s.attacker.seed or 0) + 7919 * (len(self.distillations) + 1),
            )
            usage = llm.usage()
        finally:
            await llm.aclose()
        return result.choices[0].message.content or "", usage

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
            return self._listing(), None
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
                    self._listing(),
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
                **(
                    {"distill_system": sha256_text(prompts.DISTILL_SYSTEM_PROMPT)}
                    if self.settings.distill_every > 0
                    else {}
                ),
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
        base["trajectories"] = json.loads(json.dumps(self.trajectories))
        base["distillations"] = json.loads(json.dumps(self.distillations))
        base["distilled_pairs"] = self.distilled_pairs
        base["pruned"] = json.loads(json.dumps(self.pruned))
        return base

    def load_state(self, state: dict[str, Any]) -> None:
        fmt = state.get("format")
        if fmt not in (1, 2, SKILL_AGENT_STATE_FORMAT):
            raise ValueError(f"unknown skill-agent attacker state format {fmt!r}")
        self.archive = list(state["archive"])
        self.playbook = list(state.get("playbook", []))
        self.store.load_from_state(state.get("skills") or {})
        self.trajectories = list(state.get("trajectories", []))
        self.distillations = list(state.get("distillations", []))
        self.distilled_pairs = int(state.get("distilled_pairs", 0))
        self.pruned = list(state.get("pruned", []))
