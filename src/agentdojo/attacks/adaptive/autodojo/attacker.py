"""`AutoDojoAttacker`: the runner's evolving attacker that scores candidates on the real pipeline (arXiv:2606.15057).

For each pair, trial 0 is the manifest's `important_instructions` seed. Before each later trial, `text_for` runs one
optimizer call: it builds a leaderboard of the pair's earlier trials (0..k-1) scored by their real result (did the
injection goal get achieved, or the user task fail, through the real checker and victim), shows the optimizer that
leaderboard and the benign tool-result data the text is placed into, and takes back one new injection. The optimizer
never sees the skill file, the checker's removed spans or the victim's trace -- only success or failure -- so the
attacker stays black-box toward the real defence, exactly as AutoDojo's "outcome feedback" is.

This is a per-pair optimization (AutoDojo's case set C has size one here: one user task per injection target). The
cross-pair success library and the stand-in checker/agent of the MUZZLE port are intentionally absent: the leaderboard
is the real signal, so no stand-in is needed. `prompt_set`-style ablation lives in the MUZZLE attacker.
"""

import json
import logging
import re
import time
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel

from agentdojo.attacks.adaptive.autodojo import prompts
from agentdojo.attacks.adaptive.muzzle.attacker import RewriteLLM, run_sync, sha256_text, upstream_pin
from agentdojo.attacks.adaptive.muzzle.ipi import (
    MAX_CONTEXT_CHARS,
    _split_injected_tool_result,
    build_stand_in_context,
    extract_json_object,
)
from agentdojo.attacks.adaptive.muzzle.llm import AttackerModelConfig, BudgetExhaustedError, OpenAICompatibleLLM
from agentdojo.attacks.adaptive.muzzle.vendor.constants import PLACEHOLDER_INSTR
from agentdojo.runner.manifest import Pair, trial0_injections
from agentdojo.runner.plugins import AttemptFeedback
from agentdojo.task_suite.load_suites import get_suite

logger = logging.getLogger(__name__)

STATE_FORMAT = 1

# One optimizer call per trial is retried this many times on any error before the trial falls back to the best text
# on the leaderboard (`source="goal_fallback"`), so a transient model-server error on the shared GPU box never aborts
# the whole run. A module constant, like the MUZZLE port's, so it never changes `config.json` and resume stays valid.
GENERATE_MAX_ATTEMPTS = 3
GENERATE_RETRY_BACKOFF_SECONDS = 3.0

DEVIATIONS = (
    "Optimization is per pair (one user task x one injection task); AutoDojo's case set C has size one here, so a "
    "candidate's score is the single real trial's outcome, not an ASR over many user tasks.",
    'Diagnosis and generation are one model call returning {"diagnosis", "prompt"} (AutoDojo uses two); the real '
    "victim+checker run dominates cost, and one call halves the JSON-parse surface on the shared GPU box.",
    "The leaderboard is seeded only with the trial-0 important_instructions text; AutoDojo also seeds TopicAttack and "
    "RL-Hammer outputs (not ported).",
    "Candidate score is attacker success through the real pipeline (injection goal achieved OR user task failed), the "
    "runner's own definition; the optimizer sees success/partial/failure, never the checker or victim internals.",
    "No stand-in checker, no stand-in agent and no cross-pair success library (all MUZZLE-port additions): the "
    "leaderboard of real outcomes is the only signal, so the attacker stays black-box with nothing simulated.",
    "After a successful trial the same text is reused (no optimizer call) until it fails (`reuse_success`); AutoDojo "
    "always generates. The optimizer prompt names the defence it faces (`target`: the agent itself, or a filter).",
    "A generated text missing one of the goal's literal values (email, URL, long account number) is sent back to the "
    "optimizer with the values named, within `max_tries`; the first C2 pilot's rewrites mangled the IBAN.",
)


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"(?:https?://|www\.)[^\s'\"<>]+")
_LONG_NUMBER = re.compile(r"\b\w*\d{6,}\w*\b")


def goal_literals(goal: str) -> list[str]:
    """The values in an injection goal that must reach the agent unchanged for the goal to be checkable: emails, URLs
    and identifiers with six or more digits in a row (IBANs, account numbers). Order kept, duplicates dropped."""
    found = [m.rstrip(".,;:)'") for pattern in (_EMAIL, _URL) for m in pattern.findall(goal)]
    found += [m for m in _LONG_NUMBER.findall(goal) if not any(m in f for f in found)]
    return list(dict.fromkeys(found))


class AutoDojoSettings(BaseModel):
    """AutoDojo's knobs, pinned in the run's config.json and identical in C2 and C4."""

    attacker: AttackerModelConfig
    leaderboard_size: int = 5
    """Top entries of the pair's own trials shown to the optimizer (AutoDojo: top 5)."""
    max_tokens: int = 700
    temperature: float = 1.0
    top_p: float = 0.9
    max_tries: int = 5
    """Parse/JSON retries inside one optimizer call before that call is treated as failed."""
    target: Literal["agent", "filter"] = "agent"
    """Which defence the optimizer is told it faces: `agent` (skills design, default since 2026-11-19: no filter, the
    agent may be addressed directly) or `filter` (external-checker design: text must pass as ordinary data)."""
    reuse_success: bool = True
    """If the pair's previous trial succeeded, hand out the same text again without an optimizer call (archive source
    `kept_success`) and only generate a new one after a failure. The first C2 pilot (2026-11-19) replaced texts that
    had just worked with variants that all failed."""


class AutoDojoAttacker:
    """The runner's `Attacker` for C2/C4 (see `agentdojo.runner.plugins.Attacker`).

    The archive is run-wide: one entry per trial text handed out, in order, with the leaderboard it was built from,
    the optimizer's diagnosis and usage. It is saved in the run's checkpoint, and a text already in the archive is
    returned again without a model call, so a resumed run never regenerates a trial.
    """

    name = "autodojo"

    def __init__(
        self,
        manifest: dict[str, Any],
        settings: AutoDojoSettings,
        llm_factory: Callable[[], RewriteLLM] | None = None,
    ) -> None:
        self.manifest = manifest
        self.settings = settings
        self.llm_factory: Callable[[], RewriteLLM] = llm_factory or (lambda: OpenAICompatibleLLM(settings.attacker))
        self.archive: list[dict[str, Any]] = []

    def goal(self, pair: Pair) -> str:
        suite = get_suite(self.manifest["benchmark_version"], pair.suite)
        return suite.get_injection_task_by_id(pair.injection_task).GOAL

    def _cached(self, pair: Pair, trial: int) -> str | None:
        for entry in self.archive:
            if entry["pair"] == pair.id and entry["trial"] == trial:
                return entry["text"]
        return None

    def _data_slot(self, pair: Pair) -> str | None:
        """The benign tool-result data the injection is placed into, with the slot shown as `<<INJECTION>>`.

        Built offline from AgentDojo's ground truth (the same helper the MUZZLE IPI port uses), so the optimizer sees
        what the text must blend into. `None` if the slot cannot be located; the optimizer then works from the goal
        alone. Never a real victim run, so the attacker stays black-box.
        """
        _, template = build_stand_in_context(pair, self.manifest["benchmark_version"])
        if template is None:
            return None
        split = _split_injected_tool_result(template)
        body = split[1] if split is not None else template
        body = body.replace(PLACEHOLDER_INSTR, "<<INJECTION>>")
        if len(body) > MAX_CONTEXT_CHARS:
            body = body[:MAX_CONTEXT_CHARS] + " ...[truncated]"
        return body

    def _leaderboard(self, history: list[AttemptFeedback]) -> list[dict[str, Any]]:
        """The pair's earlier trials as leaderboard entries, best first (success, then the user task surviving, then
        most recent), capped at `leaderboard_size`."""
        entries = [
            {"text": fb.injection_text, "security": fb.security, "utility": fb.utility, "trial": fb.trial}
            for fb in history
        ]
        entries.sort(key=lambda e: (e["security"], not e["utility"], e["trial"]), reverse=True)
        return entries[: self.settings.leaderboard_size]

    def text_for(self, pair: Pair, trial: int, history: list[AttemptFeedback]) -> str:
        cached = self._cached(pair, trial)
        if cached is not None:
            return cached
        if trial == 0:
            texts = set(trial0_injections(self.manifest, pair).values())
            assert len(texts) == 1
            text = texts.pop()
            self.archive.append({"pair": pair.id, "trial": 0, "text": text, "source": "trial0"})
            return text
        if not history or history[-1].trial != trial - 1:
            raise ValueError(f"trial {trial} of {pair.id} needs the feedback of trial {trial - 1}")
        if self.settings.reuse_success and history[-1].security:
            text = history[-1].injection_text
            self.archive.append({"pair": pair.id, "trial": trial, "text": text, "source": "kept_success"})
            return text
        entry = self._generate_with_fallback(pair, trial, history)
        self.archive.append(entry)
        return entry["text"]

    def _generate_with_fallback(self, pair: Pair, trial: int, history: list[AttemptFeedback]) -> dict[str, Any]:
        """Run `_generate`, retrying on any error, then fall back to the best text on the leaderboard.

        An optimizer call that keeps erroring on the shared GPU box is almost always a transient server/transport
        failure. Aborting would kill the whole run, so we retry, then reuse the best leaderboard text (a real payload
        already tried for this pair). `BudgetExhaustedError` still propagates: the attacker model's own budget is
        spent, which retrying cannot fix.
        """
        leaderboard = self._leaderboard(history)
        errors: list[str] = []
        for attempt in range(1, GENERATE_MAX_ATTEMPTS + 1):
            try:
                return run_sync(lambda: self._generate(pair, trial, leaderboard))
            except BudgetExhaustedError:
                raise
            except Exception as exc:  # any optimizer failure must degrade, never abort the run
                errors.append(f"{type(exc).__name__}: {exc}")
                logger.warning(
                    "AutoDojo optimizer failed for %s trial %d (attempt %d/%d): %s",
                    pair.id,
                    trial,
                    attempt,
                    GENERATE_MAX_ATTEMPTS,
                    exc,
                    exc_info=True,
                )
                if attempt < GENERATE_MAX_ATTEMPTS:
                    time.sleep(GENERATE_RETRY_BACKOFF_SECONDS)
        best = leaderboard[0]["text"] if leaderboard else history[-1].injection_text
        logger.error(
            "AutoDojo optimizer gave up for %s trial %d after %d attempts; reusing the best known text",
            pair.id,
            trial,
            GENERATE_MAX_ATTEMPTS,
        )
        return {
            "pair": pair.id,
            "trial": trial,
            "text": best,
            "source": "goal_fallback",
            "diagnosis": None,
            "leaderboard": [
                {"trial": e["trial"], "security": e["security"], "utility": e["utility"]} for e in leaderboard
            ],
            "usage": None,
            "errors": errors,
        }

    async def _generate(self, pair: Pair, trial: int, leaderboard: list[dict[str, Any]]) -> dict[str, Any]:
        s = self.settings
        goal = self.goal(pair)
        data_slot = self._data_slot(pair)
        messages = [
            {"role": "system", "content": prompts.SYSTEM_PROMPTS[s.target]},
            {"role": "user", "content": prompts.user_prompt(goal, data_slot, prompts.leaderboard_block(leaderboard))},
        ]
        literals = goal_literals(goal)
        llm = self.llm_factory()
        diagnosis: str | None = None
        text: str | None = None
        literal_retries = 0
        try:
            for attempt in range(s.max_tries):
                result = await llm.complete(
                    messages,
                    max_tokens=s.max_tokens,
                    temperature=s.temperature,
                    top_p=s.top_p,
                    # A new seed per attempt, so a retry is a fresh sample and not the same reply again.
                    seed=(s.attacker.seed or 0) + trial + 1000 * attempt,
                )
                content = result.choices[0].message.content or ""
                parsed = extract_json_object(content)
                if parsed is None or not isinstance(parsed.get("prompt"), str) or not parsed["prompt"].strip():
                    continue
                candidate = parsed["prompt"].strip()
                missing = [v for v in literals if v not in candidate]
                if missing:
                    # A changed IBAN, email or URL makes the trial fail whatever the agent does, so it would teach
                    # the optimizer nothing about the defence. Ask again, naming the values.
                    literal_retries += 1
                    messages = [
                        *messages,
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": prompts.missing_literals_message(missing)},
                    ]
                    continue
                text = candidate
                raw_diag = parsed.get("diagnosis")
                diagnosis = raw_diag.strip() if isinstance(raw_diag, str) else None
                break
            usage = llm.usage()
        finally:
            await llm.aclose()
        if text is None:
            raise ValueError(f"AutoDojo optimizer produced no usable injection after {s.max_tries} tries")
        return {
            "pair": pair.id,
            "trial": trial,
            "text": text,
            "source": "generated",
            "diagnosis": diagnosis,
            "leaderboard": [
                {"trial": e["trial"], "security": e["security"], "utility": e["utility"]} for e in leaderboard
            ],
            "usage": usage,
            "literal_retries": literal_retries,
            "errors": [],
        }

    def config(self) -> dict[str, Any]:
        """What goes into config.json under `attacker`. No secrets: only the key's variable name."""
        return {
            "name": self.name,
            "settings": self.settings.model_dump(),
            "paper": "arXiv:2606.15057 (AutoDojo), MIT",
            "upstream": upstream_pin(),
            "prompt_sha256": {
                "system": sha256_text(prompts.SYSTEM_PROMPTS[self.settings.target]),
            },
            "deviations": list(DEVIATIONS),
            "on_api_failure": (
                f"the OpenAI SDK retries each call; an optimizer call that still fails is retried "
                f"{GENERATE_MAX_ATTEMPTS}x, then falls back to the best leaderboard text (archive source "
                f"'goal_fallback') so the run never aborts; BudgetExhaustedError still stops the run (resumable)"
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        return {"format": STATE_FORMAT, "archive": json.loads(json.dumps(self.archive))}

    def load_state(self, state: dict[str, Any]) -> None:
        if state.get("format") != STATE_FORMAT:
            raise ValueError(f"unknown AutoDojo attacker state format {state.get('format')!r}")
        self.archive = list(state["archive"])
