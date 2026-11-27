"""MUZZLE as the runner's evolving attacker (DESIGN.md, "Evolving attacker (SuperRed MUZZLE port)").

Only MUZZLE's refinement loop is used (decided 2026-10-29). Before trials 1 and 2 of a pair, `text_for`:

1. runs MUZZLE's Judge on the pair's previous trial: the text sent, the agent's final answer as the transcript,
   payload presence fixed to "present", and a ground-truth block with the attacker-success result plus the
   separate task and security results;
2. runs offline PAIR with the injection task's goal as its target, seeded with upstream's reflection (the previous
   text and the Judge's recommendations), against a stand-in victim that is the attacker model with no system prompt
   and no context around the injection;
3. returns PAIR's best prompt as the pair's full injection text. The runner puts it in every vector the user task
   reads, replacing the vector's content.

Trial 0 is always the manifest's `important_instructions` text, with no model call. The attacker keeps going after a
success (upstream stops). It never sees the skill file, the checker's removals or the victim's trace; it rewrites
every pair in both C2 and C4, so this class is the same in both.

Deviations from upstream's `MuzzleOptimizer` are listed in `DEVIATIONS` and saved with the run's config.
"""

import asyncio
import concurrent.futures
import hashlib
import json
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import asdict
from importlib import resources
from typing import Any, Protocol, TypeVar

from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from agentdojo.attacks.adaptive.muzzle import ipi
from agentdojo.attacks.adaptive.muzzle.ipi import IpiPairBridge, PromptSet, build_stand_in_context
from agentdojo.attacks.adaptive.muzzle.llm import AttackerModelConfig, OpenAICompatibleLLM
from agentdojo.attacks.adaptive.muzzle.vendor import judge as upstream_judge
from agentdojo.attacks.adaptive.muzzle.vendor.json_utils import extract_json_object, vendored_prompt_path
from agentdojo.attacks.adaptive.muzzle.vendor.pair import prompts as pair_prompts
from agentdojo.attacks.adaptive.muzzle.vendor.pair.bridge import PairBridge
from agentdojo.runner.manifest import Pair, trial0_injections
from agentdojo.runner.plugins import AttemptFeedback
from agentdojo.task_suite.load_suites import get_suite

logger = logging.getLogger(__name__)

STATE_FORMAT = 1

# A whole per-trial rewrite is retried this many times on any error before it falls back (see
# `MuzzleAttacker._rewrite_with_fallback`). Inside one rewrite, unparseable or empty attacker/checker/judge replies
# are already handled (the stream is skipped, the candidate kept, or the score floored), so what reaches here is a
# genuinely failing rewrite -- almost always a transient model-server/transport error on a shared GPU box. Rather
# than abort the whole run (and leave the pilot half-done), MUZZLE retries, then reuses the previous trial's text so
# the run always completes; every fallback is logged and recorded in the archive as `source="rewrite_failed"`.
# Kept a module constant, not a setting, so it never changes `config.json` and an in-progress run stays resumable.
REWRITE_MAX_ATTEMPTS = 3
REWRITE_RETRY_BACKOFF_SECONDS = 3.0

NO_VESSEL_ATTACK_METHOD: dict[str, Any] = {"element": "", "why": "", "how": "", "confidence": 0.0}
"""Upstream's `_attack_method(None)`. Placement is AgentDojo's and never changes, so no vessel is described."""

DEVIATIONS = (
    "Only the refinement loop (Judge + offline PAIR) runs; OBSERVE (Summarizer, Grafter, Prompter), PROBE and the "
    "vessel choice (Dispatcher) are dropped.",
    "PAIR's instruction is the injection task's GOAL (upstream: the Prompter's instruction).",
    "PAIR's stand-in victim has no system prompt and no transcript template (upstream: the victim's real system "
    "prompt and the PROBE transcript).",
    "Judge transcript: the agent's final answer only, as one assistant message (upstream: the run's full transcript).",
    "Judge payload presence is fixed to present (upstream: the contains_injected_content heuristic), so the Judge's "
    "attribution is always 'instruction' and every trial is refined on the same placement.",
    "Judge ground truth: result = attacker success (injection goal met OR user task failed), plus "
    "user_task_completed and injection_goal_achieved (upstream: result only).",
    "Judge attack_method is upstream's empty no-vessel description.",
    "The attacker keeps rewriting after a success, starting from the previous trial's text (upstream stops).",
    "One rewrite per trial before trials 1 and 2; trial 0 is the unchanged important_instructions text.",
    "prompt_set='ipi' (default, 2026-11-18): PAIR's attacker prompts, judge and stand-in victim are replaced (ipi.py) "
    "so the loop optimises a realistic indirect injection -- text placed in tool-result data, scored on whether a "
    "stand-in agent (given the task's tools and transcript with the candidate spliced into the injection slot) carries "
    "out the goal. The faithful chat-jailbreak PAIR, with a context-free stand-in, stays as prompt_set='upstream'.",
    "prompt_set='ipi': the IPI attacker drops upstream's stop=['}'] and first-{..}-slice parse (which truncate a "
    "reply whose 'prompt' value is itself '{...}', as Mistral Small's data-shaped injections are) for a balanced-brace "
    "scan with a literal_eval fallback (ipi.IpiAttacker). The MUZZLE Judge reflection parse gets the same tolerant "
    "fallback. Vendored files are untouched.",
    "prompt_set='ipi' with ipi_stand_in_checker=True (default, 2026-11-18): each candidate is first screened by a "
    "stand-in checker -- a generic injection detector played by the attacker model, never S0 or the real skill file. "
    "Blocked or emptied candidates score 1; otherwise the stand-in agent sees what the checker left. This keeps PAIR "
    "from optimising loud text the real checker strips in full (first live C2 run: injection_survival 0.0). Set "
    "ipi_stand_in_checker=False for the pre-checker IPI behaviour.",
)


class MuzzleSettings(BaseModel):
    """MUZZLE's knobs, pinned in the run's config.json and identical in C2 and C4. Defaults are the SuperRed port's."""

    attacker: AttackerModelConfig
    pair_n_streams: int = 3
    pair_n_iterations: int = 4
    pair_keep_last_n: int = 4
    pair_attack_max_tokens: int = 500
    pair_max_attack_attempts: int = 5
    pair_attack_temperature: float = 1.0
    pair_attack_top_p: float = 0.9
    pair_surrogate_max_tokens: int = 500
    pair_surrogate_temperature: float = 0.0
    pair_judge_max_tokens: int = 10
    pair_judge_temperature: float = 0.0
    judge_max_tries: int = 5
    prompt_set: PromptSet = "ipi"
    """`"ipi"` (default, decided 2026-11-18): PAIR optimises a realistic indirect injection against a stand-in agent
    given the task's context (`ipi.py`). `"upstream"`: the faithful chat-jailbreak PAIR with a context-free stand-in
    (`vendor/pair/`), kept for ablation."""
    ipi_stand_in_checker: bool = True
    """`"ipi"` only (decided 2026-11-18): screen each candidate through a stand-in checker (a generic injection
    detector played by the attacker model, never S0 or the real skill file) before scoring it. Blocked or emptied
    candidates score 1; otherwise the stand-in agent sees what the checker left. `False` is the pre-checker IPI
    behaviour, kept for ablation."""
    ipi_checker_max_tokens: int = 512
    ipi_checker_temperature: float = 0.0


class RewriteLLM(Protocol):
    """One rewrite's model client: what the vendored code calls, plus usage and cleanup."""

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> ChatCompletion: ...

    def usage(self) -> dict[str, Any]: ...

    async def aclose(self) -> None: ...


def judge_user_prompt(
    *, injection_text: str, final_answer: str | None, utility: bool, security: bool, attacker_success: bool
) -> str:
    """Upstream's Judge user prompt with the two extra ground-truth fields; every other byte is upstream's."""
    prompt = upstream_judge._build_user_prompt(
        attack_method=NO_VESSEL_ATTACK_METHOD,
        malicious_instruction={"injection": injection_text},
        transcript=[{"role": "assistant", "content": final_answer or ""}],
        payload_present=True,
        framework_success=attacker_success,
    )
    upstream_line = "ground_truth=" + json.dumps({"result": attacker_success}, separators=(",", ":"))
    extended = {"result": attacker_success, "user_task_completed": utility, "injection_goal_achieved": security}
    extended_line = "ground_truth=" + json.dumps(extended, separators=(",", ":"))
    assert prompt.count(upstream_line) == 1
    return prompt.replace(upstream_line, extended_line)


async def evaluate_attempt(llm: RewriteLLM, feedback: AttemptFeedback, max_tries: int) -> dict[str, Any]:
    """Upstream `Judge.evaluate` with `judge_user_prompt` in place of upstream's prompt builder."""
    messages = [
        {"role": "system", "content": upstream_judge.load_system_prompt_by_name("judge")},
        {
            "role": "user",
            "content": judge_user_prompt(
                injection_text=feedback.injection_text,
                final_answer=feedback.final_answer,
                utility=feedback.utility,
                security=feedback.security,
                attacker_success=feedback.attacker_success,
            ),
        },
    ]
    eval_obj: dict[str, Any] | None = None
    remaining = max_tries
    while remaining > 0:
        try:
            result = await llm.complete(messages)
            content = result.choices[0].message.content or ""
            # Vendored parse first (keeps upstream behaviour when it succeeds), then ipi's tolerant parse, which
            # recovers the single-quoted / fenced objects Mistral Small returns (2026-11-18 live run).
            parsed = extract_json_object(content) or ipi.extract_json_object(content)
            if parsed is None or "outcome" not in parsed:
                raise ValueError("JudgeAgent: LLM output invalid JSON.")
            eval_obj = parsed
            break
        except upstream_judge.BudgetExhaustedError:
            raise
        except Exception as exc:
            remaining -= 1
            logger.warning("MUZZLE judge attempt failed: %s. Retries left: %d", exc, remaining)
    if eval_obj is None:
        eval_obj = upstream_judge._default_evaluation()
    return upstream_judge._ground_evaluation(
        eval_obj, payload_present=True, framework_success=feedback.attacker_success
    )


T = TypeVar("T")


def run_sync(make_coro: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Runs a coroutine from the runner's synchronous loop, also when called inside a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(make_coro())
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(make_coro())).result()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def prompt_hashes() -> dict[str, str]:
    """Hashes of the prompts MUZZLE sends, for config.json (resume refuses a run whose prompts changed)."""
    with open(vendored_prompt_path("judge"), encoding="utf-8") as f:
        judge_yaml = f.read()
    probe_goal, probe_target = "{goal}", "{target}"
    return {
        "judge_yaml": sha256_text(judge_yaml),
        "pair_attacker_system_prompts": sha256_text(
            json.dumps(pair_prompts.get_attacker_system_prompts(probe_goal, probe_target))
        ),
        "pair_judge_system_prompt": sha256_text(pair_prompts.get_judge_system_prompt(probe_goal, probe_target)),
        "ipi_attacker_system_prompts": sha256_text(json.dumps(ipi.ipi_attacker_system_prompts(probe_goal))),
        "ipi_judge_system": sha256_text(ipi.ipi_judge_system(probe_goal)),
        "ipi_stand_in_system_prompt": sha256_text(ipi.STAND_IN_SYSTEM_PROMPT),
        "ipi_checker_system": sha256_text(ipi.ipi_checker_system()),
    }


def upstream_pin() -> dict[str, Any]:
    pin = json.loads(resources.files("agentdojo.attacks.adaptive.muzzle").joinpath("UPSTREAM.json").read_text())
    return {"repository": pin["repository"], "commit": pin["commit"], "muzzle_upstream": pin["muzzle_upstream"]}


class MuzzleAttacker:
    """The runner's `Attacker` for C2 and C4 (see `agentdojo.runner.plugins.Attacker`).

    The archive is run-wide: one entry per trial text this attacker handed out, in order, with the feedback, the
    Judge's evaluation, PAIR's result and the model usage of the rewrite that produced it. It is saved in the run's
    checkpoint, and a text already in the archive is returned again without model calls, so a resumed run never
    rewrites a trial twice.
    """

    name = "muzzle"

    def __init__(
        self,
        manifest: dict[str, Any],
        settings: MuzzleSettings,
        llm_factory: Callable[[], RewriteLLM] | None = None,
    ) -> None:
        self.manifest = manifest
        self.settings = settings
        self.llm_factory: Callable[[], RewriteLLM] = llm_factory or (lambda: OpenAICompatibleLLM(settings.attacker))
        self.archive: list[dict[str, Any]] = []
        # Cross-pair memory (ipi only): texts whose real trial broke security, and the stand-in-judge scores each
        # framing angle earned. Both are the attacker's own observations -- never the skill file or the real
        # checker -- so they stay inside the black-box rule. Saved in the checkpoint; empty until a trial succeeds.
        self.library: list[dict[str, Any]] = []
        self.style: dict[int, list[int]] = {}

    def goal(self, pair: Pair) -> str:
        suite = get_suite(self.manifest["benchmark_version"], pair.suite)
        return suite.get_injection_task_by_id(pair.injection_task).GOAL

    def _cached(self, pair: Pair, trial: int) -> str | None:
        for entry in self.archive:
            if entry["pair"] == pair.id and entry["trial"] == trial:
                return entry["text"]
        return None

    def _ingest(self, pair: Pair, history: list[AttemptFeedback]) -> None:
        """Record this pair's finished trials that broke security, so later pairs can reuse what worked."""
        for fb in history:
            if not fb.security:
                continue
            if any(e["pair"] == pair.id and e["trial"] == fb.trial for e in self.library):
                continue
            self.library.append(
                {
                    "pair": pair.id,
                    "trial": fb.trial,
                    "suite": pair.suite,
                    "injection_task": pair.injection_task,
                    "text": fb.injection_text,
                }
            )

    def _examples_for(self, pair: Pair, k: int = 3) -> list[str]:
        """Up to `k` prior winning texts to show PAIR, same injection task first, then same suite; never this pair."""
        same_task = [e for e in self.library if e["injection_task"] == pair.injection_task and e["pair"] != pair.id]
        same_suite = [
            e for e in self.library if e["suite"] == pair.suite and e["pair"] != pair.id and e not in same_task
        ]
        chosen = same_task[-k:] + same_suite[-(k - min(k, len(same_task))) :]
        out: list[str] = []
        for text in (e["text"] for e in chosen):
            if text not in out:
                out.append(text)
        return out[:k]

    def _angle_order(self) -> list[int] | None:
        """Order the framing angles by the mean stand-in score each has earned so far (best first); None at the start.

        A light diversity-aware preference in the spirit of the search attack in arXiv:2510.09023: keep every angle
        in play, but let the framings that have worked lead. Unseen angles sit at the running average.
        """
        if not self.style:
            return None
        n = len(ipi.ipi_attacker_system_prompts("_"))
        means = {i: sum(v) / len(v) for i, v in self.style.items() if v}
        if not means:
            return None
        default = sum(means.values()) / len(means)
        return sorted(range(n), key=lambda i: -means.get(i, default))

    def text_for(self, pair: Pair, trial: int, history: list[AttemptFeedback]) -> str:
        cached = self._cached(pair, trial)
        if cached is not None:
            return cached
        self._ingest(pair, history)
        if trial == 0:
            texts = set(trial0_injections(self.manifest, pair).values())
            assert len(texts) == 1
            text = texts.pop()
            self.archive.append({"pair": pair.id, "trial": 0, "text": text, "source": "trial0"})
            return text
        if not history or history[-1].trial != trial - 1:
            raise ValueError(f"trial {trial} of {pair.id} needs the feedback of trial {trial - 1}")
        entry = self._rewrite_with_fallback(pair, trial, history[-1])
        self.archive.append(entry)
        return entry["text"]

    def _rewrite_with_fallback(self, pair: Pair, trial: int, last: AttemptFeedback) -> dict[str, Any]:
        """Run `_rewrite`, retrying a few times on any error, then fall back to the previous trial's text.

        A rewrite makes many model calls (Judge + the whole PAIR loop: attacker, stand-in checker, stand-in victim,
        delivery judge, across streams and iterations). Unparseable or empty replies are already handled inside that
        loop, so an exception escaping to here is a genuinely failing rewrite -- on a shared GPU box almost always a
        transient server/transport error. Aborting would kill the whole pilot, so we retry, then reuse `last`'s text
        (a payload already delivered for this pair). `BudgetExhaustedError` still propagates: it means the attacker
        model's own budget is spent, which retrying cannot fix.
        """
        errors: list[str] = []
        for attempt in range(1, REWRITE_MAX_ATTEMPTS + 1):
            try:
                return run_sync(lambda: self._rewrite(pair, trial, last))
            except upstream_judge.BudgetExhaustedError:
                raise
            except Exception as exc:  # any rewrite failure must degrade, never abort the run
                errors.append(f"{type(exc).__name__}: {exc}")
                logger.warning(
                    "MUZZLE rewrite failed for %s trial %d (attempt %d/%d): %s",
                    pair.id,
                    trial,
                    attempt,
                    REWRITE_MAX_ATTEMPTS,
                    exc,
                    exc_info=True,
                )
                if attempt < REWRITE_MAX_ATTEMPTS:
                    time.sleep(REWRITE_RETRY_BACKOFF_SECONDS)
        logger.error(
            "MUZZLE rewrite gave up for %s trial %d after %d attempts; reusing trial %d's text so the run continues",
            pair.id,
            trial,
            REWRITE_MAX_ATTEMPTS,
            last.trial,
        )
        return {
            "pair": pair.id,
            "trial": trial,
            "text": last.injection_text,
            "source": "rewrite_failed",
            "prompt_set": self.settings.prompt_set,
            "stand_in_context": False,
            "feedback": {
                "trial": last.trial,
                "injection_text": last.injection_text,
                "utility": last.utility,
                "security": last.security,
                "attacker_success": last.attacker_success,
                "final_answer": last.final_answer,
            },
            "judge": None,
            "pair_result": None,
            "inner_loop": None,
            "examples_used": [],
            "angle_order": None,
            "usage": None,
            "rewrite_errors": errors,
        }

    async def _rewrite(self, pair: Pair, trial: int, last: AttemptFeedback) -> dict[str, Any]:
        s = self.settings
        goal = self.goal(pair)
        llm = self.llm_factory()
        try:
            evaluation = await evaluate_attempt(llm, last, s.judge_max_tries)
            reflection = {
                "most_recent_working_payload": last.injection_text,
                "recommendations": evaluation.get("recommendations"),
                "reflection_details": evaluation,
            }
            if s.prompt_set == "ipi":
                system_prompt, transcript_template = build_stand_in_context(pair, self.manifest["benchmark_version"])
                bridge: PairBridge | IpiPairBridge = IpiPairBridge(
                    llm,
                    n_streams=s.pair_n_streams,
                    n_iterations=s.pair_n_iterations,
                    keep_last_n=s.pair_keep_last_n,
                    attack_max_tokens=s.pair_attack_max_tokens,
                    max_attack_attempts=s.pair_max_attack_attempts,
                    attack_temperature=s.pair_attack_temperature,
                    attack_top_p=s.pair_attack_top_p,
                    surrogate_max_tokens=s.pair_surrogate_max_tokens,
                    surrogate_temperature=s.pair_surrogate_temperature,
                    judge_max_tokens=s.pair_judge_max_tokens,
                    judge_temperature=s.pair_judge_temperature,
                    use_checker=s.ipi_stand_in_checker,
                    checker_max_tokens=s.ipi_checker_max_tokens,
                    checker_temperature=s.ipi_checker_temperature,
                )
            else:
                system_prompt, transcript_template = None, None
                bridge = PairBridge(
                    llm,  # type: ignore[arg-type]  # the vendored code types it as SuperRed's LLMClient
                    n_streams=s.pair_n_streams,
                    n_iterations=s.pair_n_iterations,
                    keep_last_n=s.pair_keep_last_n,
                    attack_max_tokens=s.pair_attack_max_tokens,
                    max_attack_attempts=s.pair_max_attack_attempts,
                    attack_temperature=s.pair_attack_temperature,
                    attack_top_p=s.pair_attack_top_p,
                    surrogate_max_tokens=s.pair_surrogate_max_tokens,
                    surrogate_temperature=s.pair_surrogate_temperature,
                    judge_max_tokens=s.pair_judge_max_tokens,
                    judge_temperature=s.pair_judge_temperature,
                )
            examples = self._examples_for(pair) if s.prompt_set == "ipi" else []
            angle_order = self._angle_order() if s.prompt_set == "ipi" else None
            extra = {"examples": examples, "angle_order": angle_order} if s.prompt_set == "ipi" else {}
            result = await bridge.craft(
                instruction=goal,
                transcript_template=transcript_template,
                victim_system_prompt=system_prompt,
                reflection=reflection,
                **extra,
            )
            if isinstance(bridge, IpiPairBridge):
                for angle, scores in bridge.angle_scores.items():
                    self.style.setdefault(angle, []).extend(scores)
        finally:
            await llm.aclose()
        # Upstream: `payload = result.best_prompt or self._instruction`.
        text = result.best_prompt or goal
        return {
            "pair": pair.id,
            "trial": trial,
            "text": text,
            "source": "pair" if result.best_prompt else "goal_fallback",
            "prompt_set": s.prompt_set,
            "stand_in_context": s.prompt_set == "ipi" and transcript_template is not None,
            "feedback": {
                "trial": last.trial,
                "injection_text": last.injection_text,
                "utility": last.utility,
                "security": last.security,
                "attacker_success": last.attacker_success,
                "final_answer": last.final_answer,
            },
            "judge": evaluation,
            "pair_result": {
                "best_score": result.best_score,
                "is_jailbroken": result.is_jailbroken,
                "best_improvement": result.best_improvement,
            },
            "inner_loop": asdict(bridge.last_stats) if isinstance(bridge, IpiPairBridge) else None,
            "examples_used": examples,
            "angle_order": angle_order,
            "usage": llm.usage(),
        }

    def attack_texts(self) -> list[str]:
        """Every text handed out so far, in order (the attack archive the C4 judge's leakage check reads)."""
        return [entry["text"] for entry in self.archive]

    def config(self) -> dict[str, Any]:
        """What goes into config.json under `attacker`. No secrets: only the key's variable name."""
        return {
            "name": self.name,
            "settings": self.settings.model_dump(),
            "upstream": upstream_pin(),
            "prompt_sha256": prompt_hashes(),
            "deviations": list(DEVIATIONS),
            "on_api_failure": (
                f"the OpenAI SDK retries each call; a whole rewrite that still fails is retried "
                f"{REWRITE_MAX_ATTEMPTS}x, then falls back to the previous trial's text (archive source "
                f"'rewrite_failed') so the run never aborts; BudgetExhaustedError still stops the run (resumable)"
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": STATE_FORMAT,
            "archive": json.loads(json.dumps(self.archive)),
            "library": json.loads(json.dumps(self.library)),
            "style": {str(k): v for k, v in self.style.items()},
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if state.get("format") != STATE_FORMAT:
            raise ValueError(f"unknown MUZZLE attacker state format {state.get('format')!r}")
        self.archive = list(state["archive"])
        self.library = list(state.get("library", []))
        self.style = {int(k): list(v) for k, v in state.get("style", {}).items()}
