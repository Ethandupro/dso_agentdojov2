"""Runs one condition at one seed: the unit loop, per-trial traces, skill loads, checkpoints and resume.

Requires the `deepagents` extra. The victim (and, for the `--external-checker` ablation, the checker model) are passed
in as functions, so tests can script them; `agentdojo.runner.run_condition` builds the real ones.

Defence (redesign of 2026-11-19, DESIGN.md "Victim skills"): by default (`defense="skills"`) the victim has a skill
library, the fixed `agent_skills` plus, when the condition has S0, the security skill whose body is the current skill
file. The agent sees only names and descriptions and loads skills with `load_skill`; each trial records what it
loaded and whether it loaded S0 (`skills.s0_loaded`). With `defense="checker"` the design before 2026-11-19 runs
instead: no skills, and the skill file goes to an external checker that removes injected spans from tool outputs.

Files under the run directory (DESIGN.md, "Conditions and result segmentation"):

- `config.json`: written before any model call; a run resumes only under an identical config.
- `trials.jsonl`: one record per finished unit (trial or benign task), in unit order.
- `checker_calls.jsonl`: one line per checker call, with the unit's keys (stays empty without the external checker).
- `replications.jsonl`: `--replicate-wins N` re-runs each successful injection N more times, unchanged; one line per
  re-run, with the won trial's keys. Never trials: ASR, ASR@n and pair ASR ignore them; the report adds a robust ASR.
  In C3/C4 with the batched judge trigger, each train win is also re-run `judge_confirm_runs` times (reusing
  `--replicate-wins` re-runs where there are any) so the judge only sees wins that repeat; those lines have
  `purpose: "judge_confirmation"` and their tally is the record's `judge_confirmation`, never `replication`.
- `state.json`: the checkpoint (index of the next unit, attacker and skill-file state), replaced atomically.
- `trial_<t>/<split>/...` and `benign*/...`: AgentDojo trace files, one per unit.

After each unit the runner appends its record and checker lines, then replaces `state.json`. On resume, lines of
units at or after the checkpoint (a crash between the two writes) are dropped and that unit runs again.
With `stop_on_success` (`run_condition --stop-on-success`), an evolving pair stops at its first success: its later
trials are skipped, write no record, and are listed in `state.json` under `skipped`. A pair whose trial 0 (the
original AgentDojo text) already succeeds is therefore never evolved ("baseline already works"). A unit that
raises (for example a model server that stays unreachable after the client's retries) stops the run with the
checkpoint intact; nothing is scored for it.
"""

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from agentdojo.agent_pipeline.agent_skills import (
    LOAD_SKILL_TOOL_NAME,
    AgentSkill,
    AgentSkillsMiddleware,
    SkillLibrary,
    SkillsPromptStyle,
)
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.deepagents_pipeline import ToolMiddlewareFactory
from agentdojo.checker.middleware import CheckerFn, make_checker_factory
from agentdojo.experiment_models import S0_SKILL_DESCRIPTION, S0_SKILL_NAME
from agentdojo.logging import NullLogger, TraceLogger
from agentdojo.runner.conditions import Condition, Unit
from agentdojo.runner.config import RunConfig, config_diff, sha256_text
from agentdojo.runner.plugins import Attacker, AttemptFeedback, ObservingAttacker, SkillFileSource, StaticAttacker
from agentdojo.runner.report import evolving_metrics, replication_metrics
from agentdojo.runner.score import goal_tool_names, progress_score
from agentdojo.runner.trace import judge_trace_digest, victim_own_text, victim_trace_digest
from agentdojo.skillfile.judge import TrialRecord
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.types import get_text_content_as_str

VictimFactory = Callable[[Unit, ToolMiddlewareFactory | None], BasePipelineElement]
"""Builds the victim for one unit, with the middleware factory: the skill library, or the external checker (None
without either). The real victim ignores the unit; scripted test victims use it."""

Defense = Literal["skills", "checker"]

MAX_ACTION_CHARS = 300
"""Each recorded agent action (`function(args)`) is cut to this many characters."""


def agent_actions(messages: list[Any]) -> list[str]:
    """The tool calls the agent made, in order, as `function(arg=value, ...)` strings (for the record and the judge)."""
    actions: list[str] = []
    for message in messages:
        if message["role"] != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            args = ", ".join(f"{k}={json.dumps(v, default=str)}" for k, v in dict(call.args).items())
            text = f"{call.function}({args})"
            actions.append(text if len(text) <= MAX_ACTION_CHARS else text[:MAX_ACTION_CHARS] + " [...]")
    return actions


STATE_FORMAT = 1

DEFAULT_REPLICATE_WINS = 4
"""`run_condition --replicate-wins` default (user request 2026-11-25): each evolving win is re-run 4 more times."""


class ConfigMismatchError(Exception):
    pass


def run_dir_for(
    results_root: Path,
    condition: Condition,
    seed: int,
    limit: int | None,
    suites: tuple[str, ...] | None = None,
    external_checker: bool = False,
    skills_prompt_style: str = "nudge",
    subset: str | None = None,
    single_injection_point: bool = False,
    no_attacker_memory: bool = False,
    no_graded_score: bool = False,
    no_strategy_bank: bool = False,
    no_victim_trace: bool = False,
    smoke: bool = False,
    skill_agent: bool = False,
    stop_on_success: bool = False,
    victim_temperature: float | None = None,
    attacker_temperature: float | None = None,
    distill_every: int = 0,
    distill_plain: bool = False,
    distill_trajectory: bool = False,
    free_skills: bool = False,
    no_analyst: bool = False,
    replicate_wins: int | None = None,
    judge_trigger: str | None = None,
) -> Path:
    name = condition.run_id
    if external_checker:
        name = f"{name}-external-checker"
    elif skills_prompt_style != "nudge":
        name = f"{name}-{skills_prompt_style}-skills"
    if suites:
        name = f"{name}-{'+'.join(sorted(suites))}"
    if no_attacker_memory:
        name = f"{name}-no-memory"
    if no_graded_score:
        name = f"{name}-binary-score"
    if skill_agent:
        name = f"{name}-skill-agent"
    if no_strategy_bank:
        name = f"{name}-no-strategies"
    if no_victim_trace:
        name = f"{name}-no-victim-trace"
    if distill_every:
        name = f"{name}-distill{distill_every}"
        if distill_plain:
            name = f"{name}-plain"
        elif distill_trajectory:
            name = f"{name}-traj"
    if free_skills:
        name = f"{name}-free-skills"
    if no_analyst:
        name = f"{name}-no-analyst"
    if replicate_wins is not None and replicate_wins != DEFAULT_REPLICATE_WINS:
        name = f"{name}-rep{replicate_wins}"
    if judge_trigger:
        name = f"{name}-{judge_trigger}"
    if stop_on_success:
        name = f"{name}-first-success"
    if victim_temperature is not None:
        name = f"{name}-vtemp{victim_temperature:g}"
    if attacker_temperature is not None:
        name = f"{name}-atemp{attacker_temperature:g}"
    if subset:
        name = f"{name}-subset-{subset}"
    if smoke:
        name = f"{name}-smoke"
    elif single_injection_point:
        name = f"{name}-single-point"
    if limit is not None:
        name = f"{name}-limit{limit}"
    return results_root / name / f"seed_{seed}"


def trace_dir(run_dir: Path, unit: Unit) -> Path:
    if unit.pair is None:
        return run_dir / unit.phase
    return run_dir / f"trial_{unit.trial}" / unit.pair.split


def replica_dir(run_dir: Path, unit: Unit, k: int) -> Path:
    """Trace files of the k-th replication of a won trial (kept apart from the trial's own trace)."""
    assert unit.pair is not None
    return run_dir / "replications" / f"trial_{unit.trial}" / unit.pair.split / f"replica_{k}"


ANALYST_TRACE_CHARS = 12000
ANALYST_STEP_CHARS = 1200
ANALYST_THOUGHT_CHARS = 3000
"""The analyst's (`--analyst`) longer digest of the victim's run: the leaderboard's is 4000 chars in all."""


def summarize_checker(lines: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-unit checker figures, from that unit's `checker_calls.jsonl` lines."""
    seen = [i for line in lines for i in line.get("injections", []) if i.get("present_before")]
    return {
        "calls": len(lines),
        "parse_failures": sum(bool(line["parse_failure"]) for line in lines),
        "removed_spans": [
            {"span": s["span"], "reason": s["reason"], "tool_name": line["tool_name"]}
            for line in lines
            for s in line["removed"]
        ],
        "unmatched_spans": sum(len(line["unmatched"]) for line in lines),
        "too_short_spans": sum(len(line["too_short"]) for line in lines),
        "chars_removed": sum(line["chars_removed"] for line in lines),
        "outputs_with_removal": sum(line["n_markers"] > 0 for line in lines),
        "injection_seen_by_checker": bool(seen),
        "injection_survival": max((i["surviving_fraction"] for i in seen), default=None),
        "injection_survived_whole": any(i["survived_whole"] for i in seen) if seen else None,
        "input_tokens": sum(line.get("input_tokens") or 0 for line in lines),
        "output_tokens": sum(line.get("output_tokens") or 0 for line in lines),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, path)


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class Runner:
    def __init__(
        self,
        config: RunConfig,
        condition: Condition,
        run_dir: Path,
        units: list[Unit],
        attacker: Attacker,
        static_attacker: StaticAttacker,
        skill_file: SkillFileSource,
        victim_factory: VictimFactory,
        checker_fn: CheckerFn | None = None,
        min_span_chars: int = 0,
        echo: Callable[[str], None] = print,
        defense: Defense = "skills",
        agent_skills: list[AgentSkill] | None = None,
        skills_prompt_style: SkillsPromptStyle = "nudge",
        security_skill_name: str = S0_SKILL_NAME,
        security_skill_description: str = S0_SKILL_DESCRIPTION,
        graded_score: bool = True,
        show_victim_trace: bool = True,
        stop_on_success: bool = False,
        replicate_wins: int = 0,
        judge_confirm_runs: int = 0,
        judge_evidence: bool = False,
    ) -> None:
        if defense == "checker" and condition.s0 and checker_fn is None:
            raise ValueError(f"{condition.name} with the external checker needs a checker model")
        if defense == "skills" and checker_fn is not None:
            raise ValueError("a checker model is only used with defense='checker'")
        self.defense = defense
        self.agent_skills = list(agent_skills or [])
        self.skills_prompt_style: SkillsPromptStyle = skills_prompt_style
        self.security_skill_name = security_skill_name
        self.security_skill_description = security_skill_description
        self.graded_score = graded_score
        self.show_victim_trace = show_victim_trace
        self.stop_on_success = stop_on_success
        self.replicate_wins = replicate_wins
        """Extra runs of each successful injection, unchanged (`--replicate-wins`); never counted as trials."""
        self.judge_confirm_runs = judge_confirm_runs
        self.judge_evidence = judge_evidence
        """C3/C4 rich judge evidence (PR 2): train trials carry the victim's own text (to find quoted rules; dropped
        before the record is written) and, for wins, a run digest for the judge (`judge_trace`)."""
        """C3/C4 batched trigger: unchanged re-runs of each train win (injection goal OR user task failed), so the
        judge sees only wins that repeat. Never counted as trials."""
        self._forced_replicas: list[dict[str, Any]] = []
        self._unit_index = 0
        """Security library: forced-load re-runs made during this unit's library step."""
        if hasattr(skill_file, "set_forced_load_fn"):
            skill_file.set_forced_load_fn(self.forced_load_check)  # type: ignore[attr-defined]
        self.skipped: list[int] = []
        """Unit indices skipped by `stop_on_success` (no record, no model call)."""
        self.config = config
        self.condition = condition
        self.run_dir = run_dir
        self.units = units
        self.attacker = attacker
        self.static_attacker = static_attacker
        self.skill_file = skill_file
        self.victim_factory = victim_factory
        self.checker_fn = checker_fn
        self.min_span_chars = min_span_chars
        self.echo = echo
        self.records: list[dict[str, Any]] = []
        self.units_sha256 = sha256_text("\n".join(u.id for u in units))

    @property
    def config_path(self) -> Path:
        return self.run_dir / "config.json"

    @property
    def state_path(self) -> Path:
        return self.run_dir / "state.json"

    @property
    def trials_path(self) -> Path:
        return self.run_dir / "trials.jsonl"

    @property
    def checker_path(self) -> Path:
        return self.run_dir / "checker_calls.jsonl"

    @property
    def replications_path(self) -> Path:
        return self.run_dir / "replications.jsonl"

    # --- start and resume -------------------------------------------------------------------------------------

    def _start(self) -> int:
        """Writes or checks `config.json`, restores the checkpoint, and returns the index of the next unit."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        new = json.loads(self.config.model_dump_json())
        if self.config_path.exists():
            old = json.loads(self.config_path.read_text(encoding="utf-8"))
            diffs = config_diff(old, new)
            if diffs:
                raise ConfigMismatchError(
                    f"{self.run_dir} holds a run with a different config; it resumes only an identical one. "
                    "Differences:\n  " + "\n  ".join(diffs)
                )
        else:
            _write_json_atomic(self.config_path, new)

        cursor = 0
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state["units_sha256"] != self.units_sha256:
                raise ConfigMismatchError("the unit list changed since this run started")
            cursor = state["cursor"]
            self.skipped = [i for i in state.get("skipped", []) if i < cursor]
            self.attacker.load_state(state["attacker"])
            self.skill_file.load_state(state["skill_file"])
        # Drop lines of units at or after the checkpoint: written just before a crash, before the state was saved.
        records = [r for r in _read_jsonl(self.trials_path) if r["unit"] < cursor]
        if len(records) + len(self.skipped) != cursor:
            raise ConfigMismatchError(
                f"trials.jsonl has {len(records)} records and {len(self.skipped)} skipped units before the checkpoint "
                f"at {cursor}"
            )
        _write_jsonl(self.trials_path, records)
        _write_jsonl(self.checker_path, [c for c in _read_jsonl(self.checker_path) if c["unit"] < cursor])
        _write_jsonl(self.replications_path, [r for r in _read_jsonl(self.replications_path) if r["unit"] < cursor])
        self.records = records
        return cursor

    def _commit(
        self,
        unit: Unit,
        record: dict[str, Any],
        checker_lines: list[dict[str, Any]],
        replicas: list[dict[str, Any]] | None = None,
    ) -> None:
        _append_jsonl(self.trials_path, [record])
        _append_jsonl(self.checker_path, checker_lines)
        _append_jsonl(self.replications_path, replicas or [])
        self.records.append(record)
        self._save_state(unit)

    def _skip(self, unit: Unit) -> None:
        self.skipped.append(unit.index)
        self._save_state(unit)

    def _save_state(self, unit: Unit) -> None:
        _write_json_atomic(
            self.state_path,
            {
                "format": STATE_FORMAT,
                "cursor": unit.index + 1,
                "n_units": len(self.units),
                "units_sha256": self.units_sha256,
                "complete": unit.index + 1 == len(self.units),
                "skipped": list(self.skipped),
                "attacker": self.attacker.state_dict(),
                "skill_file": self.skill_file.state_dict(),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )

    # --- one unit ---------------------------------------------------------------------------------------------

    def history(self, unit: Unit) -> list[AttemptFeedback]:
        """The pair's earlier trials in this run, as the attacker may see them."""
        assert unit.pair is not None
        return [self._feedback(r) for r in self.records if r["pair"] == unit.pair.id and r["phase"] == unit.phase]

    @staticmethod
    def _feedback(record: dict[str, Any]) -> AttemptFeedback:
        return AttemptFeedback(
            trial=record["trial"],
            injection_text=record["injection_text"],
            utility=record["utility"],
            security=record["security"],
            final_answer=record["final_answer"],
            progress=record.get("progress"),
            victim_trace=record.get("victim_trace"),
            replication=record.get("replication"),
        )

    def attacker_for(self, unit: Unit) -> Attacker:
        # C4: the evolving attacker never sees test-split injection tasks; test pairs keep the trial-0 text.
        if unit.phase == "test":
            return self.static_attacker
        return self.attacker

    def already_won(self, unit: Unit) -> bool:
        """`stop_on_success`: an evolving pair that already succeeded on an earlier trial (trial 0 included) is done.
        Static pairs (C0/C1, and C4's test pairs) always run every trial."""
        if not self.stop_on_success or unit.pair is None or self.attacker_for(unit) is self.static_attacker:
            return False
        return any(r["pair"] == unit.pair.id and r["phase"] == unit.phase and r["security"] for r in self.records)

    def run_unit(self, unit: Unit) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        start = time.time()
        suite = get_suite(self.config.benchmark_version, unit.suite)
        user_task = suite.user_tasks[unit.user_task]
        if unit.phase in ("test", "benign_after"):
            self.skill_file.freeze()

        injection_task = None
        injection_text: str | None = None
        injections: dict[str, str] = {}
        attacker_name: str | None = None
        if unit.pair is not None:
            injection_task = suite.injection_tasks[unit.pair.injection_task]
            attacker = self.attacker_for(unit)
            attacker_name = attacker.name
            assert unit.trial is not None
            injection_text = attacker.text_for(unit.pair, unit.trial, self.history(unit))
            injections = {vector: injection_text for vector in unit.pair.vectors}

        skill_version = self.skill_file.version_name
        keys: dict[str, Any] = {
            "condition": self.condition.name,
            "run_id": self.config.run_id,
            "seed": self.config.seed,
            "unit": unit.index,
            "unit_id": unit.id,
            "phase": unit.phase,
            "suite": unit.suite,
            "user_task": unit.user_task,
            "injection_task": None if unit.pair is None else unit.pair.injection_task,
            "pair": None if unit.pair is None else unit.pair.id,
            "split": unit.split,
            "trial": unit.trial,
            "skill_version": skill_version,
        }
        library_versions = getattr(self.skill_file, "library_versions", None)
        if library_versions is not None:
            keys["library_versions"] = library_versions
        self._forced_replicas = []
        self._unit_index = unit.index

        checker_lines: list[dict[str, Any]] = []
        markdown = self.skill_file.markdown
        attacked = unit.pair is not None
        run = self._victim_run(
            unit,
            user_task,
            injection_task,
            injections,
            injection_text,
            attacker_name,
            keys,
            markdown,
            trace_dir(self.run_dir, unit),
            checker_lines.append,
        )
        utility, security, messages = run["utility"], run["security"], run["messages"]
        deepagents, library = run["deepagents"], run["library"]
        final_answer = run["final_answer"]
        actions = agent_actions(list(messages))
        reasoning = run["reasoning"]
        victim_trace = (
            victim_trace_digest(messages, markdown, reasoning=reasoning)
            if attacked and self.show_victim_trace
            else None
        )
        progress = (
            progress_score(
                injection_task.GOAL,
                actions,
                final_answer,
                security,
                goal_tool_names(suite, injection_task.ID),
            )
            if attacked and self.graded_score and injection_task is not None
            else None
        )

        # Replication (user request 2026-11-25): a successful evolving-attacker injection is re-run unchanged (static
        # pairs resend one text every trial, so their trials already are replications), against the same pair and
        # the same skill-file version, `replicate_wins` more times. These runs never count as trials: they go to
        # replications.jsonl, not trials.jsonl, so trial counts, ASR, ASR@n and the pair ASR are unchanged; the record
        # only carries their tally (`replication`) for the robust-ASR figures and the attacker's leaderboard.
        replicas: list[dict[str, Any]] = []
        evolving = unit.pair is not None and self.attacker_for(unit) is not self.static_attacker
        if evolving and security and self.replicate_wins > 0 and injection_task is not None:
            for k in range(1, self.replicate_wins + 1):
                replicas.append(
                    self._replica(
                        unit,
                        suite,
                        user_task,
                        injection_task,
                        injections,
                        injection_text,
                        attacker_name,
                        keys,
                        markdown,
                        k,
                        "replicate_wins",
                    )
                )
        replication = (
            {"runs": len(replicas), "successes": sum(bool(r["security"]) for r in replicas)} if replicas else None
        )

        # Batched judge trigger (C3/C4 train, design note 2026-11-26): a win counts for the judge only if it repeats.
        # `--replicate-wins` re-runs of the same text are reused; the rest are run here, marked as confirmation runs.
        confirmation: dict[str, int] | None = None
        attacker_success = bool(security or not utility)
        if (
            self.judge_confirm_runs > 0
            and self.condition.evolving_skill
            and unit.phase == "train"
            and attacked
            and attacker_success
            and injection_task is not None
        ):
            reused = replicas[: self.judge_confirm_runs]
            extra = [
                self._replica(
                    unit,
                    suite,
                    user_task,
                    injection_task,
                    injections,
                    injection_text,
                    attacker_name,
                    keys,
                    markdown,
                    k,
                    "judge_confirmation",
                )
                for k in range(len(replicas) + 1, len(replicas) + 1 + self.judge_confirm_runs - len(reused))
            ]
            runs = reused + extra
            confirmation = {
                "runs": len(runs),
                "attacker_successes": sum(bool(r["security"] or not r["utility"]) for r in runs),
            }
            replicas = replicas + extra
        record: dict[str, Any] = {
            **keys,
            "attack": attacker_name,
            "injection_text": injection_text,
            "utility": utility,
            "security": security if attacked else None,
            "attacker_success": (security or not utility) if attacked else None,
            "progress": progress,
            "replication": replication,
            "judge_confirmation": confirmation,
            **self._judge_evidence(unit, attacked, attacker_success, list(messages), reasoning),
            "final_answer": final_answer,
            "victim_trace": victim_trace,
            "skills": (
                library.summary(self.security_skill_name if markdown is not None else None)
                if library is not None
                else None
            ),
            "agent_actions": actions,
            "checker": summarize_checker(checker_lines) if self.defense == "checker" and markdown is not None else None,
            "victim_usage": deepagents.get("usage"),
            "victim_served_models": deepagents.get("served_models"),
            "victim_error": deepagents.get("error"),
            "trace": str(
                trace_dir(self.run_dir, unit)
                .relative_to(self.run_dir)
                .joinpath(
                    (run["pipeline_name"] or "").replace("/", "_"),
                    unit.suite,
                    unit.user_task,
                    attacker_name or "none",
                    f"{unit.pair.injection_task if unit.pair else 'none'}.json",
                )
            ),
            "duration_s": round(time.time() - start, 3),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if unit.pair is not None:
            attacker = self.attacker_for(unit)
            if attacker is not self.static_attacker and isinstance(attacker, ObservingAttacker):
                # Cross-pair memory: what this trial's text achieved against the skill-file version it faced. The
                # analyst (if on) reads a longer trace than the leaderboard's, built from the same messages.
                feedback = self._feedback(record)
                if attacked and self.show_victim_trace:
                    feedback.analyst_trace = victim_trace_digest(
                        messages,
                        markdown,
                        reasoning=reasoning,
                        max_chars=ANALYST_TRACE_CHARS,
                        step_chars=ANALYST_STEP_CHARS,
                        thought_chars=ANALYST_THOUGHT_CHARS,
                    )
                attacker.observe(unit.pair, feedback, skill_version)
        record["skill_step"] = self.skill_file.after_trial(record)
        record["skill_version_after"] = self.skill_file.version_name
        if library_versions is not None:
            record["library_versions_after"] = getattr(self.skill_file, "library_versions", None)
        # Forced-load checks the library step ran (on earlier train wins) go to replications.jsonl with this unit.
        replicas.extend(self._forced_replicas)
        self._forced_replicas = []
        return record, checker_lines, replicas

    def forced_load_check(self, trial: TrialRecord, skill_name: str) -> dict[str, int]:
        """Security library: re-run a won train trial unchanged (same text, current library) except that
        `skill_name` is already loaded. Never a trial: the line goes to replications.jsonl with
        `purpose: "forced_load_check"`; its `unit` is the won trial's and `during_unit` the unit whose batch ran the
        check. Returns `{"runs": 1, "attacker_successes": 0 or 1}`."""
        unit = next(
            u
            for u in self.units
            if u.phase == "train" and u.pair is not None and u.pair.id == trial.pair and u.trial == trial.trial
        )
        assert unit.pair is not None
        suite = get_suite(self.config.benchmark_version, unit.suite)
        user_task = suite.user_tasks[unit.user_task]
        injection_task = suite.injection_tasks[unit.pair.injection_task]
        injections = {vector: trial.injection_text for vector in unit.pair.vectors}
        keys: dict[str, Any] = {
            "condition": self.condition.name,
            "run_id": self.config.run_id,
            "seed": self.config.seed,
            "unit": unit.index,
            "unit_id": unit.id,
            "phase": unit.phase,
            "suite": unit.suite,
            "user_task": unit.user_task,
            "injection_task": unit.pair.injection_task,
            "pair": unit.pair.id,
            "split": unit.split,
            "trial": unit.trial,
            "skill_version": self.skill_file.version_name,
            "library_versions": getattr(self.skill_file, "library_versions", None),
        }
        n = len(self._forced_replicas) + 1
        start = time.time()
        logdir = self.run_dir / "replications" / f"trial_{unit.trial}" / unit.pair.split / f"forced_{skill_name}_{n}"
        run = self._victim_run(
            unit,
            user_task,
            injection_task,
            injections,
            trial.injection_text,
            "forced_load",
            keys,
            self.skill_file.markdown,
            logdir,
            None,
            preload=(skill_name,),
        )
        won = bool(run["security"] or not run["utility"])
        self._forced_replicas.append(
            {
                **keys,
                "replica": n,
                "purpose": "forced_load_check",
                "forced_skill": skill_name,
                "during_unit": self._unit_index,
                "utility": run["utility"],
                "security": run["security"],
                "skills": run["library"].summary(self.security_skill_name) if run["library"] is not None else None,
                "agent_actions": agent_actions(list(run["messages"])),
                "final_answer": run["final_answer"],
                "victim_usage": run["deepagents"].get("usage"),
                "victim_error": run["deepagents"].get("error"),
                "duration_s": round(time.time() - start, 3),
            }
        )
        return {"runs": 1, "attacker_successes": int(won)}

    def _judge_evidence(
        self, unit: Unit, attacked: bool, won: bool, messages: list[Any], reasoning: dict[int, str]
    ) -> dict[str, Any]:
        """C3/C4 train trials with rich judge evidence: `victim_text` (popped by the evolving skill file before the
        record is written) and, for a win, `judge_trace`, the run digest the judge reads."""
        if not (self.judge_evidence and self.condition.evolving_skill and unit.phase == "train" and attacked):
            return {}
        hidden = frozenset({LOAD_SKILL_TOOL_NAME})
        return {
            "victim_text": victim_own_text(messages, reasoning),
            "judge_trace": judge_trace_digest(messages, reasoning, hidden) if won else None,
        }

    def _replica(
        self,
        unit: Unit,
        suite: Any,
        user_task: Any,
        injection_task: Any,
        injections: dict[str, str],
        injection_text: str | None,
        attacker_name: str | None,
        keys: dict[str, Any],
        markdown: str | None,
        k: int,
        purpose: Literal["replicate_wins", "judge_confirmation"],
    ) -> dict[str, Any]:
        """One unchanged re-run of a won trial (same text, same skill-file version), as a replications.jsonl line."""
        rep_start = time.time()
        rep = self._victim_run(
            unit,
            user_task,
            injection_task,
            injections,
            injection_text,
            attacker_name,
            keys,
            markdown,
            replica_dir(self.run_dir, unit, k),
            None,
        )
        rep_actions = agent_actions(list(rep["messages"]))
        return {
            **keys,
            "replica": k,
            "purpose": purpose,
            "utility": rep["utility"],
            "security": rep["security"],
            "progress": progress_score(
                injection_task.GOAL,
                rep_actions,
                rep["final_answer"],
                rep["security"],
                goal_tool_names(suite, injection_task.ID),
            )
            if self.graded_score
            else None,
            "skills": (
                rep["library"].summary(self.security_skill_name if markdown is not None else None)
                if rep["library"] is not None
                else None
            ),
            "agent_actions": rep_actions,
            "final_answer": rep["final_answer"],
            "victim_usage": rep["deepagents"].get("usage"),
            "victim_error": rep["deepagents"].get("error"),
            "duration_s": round(time.time() - rep_start, 3),
        }

    def _victim_run(
        self,
        unit: Unit,
        user_task: Any,
        injection_task: Any,
        injections: dict[str, str],
        injection_text: str | None,
        attacker_name: str | None,
        keys: dict[str, Any],
        markdown: str | None,
        logdir: Path,
        checker_log: Callable[[dict[str, Any]], None] | None,
        preload: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """One run of the victim on `unit` with the given injections and security-skill body (a trial, or one replica
        of a won trial). With the external checker, its lines go to `checker_log` (replicas pass None: not logged)."""
        suite = get_suite(self.config.benchmark_version, unit.suite)
        factory: ToolMiddlewareFactory | None = None
        library: SkillLibrary | None = None
        if self.defense == "skills":
            skills = list(self.agent_skills)
            if markdown is not None:
                description = getattr(self.skill_file, "description", None) or self.security_skill_description
                skills.append(AgentSkill(self.security_skill_name, description, markdown))
                skills.extend(getattr(self.skill_file, "extra_skills", []))
            if skills:
                library = SkillLibrary(skills)
                middleware = AgentSkillsMiddleware(library, self.skills_prompt_style, preload)
                factory = lambda _query: [middleware]
        elif markdown is not None:
            assert self.checker_fn is not None
            factory = make_checker_factory(
                self.checker_fn,
                markdown,
                log=checker_log or (lambda _line: None),
                log_context=keys,
                injections=[injection_text] if injection_text else [],
                min_span_chars=self.min_span_chars,
            )
        pipeline = self.victim_factory(unit, factory)

        delegate = NullLogger()
        delegate.logdir = str(logdir)  # type: ignore[attr-defined]
        with TraceLogger(
            delegate=delegate,
            suite_name=unit.suite,
            user_task_id=unit.user_task,
            injection_task_id=None if unit.pair is None else unit.pair.injection_task,
            injections=injections,
            attack_type=attacker_name or "none",
            pipeline_name=pipeline.name,
            benchmark_version=self.config.benchmark_version,
            runner=keys,
        ) as logger:
            utility, security = suite.run_task_with_pipeline(pipeline, user_task, injection_task, injections)
            logger.set_contextarg("utility", utility)
            logger.set_contextarg("security", security)

        messages = logger.messages
        final_answer = (
            get_text_content_as_str(messages[-1]["content"] or [])
            if messages and messages[-1]["role"] == "assistant"
            else None
        )
        deepagents = logger.context.get("deepagents") or {}
        # The victim's reasoning, by message index, as the harness recorded it (absent for the native loop).
        reasoning = {
            r["message"]: r["text"]
            for r in deepagents.get("reasoning") or []
            if isinstance(r, dict) and isinstance(r.get("message"), int) and isinstance(r.get("text"), str)
        }
        return {
            "utility": utility,
            "security": security,
            "messages": messages,
            "final_answer": final_answer,
            "deepagents": deepagents,
            "reasoning": reasoning,
            "library": library,
            "pipeline_name": pipeline.name,
        }

    # --- the loop ---------------------------------------------------------------------------------------------

    def run(self, max_units: int | None = None) -> list[dict[str, Any]]:
        """Runs from the checkpoint to the end (or `max_units` units, for tests). Returns all records so far."""
        cursor = self._start()
        if cursor == len(self.units):
            self.echo(f"{self.run_dir}: already complete ({cursor} units)")
            return self.records
        self.echo(f"{self.run_dir}: starting at unit {cursor} of {len(self.units)}")
        done = 0
        for unit in self.units[cursor:]:
            if max_units is not None and done >= max_units:
                break
            if self.already_won(unit):
                self._skip(unit)
                done += 1
                self.echo(f"[{unit.index + 1}/{len(self.units)}] {unit.id}: skipped (pair already succeeded)")
                continue
            record, checker_lines, replicas = self.run_unit(unit)
            self._commit(unit, record, checker_lines, replicas)
            done += 1
            outcome = (
                f"utility={record['utility']}"
                if unit.pair is None
                else f"utility={record['utility']} security={record['security']}"
            )
            skills = record.get("skills") or {}
            if skills.get("s0_loaded") is not None:
                outcome += f" s0_loaded={skills['s0_loaded']}"
            self.echo(f"[{unit.index + 1}/{len(self.units)}] {unit.id} {record['skill_version']}: {outcome}")
        return self.records


def quick_summary(records: list[dict[str, Any]], trials_per_pair: int | None = None) -> dict[str, Any]:
    """Counts for the console at the end of a run (the full report comes from `agentdojo.runner.report`)."""
    attacked = [r for r in records if r["pair"] is not None]
    benign = [r for r in records if r["pair"] is None]
    summary: dict[str, Any] = {"units": len(records)}
    if attacked:
        summary["attacked_trials"] = len(attacked)
        summary["asr"] = sum(r["security"] for r in attacked) / len(attacked)
        summary["utility_under_attack"] = sum(r["utility"] for r in attacked) / len(attacked)
    with_s0 = [r for r in records if (r.get("skills") or {}).get("s0_loaded") is not None]
    if with_s0:
        summary["s0_load_rate"] = sum(r["skills"]["s0_loaded"] for r in with_s0) / len(with_s0)
    if benign:
        summary["benign_tasks"] = len(benign)
        summary["benign_utility"] = sum(r["utility"] for r in benign) / len(benign)
    evo = evolving_metrics(records, trials_per_pair) if trials_per_pair else None
    if evo is not None:
        summary["pairs"] = {k: v for k, v in evo.items() if k != "first_success"}
    rep = replication_metrics([r for r in attacked if r["phase"] in ("attack", "train")], evo)
    if rep is not None:
        summary["replication"] = {k: v for k, v in rep.items() if k not in ("per_win", "histogram")}
    return summary
