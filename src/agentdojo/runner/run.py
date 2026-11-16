"""Runs one condition at one seed: the unit loop, per-trial traces, checker logs, checkpoints and resume.

Requires the `deepagents` extra (the checker middleware). The victim and the checker model are passed in as
functions, so tests can script them; `agentdojo.runner.run_condition` builds the real ones.

Files under the run directory (DESIGN.md, "Conditions and result segmentation"):

- `config.json`: written before any model call; a run resumes only under an identical config.
- `trials.jsonl`: one record per finished unit (trial or benign task), in unit order.
- `checker_calls.jsonl`: one line per checker call, with the unit's keys.
- `state.json`: the checkpoint (index of the next unit, attacker and skill-file state), replaced atomically.
- `trial_<t>/<split>/...` and `benign*/...`: AgentDojo trace files, one per unit.

After each unit the runner appends its record and checker lines, then replaces `state.json`. On resume, lines of
units at or after the checkpoint (a crash between the two writes) are dropped and that unit runs again. A unit that
raises (for example a model server that stays unreachable after the client's retries) stops the run with the
checkpoint intact; nothing is scored for it.
"""

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.deepagents_pipeline import ToolMiddlewareFactory
from agentdojo.checker.middleware import CheckerFn, make_checker_factory
from agentdojo.logging import NullLogger, TraceLogger
from agentdojo.runner.conditions import Condition, Unit
from agentdojo.runner.config import RunConfig, config_diff, sha256_text
from agentdojo.runner.plugins import Attacker, AttemptFeedback, SkillFileSource, StaticAttacker
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.types import get_text_content_as_str

VictimFactory = Callable[[Unit, ToolMiddlewareFactory | None], BasePipelineElement]
"""Builds the victim for one unit, with the checker middleware factory (None without a checker). The real victim
ignores the unit; scripted test victims use it."""

STATE_FORMAT = 1


class ConfigMismatchError(Exception):
    pass


def run_dir_for(results_root: Path, condition: Condition, seed: int, limit: int | None) -> Path:
    name = condition.run_id if limit is None else f"{condition.run_id}-limit{limit}"
    return results_root / name / f"seed_{seed}"


def trace_dir(run_dir: Path, unit: Unit) -> Path:
    if unit.pair is None:
        return run_dir / unit.phase
    return run_dir / f"trial_{unit.trial}" / unit.pair.split


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
        checker_fn: CheckerFn | None,
        min_span_chars: int = 0,
        echo: Callable[[str], None] = print,
    ) -> None:
        if condition.checker and checker_fn is None:
            raise ValueError(f"{condition.name} needs a checker model")
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
            self.attacker.load_state(state["attacker"])
            self.skill_file.load_state(state["skill_file"])
        # Drop lines of units at or after the checkpoint: written just before a crash, before the state was saved.
        records = [r for r in _read_jsonl(self.trials_path) if r["unit"] < cursor]
        if len(records) != cursor:
            raise ConfigMismatchError(f"trials.jsonl has {len(records)} records before the checkpoint at {cursor}")
        _write_jsonl(self.trials_path, records)
        _write_jsonl(self.checker_path, [c for c in _read_jsonl(self.checker_path) if c["unit"] < cursor])
        self.records = records
        return cursor

    def _commit(self, unit: Unit, record: dict[str, Any], checker_lines: list[dict[str, Any]]) -> None:
        _append_jsonl(self.trials_path, [record])
        _append_jsonl(self.checker_path, checker_lines)
        self.records.append(record)
        _write_json_atomic(
            self.state_path,
            {
                "format": STATE_FORMAT,
                "cursor": unit.index + 1,
                "n_units": len(self.units),
                "units_sha256": self.units_sha256,
                "complete": unit.index + 1 == len(self.units),
                "attacker": self.attacker.state_dict(),
                "skill_file": self.skill_file.state_dict(),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )

    # --- one unit ---------------------------------------------------------------------------------------------

    def history(self, unit: Unit) -> list[AttemptFeedback]:
        """The pair's earlier trials in this run, as the attacker may see them."""
        assert unit.pair is not None
        return [
            AttemptFeedback(
                trial=r["trial"],
                injection_text=r["injection_text"],
                utility=r["utility"],
                security=r["security"],
                final_answer=r["final_answer"],
            )
            for r in self.records
            if r["pair"] == unit.pair.id and r["phase"] == unit.phase
        ]

    def attacker_for(self, unit: Unit) -> Attacker:
        # C4: the evolving attacker never sees test-split injection tasks; test pairs keep the trial-0 text.
        if unit.phase == "test":
            return self.static_attacker
        return self.attacker

    def run_unit(self, unit: Unit) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
        keys = {
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

        checker_lines: list[dict[str, Any]] = []
        factory: ToolMiddlewareFactory | None = None
        markdown = self.skill_file.markdown
        if markdown is not None:
            assert self.checker_fn is not None
            factory = make_checker_factory(
                self.checker_fn,
                markdown,
                log=checker_lines.append,
                log_context=keys,
                injections=[injection_text] if injection_text else [],
                min_span_chars=self.min_span_chars,
            )
        pipeline = self.victim_factory(unit, factory)

        delegate = NullLogger()
        delegate.logdir = str(trace_dir(self.run_dir, unit))  # type: ignore[attr-defined]
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
        attacked = unit.pair is not None
        record: dict[str, Any] = {
            **keys,
            "attack": attacker_name,
            "injection_text": injection_text,
            "utility": utility,
            "security": security if attacked else None,
            "attacker_success": (security or not utility) if attacked else None,
            "final_answer": final_answer,
            "checker": summarize_checker(checker_lines) if markdown is not None else None,
            "victim_usage": deepagents.get("usage"),
            "victim_served_models": deepagents.get("served_models"),
            "victim_error": deepagents.get("error"),
            "trace": str(
                Path(delegate.logdir)  # type: ignore[attr-defined]
                .relative_to(self.run_dir)
                .joinpath(
                    (pipeline.name or "").replace("/", "_"),
                    unit.suite,
                    unit.user_task,
                    attacker_name or "none",
                    f"{unit.pair.injection_task if unit.pair else 'none'}.json",
                )
            ),
            "duration_s": round(time.time() - start, 3),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        record["skill_step"] = self.skill_file.after_trial(record)
        record["skill_version_after"] = self.skill_file.version_name
        return record, checker_lines

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
            record, checker_lines = self.run_unit(unit)
            self._commit(unit, record, checker_lines)
            done += 1
            outcome = (
                f"utility={record['utility']}"
                if unit.pair is None
                else f"utility={record['utility']} security={record['security']}"
            )
            self.echo(f"[{unit.index + 1}/{len(self.units)}] {unit.id} {record['skill_version']}: {outcome}")
        return self.records


def quick_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts for the console at the end of a run (the full report comes from `agentdojo.runner.report`)."""
    attacked = [r for r in records if r["pair"] is not None]
    benign = [r for r in records if r["pair"] is None]
    summary: dict[str, Any] = {"units": len(records)}
    if attacked:
        summary["attacked_trials"] = len(attacked)
        summary["asr"] = sum(r["security"] for r in attacked) / len(attacked)
        summary["utility_under_attack"] = sum(r["utility"] for r in attacked) / len(attacked)
    if benign:
        summary["benign_tasks"] = len(benign)
        summary["benign_utility"] = sum(r["utility"] for r in benign) / len(benign)
    return summary
