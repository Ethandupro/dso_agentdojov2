"""The evolving skill file of C3 and C4: a `SkillFileSource` over `agentdojo.skillfile.SkillFileEvolver`.

After every attacked trial the runner's record becomes the judge's `TrialRecord`: whether the agent loaded the
security skill, its tool calls and final answer (skills defence), or the removed spans and injection survival from
that trial's checker lines (`--external-checker` ablation). The evolver runs a judge step after a successful train trial
(injection goal achieved OR user task failed) and ignores test trials and benign tasks. Its state goes into the
runner's checkpoint, so a resumed run continues from the same file, history, attack archive and changelog.
"""

from typing import Any

from agentdojo.skillfile.judge import RemovedSpan, SkillFileEvolver, TrialRecord
from agentdojo.task_suite.load_suites import get_suite


def trial_record_from_run_record(record: dict[str, Any], benchmark_version: str) -> TrialRecord:
    suite = get_suite(benchmark_version, record["suite"])
    checker = record.get("checker") or {}
    skills = record.get("skills") or {}
    return TrialRecord(
        pair=record["pair"],
        trial=record["trial"],
        split=record["split"],
        skill_version=record["skill_version"],
        user_task=suite.user_tasks[record["user_task"]].PROMPT,
        injection_text=record["injection_text"],
        injection_goal_achieved=bool(record["security"]),
        user_task_succeeded=bool(record["utility"]),
        removed_spans=[RemovedSpan(span=s["span"], reason=s["reason"]) for s in checker.get("removed_spans", [])],
        injection_survival=checker.get("injection_survival"),
        defense="checker" if record.get("checker") is not None else "skills",
        s0_loaded=skills.get("s0_loaded"),
        agent_actions=list(record.get("agent_actions") or []),
        final_answer=record.get("final_answer"),
    )


class EvolvingSkillFile:
    def __init__(self, evolver: SkillFileEvolver, benchmark_version: str) -> None:
        self.evolver = evolver
        self.benchmark_version = benchmark_version

    @property
    def version_name(self) -> str:
        return self.evolver.version_name

    @property
    def markdown(self) -> str:
        return self.evolver.current_markdown

    @property
    def description(self) -> str | None:
        """S0's description as the judge last left it (C3/C4 may edit it since 2026-11-19); None if unset."""
        return self.evolver.current.description or None

    def after_trial(self, record: dict[str, Any]) -> dict[str, Any] | None:
        if record["pair"] is None:
            return None
        step = self.evolver.record_trial(trial_record_from_run_record(record, self.benchmark_version))
        if step is None:
            return None
        return {
            "step": step.step,
            "outcome": step.outcome,
            "version_before": step.version_before,
            "version_after": step.version_after,
            "applied_edits": step.applied_edits,
            "dropped_edits": step.dropped_edits,
            "judge_calls": len(step.attempts),
            "judge_input_tokens": sum(a.input_tokens or 0 for a in step.attempts),
            "judge_output_tokens": sum(a.output_tokens or 0 for a in step.attempts),
            "check_failures": [f for a in step.attempts for f in a.failures],
        }

    def freeze(self) -> None:
        self.evolver.freeze()

    def state_dict(self) -> dict[str, Any]:
        return self.evolver.state_dict()

    def load_state(self, state: dict[str, Any]) -> None:
        self.evolver.load_state(state)
