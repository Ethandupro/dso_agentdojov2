"""The evolving skill file of C3 and C4: a `SkillFileSource` over `agentdojo.skillfile.SkillFileEvolver`.

After every attacked trial the runner's record becomes the judge's `TrialRecord`: whether the agent loaded the
security skill, its tool calls and final answer (skills defence), or the removed spans and injection survival from
that trial's checker lines (`--external-checker` ablation). With the `every-win` trigger the evolver runs a judge step
after each successful train trial (injection goal achieved OR user task failed). With the `batched` trigger (the
default of `run_condition` since 2026-11-26) it collects train wins, and this class tells it when a train pair has run
its last trial (`finish_pair`): the pair's last trial index, or, with `--stop-on-success`, an evolving-attacker trial
that achieved the injection goal. Test trials and benign tasks are ignored. The evolver's state goes into the
runner's checkpoint, so a resumed run continues from the same file, history, pending wins, attack archive and
changelog.
"""

from typing import Any

from agentdojo.runner.plugins import StaticAttacker
from agentdojo.skillfile.judge import RemovedSpan, SkillFileEvolver, StepResult, TrialRecord
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
        confirmation=record.get("judge_confirmation"),
        victim_trace=record.get("judge_trace"),
        victim_text=record.get("victim_text"),
    )


def step_summary(steps: list[StepResult]) -> dict[str, Any] | None:
    """What a trial record keeps of the evolution step(s) that ran after it (a batch can run two: description, then
    rules). Judge calls and tokens are summed; `steps` lists each step's own summary."""
    if not steps:
        return None
    each = [
        {
            "step": s.step,
            "outcome": s.outcome,
            "version_before": s.version_before,
            "version_after": s.version_after,
            "applied_edits": s.applied_edits,
            "dropped_edits": s.dropped_edits,
            "judge_calls": len(s.attempts),
            "judge_input_tokens": sum(a.input_tokens or 0 for a in s.attempts),
            "judge_output_tokens": sum(a.output_tokens or 0 for a in s.attempts),
            "check_failures": [f for a in s.attempts for f in a.failures],
        }
        for s in steps
    ]
    if len(each) == 1:
        return each[0]
    applied = any(e["outcome"] == "applied" for e in each)
    return {
        "step": each[-1]["step"],
        "outcome": "applied" if applied else each[-1]["outcome"],
        "version_before": each[0]["version_before"],
        "version_after": each[-1]["version_after"],
        "applied_edits": [x for e in each for x in e["applied_edits"]],
        "dropped_edits": [x for e in each for x in e["dropped_edits"]],
        "judge_calls": sum(e["judge_calls"] for e in each),
        "judge_input_tokens": sum(e["judge_input_tokens"] for e in each),
        "judge_output_tokens": sum(e["judge_output_tokens"] for e in each),
        "check_failures": [x for e in each for x in e["check_failures"]],
        "steps": each,
    }


class EvolvingSkillFile:
    def __init__(
        self,
        evolver: SkillFileEvolver,
        benchmark_version: str,
        trials_per_pair: int | None = None,
        stop_on_success: bool = False,
    ) -> None:
        """The victim reads S0 with the index of specialised skills when `evolver.s0_index` (`--skill-loading index`
        or `both`); without specialised skills S0 is unchanged either way."""
        if evolver.trigger.batched and trials_per_pair is None:
            raise ValueError("the batched trigger needs trials_per_pair to tell when a train pair is done")
        self.evolver = evolver
        self.benchmark_version = benchmark_version
        self.trials_per_pair = trials_per_pair
        self.stop_on_success = stop_on_success

    def pair_finished(self, record: dict[str, Any]) -> bool:
        """Whether this train trial is its pair's last: the last trial index, or a `--stop-on-success` win of the
        evolving attacker (its later trials are skipped). Static pairs always run every trial."""
        if self.trials_per_pair is not None and record["trial"] >= self.trials_per_pair - 1:
            return True
        return self.stop_on_success and record.get("attack") != StaticAttacker.name and bool(record["security"])

    @property
    def version_name(self) -> str:
        return self.evolver.version_name

    @property
    def markdown(self) -> str:
        return self.evolver.library.s0_markdown(index=self.evolver.s0_index)

    @property
    def specialist_skills(self) -> list[tuple[str, str, str]]:
        return self.evolver.library.victim_skills()

    @property
    def description(self) -> str | None:
        """S0's description as the judge last left it (C3/C4 may edit it since 2026-11-19); None if unset."""
        return self.evolver.current.description or None

    def after_trial(self, record: dict[str, Any]) -> dict[str, Any] | None:
        if record["pair"] is None:
            return None
        trial = trial_record_from_run_record(record, self.benchmark_version)
        # The victim's own text is only used to find the rules it used; it never goes to trials.jsonl.
        record.pop("victim_text", None)
        train = record["phase"] == "train" and not self.evolver.frozen
        if train and self.evolver.rich_evidence and trial.victim_text is not None:
            match = self.evolver.match_rules(trial.victim_text, trial.injection_text)
            trial.cited_rules = match.rule_ids
            trial.covered_by = match.covered_by
            trial.victim_text = None
            record["s0_rule_match"] = match.to_dict()
        step = self.evolver.record_trial(trial)
        if self.evolver.rich_evidence and record["phase"] == "train":
            record["s0_rules_quoted"] = trial.cited_rules
            if self.evolver.library_policy.enabled:
                record["covered_by"] = trial.covered_by
        steps = [step] if step is not None else []
        if self.evolver.trigger.batched and record["phase"] == "train" and self.pair_finished(record):
            steps.extend(self.evolver.finish_pair())
        return step_summary(steps)

    def freeze(self) -> None:
        self.evolver.freeze()

    def state_dict(self) -> dict[str, Any]:
        return self.evolver.state_dict()

    def load_state(self, state: dict[str, Any]) -> None:
        self.evolver.load_state(state)
