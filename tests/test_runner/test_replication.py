"""`--replicate-wins`: each successful evolving injection is re-run unchanged, logged apart from the trials, and
reported as a replication rate and a robust ASR; plus the 2026-11-25 CLI defaults and flags."""

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("deepagents")

from agentdojo.runner.conditions import CONDITIONS, build_units
from agentdojo.runner.manifest import Pair
from agentdojo.runner.plugins import AttemptFeedback, NoChecker, StaticAttacker
from agentdojo.runner.report import evolving_metrics, replication_metrics, summarize_run, write_report
from agentdojo.runner.run import Runner, run_dir_for
from tests.test_runner.test_first_success import captured_cli
from tests.test_runner.test_run import MANIFEST, RecordingAttacker, ScriptedVictim, make_config, read_jsonl


class ObservingAttacker(RecordingAttacker):
    def __init__(self, manifest: dict[str, Any]) -> None:
        super().__init__(manifest)
        self.seen: list[AttemptFeedback] = []

    def observe(self, pair: Pair, feedback: AttemptFeedback, skill_version: str | None) -> None:
        self.seen.append(feedback)


def c0e_runner(tmp_path: Path, replicate_wins: int) -> tuple[Runner, ObservingAttacker]:
    cond = CONDITIONS["C0E"]
    attacker = ObservingAttacker(MANIFEST)
    runner = Runner(
        config=make_config("C0E", 4, replicate_wins=replicate_wins),
        condition=cond,
        run_dir=run_dir_for(tmp_path, cond, 0, 4, replicate_wins=replicate_wins),
        units=build_units(cond, MANIFEST, 3, 4),
        attacker=attacker,
        static_attacker=StaticAttacker(MANIFEST),
        skill_file=NoChecker(),
        victim_factory=ScriptedVictim,
        echo=lambda _: None,
        replicate_wins=replicate_wins,
    )
    return runner, attacker


def test_wins_are_replicated_outside_the_trials(tmp_path: Path):
    # The scripted victim is hijacked on the 4th of the first 4 pairs (see test_c0_has_no_checker_and_is_hijacked).
    runner, attacker = c0e_runner(tmp_path, 2)
    records = runner.run()
    attacked = [r for r in records if r["pair"] is not None]
    assert len(attacked) == 12  # trial count unchanged
    wins = [r for r in attacked if r["security"]]
    assert wins and all(r["replication"] == {"runs": 2, "successes": 2} for r in wins)
    assert all(r["replication"] is None for r in attacked if not r["security"])
    replicas = read_jsonl(runner.replications_path)
    assert len(replicas) == 2 * len(wins)
    assert {(r["pair"], r["trial"]) for r in replicas} == {(r["pair"], r["trial"]) for r in wins}
    assert all(r["security"] and r["replica"] in (1, 2) for r in replicas)
    assert (runner.run_dir / "replications").is_dir()
    # The attacker sees the tally and, for the analyst, a longer trace than the leaderboard's.
    seen_wins = [f for f in attacker.seen if f.security]
    assert all(f.replication == {"runs": 2, "successes": 2} for f in seen_wins)
    assert all(f.analyst_trace for f in attacker.seen)
    # ASR from trials only; the report adds the robust figures.
    summary = summarize_run(runner.run_dir)
    assert len(summary["running_asr"]) == 12
    rep = summary["replication"]
    assert rep["wins_replicated"] == len(wins) and rep["replica_runs"] == 2 * len(wins)
    assert rep["mean_replication_rate"] == 1.0 and rep["robust_wins"] == len(wins)
    assert "Do wins hold up?" in write_report(runner.run_dir).read_text()


def test_resume_drops_replicas_of_units_after_the_checkpoint(tmp_path: Path):
    runner, _ = c0e_runner(tmp_path, 1)
    runner.run()
    n = len(read_jsonl(runner.replications_path))
    stray = {"unit": 10_000, "pair": "x", "trial": 0, "replica": 1, "security": True}
    with runner.replications_path.open("a") as f:
        f.write(json.dumps(stray) + "\n")
    again, _ = c0e_runner(tmp_path, 1)
    again.run()
    assert len(read_jsonl(again.replications_path)) == n


def test_no_replication_when_off(tmp_path: Path):
    runner, _ = c0e_runner(tmp_path, 0)
    records = runner.run()
    assert all(r.get("replication") is None for r in records)
    assert read_jsonl(runner.replications_path) == []
    assert summarize_run(runner.run_dir)["replication"] is None


def rec(pair: str, trial: int, win: bool, successes: int | None = None, runs: int = 4) -> dict[str, Any]:
    replication = {"runs": runs, "successes": successes} if successes is not None else None
    return {"pair": pair, "trial": trial, "security": win, "phase": "attack", "replication": replication}


def test_robust_asr():
    records = [
        rec("a", 0, True, 4),  # baseline, robust
        rec("b", 0, False),
        rec("b", 1, True, 1),  # evolved, lucky
        rec("c", 0, False),
        rec("c", 1, False),
        rec("c", 2, True, 3),  # evolved, robust
        rec("d", 0, False),
        rec("d", 1, False),
        rec("d", 2, False),  # never won
    ]
    evo = evolving_metrics(records, 3)
    rep = replication_metrics(records, evo)
    assert rep is not None and evo is not None
    assert evo["evolving_asr"] == 2 / 3  # plain: b and c
    assert rep["robust_evolving_asr"] == 1 / 3  # only c holds up
    assert rep["robust_baseline_works"] == 1 and rep["robust_pair_asr_incl_baseline"] == 2 / 4
    assert rep["histogram"] == {"1/4": 1, "3/4": 1, "4/4": 1} and rep["robust_wins"] == 2
    assert rep["pooled_replication_rate"] == 8 / 12


# --- CLI ---------------------------------------------------------------------------------------------------------


def test_cli_defaults_victim_0_1_attacker_0_6_replicate_4_analyst_on(monkeypatch, tmp_path: Path):
    args = ["--condition", "C0E", "--results-root", str(tmp_path), "--limit", "2"]
    result, wiring = captured_cli(monkeypatch, args)
    assert result.exit_code == 0, result.output
    assert wiring["config"].victim["temperature"] == 0.1
    assert wiring["attacker"].settings.temperature == 0.6
    assert wiring["attacker"].settings.analyst is True
    assert wiring["replicate_wins"] == 4 and wiring["config"].replicate_wins == 4
    assert wiring["run_dir"].parent.name == "c0e_no-s0_evolving-attack-limit2"


def test_cli_static_conditions_have_no_replication(monkeypatch, tmp_path: Path):
    args = ["--condition", "C1", "--results-root", str(tmp_path), "--limit", "2"]
    result, wiring = captured_cli(monkeypatch, args)
    assert result.exit_code == 0, result.output
    assert wiring["replicate_wins"] == 0 and wiring["config"].replicate_wins is None
    assert wiring["config"].victim["temperature"] == 0.1
    result, _ = captured_cli(monkeypatch, [*args, "--replicate-wins", "2"])
    assert result.exit_code != 0 and "evolving-attack conditions only" in result.output


def test_cli_new_flags_name_their_directories(monkeypatch, tmp_path: Path):
    base = ["--condition", "C2", "--results-root", str(tmp_path), "--limit", "2", "--attacker", "skill-agent"]
    args = [*base, "--distill-every", "3", "--distill-mode", "trajectory", "--skill-choice", "free"]
    args += ["--no-analyst", "--replicate-wins", "2"]
    result, wiring = captured_cli(monkeypatch, args)
    assert result.exit_code == 0, result.output
    settings = wiring["attacker"].settings
    assert (settings.distill_mode, settings.skill_choice, settings.analyst) == ("trajectory", "free", False)
    assert wiring["replicate_wins"] == 2
    assert wiring["run_dir"].parent.name == (
        "c2_static-s0_evolving-attack-skill-agent-distill3-traj-free-skills-no-analyst-rep2-limit2"
    )


def test_cli_skill_choice_needs_the_skill_agent(monkeypatch, tmp_path: Path):
    args = ["--condition", "C2", "--results-root", str(tmp_path), "--limit", "2", "--skill-choice", "free"]
    result, _ = captured_cli(monkeypatch, args)
    assert result.exit_code != 0 and "--attacker skill-agent" in result.output
