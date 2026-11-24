"""`--stop-on-success`: evolving pairs stop at their first success, and the pair-level baseline/evolution report."""

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

pytest.importorskip("deepagents")

from agentdojo.runner import run_condition
from agentdojo.runner.conditions import CONDITIONS, Unit, build_units
from agentdojo.runner.plugins import NoChecker, StaticAttacker
from agentdojo.runner.report import evolving_metrics
from agentdojo.runner.run import Runner, run_dir_for
from agentdojo.runner.run_condition import reply_thinks
from tests.test_runner.test_run import (
    MANIFEST,
    REPO_ROOT,
    CapturedRunner,
    RecordingAttacker,
    ScriptedVictim,
    make_config,
)


class ScriptedOutcomeRunner(Runner):
    """Runs no model: each attacked trial's security comes from a script, keyed by (pair number, trial)."""

    script: dict[tuple[int, int], bool]

    def run_unit(self, unit: Unit) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        record: dict[str, Any] = {
            "unit": unit.index,
            "unit_id": unit.id,
            "phase": unit.phase,
            "skill_version": "none",
            "pair": None,
            "utility": True,
        }
        if unit.pair is not None:
            pairs = sorted({u.pair.id for u in self.units if u.pair is not None})
            assert self.attacker_for(unit) is not self.static_attacker
            record |= {
                "pair": unit.pair.id,
                "trial": unit.trial,
                "security": self.script.get(
                    (pairs.index(unit.pair.id), unit.trial if unit.trial is not None else -1), False
                ),
            }
        return record, []


def c0e_runner(tmp_path: Path, stop_on_success: bool) -> ScriptedOutcomeRunner:
    cond = CONDITIONS["C0E"]
    static = StaticAttacker(MANIFEST)
    runner = ScriptedOutcomeRunner(
        config=make_config("C0E", 2, stop_on_success=stop_on_success or None),
        condition=cond,
        run_dir=run_dir_for(tmp_path, cond, 0, 2, stop_on_success=stop_on_success),
        units=build_units(cond, MANIFEST, 3, 2),
        attacker=RecordingAttacker(MANIFEST),
        static_attacker=static,
        skill_file=NoChecker(),
        victim_factory=ScriptedVictim,
        echo=lambda _: None,
        stop_on_success=stop_on_success,
    )
    # Pair 0: the original text works at trial 0. Pair 1: fails at trial 0, wins after one evolution.
    runner.script = {(0, 0): True, (1, 1): True}
    return runner


def outcomes(records: list[dict[str, Any]]) -> list[tuple[str, int, bool]]:
    return [(r["pair"], r["trial"], r["security"]) for r in records if r["pair"] is not None]


def test_stop_on_success_stops_each_pair_at_its_first_success(tmp_path: Path):
    runner = c0e_runner(tmp_path, stop_on_success=True)
    got = outcomes(runner.run())
    pairs = sorted({p for p, _, _ in got})
    assert sorted(got) == [(pairs[0], 0, True), (pairs[1], 0, False), (pairs[1], 1, True)]
    assert len(runner.skipped) == 3
    assert "-first-success" in runner.run_dir.parent.name
    evo = evolving_metrics(runner.records, 3)
    assert evo is not None
    assert (evo["baseline_works"], evo["evolving_pairs"], evo["evolved_successes"]) == (1, 1, 1)
    assert [p["asr"] for p in evo["asr_at_n"]] == [1.0, 1.0]


def test_stop_on_success_resumes_with_skipped_units(tmp_path: Path):
    expected = outcomes(c0e_runner(tmp_path / "full", stop_on_success=True).run())
    c0e_runner(tmp_path, stop_on_success=True).run(max_units=4)
    assert outcomes(c0e_runner(tmp_path, stop_on_success=True).run()) == expected


def test_without_the_flag_every_trial_runs(tmp_path: Path):
    runner = c0e_runner(tmp_path, stop_on_success=False)
    assert len(outcomes(runner.run())) == 6
    assert runner.skipped == []


def rec(pair: str, trial: int, security: bool, phase: str = "attack") -> dict[str, Any]:
    return {"pair": pair, "trial": trial, "security": security, "phase": phase}


def test_evolving_metrics_separates_baseline_from_evolution():
    records = [
        # a: original text works at trial 0 -> baseline, not counted in the evolving ASR.
        rec("a", 0, True),
        # b: won after 2 evolutions (a later failure is ignored: first success counts).
        rec("b", 0, False),
        rec("b", 1, False),
        rec("b", 2, True),
        rec("b", 3, False),
        # c: never won in all 4 trials.
        *[rec("c", t, False) for t in range(4)],
        # d: won after 1 evolution.
        rec("d", 0, False),
        rec("d", 1, True),
        # e: still running (2 of 4 trials, no win) -> not counted yet.
        rec("e", 0, False),
        rec("e", 1, False),
        # C4's static test phase is left out.
        rec("f", 0, True, phase="test"),
    ]
    evo = evolving_metrics(records, trials_per_pair=4)
    assert evo is not None
    assert evo["finished_pairs"] == 4 and evo["baseline_works"] == 1 and evo["evolving_pairs"] == 3
    assert evo["evolved_successes"] == 2 and evo["evolving_asr"] == pytest.approx(2 / 3)
    assert evo["pair_asr_incl_baseline"] == pytest.approx(3 / 4)
    assert evo["mean_evolutions_to_success"] == pytest.approx(1.5)
    assert [p["asr"] for p in evo["asr_at_n"]] == pytest.approx([1 / 3, 2 / 3, 2 / 3])
    assert evo["first_success"] == {"b": 2, "c": None, "d": 1}


def test_reply_thinks():
    assert not reply_thinks({"content": "OK"})
    assert not reply_thinks({"content": "OK", "reasoning_content": None})
    assert not reply_thinks({"content": "<think>\n\n</think>\n\nOK"})  # Qwen's empty block with thinking off
    assert reply_thinks({"content": "OK", "reasoning_content": "Let me think."})
    assert reply_thinks({"content": "OK", "reasoning": "hmm"})
    assert reply_thinks({"content": "<think>The user wants OK.</think>OK"})


def captured_cli(monkeypatch, args: list[str]) -> tuple[Any, dict[str, Any]]:
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setattr(run_condition, "check_served", lambda role, url, model: [model])
    monkeypatch.setattr(run_condition, "check_attacker_not_thinking", lambda url, model: None)
    monkeypatch.setattr(run_condition, "Runner", CapturedRunner)
    monkeypatch.setattr(run_condition, "write_report", lambda run_dir: run_dir / "report.html")
    result = CliRunner().invoke(run_condition.main, args)
    return result, CapturedRunner.last


def test_cli_wires_stop_on_success_and_temperatures(monkeypatch, tmp_path: Path):
    args = ["--condition", "C0E", "--results-root", str(tmp_path), "--limit", "2", "--stop-on-success"]
    args += ["--victim-temperature", "0.1", "--attacker-temperature", "0.1"]
    result, wiring = captured_cli(monkeypatch, args)
    assert result.exit_code == 0, result.output
    assert wiring["stop_on_success"] is True and wiring["config"].stop_on_success is True
    assert wiring["config"].victim["temperature"] == 0.1
    assert wiring["attacker"].settings.temperature == 0.1
    assert wiring["run_dir"].parent.name == "c0e_no-s0_evolving-attack-first-success-vtemp0.1-atemp0.1-limit2"


def test_cli_refuses_stop_on_success_for_a_static_attack(monkeypatch, tmp_path: Path):
    args = ["--condition", "C0", "--results-root", str(tmp_path), "--limit", "2", "--stop-on-success"]
    result, _ = captured_cli(monkeypatch, args)
    assert result.exit_code != 0 and "evolving-attack conditions only" in result.output


def test_cli_wires_distillation_for_the_skill_agent_only(monkeypatch, tmp_path: Path):
    base = ["--condition", "C0E", "--results-root", str(tmp_path), "--limit", "2", "--distill-every", "5"]
    result, wiring = captured_cli(monkeypatch, [*base, "--attacker", "skill-agent"])
    assert result.exit_code == 0, result.output
    assert wiring["attacker"].settings.distill_every == 5
    assert wiring["run_dir"].parent.name == "c0e_no-s0_evolving-attack-skill-agent-distill5-limit2"
    result, _ = captured_cli(monkeypatch, base)
    assert result.exit_code != 0 and "--attacker skill-agent" in result.output


def test_cli_distill_mode_defaults_to_contrast(monkeypatch, tmp_path: Path):
    base = ["--condition", "C0E", "--results-root", str(tmp_path), "--limit", "2", "--attacker", "skill-agent"]
    result, wiring = captured_cli(monkeypatch, [*base, "--distill-every", "5"])
    assert result.exit_code == 0, result.output
    assert wiring["attacker"].settings.distill_mode == "contrast"
    result, wiring = captured_cli(monkeypatch, [*base, "--distill-every", "5", "--distill-mode", "plain"])
    assert result.exit_code == 0, result.output
    assert wiring["attacker"].settings.distill_mode == "plain"
    assert wiring["run_dir"].parent.name == "c0e_no-s0_evolving-attack-skill-agent-distill5-plain-limit2"
    result, _ = captured_cli(monkeypatch, [*base, "--distill-mode", "plain"])
    assert result.exit_code != 0 and "--distill-every" in result.output
