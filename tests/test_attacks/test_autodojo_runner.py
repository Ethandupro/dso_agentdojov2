"""The AutoDojo attacker through the real runner with a scripted victim and checker: a C2 run and a C4 run."""

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("deepagents")

from agentdojo.runner.conditions import CONDITIONS, build_units
from agentdojo.runner.plugins import FixedSkillFile, StaticAttacker
from agentdojo.runner.report import write_report
from agentdojo.runner.run import Runner, run_dir_for
from tests.test_attacks.test_autodojo import MANIFEST, make_attacker
from tests.test_checker.test_checker_middleware import reading_checker
from tests.test_runner.test_run import RecordingSkillFile, ScriptedVictim, make_config, read_jsonl


def make_runner(tmp_path: Path, condition: str, attacker: Any, trials: int = 3, limit: int = 1) -> Runner:
    cond = CONDITIONS[condition]
    return Runner(
        config=make_config(condition, limit, trials_per_pair=trials, attacker={"name": "autodojo"}),
        condition=cond,
        run_dir=run_dir_for(tmp_path, cond, 0, limit),
        units=build_units(cond, MANIFEST, trials, limit),
        attacker=attacker,
        static_attacker=StaticAttacker(MANIFEST),
        skill_file=FixedSkillFile("(S0)") if not cond.evolving_skill else RecordingSkillFile(),
        victim_factory=ScriptedVictim,
        checker_fn=reading_checker,
        echo=lambda _: None,
    )


def test_c2_evolves_over_nine_trials_feeding_back_real_outcomes(tmp_path: Path):
    attacker, _, _ = make_attacker()
    runner = make_runner(tmp_path, "C2", attacker, trials=9)
    records = [r for r in runner.run() if r["pair"]]
    assert [r["trial"] for r in records] == list(range(9))
    assert [r["attack"] for r in records] == ["autodojo"] * 9
    texts = [e["text"] for e in attacker.archive]
    assert len(texts) == 9 and len(set(texts)) == 9  # every trial a distinct text
    # Each rewrite's leaderboard entries carry the real security/utility of the trials that produced them.
    rewrite = attacker.archive[2]
    assert rewrite["source"] == "generated"
    seen = {e["trial"]: (e["security"], e["utility"]) for e in rewrite["leaderboard"]}
    for r in records[:2]:
        assert seen[r["trial"]] == (r["security"], r["utility"])
    state = json.loads((runner.run_dir / "state.json").read_text())
    assert [e["text"] for e in state["attacker"]["archive"]] == texts

    write_report(runner.run_dir)
    summary = json.loads((runner.run_dir / "summary.json").read_text())
    assert summary["attacker"]["rewrites"] == 8
    assert "<h2>Attacker</h2>" in (runner.run_dir / "report.html").read_text()


def test_c4_test_pairs_never_reach_the_autodojo_attacker(tmp_path: Path):
    attacker, _, _ = make_attacker()
    records = [r for r in make_runner(tmp_path, "C4", attacker, trials=3).run() if r["pair"]]
    assert [r["attack"] for r in records] == ["autodojo"] * 3 + ["static"] * 3
    train_pair = records[0]["pair"]
    assert {e["pair"] for e in attacker.archive} == {train_pair}


def test_resume_reuses_the_archive(tmp_path: Path):
    attacker, _, _ = make_attacker()
    runner = make_runner(tmp_path, "C2", attacker, trials=3)
    runner.run()
    texts = [e["text"] for e in attacker.archive]
    assert len(read_jsonl(runner.run_dir / "trials.jsonl")) == len(runner.units)
    assert texts == [e["text"] for e in attacker.archive]
