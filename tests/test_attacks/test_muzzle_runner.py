"""MUZZLE through the real runner with a scripted victim and checker: a C2 run and a C4 run, plus resume."""

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("deepagents")

from agentdojo.runner.conditions import CONDITIONS, build_units
from agentdojo.runner.plugins import FixedSkillFile, StaticAttacker
from agentdojo.runner.run import Runner, run_dir_for
from tests.test_attacks.test_muzzle import MANIFEST, make_attacker
from tests.test_checker.test_checker_middleware import reading_checker
from tests.test_runner.test_run import RecordingSkillFile, ScriptedVictim, make_config, read_jsonl


def make_runner(tmp_path: Path, condition: str, attacker: Any, limit: int = 1) -> Runner:
    cond = CONDITIONS[condition]
    return Runner(
        config=make_config(condition, limit, attacker={"name": "muzzle"}),
        condition=cond,
        run_dir=run_dir_for(tmp_path, cond, 0, limit),
        units=build_units(cond, MANIFEST, 3, limit),
        attacker=attacker,
        static_attacker=StaticAttacker(MANIFEST),
        skill_file=FixedSkillFile("(S0)") if not cond.evolving_skill else RecordingSkillFile(),
        victim_factory=ScriptedVictim,
        checker_fn=reading_checker,
        echo=lambda _: None,
    )


def test_c2_trials_use_muzzle_rewrites_and_feed_back_outcomes(tmp_path: Path):
    log: list[dict[str, Any]] = []
    attacker, _ = make_attacker(log)
    runner = make_runner(tmp_path, "C2", attacker)
    records = [r for r in runner.run() if r["pair"]]
    assert [r["trial"] for r in records] == [0, 1, 2]
    assert [r["attack"] for r in records] == ["muzzle"] * 3
    texts = attacker.attack_texts()
    assert len(texts) == 3 and texts[1] != texts[0] and texts[2] != texts[1]
    # Two rewrites, each judging the previous trial's real outcome.
    judged = [e["feedback"] for e in attacker.archive[1:]]
    assert [f["trial"] for f in judged] == [0, 1]
    for f, r in zip(judged, records[:2], strict=True):
        assert (f["utility"], f["security"]) == (r["utility"], r["security"])
    state = json.loads((runner.run_dir / "state.json").read_text())
    assert [e["text"] for e in state["attacker"]["archive"]] == texts
    assert len(read_jsonl(runner.run_dir / "trials.jsonl")) == len(runner.units)


def test_c4_test_pairs_never_reach_muzzle(tmp_path: Path):
    log: list[dict[str, Any]] = []
    attacker, _ = make_attacker(log)
    records = [r for r in make_runner(tmp_path, "C4", attacker).run() if r["pair"]]
    assert [r["attack"] for r in records] == ["muzzle"] * 3 + ["static"] * 3
    train_pair = records[0]["pair"]
    assert {e["pair"] for e in attacker.archive} == {train_pair}


def test_resume_reuses_the_archive(tmp_path: Path):
    attacker, _ = make_attacker([])
    runner = make_runner(tmp_path, "C2", attacker)
    runner.run()
    texts = attacker.attack_texts()

    fresh, clients = make_attacker([])
    again = make_runner(tmp_path, "C2", fresh)
    again.run()  # everything is done: loads the checkpoint, runs nothing
    assert fresh.attack_texts() == texts and clients == []
