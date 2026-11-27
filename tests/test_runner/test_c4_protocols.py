"""The two extra C4 protocols (`run_condition --c4-protocol`, 2026-11-27): `no-split` (one adapting phase over every
pair) and `attack-test` (the split, with the evolving attacker also on the frozen test phase), through the real
runner and the cross-run comparison."""

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("deepagents")

from agentdojo import experiment_models
from agentdojo.runner.compare import compare, find_runs
from agentdojo.runner.conditions import CONDITIONS, PairFilter, build_units
from agentdojo.runner.evolving import EvolvingSkillFile
from agentdojo.runner.plugins import FixedSkillFile, StaticAttacker
from agentdojo.runner.run import Runner, run_dir_for
from agentdojo.skillfile.checks import EditPolicy, approx_token_count
from agentdojo.skillfile.judge import SkillFileEvolver
from agentdojo.skillfile.library import LibraryPolicy
from agentdojo.skillfile.structure import load_skill_file
from agentdojo.skillfile.trigger import TriggerPolicy
from tests.test_attacks.test_autodojo import make_attacker
from tests.test_runner.test_evolving import S0_PATH, CreatingJudge, GivingUpVictim, KindMatcher, LoadingGivingUpVictim
from tests.test_runner.test_run import AGENT_SKILLS, MANIFEST, make_config

SUITES = ("banking", "travel")
SMOKE = PairFilter(single_injection_point=True, balance_suites=True)
EACH = PairFilter(single_injection_point=True, balance_suites=True, limit_each_split=True)


def _pairs(units: list[Any], phases: tuple[str, ...]) -> list[str]:
    return [u.pair.id for u in units if u.pair is not None and u.phase in phases and u.trial == 0]


def test_no_split_runs_c2_each_split_pairs_as_one_adapting_phase():
    c4 = build_units(CONDITIONS["C4"], MANIFEST, 1, 20, SUITES, SMOKE, "no-split")
    c2 = build_units(CONDITIONS["C2"], MANIFEST, 1, 20, SUITES, EACH)
    assert {u.phase for u in c4 if u.pair is not None} == {"train"}
    assert _pairs(c4, ("train",)) == _pairs(c2, ("attack",))
    assert {u.split for u in c4 if u.pair is not None} == {"train", "test"}
    # attack-test keeps the split's units; only the attacker on the test phase changes.
    split = build_units(CONDITIONS["C4"], MANIFEST, 1, 20, SUITES, SMOKE)
    assert build_units(CONDITIONS["C4"], MANIFEST, 1, 20, SUITES, SMOKE, "attack-test") == split
    assert sorted(_pairs(split, ("train", "test"))) == sorted(_pairs(c4, ("train",)))


def c4_runner(root: Path, protocol: str, trials: int = 3) -> Runner:
    cond = CONDITIONS["C4"]
    run_dir = run_dir_for(root, cond, 0, 1, c4_protocol=protocol)
    units = build_units(cond, MANIFEST, trials, 1, c4_protocol=protocol)  # type: ignore[arg-type]
    s0 = load_skill_file(S0_PATH)
    s0.description = experiment_models.S0_SKILL_DESCRIPTION
    evolver = SkillFileEvolver(
        s0,
        CreatingJudge(),
        run_dir / "skill_files",
        approx_token_count,
        policy=EditPolicy(description_editable=True),
        trigger=TriggerPolicy(mode="batched", batch_pairs=1, confirm_runs=1, min_losses=2),
        rich_evidence=True,
        rule_match_fn=KindMatcher(),
        library_policy=LibraryPolicy(enabled=True),
        s0_description=experiment_models.S0_SKILL_DESCRIPTION,
        n_train_pairs=len({u.pair.id for u in units if u.pair is not None and u.phase == "train"}),
    )
    attacker, _, _ = make_attacker()
    return Runner(
        config=make_config(
            "C4",
            1,
            trials_per_pair=trials,
            attacker={"name": "autodojo"},
            c4_protocol=None if protocol == "split" else protocol,
        ),
        condition=cond,
        run_dir=run_dir,
        units=units,
        attacker=attacker,
        static_attacker=StaticAttacker(MANIFEST),
        skill_file=EvolvingSkillFile(evolver, "v1.2.2", trials_per_pair=trials),
        victim_factory=LoadingGivingUpVictim,
        echo=lambda _: None,
        agent_skills=AGENT_SKILLS,
        judge_confirm_runs=1,
        judge_evidence=True,
        attack_test_phase=protocol == "attack-test",
    )


def c2_runner(root: Path, trials: int = 3) -> Runner:
    cond = CONDITIONS["C2"]
    attacker, _, _ = make_attacker()
    return Runner(
        config=make_config("C2", 1, trials_per_pair=trials, attacker={"name": "autodojo"}, limit_each_split=True),
        condition=cond,
        run_dir=run_dir_for(root, cond, 0, 1, limit_each_split=True),
        units=build_units(cond, MANIFEST, trials, 1, pair_filter=PairFilter(limit_each_split=True)),
        attacker=attacker,
        static_attacker=StaticAttacker(MANIFEST),
        skill_file=FixedSkillFile(S0_PATH.read_text(encoding="utf-8")),
        victim_factory=GivingUpVictim,
        echo=lambda _: None,
        agent_skills=AGENT_SKILLS,
    )


def test_attack_test_rewrites_test_pairs_against_frozen_skills(tmp_path: Path):
    records = [r for r in c4_runner(tmp_path, "attack-test").run() if r["pair"] is not None]
    test = [r for r in records if r["phase"] == "test"]
    assert test and {r["attack"] for r in test} == {"autodojo"}
    assert len({r["injection_text"] for r in test}) == len(test)  # rewritten every trial
    assert len({r["skill_version"] for r in test}) == 1 and all(r["skill_step"] is None for r in test)

    c2_runner(tmp_path).run()
    result = compare(find_runs(tmp_path, 1))
    assert result["c4_protocol"] == "attack-test" and "C4" not in result["h1"]
    assert result["h1_adaptive"] is not None and result["h1_adaptive"]["asr"]["clusters"] == 1


def test_no_split_judge_learns_from_test_split_pairs(tmp_path: Path):
    records = [r for r in c4_runner(tmp_path, "no-split").run() if r["pair"] is not None]
    assert {r["phase"] for r in records} == {"train"} and {r["attack"] for r in records} == {"autodojo"}
    on_test_split = [r for r in records if r["split"] == "test"]
    assert on_test_split and any(r["skill_step"] is not None for r in on_test_split)

    c2_runner(tmp_path).run()
    result = compare(find_runs(tmp_path, 1))
    assert result["h2"]["overall"]["asr"]["clusters"] == 2  # the train- and the test-split injection task
    assert result["h1_adaptive"] is None and "C4" not in result["h1"]
