"""The runner end to end with a scripted victim and checker: C0/C1 runs, per-trial traces, checker log keys, the C3/C4
phase hooks, checkpoint and resume, and config checks."""

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import pytest
from click.testing import CliRunner

pytest.importorskip("deepagents")

from agentdojo import experiment_models
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.deepagents_pipeline import DeepAgentsPipeline, ToolMiddlewareFactory
from agentdojo.attacks.adaptive.muzzle import MuzzleAttacker
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.runner import run_condition
from agentdojo.runner.compare import write_comparison
from agentdojo.runner.conditions import CONDITIONS, Unit, build_units
from agentdojo.runner.config import RunConfig
from agentdojo.runner.manifest import DEFAULT_MANIFEST_PATH, Pair
from agentdojo.runner.plugins import AttemptFeedback, FixedSkillFile, NoChecker, StaticAttacker
from agentdojo.runner.run import ConfigMismatchError, Runner, VictimFactory, run_dir_for
from agentdojo.runner.run_condition import PreflightError, check_judge, main
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.types import ChatMessage
from tests.test_checker.test_checker_middleware import INJECTION_START, GullibleModel, _call, reading_checker

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((REPO_ROOT / DEFAULT_MANIFEST_PATH).read_text(encoding="utf-8"))
BENCHMARK_VERSION = "v1.2.2"


class ScriptedVictim(BasePipelineElement):
    """Does the user task's ground truth; switches to the injection task's ground truth once it sees the injection."""

    def __init__(self, unit: Unit, factory: ToolMiddlewareFactory | None) -> None:
        self.unit = unit
        self.factory = factory
        self.name = "deepagents-local-Qwen_Qwen3.5-27B"

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        suite = get_suite(BENCHMARK_VERSION, self.unit.suite)
        user_task = suite.user_tasks[self.unit.user_task]
        injection_calls = []
        if self.unit.pair is not None:
            injection_task = suite.injection_tasks[self.unit.pair.injection_task]
            injection_calls = [
                _call(c.function, dict(c.args), f"i{i}") for i, c in enumerate(injection_task.ground_truth(env))
            ]
        model = GullibleModel(
            user_calls=[_call(c.function, dict(c.args), f"u{i}") for i, c in enumerate(user_task.ground_truth(env))],
            injection_calls=injection_calls,
            final=user_task.GROUND_TRUTH_OUTPUT or "Done.",
        )
        pipeline = DeepAgentsPipeline(model, name=self.name, tool_middleware_factory=self.factory)
        return pipeline.query(query, runtime, env, messages, extra_args)


def make_config(condition: str, limit: int | None, **overrides: Any) -> RunConfig:
    cond = CONDITIONS[condition]
    values: dict[str, Any] = {
        "condition": cond.name,
        "run_id": cond.run_id,
        "seed": 0,
        "checker": cond.checker,
        "skill_mode": cond.skill_mode,
        "attack_mode": cond.attack_mode,
        "trials_per_pair": 3,
        "limit": limit,
        "benchmark_version": BENCHMARK_VERSION,
        "manifest_path": str(DEFAULT_MANIFEST_PATH),
        "manifest_sha256": "test",
        "s0_path": MANIFEST["s0_path"],
        "s0_sha256": MANIFEST["s0_sha256"],
        "victim": {"model_id": "scripted"},
        "checker_model": {"model_id": "scripted"} if cond.checker else None,
        "served_models": {},
        "code": {},
        "versions": {},
        "created_at": "now",
    }
    values.update(overrides)
    return RunConfig(**values)


def make_runner(
    tmp_path: Path,
    condition: str,
    limit: int = 2,
    victim_factory: VictimFactory = ScriptedVictim,
    skill_file=None,
    **overrides: Any,
) -> Runner:
    cond = CONDITIONS[condition]
    static = StaticAttacker(MANIFEST)
    if skill_file is None:
        skill_file = FixedSkillFile("(S0)") if cond.checker else NoChecker()
    return Runner(
        config=make_config(condition, limit, **overrides),
        condition=cond,
        run_dir=run_dir_for(tmp_path, cond, 0, limit),
        units=build_units(cond, MANIFEST, 3, limit),
        attacker=static,
        static_attacker=static,
        skill_file=skill_file,
        victim_factory=victim_factory,
        checker_fn=reading_checker if cond.checker else None,
        echo=lambda _: None,
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_unit_lists():
    c1 = build_units(CONDITIONS["C1"], MANIFEST)
    assert len(c1) == 97 + 949 * 3
    assert [u.phase for u in c1[:97]] == ["benign"] * 97
    attacked = c1[97:]
    assert [u.pair.id for u in attacked[:3] if u.pair] == [MANIFEST["pairs"][0]["id"]] * 3
    assert [u.trial for u in attacked[:3]] == [0, 1, 2]

    c3 = build_units(CONDITIONS["C3"], MANIFEST)
    phases = [u.phase for u in c3]
    assert phases == ["benign_before"] * 97 + ["train"] * 559 * 3 + ["test"] * 390 * 3 + ["benign_after"] * 97
    assert all(u.split == "train" for u in c3 if u.phase == "train")
    assert all(u.split == "test" for u in c3 if u.phase == "test")
    assert [u.index for u in c3] == list(range(len(c3)))


def test_c1_run_removes_injections_and_logs_per_trial(tmp_path: Path):
    runner = make_runner(tmp_path, "C1")
    records = runner.run()
    run_dir = runner.run_dir
    assert run_dir == tmp_path / "c1_static-skill_static-attack-limit2" / "seed_0"
    assert len(records) == len(runner.units) == len({(p["suite"], p["user_task"]) for p in MANIFEST["pairs"][:2]}) + 6

    attacked = [r for r in records if r["pair"] is not None]
    assert [(r["pair"], r["trial"]) for r in attacked] == [
        (p["id"], t) for p in MANIFEST["pairs"][:2] for t in range(3)
    ]
    for r in attacked:
        assert r["security"] is False and r["utility"] is True and r["attacker_success"] is False
        assert INJECTION_START in r["injection_text"]
        assert r["checker"]["injection_seen_by_checker"] and r["checker"]["injection_survival"] == 0.0
        assert r["skill_version"] == r["skill_version_after"] == "S0"
        assert (run_dir / r["trace"]).exists(), r["trace"]
        assert r["trace"].startswith(f"trial_{r['trial']}/{r['split']}/")
        trace = json.loads((run_dir / r["trace"]).read_text())
        assert trace["runner"]["unit"] == r["unit"] and trace["security"] is False
    for r in records:
        if r["pair"] is None:
            assert r["phase"] == "benign" and r["security"] is None and r["utility"] is True
            assert (run_dir / r["trace"]).exists()

    lines = read_jsonl(run_dir / "checker_calls.jsonl")
    assert lines and {line["unit"] for line in lines} <= {r["unit"] for r in records}
    assert all(line["condition"] == "C1" and line["skill_version"] == "S0" for line in lines)
    state = json.loads((run_dir / "state.json").read_text())
    assert state["complete"] and state["cursor"] == len(records)
    assert json.loads((run_dir / "config.json").read_text())["condition"] == "C1"

    # Running again does nothing.
    assert len(make_runner(tmp_path, "C1").run()) == len(records)


def test_c0_has_no_checker_and_is_hijacked(tmp_path: Path):
    # The scripted victim reaches the injection goal only where the injection task's ground truth works on its own;
    # among the first 4 pairs that is the 4th (user_task_27 x injection_task_5).
    runner = make_runner(tmp_path, "C0", limit=4)
    records = runner.run()
    attacked = [r for r in records if r["pair"] is not None]
    assert len(attacked) == 12 and any(r["security"] for r in attacked)
    assert all(r["checker"] is None and r["skill_version"] == "none" for r in records)
    assert not (runner.checker_path.read_text())
    for r in attacked:
        trace = json.loads((runner.run_dir / r["trace"]).read_text())
        tool_texts = [str(m["content"]) for m in trace["messages"] if m["role"] == "tool"]
        assert any(INJECTION_START in text for text in tool_texts)


def test_compare_reads_real_run_output(tmp_path: Path):
    # C0 and C1 smoke runs side by side: the checker with S0 removes the one hijack C0 has among the first 4 pairs.
    make_runner(tmp_path, "C0", limit=4).run()
    make_runner(tmp_path, "C1", limit=4).run()
    out = write_comparison(tmp_path, tmp_path / "comparison.html", limit=4)
    result = json.loads(out.with_suffix(".json").read_text())
    asr = result["c0_vs_c1"]["asr"]
    assert asr["a"] > 0 and asr["b"] == 0 and asr["diff"] == asr["a"]
    assert asr["n_a"] == asr["n_b"] == 12
    assert result["h1"] == {} and result["h2"] is None


def test_resume_after_stop_and_after_a_crash_between_writes(tmp_path: Path):
    full = make_runner(tmp_path / "full", "C1").run()

    runner = make_runner(tmp_path, "C1")
    assert len(runner.run(max_units=3)) == 3
    # A crash after the record was appended but before the checkpoint: the line must be dropped and the unit rerun.
    with (runner.trials_path).open("a") as f:
        f.write(json.dumps({"unit": 3, "half": True}) + "\n")
    records = make_runner(tmp_path, "C1").run()
    assert [r["unit"] for r in records] == list(range(len(full)))
    assert not any(r.get("half") for r in records)
    assert [(r["unit_id"], r["utility"], r["security"]) for r in records] == [
        (r["unit_id"], r["utility"], r["security"]) for r in full
    ]
    assert {line["unit"] for line in read_jsonl(runner.checker_path)} <= set(range(len(full)))


def test_failed_unit_stops_the_run_with_the_checkpoint_intact(tmp_path: Path):
    calls = {"n": 0}

    def flaky(unit: Unit, factory: ToolMiddlewareFactory | None) -> BasePipelineElement:
        calls["n"] += 1
        if calls["n"] == 2:
            raise ConnectionError("victim server down")
        return ScriptedVictim(unit, factory)

    runner = make_runner(tmp_path, "C1", victim_factory=flaky)
    with pytest.raises(ConnectionError):
        runner.run()
    assert json.loads(runner.state_path.read_text())["cursor"] == 1
    assert len(read_jsonl(runner.trials_path)) == 1
    records = make_runner(tmp_path, "C1").run()
    assert len(records) == len(runner.units)


def test_resume_refuses_a_different_config(tmp_path: Path):
    make_runner(tmp_path, "C1").run(max_units=1)
    with pytest.raises(ConfigMismatchError, match="victim.model_id"):
        make_runner(tmp_path, "C1", victim={"model_id": "other"}).run()
    # Only the creation time may differ.
    make_runner(tmp_path, "C1", created_at="later").run(max_units=1)


class RecordingSkillFile(FixedSkillFile):
    def __init__(self) -> None:
        super().__init__("(S0)")
        self.frozen_at: int | None = None
        self.seen: list[tuple[str, int]] = []

    def after_trial(self, record: dict[str, Any]) -> dict[str, Any] | None:
        self.seen.append((record["phase"], record["unit"]))
        return None

    def freeze(self) -> None:
        if self.frozen_at is None:
            self.frozen_at = len(self.seen)


class RecordingAttacker(StaticAttacker):
    name = "recording"

    def __init__(self, manifest: dict[str, Any]) -> None:
        super().__init__(manifest)
        self.calls: list[tuple[str, int, int]] = []

    def text_for(self, pair: Pair, trial: int, history: list[AttemptFeedback]) -> str:
        self.calls.append((pair.id, trial, len(history)))
        assert [h.trial for h in history] == list(range(trial))
        return super().text_for(pair, trial, history)


def test_c4_phases_freeze_and_keep_test_pairs_from_the_attacker(tmp_path: Path):
    cond = CONDITIONS["C4"]
    skill_file = RecordingSkillFile()
    attacker = RecordingAttacker(MANIFEST)
    runner = Runner(
        config=make_config("C4", 1),
        condition=cond,
        run_dir=run_dir_for(tmp_path, cond, 0, 1),
        units=build_units(cond, MANIFEST, 3, 1),
        attacker=attacker,
        static_attacker=StaticAttacker(MANIFEST),
        skill_file=skill_file,
        victim_factory=ScriptedVictim,
        checker_fn=reading_checker,
        echo=lambda _: None,
    )
    records = runner.run()
    phases = [r["phase"] for r in records]
    assert phases == ["benign_before", "train", "train", "train", "test", "test", "test", "benign_after"]
    # Frozen before the first test unit; the evolving attacker saw only the train pair, with its own history.
    assert skill_file.frozen_at == 4
    train_pair = records[1]["pair"]
    assert attacker.calls == [(train_pair, 0, 0), (train_pair, 1, 1), (train_pair, 2, 2)]
    assert [r["attack"] for r in records if r["pair"]] == ["recording"] * 3 + ["static"] * 3
    assert skill_file.seen == [(p, i) for i, p in enumerate(phases)]


def test_cli_preflight(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    result = CliRunner().invoke(main, ["--condition", "C1", "--victim-base-url", "http://127.0.0.1:9/v1"])
    assert result.exit_code == 1 and "did not answer" in result.output


class CapturedRunner:
    """Stands in for `Runner` in the CLI: keeps the wiring, runs nothing."""

    last: ClassVar[dict[str, Any]] = {}

    def __init__(self, **kwargs: Any) -> None:
        CapturedRunner.last = kwargs
        self.run_dir = kwargs["run_dir"]

    def run(self) -> list[dict]:
        return []


@pytest.mark.parametrize("condition", ["C2", "C4"])
def test_cli_wires_local_muzzle_for_c2_and_c4(monkeypatch, tmp_path: Path, condition: str):
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")  # C4's judge only
    served: list[tuple[str, str, str]] = []
    judged: list[str] = []

    def check_served(role: str, url: str, model: str) -> list[str]:
        served.append((role, url, model))
        return [model]

    monkeypatch.setattr(run_condition, "check_served", check_served)
    monkeypatch.setattr(run_condition, "check_judge", judged.append)
    monkeypatch.setattr(run_condition, "Runner", CapturedRunner)
    monkeypatch.setattr(run_condition, "write_report", lambda run_dir: run_dir / "report.html")
    monkeypatch.setattr(run_condition, "vllm_token_counter", lambda url, model: len)
    args = ["--condition", condition, "--results-root", str(tmp_path), "--limit", "2", "--seed", "4"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output

    wiring = CapturedRunner.last
    attacker = wiring["attacker"]
    assert isinstance(attacker, MuzzleAttacker) and wiring["static_attacker"].name == "static"
    model = attacker.settings.attacker
    assert (model.model, model.base_url) == (experiment_models.ATTACKER_MODEL_ID, experiment_models.ATTACKER_BASE_URL)
    assert model.api_key_env is None and model.seed == 4
    # Preflight checks the local attacker server like the victim and checker; only C4 pings the judge.
    assert ("attacker", experiment_models.ATTACKER_BASE_URL, experiment_models.ATTACKER_MODEL_ID) in served
    assert judged == ([] if condition == "C2" else [experiment_models.JUDGE_MODEL_ID])
    config = wiring["config"]
    assert config.attacker["name"] == "muzzle" and config.attacker["settings"]["attacker"]["seed"] == 4
    assert config.served_models["attacker"] == [experiment_models.ATTACKER_MODEL_ID]
    assert (config.judge is not None) == (condition == "C4")
    assert "sk-or-test" not in json.dumps(config.model_dump(), default=str)


def test_judge_preflight_and_env_file_override(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(PreflightError, match="OPENROUTER_API_KEY is not set"):
        check_judge("deepseek/deepseek-v4.1-flash")

    # The env file wins over a key already exported in the shell.
    monkeypatch.setenv("OPENROUTER_API_KEY", "old")
    env_file = tmp_path / "api.env"
    env_file.write_text("OPENROUTER_API_KEY=sk-or-new\n")
    monkeypatch.chdir(REPO_ROOT)
    args = ["--condition", "C3", "--env-file", str(env_file), "--victim-base-url", "http://127.0.0.1:9/v1"]
    CliRunner().invoke(main, args)
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-new"
