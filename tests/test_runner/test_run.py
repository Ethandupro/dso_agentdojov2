"""The runner end to end with a scripted victim: C0/C1 runs with the victim's skill library (and the external-checker
ablation), per-trial traces, skill loads, checker log keys, the C3/C4 phase hooks, checkpoint and resume, and config
checks."""

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import pytest
from click.testing import CliRunner

pytest.importorskip("deepagents")

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage

from agentdojo import experiment_models
from agentdojo.agent_pipeline.agent_skills import LOAD_SKILL_TOOL_NAME, load_skills_dir
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.deepagents_pipeline import DeepAgentsPipeline, ToolMiddlewareFactory
from agentdojo.attacks.adaptive.autodojo import AutoDojoAttacker
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
AGENT_SKILLS = load_skills_dir(REPO_ROOT / experiment_models.AGENT_SKILLS_DIR)
S0_NAME = experiment_models.S0_SKILL_NAME


class SkillAwareModel(GullibleModel):
    """A `GullibleModel` that, when its system prompt lists the security skill, loads it first and is then never
    hijacked (the scripted stand-in for an agent that reads S0 and follows it)."""

    load_security_skill: bool = True
    loaded: bool = False

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        system = " ".join(str(m.content) for m in messages if isinstance(m, SystemMessage))
        if self.load_security_skill and not self.loaded and f"**{S0_NAME}**" in system:
            self.loaded = True
            self.user_calls = [_call(LOAD_SKILL_TOOL_NAME, {"name": S0_NAME}, "s0"), *self.user_calls]
        if any(
            isinstance(m, ToolMessage) and m.name == LOAD_SKILL_TOOL_NAME and "S0" in str(m.content) for m in messages
        ):
            self.hijacked = True  # never switches to the injection calls
        return super()._generate(messages, *args, **kwargs)


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
        model = SkillAwareModel(
            user_calls=[_call(c.function, dict(c.args), f"u{i}") for i, c in enumerate(user_task.ground_truth(env))],
            injection_calls=injection_calls,
            final=user_task.GROUND_TRUTH_OUTPUT or "Done.",
        )
        pipeline = DeepAgentsPipeline(model, name=self.name, tool_middleware_factory=self.factory)
        return pipeline.query(query, runtime, env, messages, extra_args)


def make_config(condition: str, limit: int | None, defense: str = "skills", **overrides: Any) -> RunConfig:
    cond = CONDITIONS[condition]
    checker = defense == "checker" and cond.s0
    values: dict[str, Any] = {
        "condition": cond.name,
        "run_id": cond.run_id,
        "seed": 0,
        "checker": checker,
        "defense": defense,
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
        "checker_model": {"model_id": "scripted"} if checker else None,
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
    defense: str = "skills",
    **overrides: Any,
) -> Runner:
    cond = CONDITIONS[condition]
    static = StaticAttacker(MANIFEST)
    if skill_file is None:
        skill_file = FixedSkillFile("(S0)") if cond.s0 else NoChecker()
    checker = defense == "checker"
    return Runner(
        config=make_config(condition, limit, defense=defense, **overrides),
        condition=cond,
        run_dir=run_dir_for(tmp_path, cond, 0, limit, external_checker=checker),
        units=build_units(cond, MANIFEST, 3, limit),
        attacker=static,
        static_attacker=static,
        skill_file=skill_file,
        victim_factory=victim_factory,
        checker_fn=reading_checker if checker and cond.s0 else None,
        echo=lambda _: None,
        defense=defense,  # type: ignore[arg-type]
        agent_skills=[] if checker else AGENT_SKILLS,
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_suites_filter_keeps_only_those_suites_and_names_the_dir(tmp_path: Path):
    c2 = build_units(CONDITIONS["C2"], MANIFEST, suites=("banking", "travel"))
    suites = {u.suite for u in c2}
    assert suites == {"banking", "travel"}  # workspace and slack dropped
    attacked = [u for u in c2 if u.pair is not None]
    assert attacked and all(u.pair.suite in ("banking", "travel") for u in attacked if u.pair)
    assert [u.index for u in c2] == list(range(len(c2)))  # indices stay contiguous after filtering
    # The filter combines with --limit and gets its own results directory.
    limited = build_units(CONDITIONS["C2"], MANIFEST, limit=5, suites=("banking",))
    assert len({u.pair.id for u in limited if u.pair}) == 5
    rd = run_dir_for(tmp_path, CONDITIONS["C2"], 0, 5, ("travel", "banking"))
    assert rd.parent.name == f"{CONDITIONS['C2'].run_id}-banking+travel-limit5"


def test_plain_skills_prompt_runs_get_their_own_directory(tmp_path: Path):
    c1 = CONDITIONS["C1"]
    assert run_dir_for(tmp_path, c1, 0, 30).parent.name == "c1_static-s0_static-attack-limit30"
    plain = run_dir_for(tmp_path, c1, 0, 30, skills_prompt_style="plain")
    assert plain.parent.name == "c1_static-s0_static-attack-plain-skills-limit30"


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


def test_c1_victim_lists_and_loads_s0_and_logs_per_trial(tmp_path: Path):
    runner = make_runner(tmp_path, "C1")
    records = runner.run()
    run_dir = runner.run_dir
    assert run_dir == tmp_path / "c1_static-s0_static-attack-limit2" / "seed_0"
    attacked = [r for r in records if r["pair"] is not None]
    assert len(attacked) == 6
    listed = sorted([s.name for s in AGENT_SKILLS] + [S0_NAME])
    for r in records:
        assert r["checker"] is None
        assert r["skills"]["listed"] == listed and r["skills"]["security_skill"] == S0_NAME
        assert r["skills"]["s0_loaded"] is True and r["skills"]["loaded"] == [S0_NAME]
        assert r["agent_actions"][0] == f'{LOAD_SKILL_TOOL_NAME}(name="{S0_NAME}")'
        assert r["skill_version"] == "S0"
        trace = json.loads((run_dir / r["trace"]).read_text())
        system_prompt = trace["deepagents"]["agent_setup"]["main"]["system_prompt"]
        assert system_prompt is not None
        assert LOAD_SKILL_TOOL_NAME in trace["deepagents"]["agent_setup"]["main"]["tools"]
    for r in attacked:
        assert r["security"] is False and r["utility"] is True and r["attacker_success"] is False
    assert not runner.checker_path.read_text()
    config = json.loads((run_dir / "config.json").read_text())
    assert config["defense"] == "skills" and config["checker"] is False


def test_c0_lists_the_fixed_skills_without_s0(tmp_path: Path):
    records = make_runner(tmp_path, "C0", limit=1).run()
    for r in records:
        assert r["skills"]["listed"] == sorted(s.name for s in AGENT_SKILLS)
        assert r["skills"]["security_skill"] is None and r["skills"]["s0_loaded"] is None
        assert r["skills"]["loaded"] == []


def test_external_checker_ablation_removes_injections_and_logs_per_trial(tmp_path: Path):
    runner = make_runner(tmp_path, "C1", defense="checker")
    records = runner.run()
    run_dir = runner.run_dir
    assert run_dir == tmp_path / "c1_static-s0_static-attack-external-checker-limit2" / "seed_0"
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

    assert all(r["skills"] is None for r in records)

    # Running again does nothing.
    assert len(make_runner(tmp_path, "C1", defense="checker").run()) == len(records)


def test_c0_has_no_checker_and_is_hijacked(tmp_path: Path):
    # The scripted victim reaches the injection goal only where the injection task's ground truth works on its own;
    # among the first 4 pairs that is the 4th (user_task_27 x injection_task_5).
    runner = make_runner(tmp_path, "C0", limit=4)
    records = runner.run()
    attacked = [r for r in records if r["pair"] is not None]
    assert len(attacked) == 12 and any(r["security"] for r in attacked)
    assert all(r["checker"] is None and r["skill_version"] == "none" for r in records)
    assert all(r["skills"]["s0_loaded"] is None for r in records)
    assert not (runner.checker_path.read_text())
    for r in attacked:
        trace = json.loads((runner.run_dir / r["trace"]).read_text())
        tool_texts = [str(m["content"]) for m in trace["messages"] if m["role"] == "tool"]
        assert any(INJECTION_START in text for text in tool_texts)


def test_compare_reads_real_run_output(tmp_path: Path):
    # C0 and C1 smoke runs side by side: the (scripted) victim that loads S0 avoids the one hijack C0 has among the
    # first 4 pairs.
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


def test_resume_with_the_external_checker(tmp_path: Path):
    runner = make_runner(tmp_path, "C1", defense="checker")
    assert len(runner.run(max_units=3)) == 3
    records = make_runner(tmp_path, "C1", defense="checker").run()
    assert len(records) == len(runner.units)
    assert {line["unit"] for line in read_jsonl(runner.checker_path)} <= set(range(len(records)))


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
        echo=lambda _: None,
        agent_skills=AGENT_SKILLS,
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


@pytest.mark.parametrize("condition", ["C0E", "C2", "C4"])
@pytest.mark.parametrize(
    ("attacker_flag", "expected_type"),
    [(None, AutoDojoAttacker), ("autodojo", AutoDojoAttacker), ("muzzle", MuzzleAttacker)],
)
def test_cli_wires_local_attacker_for_c0e_c2_and_c4(
    monkeypatch, tmp_path: Path, condition: str, attacker_flag: str | None, expected_type: type
):
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
    if attacker_flag is not None:
        args += ["--attacker", attacker_flag]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output

    wiring = CapturedRunner.last
    attacker = wiring["attacker"]
    assert isinstance(attacker, expected_type) and wiring["static_attacker"].name == "static"
    model = attacker.settings.attacker
    assert (model.model, model.base_url) == (experiment_models.ATTACKER_MODEL_ID, experiment_models.ATTACKER_BASE_URL)
    assert model.api_key_env is None and model.seed == 4
    # Preflight checks the local attacker server like the victim and checker; only C4 pings the judge.
    assert ("attacker", experiment_models.ATTACKER_BASE_URL, experiment_models.ATTACKER_MODEL_ID) in served
    assert judged == ([experiment_models.JUDGE_MODEL_ID] if condition == "C4" else [])
    config = wiring["config"]
    assert config.attacker["name"] == expected_type.name and config.attacker["settings"]["attacker"]["seed"] == 4
    assert config.served_models["attacker"] == [experiment_models.ATTACKER_MODEL_ID]
    assert (config.judge is not None) == (condition == "C4")
    if config.judge is not None:  # C4: the judge may also rewrite S0's description (decided 2026-11-19)
        assert config.judge["policy"]["description_editable"] is True
    assert "sk-or-test" not in json.dumps(config.model_dump(), default=str)
    # The victim's skill library replaces the external checker: no checker server, S0 listed as a skill.
    assert wiring["defense"] == "skills" and wiring["checker_fn"] is None
    assert [s.name for s in wiring["agent_skills"]] == [s.name for s in AGENT_SKILLS]
    assert "checker" not in config.served_models and config.checker is False
    assert config.skills is not None and config.skills["prompt_style"] == "nudge"
    assert wiring["skills_prompt_style"] == "nudge"
    if condition == "C0E":  # C0's victim: the fixed skills only, no S0, against the evolving attacker
        assert config.skills["security_skill"] is None and isinstance(wiring["skill_file"], NoChecker)
        assert wiring["run_dir"].parent.name == "c0e_no-s0_evolving-attack-limit2"
    else:
        assert config.skills["security_skill"]["name"] == S0_NAME


def test_cli_external_checker_ablation(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(REPO_ROOT)
    served: list[str] = []

    def check_served(role: str, url: str, model: str) -> list[str]:
        served.append(role)
        return [model]

    monkeypatch.setattr(run_condition, "check_served", check_served)
    monkeypatch.setattr(run_condition, "Runner", CapturedRunner)
    monkeypatch.setattr(run_condition, "write_report", lambda run_dir: run_dir / "report.html")
    args = ["--condition", "C1", "--results-root", str(tmp_path), "--limit", "2", "--external-checker"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    wiring = CapturedRunner.last
    assert wiring["defense"] == "checker" and wiring["checker_fn"] is not None and wiring["agent_skills"] == []
    assert served == ["victim", "checker"]
    config = wiring["config"]
    assert config.checker is True and config.skills is None
    assert config.s0_path == str(experiment_models.CHECKER_SKILL_FILE)
    assert wiring["skill_file"].markdown == experiment_models.CHECKER_SKILL_FILE.read_text()
    assert wiring["run_dir"].parent.name == "c1_static-s0_static-attack-external-checker-limit2"


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
