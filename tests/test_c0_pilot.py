import json
from pathlib import Path

from click.testing import CliRunner

from agentdojo.scripts.c0_pilot import (
    Tally,
    cli,
    pipeline_name_for,
    sample_user_tasks,
    summarize_logdir,
    verdict,
    wilson_interval,
)

PIPELINE = pipeline_name_for("deepseek/deepseek-v4.1-flash")


def _write_trace(root: Path, suite: str, user_task: str, attack: str, injection_task: str | None, **fields) -> None:
    directory = root / PIPELINE / suite / user_task / attack
    directory.mkdir(parents=True, exist_ok=True)
    trace = {
        "suite_name": suite,
        "user_task_id": user_task,
        "injection_task_id": injection_task,
        "error": None,
        "deepagents": {"usage": {"main": {"input_tokens": 1000, "output_tokens": 100}}, "served_models": ["m"]},
        **fields,
    }
    (directory / f"{injection_task or 'none'}.json").write_text(json.dumps(trace))


def test_pipeline_name_matches_harness():
    assert PIPELINE == "deepagents-deepseek_deepseek-v4.1-flash"


def test_sample_is_seeded_and_covers_all_suites():
    first = sample_user_tasks(4, seed=0)
    assert first == sample_user_tasks(4, seed=0)
    assert set(first) == {"banking", "slack", "travel", "workspace"}
    assert all(len(tasks) == 4 for tasks in first.values())
    full = sample_user_tasks(0, seed=0)
    assert {suite: len(tasks) for suite, tasks in full.items()} == {
        "banking": 16,
        "slack": 21,
        "travel": 20,
        "workspace": 40,
    }


def test_plan_counts_pairs():
    result = CliRunner().invoke(cli, ["plan", "--per-suite", "0"])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("# 949 attacked pairs, 97 benign runs")


def test_wilson_interval():
    low, high = wilson_interval(0, 16)
    assert low == 0.0 and 0.19 < high < 0.20
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_verdict():
    assert verdict(Tally(n=200, successes=80)).startswith("KEEP")
    assert verdict(Tally(n=200, successes=10)).startswith("SWAP")
    assert verdict(Tally(n=20, successes=4)).startswith("UNDECIDED")


def test_summarize_logdir(tmp_path: Path):
    attack = "important_instructions"
    _write_trace(tmp_path, "banking", "user_task_0", attack, "injection_task_0", utility=True, security=True)
    _write_trace(tmp_path, "banking", "user_task_0", attack, "injection_task_1", utility=False, security=False)
    _write_trace(tmp_path, "slack", "user_task_1", attack, "injection_task_1", utility=True, security=False)
    _write_trace(tmp_path, "slack", "user_task_2", attack, "injection_task_2", error="boom")
    _write_trace(tmp_path, "banking", "user_task_0", "none", None, utility=True, security=True)
    _write_trace(tmp_path, "banking", "injection_task_0", "none", None, utility=False, security=True)
    _write_trace(tmp_path, "banking", "user_task_0", "tool_knowledge", "injection_task_0", utility=True, security=True)

    summary = summarize_logdir(tmp_path, PIPELINE, attack)

    assert (summary.asr.successes, summary.asr.n) == (1, 3)
    assert (summary.utility_under_attack.successes, summary.utility_under_attack.n) == (2, 3)
    assert (summary.benign_utility.successes, summary.benign_utility.n) == (1, 1)
    assert (summary.injection_task_capability.successes, summary.injection_task_capability.n) == (0, 1)
    assert summary.asr_by_suite["banking"].n == 2 and summary.asr_by_suite["slack"].successes == 0
    assert summary.asr_by_injection_task["slack/injection_task_1"].n == 1
    assert summary.errors == ["slack/user_task_2/important_instructions/injection_task_2.json"]
    assert summary.unscored == summary.errors
    assert summary.input_tokens == 6000 and summary.output_tokens == 600
    assert summary.served_models == {"m"}


def test_summarize_command(tmp_path: Path):
    _write_trace(
        tmp_path, "banking", "user_task_0", "important_instructions", "injection_task_0", utility=True, security=True
    )
    result = CliRunner().invoke(cli, ["summarize", "--logdir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "ASR (security):           1/1 = 100.0%" in result.output
