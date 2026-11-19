import json
from pathlib import Path

import pytest

pytest.importorskip("deepagents")

from agentdojo.runner.report import summarize_run, write_report
from tests.test_runner.test_run import make_runner


def test_report_shows_victim_skills(tmp_path: Path):
    runner = make_runner(tmp_path, "C1", limit=2)
    runner.run()
    path = write_report(runner.run_dir)
    skills = json.loads((runner.run_dir / "summary.json").read_text())["skills"]
    assert skills["security_skill"] == "tool-output-safety"
    assert skills["s0_load_rate_attacked"] == 1.0 and skills["s0_load_rate_benign"] == 1.0
    assert skills["asr_when_s0_loaded"] == 0.0 and skills["asr_when_s0_not_loaded"] is None
    assert skills["units_loading_each_skill"] == {"tool-output-safety": skills["units"]}
    page = path.read_text()
    assert "Victim skills" in page and "S0 load rate, attacked trials" in page
    assert "&quot;run_id&quot;: &quot;c1_static-s0_static-attack&quot;" in page  # full config shown


def test_report_from_an_external_checker_c1_run(tmp_path: Path):
    runner = make_runner(tmp_path, "C1", limit=4, defense="checker")
    records = runner.run()
    path = write_report(runner.run_dir)
    summary = json.loads((runner.run_dir / "summary.json").read_text())

    attacked = [r for r in records if r["pair"] is not None]
    entry = summary["attacked"]["attack"]
    assert entry["trials"] == len(attacked) == 12
    assert entry["asr"]["value"] == sum(r["security"] for r in attacked) / 12
    assert entry["utility_under_attack"]["value"] == sum(r["utility"] for r in attacked) / 12
    assert set(entry["by_trial"]) == {"0", "1", "2"}
    assert summary["benign"]["benign"]["tasks"] == len(records) - 12
    assert summary["run"]["complete"] and summary["checker"]["calls"] > 0
    assert summary["checker"]["attacked_trials_injection_seen"] == 12
    assert summary["checker"]["injection_fully_removed"] == 12
    assert {row["injection_task"] for row in summary["per_injection_task"]} == {
        f"{r['suite']}/{r['injection_task']}" for r in attacked
    }

    page = path.read_text()
    assert page.startswith("<!DOCTYPE html>")
    assert "<script src" not in page and "<link" not in page  # self-contained
    assert summary["skills"] is None
    assert "<svg" in page and "Per injection task" in page
    assert "Tokens by component" in page
    by_component = summary["cost"]["by_component"]
    assert by_component["attacked_trials"] == 12
    assert by_component["components"]["checker"]["calls"] == summary["checker"]["calls"]
    assert by_component["components"]["judge"]["calls"] == 0  # C1: no judge


def test_partial_run_report_uses_only_checkpointed_units(tmp_path: Path):
    runner = make_runner(tmp_path, "C0", limit=2)
    runner.run(max_units=4)
    with runner.trials_path.open("a") as f:  # a line written after the checkpoint
        f.write(json.dumps({"unit": 4, "phase": "attack"}) + "\n")
    summary = summarize_run(runner.run_dir)
    assert summary["run"]["units_done"] == 4 and not summary["run"]["complete"]
    assert summary["checker"] is None
    write_report(runner.run_dir)
    assert "stopped after unit 4" in (runner.run_dir / "report.html").read_text()
