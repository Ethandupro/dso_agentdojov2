import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from agentdojo.runner.cost import component_usage, cost_summary, load_prices, main

CONFIG = {
    "victim": {"model_id": "Qwen/Qwen3.5-27B"},
    "checker_model": {"model_id": "Qwen/Qwen3.5-9B"},
    "judge": {"model": "deepseek/deepseek-v4.1-flash"},
    "attacker": {"settings": {"attacker": {"model": "mistralai/Mistral-Small"}}},
}
RECORDS = [
    {"unit": 0, "pair": None, "victim_usage": {"main": {"model_calls": 2, "input_tokens": 100, "output_tokens": 10}}},
    {
        "unit": 1,
        "pair": "banking/user_task_0/injection_task_0",
        "victim_usage": {
            "main": {"model_calls": 3, "input_tokens": 300, "output_tokens": 30},
            "summarizer": {"model_calls": 1, "input_tokens": 50, "output_tokens": 5},
        },
        "skill_step": {"judge_calls": 2, "judge_input_tokens": 400, "judge_output_tokens": 40},
    },
    {"unit": 2, "pair": "banking/user_task_0/injection_task_0", "victim_usage": None, "skill_step": None},
]
LINES = [
    {"unit": 1, "input_tokens": 60, "output_tokens": 6},
    {"unit": 1, "input_tokens": None, "output_tokens": None},
]
STATE = {
    "cursor": 3,
    "attacker": {
        "archive": [
            {"pair": "p", "trial": 0, "source": "trial0"},
            {"pair": "p", "trial": 1, "usage": {"calls": 5, "prompt_tokens": 700, "completion_tokens": 70}},
        ]
    },
}


def test_component_usage_adds_each_source():
    usage = component_usage(CONFIG, STATE, RECORDS, LINES)
    assert usage["victim"] == {
        "model": "Qwen/Qwen3.5-27B",
        "calls": 6,
        "input_tokens": 450,
        "output_tokens": 45,
        "total_tokens": 495,
    }
    assert usage["checker"]["calls"] == 2 and usage["checker"]["total_tokens"] == 66
    assert usage["judge"]["calls"] == 2 and usage["judge"]["total_tokens"] == 440
    assert usage["attacker"]["calls"] == 5 and usage["attacker"]["total_tokens"] == 770
    assert usage["attacker"]["model"] == "mistralai/Mistral-Small"


def test_c0_run_has_only_victim_usage():
    config = {"victim": {"model_id": "v"}, "checker_model": None, "judge": None, "attacker": None}
    usage = component_usage(config, {"attacker": {}}, RECORDS[:1], [])
    assert usage["victim"]["total_tokens"] == 110
    for name in ("checker", "judge", "attacker"):
        assert usage[name] == {"model": None, "calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    summary = cost_summary(usage, attacked_trials=0)
    assert summary["defence_overhead"] == 0.0
    assert summary["components"]["victim"]["share_of_tokens"] == 1.0
    assert summary["components"]["victim"]["tokens_per_attacked_trial"] is None


def test_cost_summary_prices_and_overhead():
    usage = component_usage(CONFIG, STATE, RECORDS, LINES)
    prices = {"deepseek/deepseek-v4.1-flash": {"input": 1.0, "output": 10.0}}
    summary = cost_summary(usage, attacked_trials=2, prices=prices)
    assert summary["total_tokens"] == 495 + 66 + 440 + 770
    assert summary["defence_overhead"] == pytest.approx((66 + 440) / 495)
    assert summary["components"]["judge"]["usd"] == pytest.approx((400 * 1.0 + 40 * 10.0) / 1e6)
    assert summary["usd_priced_total"] == summary["components"]["judge"]["usd"]
    assert summary["unpriced"] == ["victim", "checker", "attacker"]
    assert summary["components"]["attacker"]["tokens_per_attacked_trial"] == 385


def test_load_prices_skips_comments_and_unfilled(tmp_path: Path):
    path = tmp_path / "prices.json"
    path.write_text(json.dumps({"_comment": "x", "a": {"input": 1, "output": 2}, "b": {"input": None, "output": None}}))
    assert load_prices(path) == {"a": {"input": 1.0, "output": 2.0}}
    path.write_text(json.dumps({"a": {"input": 1}}))
    with pytest.raises(ValueError):
        load_prices(path)


def test_cli_writes_cost_json_and_ignores_units_past_checkpoint(tmp_path: Path):
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    (tmp_path / "state.json").write_text(json.dumps({**STATE, "cursor": 2}))
    (tmp_path / "trials.jsonl").write_text("\n".join(json.dumps(r) for r in RECORDS))
    (tmp_path / "checker_calls.jsonl").write_text("\n".join(json.dumps(line) for line in LINES))
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps({"deepseek/deepseek-v4.1-flash": {"input": 1.0, "output": 10.0}}))

    result = CliRunner().invoke(main, [str(tmp_path), "--prices", str(prices)])
    assert result.exit_code == 0, result.output
    assert "Defence overhead" in result.output and "Judge" in result.output
    summary = json.loads((tmp_path / "cost.json").read_text())
    assert summary["attacked_trials"] == 1  # unit 2 is past the checkpoint
    assert summary["components"]["victim"]["calls"] == 6
    assert summary["usd_priced_total"] == pytest.approx(0.0008)


def test_example_prices_file_loads():
    path = Path(__file__).parents[2] / "skillfile_evolution" / "prices.example.json"
    assert load_prices(path) == {}
