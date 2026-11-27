"""The victim's per-model settings (gpt-oss-20b default since 2026-11-27) and the reasoning-effort results directory."""

from pathlib import Path

import pytest

pytest.importorskip("deepagents")

from agentdojo import experiment_models
from agentdojo.runner.conditions import CONDITIONS
from agentdojo.runner.run import run_dir_for
from agentdojo.runner.run_condition import victim_deepagents_config

URL = "http://localhost:8000/v1"


def test_gpt_oss_victim_reasons_at_the_default_effort():
    config = victim_deepagents_config(experiment_models.VICTIM_MODEL_ID, URL)
    assert config.model_id == "openai/gpt-oss-20b"
    assert config.reasoning_effort == experiment_models.VICTIM_REASONING_EFFORT == "medium"
    assert config.send_reasoning_back and config.enable_thinking is None
    assert config.reasoning_system_prompt is None
    assert (config.temperature, config.top_p) == (experiment_models.VICTIM_TEMPERATURE, 1.0)
    assert config.extra_body() == {"reasoning_effort": "medium"}
    assert victim_deepagents_config("openai/gpt-oss-20b", URL, reasoning_effort="high").reasoning_effort == "high"
    with pytest.raises(ValueError, match="effort"):
        victim_deepagents_config("openai/gpt-oss-20b", URL, reasoning_effort="max")


def test_reasoning_effort_is_a_gpt_oss_setting_only():
    with pytest.raises(ValueError, match="gpt-oss"):
        victim_deepagents_config("Qwen/Qwen3.5-27B", URL, reasoning_effort="low")
    qwen = victim_deepagents_config("Qwen/Qwen3.5-27B", URL)
    assert qwen.reasoning_effort is None and not qwen.send_reasoning_back and qwen.enable_thinking is False


def test_non_default_effort_gets_its_own_results_directory(tmp_path: Path):
    cond = CONDITIONS["C0"]
    assert run_dir_for(tmp_path, cond, 0, 20).parent.name == "c0_no-s0_static-attack-limit20"
    assert run_dir_for(tmp_path, cond, 0, 20, victim_reasoning_effort="high").parent.name == (
        "c0_no-s0_static-attack-efforthigh-limit20"
    )
