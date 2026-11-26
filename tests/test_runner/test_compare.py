"""The cross-run comparison on hand-made run directories: which rows feed H1, H2 and C0 vs C1, partial runs, and the
refusal to mix manifests."""

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from agentdojo.runner.compare import (
    RunMismatchError,
    compare,
    find_runs,
    main,
    position_bins,
    render_html,
    write_comparison,
)
from agentdojo.runner.conditions import CONDITIONS

TRAIN_INJ = ["injection_task_1", "injection_task_2", "injection_task_3"]
TEST_INJ = ["injection_task_4", "injection_task_5", "injection_task_6"]
USER_TASKS = ["user_task_0", "user_task_1"]


def attacked(phase: str, split: str, inj: str, trial: int, security: bool, utility: bool = True) -> dict[str, Any]:
    return {
        "phase": phase,
        "split": split,
        "suite": "banking",
        "user_task": "user_task_0",
        "injection_task": inj,
        "pair": f"banking/user_task_0/{inj}",
        "trial": trial,
        "security": security,
        "utility": utility,
        "attacker_success": security or not utility,
    }


def benign(phase: str, user_task: str, utility: bool) -> dict[str, Any]:
    return {
        "phase": phase,
        "split": None,
        "suite": "banking",
        "user_task": user_task,
        "injection_task": None,
        "pair": None,
        "trial": None,
        "security": None,
        "utility": utility,
        "attacker_success": None,
    }


def write_run(
    root: Path, condition: str, records: list[dict[str, Any]], manifest: str = "m", complete: bool = True, **cfg: Any
) -> Path:
    run_dir = root / CONDITIONS[condition].run_id / "seed_0"
    run_dir.mkdir(parents=True)
    config = {"condition": condition, "seed": 0, "limit": None, "manifest_sha256": manifest, "trials_per_pair": 3}
    config.update(cfg)
    (run_dir / "config.json").write_text(json.dumps(config))
    rows = [{**r, "unit": i} for i, r in enumerate(records)]
    (run_dir / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    n = len(rows) if complete else len(rows) - 1
    (run_dir / "state.json").write_text(json.dumps({"cursor": n, "n_units": len(rows), "complete": complete}))
    return run_dir


def fixed_attack_run(security: bool, phase: str = "attack") -> list[dict[str, Any]]:
    out = [benign("benign", u, True) for u in USER_TASKS]
    for split, injs in (("train", TRAIN_INJ), ("test", TEST_INJ)):
        out += [attacked(phase, split, inj, t, security) for inj in injs for t in range(3)]
    return out


def evolving_run(train_security: list[bool], test_security: bool, benign_after: bool) -> list[dict[str, Any]]:
    out = [benign("benign_before", u, True) for u in USER_TASKS]
    out += [attacked("train", "train", inj, t, train_security[t]) for inj in TRAIN_INJ for t in range(3)]
    out += [attacked("test", "test", inj, t, test_security) for inj in TEST_INJ for t in range(3)]
    return out + [benign("benign_after", u, benign_after) for u in USER_TASKS]


def test_h1_h2_and_c0_vs_c1(tmp_path: Path):
    write_run(tmp_path, "C0", fixed_attack_run(True))
    write_run(tmp_path, "C1", fixed_attack_run(True))
    write_run(tmp_path, "C2", fixed_attack_run(True))
    # C3: blocks every test attack but fails both benign tasks after. C4: the attacker wins only on trial 0.
    write_run(tmp_path, "C3", evolving_run([True, True, True], False, False))
    write_run(tmp_path, "C4", evolving_run([True, False, False], True, True))
    result = compare(find_runs(tmp_path))

    c3 = result["h1"]["C3"]
    asr = c3["test_vs_c1"]["asr"]
    # Compared on test-split injection tasks only: C1's train rows are left out.
    assert (asr["a"], asr["b"], asr["diff"], asr["clusters"], asr["n_b"]) == (0.0, 1.0, -1.0, 3, 9)
    assert c3["benign_utility_after_vs_before"]["diff"] == -1.0
    assert result["h1"]["C4"]["test_vs_c1"]["asr"]["diff"] == 0.0
    assert result["c1_test_asr"]["value"] == 1.0

    # H2: C2's train-split attack rows against C4's train phase, never test pairs.
    h2 = result["h2"]
    assert h2["overall"]["asr"]["n_a"] == h2["overall"]["asr"]["n_b"] == 9
    assert abs(h2["overall"]["asr"]["diff"] - 2 / 3) < 1e-9
    assert {t: v["asr"]["diff"] for t, v in h2["by_trial"].items()} == {"0": 0.0, "1": 1.0, "2": 1.0}
    # By run position: 3 train pairs give 3 bins of one pair each; C2 always wins, C4 only on trial 0.
    assert [(b["pairs"], b["c2_asr"]["value"]) for b in h2["by_position"]] == [(1, 1.0)] * 3
    assert all(abs(b["asr"]["diff"] - 2 / 3) < 1e-9 for b in h2["by_position"])

    assert result["c0_vs_c1"]["asr"]["diff"] == 0.0 and result["c0_vs_c1"]["asr"]["clusters"] == 6
    assert result["missing"] == []
    page = render_html(result)
    assert "ASR over the run" in page and "bin 3 (1 pairs)" in page and "ASR by trial index" in page


def test_missing_conditions_and_partial_runs(tmp_path: Path):
    write_run(tmp_path, "C1", fixed_attack_run(True), complete=False)
    write_run(tmp_path, "C3", evolving_run([True] * 3, False, True))
    runs = find_runs(tmp_path)
    c1 = next(r for r in runs if r["config"]["condition"] == "C1")
    # Rows past the checkpoint cursor are not counted.
    assert len(c1["records"]) == len(fixed_attack_run(True)) - 1 and not c1["complete"]
    result = compare(runs)
    assert set(result["h1"]) == {"C3"}
    assert result["h2"] is None and result["c0_vs_c1"] is None
    assert result["missing"] == ["H1 C4 vs C1 needs both runs", "H2 needs C2 and C4 runs", "C0 vs C1 needs both runs"]

    out = write_comparison(tmp_path, tmp_path / "comparison.html")
    page = out.read_text()
    assert "Some runs are partial" in page and "Not available yet: needs C2 and C4" in page
    assert json.loads(out.with_suffix(".json").read_text())["h1"]["C3"]["test_vs_c1"]["asr"]["diff"] == -1.0


def test_refuses_mixed_manifests_and_filters_on_limit(tmp_path: Path):
    write_run(tmp_path, "C1", fixed_attack_run(True), manifest="a")
    write_run(tmp_path, "C3", evolving_run([True] * 3, False, True), manifest="b")
    with pytest.raises(RunMismatchError, match="same manifest"):
        find_runs(tmp_path)
    result = CliRunner().invoke(main, [str(tmp_path)])
    assert result.exit_code != 0 and "same manifest" in result.output

    smoke = tmp_path / "smoke"
    write_run(smoke, "C1", fixed_attack_run(True), limit=5)
    assert find_runs(smoke) == []
    assert len(find_runs(smoke, limit=5)) == 1
    ok = CliRunner().invoke(main, [str(smoke), "--limit", "5"])
    assert ok.exit_code == 0, ok.output
    assert (smoke / "comparison-limit5.html").exists() and (smoke / "comparison-limit5.json").exists()


def test_refuses_mixed_skill_loading(tmp_path: Path):
    write_run(tmp_path, "C1", fixed_attack_run(True))  # before 2026-11-26: no key, i.e. "index"
    write_run(tmp_path, "C3", evolving_run([True] * 3, False, True), skills={"loading": "prompt"})
    with pytest.raises(RunMismatchError, match="--skill-loading"):
        find_runs(tmp_path)
    same = tmp_path / "same"
    write_run(same, "C1", fixed_attack_run(True), skills={"loading": "prompt"})
    write_run(same, "C3", evolving_run([True] * 3, False, True), skills={"loading": "prompt"})
    assert len(find_runs(same)) == 2


def test_position_bins_follow_run_order():
    rows = [{"unit": 100 - i, "pair": f"p{i}"} for i in range(25)]  # written in reverse unit order
    bins = position_bins(rows)
    assert [len(b) for b in bins] == [2, 3, 2, 3, 2, 3, 2, 3, 2, 3]
    assert bins[0] == ["p24", "p23"] and bins[-1][-1] == "p0"
    assert position_bins(rows[:3]) == [["p2"], ["p1"], ["p0"]]
