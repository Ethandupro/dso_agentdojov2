import json
import os
from collections import defaultdict
from pathlib import Path

import pytest

from agentdojo.attacks.attack_registry import load_attack
from agentdojo.runner.manifest import (
    DEFAULT_MANIFEST_PATH,
    ManifestMismatchError,
    _NamedPipeline,
    allocate_train_counts,
    build_manifest,
    check_manifest,
    dumps_manifest,
    load_pairs,
    trial0_injections,
    victim_pipeline_name,
)
from agentdojo.task_suite.load_suites import get_suites

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def manifest():
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        yield build_manifest()
    finally:
        os.chdir(cwd)


@pytest.fixture
def in_repo_root(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)


def test_committed_manifest_is_up_to_date(in_repo_root, manifest):
    assert DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8") == dumps_manifest(manifest)


def test_every_pair_once(manifest):
    suites = get_suites("v1.2.2")
    expected = {
        f"{name}/{user}/{inj}"
        for name, suite in suites.items()
        for user in suite.user_tasks
        for inj in suite.injection_tasks
    }
    ids = [p["id"] for p in manifest["pairs"]]
    assert len(ids) == len(set(ids)) == len(expected) == 949
    assert set(ids) == expected
    assert [p["index"] for p in manifest["pairs"]] == list(range(949))


def test_split_is_by_injection_task(manifest):
    sides = defaultdict(set)
    for pair in manifest["pairs"]:
        sides[(pair["suite"], pair["injection_task"])].add(pair["split"])
    assert all(len(s) == 1 for s in sides.values())
    for suite, split in manifest["split"].items():
        assert not set(split["train"]) & set(split["test"])
        for side in ("train", "test"):
            for inj in split[side]:
                assert sides[(suite, inj)] == {side}
    assert manifest["counts"]["injection_tasks"] == {"train": 21, "test": 14}


def test_allocation_is_exact_and_stratified():
    counts = allocate_train_counts({"banking": 9, "slack": 5, "travel": 7, "workspace": 14}, 0.6)
    assert counts == {"banking": 6, "slack": 3, "travel": 4, "workspace": 8}


def test_train_runs_before_test_with_suites_mixed(manifest):
    splits = [p["split"] for p in manifest["pairs"]]
    n_train = manifest["counts"]["train_pairs"]
    assert splits == ["train"] * n_train + ["test"] * (949 - n_train)
    assert len({p["suite"] for p in manifest["pairs"][:20]}) > 1


def test_trial0_text_is_important_instructions_for_the_victim(manifest):
    suites = get_suites("v1.2.2")
    pipeline = _NamedPipeline(victim_pipeline_name())
    for pair in load_pairs(manifest)[::97]:
        suite = suites[pair.suite]
        attack = load_attack("important_instructions", suite, pipeline)
        expected = attack.attack(suite.user_tasks[pair.user_task], suite.injection_tasks[pair.injection_task])
        assert trial0_injections(manifest, pair) == expected
    assert manifest["attack_model_name"] == "Qwen"


def test_check_detects_a_changed_manifest(in_repo_root, manifest, tmp_path):
    changed = json.loads(dumps_manifest(manifest))
    changed["pairs"][0]["split"] = "test" if changed["pairs"][0]["split"] == "train" else "train"
    path = tmp_path / "pairs_manifest.json"
    path.write_text(dumps_manifest(changed), encoding="utf-8")
    with pytest.raises(ManifestMismatchError, match="pairs"):
        check_manifest(path)
    path.write_text(dumps_manifest(manifest), encoding="utf-8")
    assert check_manifest(path) == json.loads(dumps_manifest(manifest))
