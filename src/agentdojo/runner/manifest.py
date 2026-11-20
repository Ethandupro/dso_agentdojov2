"""The fixed pair manifest of the skill-file experiments (`skillfile_evolution/DESIGN.md`, "Splits" and "Attacks").

Built once from the benchmark, committed as `skillfile_evolution/pairs_manifest.json`, and checked by every run before
any model call. It fixes everything that must be identical across conditions:

- the pairs: every (user task, injection task) pair of every suite (decided 2026-11-16 by the user: all 949);
- the split, used only in C3 and C4: 60/40 by injection task, stratified by suite, so every test pair is an attack
  the judge never saw. C0-C2 run every pair and keep each pair's label;
- the order: train pairs shuffled with the fixed seed (suites mixed), then test pairs shuffled the same way. C3 and
  C4 run train then test in this order; C0-C2 run the same sequence;
- the trial-0 injection text of each pair: `important_instructions` addressed to the victim model, placed in every
  injection vector the user task reads;
- the hash of the committed starting skill file S0.

Nothing here depends on the condition or the run seed.
"""

import hashlib
import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.attacks.base_attacks import FixedJailbreakAttack, get_model_name_from_pipeline
from agentdojo.experiment_models import DEFAULT_SKILL_FILE, VICTIM_MODEL_ID
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suites
from agentdojo.types import ChatMessage

MANIFEST_FORMAT = 1
DEFAULT_MANIFEST_PATH = Path("skillfile_evolution/pairs_manifest.json")
BENCHMARK_VERSION = "v1.2.2"
SPLIT_SEED = 0
TRAIN_FRACTION = 0.6
TRIAL0_ATTACK = "important_instructions"

Split = Literal["train", "test"]


def victim_pipeline_name(model_id: str = VICTIM_MODEL_ID) -> str:
    """The victim's pipeline name, as `DeepAgentsConfig.pipeline_name` builds it for the local vLLM server. The attack
    reads the model's name from it ("Qwen" for Qwen3.5-27B)."""
    return "deepagents-local-" + model_id.replace("/", "_")


class _NamedPipeline(BasePipelineElement):
    """Stands in for the victim when building attack texts: attacks only read its name."""

    def __init__(self, name: str) -> None:
        self.name = name

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        raise NotImplementedError("only used for its name")


def task_sort_key(task_id: str) -> tuple[str, int]:
    """`user_task_2` before `user_task_10`."""
    match = re.fullmatch(r"(.*?)(\d+)", task_id)
    if match is None:
        return task_id, -1
    return match.group(1), int(match.group(2))


def allocate_train_counts(sizes: dict[str, int], fraction: float) -> dict[str, int]:
    """Train injection tasks per suite: `round(fraction * total)` overall, shared out by largest remainder.

    Exact arithmetic (0.6 * 14 is 8.4, not 8.399999...), and ties go to the suite that comes first by name, so the
    result doesn't depend on float rounding.
    """
    share = Fraction(str(fraction))
    total = round(share * sum(sizes.values()))
    exact = {suite: share * n for suite, n in sizes.items()}
    counts = {suite: int(value) for suite, value in exact.items()}
    by_remainder = sorted(sizes, key=lambda suite: (-(exact[suite] - counts[suite]), suite))
    for suite in by_remainder[: total - sum(counts.values())]:
        counts[suite] += 1
    return counts


def split_injection_tasks(
    injection_tasks: dict[str, list[str]], seed: int = SPLIT_SEED, fraction: float = TRAIN_FRACTION
) -> dict[str, dict[Split, list[str]]]:
    """Splits each suite's injection tasks into train and test (stratified by suite, fixed seed)."""
    counts = allocate_train_counts({suite: len(ids) for suite, ids in injection_tasks.items()}, fraction)
    split: dict[str, dict[Split, list[str]]] = {}
    for suite in sorted(injection_tasks):
        ids = sorted(injection_tasks[suite], key=task_sort_key)
        random.Random(f"split:{seed}:{suite}").shuffle(ids)
        n_train = counts[suite]
        split[suite] = {
            "train": sorted(ids[:n_train], key=task_sort_key),
            "test": sorted(ids[n_train:], key=task_sort_key),
        }
    return split


def pair_id(suite: str, user_task: str, injection_task: str) -> str:
    return f"{suite}/{user_task}/{injection_task}"


def build_manifest(
    benchmark_version: str = BENCHMARK_VERSION,
    seed: int = SPLIT_SEED,
    train_fraction: float = TRAIN_FRACTION,
    victim_model_id: str = VICTIM_MODEL_ID,
    s0_path: Path = DEFAULT_SKILL_FILE,
) -> dict[str, Any]:
    suites = get_suites(benchmark_version)
    split = split_injection_tasks(
        {name: list(suite.injection_tasks) for name, suite in suites.items()}, seed, train_fraction
    )
    pipeline = _NamedPipeline(victim_pipeline_name(victim_model_id))

    user_names: set[str] = set()
    injection_texts: dict[str, str] = {}
    pairs: dict[Split, list[dict[str, Any]]] = {"train": [], "test": []}
    for suite_name in sorted(suites):
        suite = suites[suite_name]
        attack = load_attack(TRIAL0_ATTACK, suite, pipeline)
        assert isinstance(attack, FixedJailbreakAttack)
        user_names.add(attack.user_name)
        side_of = {inj: side for side in ("train", "test") for inj in split[suite_name][side]}
        for user_task_id in sorted(suite.user_tasks, key=task_sort_key):
            user_task = suite.user_tasks[user_task_id]
            for injection_task_id in sorted(suite.injection_tasks, key=task_sort_key):
                injections = attack.attack(user_task, suite.injection_tasks[injection_task_id])
                texts = set(injections.values())
                if len(texts) != 1:
                    raise ValueError(f"expected one trial-0 text for {user_task_id} x {injection_task_id}")
                text = texts.pop()
                key = f"{suite_name}/{injection_task_id}"
                if injection_texts.setdefault(key, text) != text:
                    raise ValueError(f"trial-0 text of {key} depends on the user task")
                side: Split = side_of[injection_task_id]  # type: ignore[assignment]
                pairs[side].append(
                    {
                        "id": pair_id(suite_name, user_task_id, injection_task_id),
                        "suite": suite_name,
                        "user_task": user_task_id,
                        "injection_task": injection_task_id,
                        "split": side,
                        "vectors": sorted(injections),
                    }
                )

    ordered: list[dict[str, Any]] = []
    for side in ("train", "test"):
        side_pairs = list(pairs[side])
        random.Random(f"order:{seed}:{side}").shuffle(side_pairs)
        ordered.extend(side_pairs)
    for index, pair in enumerate(ordered):
        pair["index"] = index

    s0_bytes = Path(s0_path).read_bytes()
    return {
        "format": MANIFEST_FORMAT,
        "benchmark_version": benchmark_version,
        "split_seed": seed,
        "train_fraction": train_fraction,
        "split_by": "injection_task, stratified by suite",
        "order": "train pairs shuffled with the split seed, then test pairs shuffled the same way",
        "trial0_attack": TRIAL0_ATTACK,
        "victim_model_id": victim_model_id,
        "attack_model_name": get_model_name_from_pipeline(pipeline),
        "attack_user_name": sorted(user_names),
        "s0_path": str(s0_path),
        "s0_sha256": hashlib.sha256(s0_bytes).hexdigest(),
        "counts": {
            "pairs": len(ordered),
            "train_pairs": len(pairs["train"]),
            "test_pairs": len(pairs["test"]),
            "injection_tasks": {side: sum(len(s[side]) for s in split.values()) for side in ("train", "test")},
            "per_suite": {
                suite: {
                    "pairs": sum(p["suite"] == suite for p in ordered),
                    **{k: len(v) for k, v in split[suite].items()},
                }
                for suite in sorted(split)
            },
        },
        "split": split,
        "injection_texts": injection_texts,
        "pairs": ordered,
    }


def dumps_manifest(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=1, ensure_ascii=False) + "\n"


def manifest_sha256(path: Path = DEFAULT_MANIFEST_PATH) -> str:
    """The hash a run records in its `config.json`."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ManifestMismatchError(Exception):
    pass


def check_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    """Rebuilds the manifest from the benchmark and the committed S0 and compares it with the file at `path`.

    Raises `ManifestMismatchError` naming the first differing fields; returns the loaded manifest otherwise. Runs call
    this before any model call, so changed tasks, attack text, victim name or S0 stop the run.
    """
    path = Path(path)
    if not path.exists():
        raise ManifestMismatchError(
            f"{path} does not exist. Build it with `python -m agentdojo.runner.build_manifest`."
        )
    stored = json.loads(path.read_text(encoding="utf-8"))
    rebuilt = build_manifest(
        benchmark_version=stored["benchmark_version"],
        seed=stored["split_seed"],
        train_fraction=stored["train_fraction"],
        victim_model_id=stored["victim_model_id"],
        s0_path=Path(stored["s0_path"]),
    )
    differing = [key for key in sorted(set(stored) | set(rebuilt)) if stored.get(key) != rebuilt.get(key)]
    if differing:
        raise ManifestMismatchError(
            f"{path} does not match the benchmark and S0 as they are now; differing: {differing}"
        )
    return stored


@dataclass(frozen=True)
class Pair:
    index: int
    id: str
    suite: str
    user_task: str
    injection_task: str
    split: Split
    vectors: tuple[str, ...]


def load_pairs(manifest: dict[str, Any]) -> list[Pair]:
    """The pairs in run order."""
    return [
        Pair(
            index=p["index"],
            id=p["id"],
            suite=p["suite"],
            user_task=p["user_task"],
            injection_task=p["injection_task"],
            split=p["split"],
            vectors=tuple(p["vectors"]),
        )
        for p in manifest["pairs"]
    ]


def trial0_injections(manifest: dict[str, Any], pair: Pair) -> dict[str, str]:
    """Trial-0 injections of a pair: the same `important_instructions` text in every vector the user task reads."""
    text = manifest["injection_texts"][f"{pair.suite}/{pair.injection_task}"]
    return {vector: text for vector in pair.vectors}
