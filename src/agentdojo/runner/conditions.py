"""The conditions and the ordered unit list of one run (`skillfile_evolution/DESIGN.md`, "Conditions", "Trial protocol").

A run is a fixed list of units worked through in order: one trial of one pair, or one benign task. The list depends
only on the condition, the manifest, the trials per pair and `--limit`, never on results, so a checkpoint is just the
index of the next unit.
"""

from dataclasses import dataclass
from typing import Any, Literal

from agentdojo.runner.manifest import Pair, load_pairs, task_sort_key

ConditionName = Literal["C0", "C0E", "C1", "C2", "C3", "C4"]
Phase = Literal["benign", "benign_before", "benign_after", "attack", "train", "test"]


@dataclass(frozen=True)
class Condition:
    name: ConditionName
    run_id: str
    s0: bool
    """Whether the security skill S0 is in play: in the victim's skill list (default), or given to the external
    checker (`--external-checker` ablation). C0 has no S0."""
    skill_mode: Literal["none", "static", "evolving"]
    attack_mode: Literal["static", "evolving"]

    @property
    def evolving_skill(self) -> bool:
        return self.skill_mode == "evolving"


CONDITIONS: dict[str, Condition] = {
    c.name: c
    for c in [
        Condition("C0", "c0_no-s0_static-attack", s0=False, skill_mode="none", attack_mode="static"),
        # Added 2026-11-20 by the user: C0's victim (5 general skills, no S0) against C2's evolving attacker, so C2
        # vs C0E shows what S0 adds under adaptive attack.
        Condition("C0E", "c0e_no-s0_evolving-attack", s0=False, skill_mode="none", attack_mode="evolving"),
        Condition("C1", "c1_static-s0_static-attack", s0=True, skill_mode="static", attack_mode="static"),
        Condition("C2", "c2_static-s0_evolving-attack", s0=True, skill_mode="static", attack_mode="evolving"),
        Condition("C3", "c3_evolving-s0_static-attack", s0=True, skill_mode="evolving", attack_mode="static"),
        Condition("C4", "c4_evolving-s0_evolving-attack", s0=True, skill_mode="evolving", attack_mode="evolving"),
    ]
}


@dataclass(frozen=True)
class Unit:
    index: int
    phase: Phase
    suite: str
    user_task: str
    pair: Pair | None = None
    """None for benign units."""
    trial: int | None = None

    @property
    def id(self) -> str:
        if self.pair is None:
            return f"{self.phase}/{self.suite}/{self.user_task}"
        return f"{self.phase}/{self.pair.id}/trial_{self.trial}"

    @property
    def split(self) -> str | None:
        return None if self.pair is None else self.pair.split


def benign_tasks(
    manifest: dict[str, Any], limit: int | None = None, suites: tuple[str, ...] | None = None
) -> list[tuple[str, str]]:
    """Every user task (suite, id), in suite then task order. With `suites`, only those suites. With `limit`, the
    user tasks of the first `limit` pairs of the (suite-filtered) run order instead, so a smoke run stays small."""
    if limit is not None:
        seen: list[tuple[str, str]] = []
        for pair in _select_pairs(manifest, limit, suites):
            if (pair.suite, pair.user_task) not in seen:
                seen.append((pair.suite, pair.user_task))
        return seen
    tasks = {(p["suite"], p["user_task"]) for p in manifest["pairs"] if suites is None or p["suite"] in suites}
    return sorted(tasks, key=lambda t: (t[0], task_sort_key(t[1])))


def _select_pairs(manifest: dict[str, Any], limit: int | None, suites: tuple[str, ...] | None) -> list[Pair]:
    """Run-order pairs, kept to `suites` if given, then cut to the first `limit`."""
    pairs = [p for p in load_pairs(manifest) if suites is None or p.suite in suites]
    return pairs if limit is None else pairs[:limit]


def build_units(
    condition: Condition,
    manifest: dict[str, Any],
    trials_per_pair: int = 3,
    limit: int | None = None,
    suites: tuple[str, ...] | None = None,
) -> list[Unit]:
    """The run's units in order.

    - C0-C2: benign tasks once (proposed 2026-11-16, DESIGN open question), then every pair x trials in manifest
      order, each pair keeping its train/test label.
    - C3/C4: benign before, train pairs x trials, (freeze), test pairs x trials, benign after.

    `suites` keeps only pairs in those suites (a pilot cut, e.g. the C0-vulnerable banking/travel pairs). `limit`
    keeps only the first `limit` pairs of the (suite-filtered) run order (smoke runs); in C3/C4 that means the first
    `limit` train pairs and the first `limit` test pairs.
    """
    pairs = _select_pairs(manifest, None, suites)
    units: list[Unit] = []

    def add_benign(phase: Phase) -> None:
        for suite, user_task in benign_tasks(manifest, limit, suites):
            units.append(Unit(len(units), phase, suite, user_task))

    def add_pairs(phase: Phase, selected: list[Pair]) -> None:
        for pair in selected:
            for trial in range(trials_per_pair):
                units.append(Unit(len(units), phase, pair.suite, pair.user_task, pair, trial))

    if condition.evolving_skill:
        train = [p for p in pairs if p.split == "train"]
        test = [p for p in pairs if p.split == "test"]
        if limit is not None:
            train, test = train[:limit], test[:limit]
        add_benign("benign_before")
        add_pairs("train", train)
        add_pairs("test", test)
        add_benign("benign_after")
    else:
        add_benign("benign")
        add_pairs("attack", pairs if limit is None else pairs[:limit])
    return units
