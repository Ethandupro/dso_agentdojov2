"""The conditions and the ordered unit list of one run (`skillfile_evolution/DESIGN.md`, "Conditions", "Trial protocol").

A run is a fixed list of units worked through in order: one trial of one pair, or one benign task. The list depends
only on the condition, the manifest, the trials per pair and `--limit`, never on results, so a checkpoint is just the
index of the next unit.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from agentdojo.runner.manifest import Pair, load_pairs, task_sort_key

ConditionName = Literal["C0", "C0E", "C1", "C2", "C3", "C4"]
Phase = Literal["benign", "benign_before", "benign_after", "attack", "train", "test"]
LEARNING_PHASES: tuple[Phase, ...] = ("train", "attack")
"""Attacked phases whose trials the C3/C4 judge learns from: C3's train phase, and C4's single attack phase (C4 has
no train/test split since 2026-11-27)."""


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
        # Run id changed 2026-11-27 (was `c4_evolving-s0_evolving-attack`) when C4 lost its train/test split, so old
        # C4 runs are never resumed or pooled with new ones.
        Condition("C4", "c4_evolving-s0_evolving-attack_live", s0=True, skill_mode="evolving", attack_mode="evolving"),
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


@dataclass(frozen=True)
class PairFilter:
    """Pilot cuts on the manifest's pairs, applied before `--limit` (added 2026-11-20 for the small, fast attacker
    experiments: a few single-injection-point user tasks from two suites)."""

    user_tasks: frozenset[str] | None = None
    """Keep only these user tasks, as `suite/user_task` (from a `--subset` file)."""
    single_injection_point: bool = False
    """Keep only pairs whose user task reads exactly one injection vector."""
    limit_each_split: bool = False
    """C0-C2 with `--limit`: take the first `limit` train pairs and the first `limit` test pairs, as C3 always does
    (`run_condition --limit-each-split`, added 2026-11-27). Without it the first `limit` pairs of the run order are
    taken, and in the smoke order those are all train pairs (the first 136), so C1 has no test pairs to set against
    C3's test phase (H1)."""
    balance_suites: bool = False
    """Not a cut but an order (added 2026-11-20 for `--smoke`): interleave the kept pairs suite by suite (one banking,
    one travel, ...), each suite keeping its run order, so `--limit N` takes about N / suites pairs from each suite
    instead of the first N of the run order."""

    def keep(self, pair: Pair) -> bool:
        if self.user_tasks is not None and f"{pair.suite}/{pair.user_task}" not in self.user_tasks:
            return False
        return not (self.single_injection_point and len(pair.vectors) != 1)


NO_FILTER = PairFilter()


def interleave_by_suite(pairs: list[Pair]) -> list[Pair]:
    """Round-robin over the suites in order of first appearance, each suite keeping its own order."""
    by_suite: dict[str, list[Pair]] = {}
    for pair in pairs:
        by_suite.setdefault(pair.suite, []).append(pair)
    queues = list(by_suite.values())
    out: list[Pair] = []
    for i in range(max((len(q) for q in queues), default=0)):
        out.extend(q[i] for q in queues if i < len(q))
    return out


def load_subset(path: Path, manifest: dict[str, Any]) -> frozenset[str]:
    """The user tasks listed in a `--subset` file: one `suite/user_task` per line; blank lines and `#` comments are
    skipped. Every entry must be a user task of the manifest."""
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry:
            entries.append(entry)
    if not entries:
        raise ValueError(f"{path} lists no user tasks")
    known = {f"{p['suite']}/{p['user_task']}" for p in manifest["pairs"]}
    unknown = [e for e in entries if e not in known]
    if unknown:
        raise ValueError(f"{path} lists user tasks that are not in the manifest: {unknown}")
    return frozenset(entries)


def benign_tasks(
    manifest: dict[str, Any],
    limit: int | None = None,
    suites: tuple[str, ...] | None = None,
    pair_filter: PairFilter = NO_FILTER,
) -> list[tuple[str, str]]:
    """The user tasks (suite, id) of the selected pairs, in suite then task order. With `limit`, the user tasks of
    the first `limit` pairs of the (filtered) run order instead, in that order, so a smoke run stays small."""
    if limit is not None:
        seen: list[tuple[str, str]] = []
        for pair in _select_pairs(manifest, limit, suites, pair_filter):
            if (pair.suite, pair.user_task) not in seen:
                seen.append((pair.suite, pair.user_task))
        return seen
    tasks = {(p.suite, p.user_task) for p in _select_pairs(manifest, None, suites, pair_filter)}
    return sorted(tasks, key=lambda t: (t[0], task_sort_key(t[1])))


def _select_pairs(
    manifest: dict[str, Any],
    limit: int | None,
    suites: tuple[str, ...] | None,
    pair_filter: PairFilter = NO_FILTER,
) -> list[Pair]:
    """Run-order pairs, kept to `suites` and `pair_filter` if given, then cut to the first `limit`."""
    pairs = [p for p in load_pairs(manifest) if (suites is None or p.suite in suites) and pair_filter.keep(p)]
    if pair_filter.balance_suites:
        pairs = interleave_by_suite(pairs)
    return pairs if limit is None else pairs[:limit]


def build_units(
    condition: Condition,
    manifest: dict[str, Any],
    trials_per_pair: int = 3,
    limit: int | None = None,
    suites: tuple[str, ...] | None = None,
    pair_filter: PairFilter = NO_FILTER,
) -> list[Unit]:
    """The run's units in order.

    - C0-C2: benign tasks once (proposed 2026-11-16, DESIGN open question), then every pair x trials in manifest
      order, each pair keeping its train/test label.
    - C3: benign before, train pairs x trials, (freeze), test pairs x trials, benign after.
    - C4 (no split, user decision 2026-11-27): benign before, every pair x trials in C2's order (phase `attack`; the
      judge learns from all of them and the attacker rewrites all of them), (freeze), benign after.

    `suites` keeps only pairs in those suites (a pilot cut, e.g. the C0-vulnerable banking/travel pairs). `limit`
    keeps only the first `limit` pairs of the (suite-filtered) run order (smoke runs), as in C2; in C3 that means the
    first `limit` train pairs and the first `limit` test pairs. `pair_filter` (a `--subset` file and/or
    `--single-injection-point`) is applied together with `suites`, also before `limit`.
    """
    pairs = _select_pairs(manifest, None, suites, pair_filter)

    def order(selected: list[Pair]) -> list[Pair]:
        return interleave_by_suite(selected) if pair_filter.balance_suites else selected

    units: list[Unit] = []

    def add_benign(phase: Phase) -> None:
        for suite, user_task in benign_tasks(manifest, limit, suites, pair_filter):
            units.append(Unit(len(units), phase, suite, user_task))

    def add_pairs(phase: Phase, selected: list[Pair]) -> None:
        for pair in selected:
            for trial in range(trials_per_pair):
                units.append(Unit(len(units), phase, pair.suite, pair.user_task, pair, trial))

    if condition.name == "C4":
        # One adapting phase over C2's pairs in C2's order, so H2 compares the same pairs at every position.
        add_benign("benign_before")
        add_pairs("attack", pairs if limit is None else pairs[:limit])
        add_benign("benign_after")
    elif condition.evolving_skill:
        # Balanced within each split, so a smoke run's train and test cuts both cover every suite.
        train = order([p for p in pairs if p.split == "train"])
        test = order([p for p in pairs if p.split == "test"])
        if limit is not None:
            train, test = train[:limit], test[:limit]
        add_benign("benign_before")
        add_pairs("train", train)
        add_pairs("test", test)
        add_benign("benign_after")
    else:
        add_benign("benign")
        if limit is not None and pair_filter.limit_each_split:
            train = order([p for p in pairs if p.split == "train"])[:limit]
            test = order([p for p in pairs if p.split == "test"])[:limit]
            add_pairs("attack", train + test)
        else:
            add_pairs("attack", pairs if limit is None else pairs[:limit])
    return units
