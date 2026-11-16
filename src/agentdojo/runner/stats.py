"""Cluster bootstrap confidence intervals (decided 2026-11-16 by the user: bootstrapped over injection tasks).

Trials of one injection task are strongly correlated (same goal, same text in trial 0, and in C3/C4 the split itself
is by injection task), so the resampling unit is the cluster, not the trial: draw clusters with replacement, keep all
of each drawn cluster's trials, recompute the pooled rate. Benign tasks have no injection task; they are clustered by
user task.
"""

from collections.abc import Callable, Hashable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

DEFAULT_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 0
CONFIDENCE = 0.95


@dataclass
class Estimate:
    value: float | None
    """Pooled rate over all trials."""
    lo: float | None
    hi: float | None
    n: int
    """Trials."""
    clusters: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def cluster_bootstrap(
    rows: Sequence[dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
    cluster: Callable[[dict[str, Any]], Hashable],
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = CONFIDENCE,
) -> Estimate:
    """Pooled mean of `value` over `rows`, with a percentile CI from resampling clusters."""
    if not rows:
        return Estimate(None, None, None, 0, 0)
    sums: dict[Hashable, float] = {}
    counts: dict[Hashable, int] = {}
    for row in rows:
        key = cluster(row)
        sums[key] = sums.get(key, 0.0) + float(value(row))
        counts[key] = counts.get(key, 0) + 1
    keys = list(sums)
    s = np.array([sums[k] for k in keys])
    c = np.array([counts[k] for k in keys], dtype=float)
    point = float(s.sum() / c.sum())
    if len(keys) == 1:
        return Estimate(point, None, None, int(c.sum()), 1)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(resamples, len(keys)))
    rates = s[draws].sum(axis=1) / c[draws].sum(axis=1)
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(rates, [alpha, 1 - alpha])
    return Estimate(point, float(lo), float(hi), int(c.sum()), len(keys))


def injection_cluster(row: dict[str, Any]) -> str:
    return f"{row['suite']}/{row['injection_task']}"


def user_task_cluster(row: dict[str, Any]) -> str:
    return f"{row['suite']}/{row['user_task']}"
