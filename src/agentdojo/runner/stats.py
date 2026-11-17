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


@dataclass
class Difference:
    """Rate in A minus rate in B, with a paired cluster-bootstrap CI."""

    a: float | None
    b: float | None
    diff: float | None
    lo: float | None
    hi: float | None
    n_a: int
    n_b: int
    clusters: int
    """Clusters present on both sides (the ones resampled)."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def paired_cluster_bootstrap(
    rows_a: Sequence[dict[str, Any]],
    rows_b: Sequence[dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
    cluster: Callable[[dict[str, Any]], Hashable],
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = CONFIDENCE,
) -> Difference:
    """Difference of pooled rates between two conditions run on the same clusters (injection tasks).

    Only clusters present on both sides are used, and each resample draws the same clusters for A and B, so an
    injection task is always compared with itself. That removes the between-task spread both conditions share.
    """

    def per_cluster(rows: Sequence[dict[str, Any]]) -> dict[Hashable, tuple[float, int]]:
        acc: dict[Hashable, tuple[float, int]] = {}
        for row in rows:
            s, c = acc.get(cluster(row), (0.0, 0))
            acc[cluster(row)] = (s + float(value(row)), c + 1)
        return acc

    a, b = per_cluster(rows_a), per_cluster(rows_b)
    keys = [k for k in a if k in b]
    if not keys:
        return Difference(None, None, None, None, None, len(rows_a), len(rows_b), 0)
    sa = np.array([a[k][0] for k in keys])
    ca = np.array([a[k][1] for k in keys], dtype=float)
    sb = np.array([b[k][0] for k in keys])
    cb = np.array([b[k][1] for k in keys], dtype=float)
    rate_a, rate_b = float(sa.sum() / ca.sum()), float(sb.sum() / cb.sum())
    n_a, n_b = int(ca.sum()), int(cb.sum())
    if len(keys) == 1:
        return Difference(rate_a, rate_b, rate_a - rate_b, None, None, n_a, n_b, 1)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(resamples, len(keys)))
    diffs = sa[draws].sum(axis=1) / ca[draws].sum(axis=1) - sb[draws].sum(axis=1) / cb[draws].sum(axis=1)
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(diffs, [alpha, 1 - alpha])
    return Difference(rate_a, rate_b, rate_a - rate_b, float(lo), float(hi), n_a, n_b, len(keys))
