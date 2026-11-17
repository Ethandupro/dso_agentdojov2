from agentdojo.runner.stats import cluster_bootstrap, paired_cluster_bootstrap


def rows(spec: dict[str, list[int]]) -> list[dict]:
    return [{"cluster": key, "v": v} for key, values in spec.items() for v in values]


def test_pooled_rate_and_interval():
    data = rows({"a": [1, 1, 1, 0], "b": [0, 0, 0, 0], "c": [1, 0, 1, 0], "d": [1, 1, 0, 0]})
    est = cluster_bootstrap(data, lambda r: r["v"], lambda r: r["cluster"], resamples=2000)
    assert est.value == 7 / 16 and est.n == 16 and est.clusters == 4
    assert est.lo is not None and est.hi is not None and est.value is not None
    assert 0 <= est.lo <= est.value <= est.hi <= 1
    # Fixed seed: the same interval every time.
    again = cluster_bootstrap(data, lambda r: r["v"], lambda r: r["cluster"], resamples=2000)
    assert (again.lo, again.hi) == (est.lo, est.hi)


def test_clustering_widens_the_interval():
    # Same 50% rate; in one layout every cluster is all-or-nothing, so the cluster bootstrap must be wider.
    mixed = rows({str(i): [1, 0] * 5 for i in range(10)})
    clumped = rows({str(i): [i % 2] * 10 for i in range(10)})
    a = cluster_bootstrap(mixed, lambda r: r["v"], lambda r: r["cluster"])
    b = cluster_bootstrap(clumped, lambda r: r["v"], lambda r: r["cluster"])
    assert a.value == b.value == 0.5
    assert a.lo is not None and a.hi is not None and b.lo is not None and b.hi is not None
    assert (b.hi - b.lo) > (a.hi - a.lo)


def test_edge_cases():
    assert cluster_bootstrap([], lambda r: r["v"], lambda r: r["cluster"]).value is None
    one = cluster_bootstrap(rows({"a": [1, 0]}), lambda r: r["v"], lambda r: r["cluster"])
    assert one.value == 0.5 and one.lo is None and one.clusters == 1


def test_paired_difference_uses_shared_clusters_only():
    a = [{**r, "side": "a"} for r in rows({"x": [1, 1], "y": [1, 0], "z": [0, 0], "only_a": [1, 1]})]
    b = [{**r, "side": "b"} for r in rows({"x": [0, 0], "y": [0, 0], "z": [0, 0], "only_b": [1]})]
    d = paired_cluster_bootstrap(a, b, lambda r: r["v"], lambda r: r["cluster"], resamples=2000)
    assert d.clusters == 3 and d.n_a == 6 and d.n_b == 6
    assert d.a == 0.5 and d.b == 0.0 and d.diff == 0.5
    assert d.lo is not None and d.hi is not None and d.diff is not None
    assert 0 <= d.lo <= d.diff <= d.hi <= 1


def test_pairing_removes_shared_spread():
    # A is B plus 0.1 in every cluster, but clusters differ a lot: the paired interval is much tighter than the
    # unpaired intervals and sits on +0.1.
    b = rows({str(i): [1] * i + [0] * (10 - i) for i in range(10)})
    a = rows({str(i): [1] * (i + 1) + [0] * (9 - i) for i in range(10)})
    d = paired_cluster_bootstrap(a, b, lambda r: r["v"], lambda r: r["cluster"])
    assert d.diff is not None and abs(d.diff - 0.1) < 1e-9
    assert d.lo is not None and d.hi is not None
    assert abs(d.lo - 0.1) < 1e-9 and abs(d.hi - 0.1) < 1e-9


def test_paired_edge_cases():
    none = paired_cluster_bootstrap(rows({"x": [1]}), rows({"y": [0]}), lambda r: r["v"], lambda r: r["cluster"])
    assert none.diff is None and none.clusters == 0 and none.n_a == 1
    one = paired_cluster_bootstrap(rows({"x": [1]}), rows({"x": [0]}), lambda r: r["v"], lambda r: r["cluster"])
    assert one.diff == 1.0 and one.lo is None and one.clusters == 1
