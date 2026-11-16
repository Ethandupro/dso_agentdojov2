from agentdojo.runner.stats import cluster_bootstrap


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
