from agentdojo.runner.report import REFUSAL_WARN_RATE, attacker_metrics


def rewrite(source: str, jailbroken: bool = False, empty: int = 0, truncated: int = 0) -> dict:
    return {
        "source": source,
        "pair_result": {"is_jailbroken": jailbroken},
        "usage": {"calls": 4, "empty_replies": empty, "truncated_replies": truncated},
    }


def test_no_archive_means_no_attacker_section():
    assert attacker_metrics({}) is None
    assert attacker_metrics({"attacker": {}}) is None  # static attacker: empty state


def test_counts_fallbacks_over_rewrites_only():
    archive = [
        {"source": "trial0"},
        rewrite("pair", jailbroken=True),
        rewrite("goal_fallback", empty=2),
        rewrite("pair", truncated=1),
        {"source": "trial0"},
    ]
    m = attacker_metrics({"attacker": {"archive": archive}})
    assert m is not None
    assert m["rewrites"] == 3 and m["goal_fallbacks"] == 1
    assert m["refusal_rate"] == 1 / 3 and m["refusal_warning"]
    assert m["jailbroken_per_pair_judge"] == 1
    assert (m["model_calls"], m["empty_replies"], m["truncated_replies"]) == (12, 2, 1)


def test_warning_only_above_threshold():
    n = 100
    fallbacks = int(REFUSAL_WARN_RATE * n)  # exactly at the threshold: no warning
    archive = [rewrite("goal_fallback")] * fallbacks + [rewrite("pair")] * (n - fallbacks)
    m = attacker_metrics({"attacker": {"archive": archive}})
    assert m is not None and not m["refusal_warning"]
    m = attacker_metrics({"attacker": {"archive": [*archive, rewrite("goal_fallback")]}})
    assert m is not None and m["refusal_warning"]


def test_only_trial0_entries_gives_no_rate():
    m = attacker_metrics({"attacker": {"archive": [{"source": "trial0"}]}})
    assert m is not None and m["rewrites"] == 0 and m["refusal_rate"] is None and not m["refusal_warning"]
