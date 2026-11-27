"""Cross-run comparison: the hypotheses, computed from finished per-condition runs (DESIGN.md, "Conditions and result
segmentation": cross-condition comparisons come from the per-condition results, never from mixed logs).

    python -m agentdojo.runner.compare results/ --out results/comparison.html

It reads every `results/<run_id>/seed_<n>/` run (full runs only; `--limit N` compares smoke runs instead), refuses to
mix runs built from different manifests, pools the seeds of each condition, and computes:

- H1: test-phase ASR on the final file in C3 and C4 against C1 on the same test pairs, with benign utility before
  and after evolution.
- H2: C2 against C4 on train-split injection tasks only (C4's attacker never rewrites test pairs), overall and by
  position in the run (train pairs in run order, 10 bins; the main curve) and by trial index 0/1/2.
- C0 against C1: what the checker with S0 adds.

Differences are A minus B with 95% paired cluster-bootstrap intervals over injection tasks (benign: user tasks).
It only reads results and never runs anything.
"""

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import click

from agentdojo.runner.report import CSS, _e, _pct, _point_whisker_chart
from agentdojo.runner.stats import (
    CONFIDENCE,
    DEFAULT_RESAMPLES,
    cluster_bootstrap,
    injection_cluster,
    paired_cluster_bootstrap,
    user_task_cluster,
)

Row = dict[str, Any]


class RunMismatchError(Exception):
    pass


def _read_jsonl(path: Path) -> list[Row]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_run(run_dir: Path) -> dict[str, Any]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    cursor = state.get("cursor", 0)
    return {
        "dir": str(run_dir),
        "config": config,
        "complete": bool(state.get("complete")),
        "units_done": cursor,
        "units_total": state.get("n_units"),
        "records": [r for r in _read_jsonl(run_dir / "trials.jsonl") if r["unit"] < cursor],
    }


def find_runs(results_root: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Every run under `results_root` with the given `--limit` (None: full runs), checked to share one manifest."""
    runs = []
    for config_path in sorted(results_root.glob("*/seed_*/config.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("limit") != limit:
            continue
        runs.append(load_run(config_path.parent))
    keys = {(r["config"]["manifest_sha256"], r["config"]["trials_per_pair"]) for r in runs}
    if len(keys) > 1:
        detail = "\n  ".join(
            f"{r['dir']}: manifest {r['config']['manifest_sha256'][:12]}, {r['config']['trials_per_pair']} trials/pair"
            for r in runs
        )
        raise RunMismatchError(f"these runs were not built from the same manifest and trial count:\n  {detail}")
    loadings = {skill_loading(r["config"]) for r in runs}
    if len(loadings) > 1:
        detail = "\n  ".join(f"{r['dir']}: --skill-loading {skill_loading(r['config'])}" for r in runs)
        raise RunMismatchError(
            "these runs use different --skill-loading settings, so their victims saw different skills prompts. "
            f"Re-run with one setting, or move the other runs aside:\n  {detail}"
        )
    return runs


def skill_loading(config: dict[str, Any]) -> str:
    """`--skill-loading` of a run; runs from before 2026-11-26 (no key) used the default, `index`."""
    return (config.get("skills") or {}).get("loading", "index")


def pooled(runs: list[dict[str, Any]], condition: str) -> list[Row]:
    return [r for run in runs if run["config"]["condition"] == condition for r in run["records"]]


def _select(rows: Sequence[Row], phases: tuple[str, ...], split: str | None = None) -> list[Row]:
    return [r for r in rows if r["phase"] in phases and (split is None or r["split"] == split)]


def _diff(a: Sequence[Row], b: Sequence[Row], value: Callable[[Row], float], benign: bool = False) -> dict[str, Any]:
    return paired_cluster_bootstrap(a, b, value, user_task_cluster if benign else injection_cluster).to_dict()


def _asr(rows: Sequence[Row]) -> dict[str, Any]:
    return cluster_bootstrap(rows, lambda r: r["security"], injection_cluster).to_dict()


def _attacked_comparison(a: Sequence[Row], b: Sequence[Row]) -> dict[str, Any]:
    return {
        "asr": _diff(a, b, lambda r: r["security"]),
        "utility_under_attack": _diff(a, b, lambda r: r["utility"]),
        "attacker_success": _diff(a, b, lambda r: r["attacker_success"]),
    }


H2_BINS = 10


def position_bins(rows: Sequence[Row], bins: int = H2_BINS) -> list[list[str]]:
    """Train pairs in run order, cut into `bins` near-equal groups (fewer when there are fewer pairs).

    C2 and C4 run the train pairs in the same manifest order, so bin k holds the same pairs in both runs.
    """
    order: dict[str, int] = {}
    for r in sorted(rows, key=lambda r: r["unit"]):
        order.setdefault(r["pair"], len(order))
    pairs = sorted(order, key=order.__getitem__)
    k = min(bins, len(pairs))
    return [pairs[len(pairs) * i // k : len(pairs) * (i + 1) // k] for i in range(k)]


def _h2_by_position(c2: Sequence[Row], c4: Sequence[Row]) -> list[dict[str, Any]]:
    out = []
    for i, pairs in enumerate(position_bins(c4)):
        members = set(pairs)
        a = [r for r in c2 if r["pair"] in members]
        b = [r for r in c4 if r["pair"] in members]
        out.append(
            {
                "bin": i,
                "pairs": len(pairs),
                "asr": _diff(a, b, lambda r: r["security"]),
                "c2_asr": _asr(a),
                "c4_asr": _asr(b),
            }
        )
    return out


def compare(runs: list[dict[str, Any]]) -> dict[str, Any]:
    conditions = sorted({run["config"]["condition"] for run in runs})
    rows = {c: pooled(runs, c) for c in conditions}
    result: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runs": [
            {
                "dir": run["dir"],
                "condition": run["config"]["condition"],
                "seed": run["config"]["seed"],
                "complete": run["complete"],
                "units_done": run["units_done"],
                "units_total": run["units_total"],
                "git_commit": (run["config"].get("code") or {}).get("git_commit"),
            }
            for run in runs
        ],
        "manifest_sha256": runs[0]["config"]["manifest_sha256"] if runs else None,
        "bootstrap": {"resamples": DEFAULT_RESAMPLES, "confidence": CONFIDENCE, "paired": True},
        "h1": {},
        "h2": None,
        "c0_vs_c1": None,
        "missing": [],
    }

    # H1: test ASR on the final file vs C1 on the same test pairs, at benign utility before/after.
    c1_test = _select(rows.get("C1", []), ("attack",), "test")
    for cond in ("C3", "C4"):
        test = _select(rows.get(cond, []), ("test",))
        if not test or not c1_test:
            result["missing"].append(f"H1 {cond} vs C1 needs both runs")
            continue
        before = _select(rows[cond], ("benign_before",))
        after = _select(rows[cond], ("benign_after",))
        result["h1"][cond] = {
            "test_vs_c1": _attacked_comparison(test, c1_test),
            "benign_utility_after_vs_before": _diff(after, before, lambda r: r["utility"], benign=True)
            if before and after
            else None,
        }
    if c1_test:
        result["c1_test_asr"] = _asr(c1_test)
        c1_benign = _select(rows["C1"], ("benign",))
        result["c1_benign_utility"] = (
            cluster_bootstrap(c1_benign, lambda r: r["utility"], user_task_cluster).to_dict() if c1_benign else None
        )

    # H2: C2 vs C4 on train-split injection tasks, overall and by trial index.
    c2_train = _select(rows.get("C2", []), ("attack",), "train")
    c4_train = _select(rows.get("C4", []), ("train",))
    if c2_train and c4_train:
        trials = sorted({r["trial"] for r in c2_train + c4_train})
        result["h2"] = {
            "overall": _attacked_comparison(c2_train, c4_train),
            "by_position": _h2_by_position(c2_train, c4_train),
            "by_trial": {
                str(t): {
                    "asr": _diff(
                        [r for r in c2_train if r["trial"] == t],
                        [r for r in c4_train if r["trial"] == t],
                        lambda r: r["security"],
                    ),
                    "c2_asr": _asr([r for r in c2_train if r["trial"] == t]),
                    "c4_asr": _asr([r for r in c4_train if r["trial"] == t]),
                }
                for t in trials
            },
        }
    else:
        result["missing"].append("H2 needs C2 and C4 runs")

    c0 = _select(rows.get("C0", []), ("attack",))
    c1 = _select(rows.get("C1", []), ("attack",))
    if c0 and c1:
        result["c0_vs_c1"] = _attacked_comparison(c0, c1)
    else:
        result["missing"].append("C0 vs C1 needs both runs")
    return result


# --- HTML -----------------------------------------------------------------------------------------------------


def _signed(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:+.1f} pts"


def _diff_cell(d: dict[str, Any] | None) -> str:
    if d is None or d["diff"] is None:
        return "n/a"
    ci = "" if d["lo"] is None else f" <span class='muted'>[{_signed(d['lo'])}, {_signed(d['hi'])}]</span>"
    return f"<b>{_signed(d['diff'])}</b>{ci}"


def _diff_row(label: str, d: dict[str, Any] | None) -> str:
    if d is None:
        return f"<tr><td>{label}</td><td colspan='5'>n/a</td></tr>"
    return (
        f"<tr><td>{label}</td><td>{_pct(d['a'])}</td><td>{_pct(d['b'])}</td><td>{_diff_cell(d)}</td>"
        f"<td>{d['n_a']} / {d['n_b']}</td><td>{d['clusters']}</td></tr>"
    )


TABLE_HEAD = (
    "<table><thead><tr><th>Measure</th><th>{a}</th><th>{b}</th><th>Difference [95% CI]</th><th>Trials</th>"
    "<th>Clusters</th></tr></thead><tbody>"
)


def _comparison_table(a: str, b: str, comp: dict[str, Any]) -> str:
    return (
        TABLE_HEAD.format(a=_e(a), b=_e(b))
        + _diff_row("ASR", comp["asr"])
        + _diff_row("Utility under attack", comp["utility_under_attack"])
        + _diff_row("Attacker success", comp["attacker_success"])
        + "</tbody></table>"
    )


def render_html(result: dict[str, Any]) -> str:
    out = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><title>Cross-run comparison</title>",
        f"<style>{CSS}</style></head><body><h1>Cross-run comparison</h1>",
        f"<p class='muted'>Generated {_e(result['generated_at'])}. Manifest {_e(result['manifest_sha256'])}. "
        "Differences are A minus B in percentage points, with 95% paired cluster-bootstrap intervals over injection "
        f"tasks (benign: user tasks), {result['bootstrap']['resamples']:,} resamples. Seeds of a condition are "
        "pooled. Only injection tasks present in both conditions are compared.</p>",
        "<h2>Runs</h2><table><thead><tr><th>Run</th><th>Condition</th><th>Seed</th><th>Status</th><th>Commit</th>"
        "</tr></thead><tbody>",
    ]
    for run in result["runs"]:
        status = "complete" if run["complete"] else f"partial ({run['units_done']} of {run['units_total']} units)"
        out.append(
            f"<tr><td>{_e(run['dir'])}</td><td>{_e(run['condition'])}</td><td>{run['seed']}</td>"
            f"<td>{_e(status)}</td><td>{_e((run['git_commit'] or '')[:10])}</td></tr>"
        )
    out.append("</tbody></table>")
    if any(not run["complete"] for run in result["runs"]):
        out.append("<p><span class='badge'>Some runs are partial; their numbers will change.</span></p>")

    out.append(
        "<h2>H1: final skill file vs S0 on held-out injection tasks</h2>"
        "<p class='muted'>C3/C4 test phase (final, frozen file) against C1 on the same test pairs. "
        "A negative ASR difference means the evolved file blocks more. H1 claims a gain only if benign utility "
        "holds (after minus before).</p>"
    )
    if not result["h1"]:
        out.append("<p>Not available yet: needs C1 and C3 or C4 results.</p>")
    for cond, h1 in result["h1"].items():
        out.append(f"<h3>{_e(cond)} vs C1</h3>")
        out.append(_comparison_table(f"{cond} test", "C1 test pairs", h1["test_vs_c1"]))
        if h1["benign_utility_after_vs_before"] is not None:
            out.append(
                TABLE_HEAD.format(a="After (final file)", b="Before (S0)")
                + _diff_row("Benign utility", h1["benign_utility_after_vs_before"])
                + "</tbody></table>"
            )
    if result.get("c1_test_asr"):
        series = {"ASR on test pairs": [result["c1_test_asr"]]}
        cats = ["C1"]
        for cond, h1 in result["h1"].items():
            d = h1["test_vs_c1"]["asr"]
            cats.append(cond)
            series["ASR on test pairs"].append({"value": d["a"], "lo": None, "hi": None})
        out.append(_point_whisker_chart("Test-pair ASR by condition", cats, series))

    out.append(
        "<h2>H2: static vs evolving skill file under the evolving attacker</h2>"
        "<p class='muted'>C2 minus C4 on train-split injection tasks only. A positive difference means the evolving "
        "file (C4) holds up better against the adapting attacker. Main curve: position in the run (train pairs in run "
        "order, cut into equal bins; the same pairs in each bin for C2 and C4), where the C4 file learns as it goes. "
        "Second view: trial index 0/1/2, the attacker refining one pair.</p>"
    )
    if result["h2"] is None:
        out.append("<p>Not available yet: needs C2 and C4 results (MUZZLE).</p>")
    else:
        out.append(_comparison_table("C2 (train inj.)", "C4 (train phase)", result["h2"]["overall"]))
        by_pos = result["h2"]["by_position"]
        out.append(
            _point_whisker_chart(
                "ASR over the run (train pairs in run order)",
                [f"{b['bin'] + 1}" for b in by_pos],
                {"C2": [b["c2_asr"] for b in by_pos], "C4": [b["c4_asr"] for b in by_pos]},
            )
        )
        out.append(TABLE_HEAD.format(a="C2", b="C4"))
        out += [_diff_row(f"ASR, bin {b['bin'] + 1} ({b['pairs']} pairs)", b["asr"]) for b in by_pos]
        out.append("</tbody></table>")
        by_trial = result["h2"]["by_trial"]
        out.append(TABLE_HEAD.format(a="C2", b="C4"))
        out += [_diff_row(f"ASR, trial {t}", v["asr"]) for t, v in by_trial.items()]
        out.append("</tbody></table>")
        out.append(
            _point_whisker_chart(
                "ASR by trial index",
                [f"trial {t}" for t in by_trial],
                {"C2": [v["c2_asr"] for v in by_trial.values()], "C4": [v["c4_asr"] for v in by_trial.values()]},
            )
        )

    out.append("<h2>C0 vs C1: the checker with S0</h2>")
    if result["c0_vs_c1"] is None:
        out.append("<p>Not available yet: needs C0 and C1 results.</p>")
    else:
        out.append(_comparison_table("C0 (no checker)", "C1 (S0)", result["c0_vs_c1"]))

    if result["missing"]:
        out.append("<h2>Not computed</h2><ul>" + "".join(f"<li>{_e(m)}</li>" for m in result["missing"]) + "</ul>")
    out.append("</body></html>")
    return "\n".join(out)


def write_comparison(results_root: Path, out: Path, limit: int | None = None) -> Path:
    runs = find_runs(results_root, limit)
    if not runs:
        raise RunMismatchError(f"no runs with limit={limit} under {results_root}")
    result = compare(runs)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    out.write_text(render_html(result), encoding="utf-8")
    return out


@click.command()
@click.argument("results_root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=Path("results"))
@click.option("--out", type=click.Path(path_type=Path), default=None, help="Default: <results_root>/comparison.html")
@click.option("--limit", type=int, default=None, help="Compare smoke runs made with this --limit instead.")
def main(results_root: Path, out: Path | None, limit: int | None) -> None:
    out = out or results_root / ("comparison.html" if limit is None else f"comparison-limit{limit}.html")
    try:
        path = write_comparison(results_root, out, limit)
    except RunMismatchError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"wrote {path} and {path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
