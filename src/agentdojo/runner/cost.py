"""Token use and estimated cost of a run, per component: victim, checker, judge and attacker.

Every component already logs its own tokens, in different places:

- victim: `victim_usage` on each trial record (per DeepAgents agent);
- checker: `input_tokens`/`output_tokens` on each `checker_calls.jsonl` line;
- judge (C3/C4): `judge_calls`, `judge_input_tokens`, `judge_output_tokens` on each record's `skill_step`, plus the
  judge model's rule-matching call on each train trial (`s0_rule_match`, rich evidence);
- attacker (MUZZLE, C2/C4): `usage` on each rewrite in the attacker's archive in `state.json`.

This module adds them up so a run report can say what the defence costs next to what the victim costs.

    python -m agentdojo.runner.cost results/c3_evolving-skill_static-attack/seed_0 --prices prices.json

A prices file maps model ids to US dollars per million tokens. Models it does not list get no dollar figure (self-hosted
models have no per-token price; their cost is GPU time, which the report shows as wall time):

    {"provider/model-id": {"input": 0.25, "output": 1.00}}
"""

import json
from pathlib import Path
from typing import Any

import click

COMPONENTS = ("victim", "checker", "judge", "attacker")
COMPONENT_LABELS = {
    "victim": "Victim",
    "checker": "Checker",
    "judge": "Judge (skill-file rewrites)",
    "attacker": "Attacker (MUZZLE rewrites)",
}
DEFENCE_COMPONENTS = ("checker", "judge")


def _empty(model: str | None) -> dict[str, Any]:
    return {"model": model, "calls": 0, "input_tokens": 0, "output_tokens": 0}


def _model_ids(config: dict[str, Any]) -> dict[str, str | None]:
    attacker = config.get("attacker") or {}
    return {
        "victim": (config.get("victim") or {}).get("model_id"),
        "checker": (config.get("checker_model") or {}).get("model_id"),
        "judge": (config.get("judge") or {}).get("model"),
        "attacker": ((attacker.get("settings") or {}).get("attacker") or {}).get("model"),
    }


def component_usage(
    config: dict[str, Any],
    state: dict[str, Any],
    records: list[dict[str, Any]],
    checker_lines: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Calls and tokens per component, from the checkpointed part of a run.

    `records` and `checker_lines` must already be cut at the checkpoint (as `report.summarize_run` does). The
    attacker's archive is read from `state`, which is the checkpoint itself."""
    models = _model_ids(config)
    usage = {c: _empty(models[c]) for c in COMPONENTS}

    victim = usage["victim"]
    for r in records:
        for counts in (r.get("victim_usage") or {}).values():
            victim["calls"] += counts.get("model_calls") or 0
            victim["input_tokens"] += counts.get("input_tokens") or 0
            victim["output_tokens"] += counts.get("output_tokens") or 0

    checker = usage["checker"]
    checker["calls"] = len(checker_lines)
    checker["input_tokens"] = sum(line.get("input_tokens") or 0 for line in checker_lines)
    checker["output_tokens"] = sum(line.get("output_tokens") or 0 for line in checker_lines)

    judge = usage["judge"]
    for r in records:
        step = r.get("skill_step") or {}
        judge["calls"] += step.get("judge_calls") or 0
        judge["input_tokens"] += step.get("judge_input_tokens") or 0
        judge["output_tokens"] += step.get("judge_output_tokens") or 0
        # Paraphrase-aware rule matching (C3/C4 rich evidence) calls the judge model once per train trial.
        match = r.get("s0_rule_match") or {}
        if match.get("input_tokens") or match.get("output_tokens"):
            judge["calls"] += 1
            judge["input_tokens"] += match.get("input_tokens") or 0
            judge["output_tokens"] += match.get("output_tokens") or 0

    attacker = usage["attacker"]
    for entry in (state.get("attacker") or {}).get("archive") or []:
        counts = entry.get("usage") or {}
        attacker["calls"] += counts.get("calls") or 0
        attacker["input_tokens"] += counts.get("prompt_tokens") or 0
        attacker["output_tokens"] += counts.get("completion_tokens") or 0

    for entry in usage.values():
        entry["total_tokens"] = entry["input_tokens"] + entry["output_tokens"]
    return usage


def load_prices(path: Path) -> dict[str, dict[str, float]]:
    """Reads a prices file: `{model_id: {"input": usd_per_million, "output": usd_per_million}}`."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    prices: dict[str, dict[str, float]] = {}
    for model, price in raw.items():
        if model.startswith("_"):  # comments
            continue
        if isinstance(price, dict) and price.get("input") is None and price.get("output") is None:
            continue  # not filled in yet
        if not isinstance(price, dict) or not {"input", "output"} <= set(price):
            raise ValueError(f"{path}: price for {model!r} needs 'input' and 'output' (USD per million tokens)")
        prices[model] = {"input": float(price["input"]), "output": float(price["output"])}
    return prices


def usd(entry: dict[str, Any], prices: dict[str, dict[str, float]]) -> float | None:
    """Estimated dollars for one component, or None when its model has no price."""
    price = prices.get(entry["model"] or "")
    if price is None:
        return None
    return (entry["input_tokens"] * price["input"] + entry["output_tokens"] * price["output"]) / 1_000_000


def cost_summary(
    usage: dict[str, dict[str, Any]],
    attacked_trials: int,
    prices: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Per-component usage with shares, per-trial averages, dollars where priced, and the defence overhead
    (checker + judge tokens as a fraction of victim tokens)."""
    prices = prices or {}
    grand = sum(e["total_tokens"] for e in usage.values())
    components: dict[str, Any] = {}
    for name in COMPONENTS:
        e = usage[name]
        components[name] = {
            **e,
            "share_of_tokens": (e["total_tokens"] / grand) if grand else None,
            "tokens_per_attacked_trial": (e["total_tokens"] / attacked_trials) if attacked_trials else None,
            "usd": usd(e, prices),
        }
    priced = [c["usd"] for c in components.values() if c["usd"] is not None]
    victim_tokens = usage["victim"]["total_tokens"]
    defence_tokens = sum(usage[c]["total_tokens"] for c in DEFENCE_COMPONENTS)
    return {
        "components": components,
        "total_tokens": grand,
        "attacked_trials": attacked_trials,
        "defence_overhead": (defence_tokens / victim_tokens) if victim_tokens else None,
        "usd_priced_total": sum(priced) if priced else None,
        "unpriced": [c for c in COMPONENTS if components[c]["usd"] is None and components[c]["total_tokens"]],
    }


def format_table(summary: dict[str, Any]) -> str:
    """Plain-text table for the terminal."""
    head = f"{'Component':<28}{'Model':<34}{'Calls':>9}{'In tok':>13}{'Out tok':>12}{'Share':>8}{'USD':>10}"
    lines = [head, "-" * len(head)]
    for name in COMPONENTS:
        c = summary["components"][name]
        share = "-" if c["share_of_tokens"] is None else f"{c['share_of_tokens']:.1%}"
        dollars = "-" if c["usd"] is None else f"{c['usd']:.2f}"
        lines.append(
            f"{COMPONENT_LABELS[name]:<28}{(c['model'] or '-')[:33]:<34}{c['calls']:>9,}"
            f"{c['input_tokens']:>13,}{c['output_tokens']:>12,}{share:>8}{dollars:>10}"
        )
    overhead = summary["defence_overhead"]
    lines.append("")
    lines.append(f"Total tokens: {summary['total_tokens']:,} over {summary['attacked_trials']:,} attacked trials")
    lines.append(
        "Defence overhead (checker + judge tokens / victim tokens): "
        + ("n/a" if overhead is None else f"{overhead:.1%}")
    )
    if summary["usd_priced_total"] is not None:
        lines.append(f"Estimated USD (priced components only): {summary['usd_priced_total']:.2f}")
    if summary["unpriced"]:
        lines.append("No price for: " + ", ".join(summary["unpriced"]) + " (self-hosted, or missing from --prices)")
    return "\n".join(lines)


@click.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--prices",
    "prices_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="JSON file of USD per million tokens per model id.",
)
def main(run_dir: Path, prices_path: Path | None) -> None:
    """Print token use and estimated cost per component, and write `cost.json` into RUN_DIR."""
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    cursor = state.get("cursor", 0)
    records = [r for r in _read_jsonl(run_dir / "trials.jsonl") if r["unit"] < cursor]
    lines = [line for line in _read_jsonl(run_dir / "checker_calls.jsonl") if line["unit"] < cursor]
    attacked = sum(r.get("pair") is not None for r in records)
    prices = load_prices(prices_path) if prices_path else {}
    summary = cost_summary(component_usage(config, state, records, lines), attacked, prices)
    (run_dir / "cost.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    click.echo(format_table(summary))
    click.echo(f"\nwrote {run_dir / 'cost.json'}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
