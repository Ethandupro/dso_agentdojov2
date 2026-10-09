"""Export AgentDojo results into a self-contained, interactive security-grid page.

Walks `<logdir>/<pipeline>/<suite>/<user_task>/<attack>/<injection_task>.json` and embeds a compact
summary of every run into `results_grid.html`, which shows one user-task x injection-task table per suite
with attack success rate (ASR) per cell, row, column, suite and overall.

Polarity: in the result files `security == True` means the injection SUCCEEDED (the attacker won), so the
ASR numerator is the runs with `security == True`.

Usage:
    PYTHONPATH=src python util_scripts/export_results_grid.py runs --out runs/results_grid.html
    # Pool several pipelines (e.g. the final pipeline of each evolving seed) into one selectable arm:
    ... --group "Evolving (pooled)=m-evolving_guardrail-aaaa1111,m-evolving_guardrail-bbbb2222"
"""

import datetime
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import click

from agentdojo.task_suite.load_suites import get_suites

TEMPLATE = Path(__file__).parent / "results_grid.html"
PLACEHOLDER = "/*__GRID_DATA__*/null"
V1_VERSION = "v1.1.2"
V1_2_VERSION = "v1.2.2"
DEFENSE_SUFFIXES = ("tool_filter", "repeat_user_prompt", "spotlighting_with_delimiting", "transformers_pi_detector")
MAX_ARG_CHARS = 48
MAX_CALL_CHARS = 160
MAX_DETECTION_CHARS = 400


def natural_key(text: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


class Interner:
    def __init__(self) -> None:
        self.values: list[str] = []
        self.index: dict[str, int] = {}

    def add(self, value: str) -> int:
        if value not in self.index:
            self.index[value] = len(self.values)
            self.values.append(value)
        return self.index[value]


def shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def format_call(call: dict[str, Any]) -> str:
    args = call.get("args") or {}
    rendered = ", ".join(
        f"{key}={shorten(json.dumps(value, ensure_ascii=False), MAX_ARG_CHARS)}" for key, value in args.items()
    )
    return shorten(f"{call.get('function', '?')}({rendered})", MAX_CALL_CHARS)


def tool_calls(messages: list[dict[str, Any]]) -> list[str]:
    calls = []
    for message in messages:
        if message.get("role") == "assistant":
            calls.extend(format_call(call) for call in message.get("tool_calls") or [])
    return calls


def format_detections(detections: Iterable[Any]) -> list[str]:
    """Summarise `guardrail_detections` as one line per detection: level, rules fired and matched text."""
    lines = []
    for detection in detections:
        if not isinstance(detection, dict):
            lines.append(shorten(str(detection), MAX_DETECTION_CHARS))
            continue
        level = detection.get("threat_level") or detection.get("level") or ""
        rules = [
            rule.get("name", str(rule)) if isinstance(rule, dict) else str(rule)
            for rule in detection.get("rules") or []
        ]
        matches = [
            match.get("text", "") if isinstance(match, dict) else str(match) for match in detection.get("matches") or []
        ]
        parts = [str(level)] if level else []
        if rules:
            parts.append("rules: " + ", ".join(rules))
        if matches:
            parts.append("matched: " + " | ".join(repr(match) for match in matches if match))
        lines.append(shorten(" · ".join(parts) or json.dumps(detection), MAX_DETECTION_CHARS))
    return lines


def pipeline_kind(name: str) -> str:
    if "evolving_guardrail" in name:
        return "evolving_guardrail"
    if "static_guardrail" in name:
        return "static_guardrail"
    if name.endswith(DEFENSE_SUFFIXES):
        return "defense"
    return "model"


def detect_version(records: list[dict[str, Any]], v1_injection_ids: set[str], default: str) -> str:
    """Pick the benchmark version a pipeline ran, from the files' own field or the workspace tasks it touched."""
    declared = {record["benchmark_version"] for record in records if record.get("benchmark_version")}
    if len(declared) == 1:
        return declared.pop()
    workspace_ids = {record["injection_task_id"] for record in records if record["suite_name"] == "workspace"}
    workspace_ids.discard(None)
    if workspace_ids - v1_injection_ids:
        return V1_2_VERSION
    if workspace_ids:
        return V1_VERSION
    return default


def build_dataset(logdir: Path, default_version: str, groups: list[tuple[str, list[str]]]) -> dict[str, Any]:
    strings = Interner()
    attacks = Interner()
    suites_index = Interner()
    task_ids: dict[str, set[str]] = {}
    raw_pipelines: dict[str, list[dict[str, Any]]] = {}

    for path in sorted(logdir.glob("*/*/*/*/*.json")):
        pipeline_dir, suite_dir, _, attack_dir, _ = path.relative_to(logdir).parts
        if "dos" in attack_dir:  # DoS attacks only run on one injection task and score security as `not utility`
            continue
        data = json.loads(path.read_text())
        user_task_id = data.get("user_task_id")
        if user_task_id is None or not str(user_task_id).startswith("user_task"):
            continue  # `injection_task_*/none/none.json` checks the injection task itself, not a pair
        raw_pipelines.setdefault(pipeline_dir, []).append(
            {
                "suite_name": data.get("suite_name") or suite_dir,
                "user_task_id": user_task_id,
                "injection_task_id": data.get("injection_task_id"),
                "attack_type": data.get("attack_type") or attack_dir,
                "utility": bool(data.get("utility")),
                "security": bool(data.get("security")),
                "error": data.get("error"),
                "calls": tool_calls(data.get("messages") or []),
                "detections": format_detections(data.get("guardrail_detections") or []),
                "benchmark_version": data.get("benchmark_version"),
            }
        )

    v1_injection_ids = set(get_suites(V1_VERSION)["workspace"].injection_tasks)
    versions: dict[str, Any] = {}
    pipelines = []
    for name in sorted(raw_pipelines, key=str.lower):
        records = raw_pipelines[name]
        version = detect_version(records, v1_injection_ids, default_version)
        if version not in versions:
            versions[version] = {"suites": {}}
            for suite_name, suite in get_suites(version).items():
                user_tasks = sorted(suite.user_tasks, key=natural_key)
                injection_tasks = sorted(suite.injection_tasks, key=natural_key)
                versions[version]["suites"][suite_name] = {
                    "user_tasks": user_tasks,
                    "injection_tasks": injection_tasks,
                    "prompts": {task_id: suite.user_tasks[task_id].PROMPT for task_id in user_tasks},
                    "goals": {task_id: suite.injection_tasks[task_id].GOAL for task_id in injection_tasks},
                }
                task_ids.setdefault(suite_name, set()).update(user_tasks, injection_tasks)

        runs, benign = [], []
        for record in records:
            suite = suites_index.add(record["suite_name"])
            task_ids.setdefault(record["suite_name"], set()).add(record["user_task_id"])
            detections = [strings.add(line) for line in record["detections"]]
            error = strings.add(shorten(str(record["error"]), MAX_DETECTION_CHARS)) if record["error"] else -1
            calls = [strings.add(call) for call in record["calls"]]
            if record["attack_type"] in (None, "none") or record["injection_task_id"] is None:
                benign.append([suite, record["user_task_id"], int(record["utility"]), error, calls])
                continue
            task_ids[record["suite_name"]].add(record["injection_task_id"])
            runs.append(
                [
                    suite,
                    record["user_task_id"],
                    record["injection_task_id"],
                    attacks.add(record["attack_type"]),
                    int(record["utility"]),
                    int(record["security"]),
                    error,
                    calls,
                    detections,
                ]
            )
        pipelines.append(
            {"name": name, "kind": pipeline_kind(name), "version": version, "runs": runs, "benign": benign}
        )

    # Replace task ID strings with indices into one naturally sorted ID list per suite.
    ids = {suite: sorted(values, key=natural_key) for suite, values in task_ids.items()}
    positions = {suite: {task_id: i for i, task_id in enumerate(values)} for suite, values in ids.items()}
    suite_names = suites_index.values
    for pipeline in pipelines:
        for run in pipeline["runs"]:
            lookup = positions[suite_names[run[0]]]
            run[1], run[2] = lookup[run[1]], lookup[run[2]]
        for run in pipeline["benign"]:
            run[1] = positions[suite_names[run[0]]][run[1]]

    known = {pipeline["name"] for pipeline in pipelines}
    for label, members in groups:
        missing = [member for member in members if member not in known]
        if missing:
            raise click.BadParameter(f"group {label!r} names unknown pipelines: {', '.join(missing)}")

    return {
        "meta": {
            "generated": datetime.datetime.now().astimezone().isoformat(timespec="minutes"),
            "logdir": str(logdir),
            "polarity": "security == true means the attacker succeeded",
        },
        "suites": suite_names,
        "ids": ids,
        "attacks": attacks.values,
        "strings": strings.values,
        "versions": versions,
        "pipelines": pipelines,
        "groups": [{"name": label, "members": members} for label, members in groups],
    }


def parse_group(value: str) -> tuple[str, list[str]]:
    label, sep, members = value.partition("=")
    if not sep or not label.strip() or not members.strip():
        raise click.BadParameter(f"expected LABEL=pipeline1,pipeline2 but got {value!r}")
    return label.strip(), [member.strip() for member in members.split(",") if member.strip()]


def render(dataset: dict[str, Any], template: Path, fragment: bool) -> str:
    page = template.read_text()
    if PLACEHOLDER not in page:
        raise click.ClickException(f"{template} has no {PLACEHOLDER} placeholder")
    payload = json.dumps(dataset, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = page.replace(PLACEHOLDER, payload)
    if fragment:
        return page
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        f"</head>\n<body>\n{page}\n</body>\n</html>\n"
    )


@click.command()
@click.argument("logdir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=Path("runs"))
@click.option(
    "--out", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Defaults to LOGDIR/results_grid.html."
)
@click.option("--template", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=TEMPLATE)
@click.option(
    "--benchmark-version",
    "default_version",
    default=V1_2_VERSION,
    help="Version assumed for pipelines whose results don't reveal it (e.g. banking-only runs).",
)
@click.option("--group", "groups", multiple=True, help="LABEL=pipeline1,pipeline2 pools runs into one arm. Repeatable.")
@click.option(
    "--json",
    "json_out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Also write the dataset as JSON.",
)
@click.option("--fragment", is_flag=True, help="Omit the <!doctype>/<html>/<head>/<body> wrapper.")
def main(
    logdir: Path,
    out: Path | None,
    template: Path,
    default_version: str,
    groups: tuple[str, ...],
    json_out: Path | None,
    fragment: bool,
) -> None:
    dataset = build_dataset(logdir, default_version, [parse_group(group) for group in groups])
    if json_out is not None:
        json_out.write_text(json.dumps(dataset, ensure_ascii=False, separators=(",", ":")))
    out = out or logdir / "results_grid.html"
    out.write_text(render(dataset, template, fragment))
    runs = sum(len(pipeline["runs"]) for pipeline in dataset["pipelines"])
    click.echo(
        f"{len(dataset['pipelines'])} pipelines, {runs} attack runs -> {out} ({out.stat().st_size / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
