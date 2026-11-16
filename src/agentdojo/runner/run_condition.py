"""One command per run: one condition at one seed, resumable.

    uv run --extra deepagents python -m agentdojo.runner.run_condition --condition C1
    uv run --extra deepagents python -m agentdojo.runner.run_condition --condition C1 --limit 5   # smoke run

Runs on the GPU VM against the local vLLM servers (`local_victim/serve_all.sh`): victim on port 8000, checker on
port 8001. Results go to `results/<run_id>/seed_<n>/` (`results/<run_id>-limit<N>/...` for smoke runs). Run the
same command again to resume after a crash or Ctrl-C; it refuses to resume under a different config.

This version runs C0 and C1. C2 and C4 need the MUZZLE attacker, C3 and C4 the skill-file judge; they are wired in
by later PRs.
"""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import click
from dotenv import load_dotenv

from agentdojo import experiment_models
from agentdojo.agent_pipeline.deepagents_pipeline import DeepAgentsConfig, DeepAgentsPipeline, ToolMiddlewareFactory
from agentdojo.checker.middleware import CHECKER_REPLY_SCHEMA, CHECKER_USER_TEMPLATE, local_checker_fn
from agentdojo.runner.conditions import CONDITIONS, Unit, build_units
from agentdojo.runner.config import RunConfig, git_state, now, package_versions, sha256_text
from agentdojo.runner.manifest import DEFAULT_MANIFEST_PATH, ManifestMismatchError, check_manifest, manifest_sha256
from agentdojo.runner.plugins import FixedSkillFile, NoChecker, SkillFileSource, StaticAttacker
from agentdojo.runner.run import ConfigMismatchError, Runner, quick_summary, run_dir_for

SUPPORTED_CONDITIONS = ("C0", "C1")


class PreflightError(Exception):
    pass


def served_models(base_url: str, timeout: float = 10.0) -> list[str]:
    """The model ids an OpenAI-compatible server reports at `/models`."""
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = json.load(response)
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise PreflightError(f"{url} did not answer ({e}). Is the server up? See local_victim/README.md.") from e
    return sorted(m["id"] for m in data.get("data", []))


def check_served(role: str, base_url: str, model_id: str) -> list[str]:
    models = served_models(base_url)
    if model_id not in models:
        raise PreflightError(f"the {role} server at {base_url} serves {models}, not {model_id}")
    return models


@click.command()
@click.option("--condition", type=click.Choice(sorted(CONDITIONS)), required=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--results-root", type=click.Path(path_type=Path), default=Path("results"), show_default=True)
@click.option("--limit", type=int, default=None, help="Smoke run: only the first N pairs of the run order.")
@click.option("--trials-per-pair", type=int, default=3, show_default=True)
@click.option("--manifest", type=click.Path(path_type=Path), default=DEFAULT_MANIFEST_PATH, show_default=True)
@click.option("--victim-model-id", default=experiment_models.VICTIM_MODEL_ID, show_default=True)
@click.option("--victim-base-url", default=experiment_models.VICTIM_BASE_URL, show_default=True)
@click.option("--checker-model-id", default=experiment_models.CHECKER_MODEL_ID, show_default=True)
@click.option("--checker-base-url", default=experiment_models.CHECKER_BASE_URL, show_default=True)
@click.option("--min-span-chars", type=int, default=0, show_default=True)
@click.option("--env-file", type=click.Path(path_type=Path), default=None, help="Loads API keys (judge, attacker).")
def main(
    condition: str,
    seed: int,
    results_root: Path,
    limit: int | None,
    trials_per_pair: int,
    manifest: Path,
    victim_model_id: str,
    victim_base_url: str,
    checker_model_id: str,
    checker_base_url: str,
    min_span_chars: int,
    env_file: Path | None,
) -> None:
    if condition not in SUPPORTED_CONDITIONS:
        raise click.UsageError(f"{condition} is not wired in yet; this version runs {', '.join(SUPPORTED_CONDITIONS)}")
    if env_file is not None:
        load_dotenv(env_file)
    cond = CONDITIONS[condition]

    try:
        manifest_data = check_manifest(manifest)
    except ManifestMismatchError as e:
        raise click.ClickException(str(e)) from e
    if manifest_data["victim_model_id"] != victim_model_id:
        raise click.ClickException(
            f"the manifest's trial-0 texts address {manifest_data['victim_model_id']}, not {victim_model_id}; "
            "rebuild it with `python -m agentdojo.runner.build_manifest`"
        )

    victim_config = DeepAgentsConfig(
        model_id=victim_model_id,
        base_url=victim_base_url,
        enable_thinking=experiment_models.VICTIM_ENABLE_THINKING,
    )
    try:
        served = {"victim": check_served("victim", victim_base_url, victim_model_id)}
        if cond.checker:
            served["checker"] = check_served("checker", checker_base_url, checker_model_id)
    except PreflightError as e:
        raise click.ClickException(str(e)) from e

    s0_path = Path(manifest_data["s0_path"])
    config = RunConfig(
        condition=cond.name,
        run_id=cond.run_id,
        seed=seed,
        checker=cond.checker,
        skill_mode=cond.skill_mode,
        attack_mode=cond.attack_mode,
        trials_per_pair=trials_per_pair,
        limit=limit,
        benchmark_version=manifest_data["benchmark_version"],
        manifest_path=str(manifest),
        manifest_sha256=manifest_sha256(manifest),
        s0_path=str(s0_path),
        s0_sha256=manifest_data["s0_sha256"],
        victim=victim_config.model_dump(),
        checker_model=(
            {
                "model_id": checker_model_id,
                "base_url": checker_base_url,
                "enable_thinking": experiment_models.CHECKER_ENABLE_THINKING,
                "min_span_chars": min_span_chars,
                "user_template_sha256": sha256_text(CHECKER_USER_TEMPLATE),
                "reply_schema_sha256": sha256_text(json.dumps(CHECKER_REPLY_SCHEMA, sort_keys=True)),
            }
            if cond.checker
            else None
        ),
        served_models=served,
        code=git_state(),
        versions=package_versions(),
        created_at=now(),
    )

    skill_file: SkillFileSource = FixedSkillFile(s0_path.read_text(encoding="utf-8")) if cond.checker else NoChecker()
    checker_fn = (
        local_checker_fn(checker_model_id, checker_base_url, enable_thinking=experiment_models.CHECKER_ENABLE_THINKING)
        if cond.checker
        else None
    )

    def victim_factory(unit: Unit, factory: ToolMiddlewareFactory | None) -> DeepAgentsPipeline:
        return DeepAgentsPipeline.from_config(victim_config, tool_middleware_factory=factory)

    static = StaticAttacker(manifest_data)
    runner = Runner(
        config=config,
        condition=cond,
        run_dir=run_dir_for(results_root, cond, seed, limit),
        units=build_units(cond, manifest_data, trials_per_pair, limit),
        attacker=static,
        static_attacker=static,
        skill_file=skill_file,
        victim_factory=victim_factory,
        checker_fn=checker_fn,
        min_span_chars=min_span_chars,
        echo=click.echo,
    )
    try:
        records = runner.run()
    except ConfigMismatchError as e:
        raise click.ClickException(str(e)) from e
    except KeyboardInterrupt:
        click.echo("Interrupted. Run the same command again to resume from the last finished unit.", err=True)
        sys.exit(130)
    click.echo(json.dumps(quick_summary(records), indent=2))


if __name__ == "__main__":
    main()
