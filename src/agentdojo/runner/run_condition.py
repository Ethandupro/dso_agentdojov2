"""One command per run: one condition at one seed, resumable.

    uv run --extra deepagents python -m agentdojo.runner.run_condition --condition C1
    uv run --extra deepagents python -m agentdojo.runner.run_condition --condition C1 --limit 5   # smoke run

Runs on the GPU VM against the local vLLM servers (`local_victim/serve_all.sh`): victim on port 8000, checker on
port 8002. Results go to `results/<run_id>/seed_<n>/` (`results/<run_id>-limit<N>/...` for smoke runs). Run the
same command again to resume after a crash or Ctrl-C; it refuses to resume under a different config.

All five conditions run here. C2 and C4 also need the MUZZLE attacker's local vLLM server (started by
`local_victim/serve_all.sh` next to the checker). C3 and C4 also need the judge's OpenRouter key (`OPENROUTER_API_KEY`,
e.g. from `--env-file ~/api.env`, whose values win over variables already set in the shell); the key is never written
to the run directory.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path

import click
import openai
from dotenv import load_dotenv

from agentdojo import experiment_models
from agentdojo.agent_pipeline.deepagents_pipeline import DeepAgentsConfig, DeepAgentsPipeline, ToolMiddlewareFactory
from agentdojo.attacks.adaptive.autodojo import AutoDojoAttacker, AutoDojoSettings
from agentdojo.attacks.adaptive.muzzle import AttackerModelConfig, MuzzleAttacker, MuzzleSettings
from agentdojo.checker.middleware import CHECKER_REPLY_SCHEMA, CHECKER_USER_TEMPLATE, local_checker_fn
from agentdojo.runner.conditions import CONDITIONS, Unit, build_units
from agentdojo.runner.config import RunConfig, git_state, now, package_versions, sha256_text
from agentdojo.runner.evolving import EvolvingSkillFile
from agentdojo.runner.manifest import DEFAULT_MANIFEST_PATH, ManifestMismatchError, check_manifest, manifest_sha256
from agentdojo.runner.plugins import Attacker, FixedSkillFile, NoChecker, SkillFileSource, StaticAttacker
from agentdojo.runner.report import write_report
from agentdojo.runner.run import ConfigMismatchError, Runner, quick_summary, run_dir_for
from agentdojo.skillfile.checks import EditPolicy
from agentdojo.skillfile.judge import (
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_JUDGE_MODEL,
    JUDGE_SYSTEM_PROMPT,
    SkillFileEvolver,
    openrouter_client,
    openrouter_judge_fn,
)
from agentdojo.skillfile.structure import load_skill_file
from agentdojo.skillfile.tokens import vllm_token_counter

JUDGE_KEY_ENV = experiment_models.JUDGE_API_KEY_ENV


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


def check_judge(model: str) -> None:
    """The judge's key is set and OpenRouter answers a short request with that model."""
    if not os.environ.get(JUDGE_KEY_ENV):
        raise PreflightError(f"{JUDGE_KEY_ENV} is not set; pass --env-file with the judge's key")
    try:
        openrouter_client(max_retries=2).chat.completions.create(
            model=model, max_tokens=16, messages=[{"role": "user", "content": "ping"}]
        )
    except openai.APIError as e:
        raise PreflightError(f"the judge model {model} did not answer on OpenRouter: {e}") from e


def attacker_model_config(model_id: str, base_url: str, seed: int) -> AttackerModelConfig:
    """The local attacker model (decided 2026-11-17 by the user: no API), shared by both evolving attackers."""
    return AttackerModelConfig(model=model_id, base_url=base_url, api_key_env=None, seed=seed)


def muzzle_settings(model_id: str, base_url: str, seed: int) -> MuzzleSettings:
    return MuzzleSettings(attacker=attacker_model_config(model_id, base_url, seed))


def autodojo_settings(model_id: str, base_url: str, seed: int) -> AutoDojoSettings:
    return AutoDojoSettings(attacker=attacker_model_config(model_id, base_url, seed))


@click.command()
@click.option("--condition", type=click.Choice(sorted(CONDITIONS)), required=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--results-root", type=click.Path(path_type=Path), default=Path("results"), show_default=True)
@click.option("--limit", type=int, default=None, help="Smoke run: only the first N pairs of the run order.")
@click.option(
    "--suites",
    default=None,
    help="Pilot cut: comma-separated suites to keep (e.g. banking,travel). Gets its own results directory.",
)
@click.option("--trials-per-pair", type=int, default=3, show_default=True)
@click.option("--manifest", type=click.Path(path_type=Path), default=DEFAULT_MANIFEST_PATH, show_default=True)
@click.option("--victim-model-id", default=experiment_models.VICTIM_MODEL_ID, show_default=True)
@click.option("--victim-base-url", default=experiment_models.VICTIM_BASE_URL, show_default=True)
@click.option("--checker-model-id", default=experiment_models.CHECKER_MODEL_ID, show_default=True)
@click.option("--checker-base-url", default=experiment_models.CHECKER_BASE_URL, show_default=True)
@click.option("--min-span-chars", type=int, default=0, show_default=True)
@click.option("--env-file", type=click.Path(path_type=Path), default=None, help="Loads API keys (judge, attacker).")
@click.option("--judge-model", default=DEFAULT_JUDGE_MODEL, show_default=True, help="C3/C4 only.")
@click.option("--digest-size", type=int, default=20, show_default=True, help="Trials in the judge's history digest.")
@click.option("--attacker-model-id", default=experiment_models.ATTACKER_MODEL_ID, show_default=True, help="C2/C4.")
@click.option("--attacker-base-url", default=experiment_models.ATTACKER_BASE_URL, show_default=True, help="C2/C4.")
@click.option(
    "--attacker",
    "attacker_loop",
    type=click.Choice(["autodojo", "muzzle"]),
    default="autodojo",
    show_default=True,
    help="C2/C4 evolving attacker: 'autodojo' scores each rewrite on the real pipeline (leaderboard of the pair's "
    "own trials); 'muzzle' is the earlier offline stand-in loop, kept for ablation.",
)
def main(
    condition: str,
    seed: int,
    results_root: Path,
    limit: int | None,
    suites: str | None,
    trials_per_pair: int,
    manifest: Path,
    victim_model_id: str,
    victim_base_url: str,
    checker_model_id: str,
    checker_base_url: str,
    min_span_chars: int,
    env_file: Path | None,
    judge_model: str,
    digest_size: int,
    attacker_model_id: str,
    attacker_base_url: str,
    attacker_loop: str,
) -> None:
    if env_file is not None:
        load_dotenv(env_file, override=True)
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

    suites_tuple: tuple[str, ...] | None = None
    if suites is not None:
        requested = tuple(s.strip() for s in suites.split(",") if s.strip())
        known = {p["suite"] for p in manifest_data["pairs"]}
        unknown = [s for s in requested if s not in known]
        if unknown:
            raise click.ClickException(f"unknown suite(s) {unknown}; the manifest has {sorted(known)}")
        suites_tuple = requested or None

    victim_config = DeepAgentsConfig(
        model_id=victim_model_id,
        base_url=victim_base_url,
        enable_thinking=experiment_models.VICTIM_ENABLE_THINKING,
    )
    try:
        served = {"victim": check_served("victim", victim_base_url, victim_model_id)}
        if cond.checker:
            served["checker"] = check_served("checker", checker_base_url, checker_model_id)
        if cond.evolving_skill:
            check_judge(judge_model)
        evolving_attack = cond.attack_mode == "evolving"
        if evolving_attack:
            served["attacker"] = check_served("attacker", attacker_base_url, attacker_model_id)
    except PreflightError as e:
        raise click.ClickException(str(e)) from e

    s0_path = Path(manifest_data["s0_path"])
    policy = EditPolicy()
    static = StaticAttacker(manifest_data)
    attacker: Attacker = static
    if evolving_attack:
        if attacker_loop == "muzzle":
            attacker = MuzzleAttacker(manifest_data, muzzle_settings(attacker_model_id, attacker_base_url, seed))
        else:
            attacker = AutoDojoAttacker(manifest_data, autodojo_settings(attacker_model_id, attacker_base_url, seed))
    run_dir = run_dir_for(results_root, cond, seed, limit, suites_tuple)
    config = RunConfig(
        condition=cond.name,
        run_id=cond.run_id,
        seed=seed,
        checker=cond.checker,
        skill_mode=cond.skill_mode,
        attack_mode=cond.attack_mode,
        trials_per_pair=trials_per_pair,
        limit=limit,
        suites=list(suites_tuple) if suites_tuple else None,
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
        judge=(
            {
                "provider": "openrouter",
                "base_url": experiment_models.JUDGE_BASE_URL,
                "model": judge_model,
                "api_key_env": JUDGE_KEY_ENV,
                "max_tokens": DEFAULT_JUDGE_MAX_TOKENS,
                "seed": seed,
                "system_prompt_sha256": sha256_text(JUDGE_SYSTEM_PROMPT),
                "digest_size": digest_size,
                "policy": {k: v for k, v in asdict(policy).items() if k != "common_words"},
                "common_words": len(policy.common_words),
                "token_counter": f"vLLM /tokenize of {checker_model_id}",
                "on_api_failure": "retry 5 times, then stop the run (resumable)",
            }
            if cond.evolving_skill
            else None
        ),
        attacker=attacker.config() if isinstance(attacker, MuzzleAttacker | AutoDojoAttacker) else None,
        served_models=served,
        code=git_state(),
        versions=package_versions(),
        created_at=now(),
    )

    skill_file: SkillFileSource
    if cond.evolving_skill:
        evolver = SkillFileEvolver(
            load_skill_file(s0_path),
            openrouter_judge_fn(judge_model, seed=seed),
            run_dir / "skill_files",
            vllm_token_counter(checker_base_url, checker_model_id),
            policy=policy,
            digest_size=digest_size,
        )
        skill_file = EvolvingSkillFile(evolver, manifest_data["benchmark_version"])
    elif cond.checker:
        skill_file = FixedSkillFile(s0_path.read_text(encoding="utf-8"))
    else:
        skill_file = NoChecker()
    checker_fn = (
        local_checker_fn(checker_model_id, checker_base_url, enable_thinking=experiment_models.CHECKER_ENABLE_THINKING)
        if cond.checker
        else None
    )

    def victim_factory(unit: Unit, factory: ToolMiddlewareFactory | None) -> DeepAgentsPipeline:
        return DeepAgentsPipeline.from_config(victim_config, tool_middleware_factory=factory)

    runner = Runner(
        config=config,
        condition=cond,
        run_dir=run_dir,
        units=build_units(cond, manifest_data, trials_per_pair, limit, suites_tuple),
        attacker=attacker,
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
        click.echo(f"Interrupted. Partial report: {write_report(runner.run_dir)}", err=True)
        click.echo("Run the same command again to resume from the last finished unit.", err=True)
        sys.exit(130)
    click.echo(json.dumps(quick_summary(records), indent=2))
    click.echo(f"report: {write_report(runner.run_dir)}")


if __name__ == "__main__":
    main()
