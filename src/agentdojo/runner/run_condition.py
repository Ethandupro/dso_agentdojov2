"""One command per run: one condition at one seed, resumable.

    uv run --extra deepagents python -m agentdojo.runner.run_condition --condition C1
    uv run --extra deepagents python -m agentdojo.runner.run_condition --condition C1 --limit 5   # smoke run

Runs on the GPU VM against the local vLLM servers (`local_victim/serve_all.sh`): the victim (Magistral Small 1.2, thinking,
no draft model) on port 8000. Results go to `results/<run_id>/seed_<n>/` (`results/<run_id>-limit<N>/...` for smoke
runs). Run the same command again to resume after a crash or Ctrl-C; it refuses to resume under a different config.

Defence (redesign of 2026-11-19): the victim has a skill library, the fixed skills in
`skillfile_evolution/skill_files/agent_skills/` plus, in C1-C4, the security skill S0 (evolved by the judge in C3/C4).
`--external-checker` runs the earlier design instead (no skills; S0_checker.md read by the Qwen3.5-9B checker on
port 8002) and writes to its own `-external-checker` results directory.

All five conditions run here. C2 and C4 also need the attacker's local vLLM server (started by
`local_victim/serve_all.sh` on GPU 1). C3 and C4 also need the judge's OpenRouter key (`OPENROUTER_API_KEY`,
e.g. from `--env-file ~/api.env`, whose values win over variables already set in the shell); the key is never written
to the run directory.
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import click
import openai
from dotenv import load_dotenv

from agentdojo import experiment_models
from agentdojo.agent_pipeline.agent_skills import SKILLS_PROMPTS, load_skills_dir
from agentdojo.agent_pipeline.deepagents_pipeline import (
    DeepAgentsConfig,
    DeepAgentsPipeline,
    ToolMiddlewareFactory,
    fetch_hf_text,
)
from agentdojo.attacks.adaptive.autodojo import (
    AutoDojoAttacker,
    AutoDojoSettings,
    SkillAgentAttacker,
    SkillAgentSettings,
)
from agentdojo.attacks.adaptive.muzzle import AttackerModelConfig, MuzzleAttacker, MuzzleSettings
from agentdojo.checker.middleware import CHECKER_REPLY_SCHEMA, CHECKER_USER_TEMPLATE, local_checker_fn
from agentdojo.runner.conditions import CONDITIONS, PairFilter, Unit, build_units, load_subset
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
    JUDGE_SYSTEM_PROMPTS,
    SkillFileEvolver,
    openrouter_client,
    openrouter_judge_fn,
)
from agentdojo.skillfile.structure import load_skill_file
from agentdojo.skillfile.tokens import vllm_token_counter

JUDGE_KEY_ENV = experiment_models.JUDGE_API_KEY_ENV

AUTODOJO_LOOPS = ("autodojo", "skill-agent")
"""Evolving attacker loops that share AutoDojo's settings (memory, strategy bank, victim trace); 'muzzle' does not."""


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


THINKING_TEXT = re.compile(r"<think>\s*(?!</think>)\S")


def reply_thinks(message: dict[str, Any]) -> bool:
    """Whether a chat-completion message shows thinking: a non-empty reasoning field (vLLM's reasoning parser fills
    `reasoning_content`, newer builds `reasoning`), or a `<think>` block with text in the content."""
    for key in ("reasoning_content", "reasoning"):
        if str(message.get(key) or "").strip():
            return True
    return bool(THINKING_TEXT.search(str(message.get("content") or "")))


def check_attacker_not_thinking(base_url: str, model_id: str, timeout: float = 120.0) -> None:
    """The attacker never thinks (user decision 2026-11-20). One short request, sent exactly as the attacker's own
    (`enable_thinking: false`), must come back without thinking, or the run does not start."""
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Reply with the single word OK."}],
        "max_tokens": 64,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": experiment_models.ATTACKER_ENABLE_THINKING},
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise PreflightError(f"the attacker at {base_url} did not answer a test request ({e})") from e
    message = (data.get("choices") or [{}])[0].get("message") or {}
    if reply_thinks(message):
        raise PreflightError(
            f"the attacker {model_id} thinks although every request sends enable_thinking=false; restart it with "
            "local_victim/serve_all.sh (its chat template must honour enable_thinking)"
        )


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


def victim_deepagents_config(
    model_id: str,
    base_url: str,
    fetch: Callable[[str, str], str] | None = None,
    temperature: float | None = None,
) -> DeepAgentsConfig:
    """The victim's settings. A Magistral victim (default since 2026-11-20) thinks: its model card's reasoning system
    prompt goes ahead of the agent's, with the card's sampling. Other (Qwen) victims run non-thinking at temperature
    0, as before. `temperature` (`--victim-temperature`) overrides the default, e.g. a low value so the same
    injection gets (nearly) the same answer every time."""
    if experiment_models.is_magistral(model_id):
        return DeepAgentsConfig(
            model_id=model_id,
            base_url=base_url,
            enable_thinking=None,
            temperature=experiment_models.MAGISTRAL_TEMPERATURE if temperature is None else temperature,
            top_p=experiment_models.MAGISTRAL_TOP_P,
            reasoning_system_prompt=(fetch or fetch_hf_text)(model_id, experiment_models.REASONING_PROMPT_FILE),
        )
    return DeepAgentsConfig(
        model_id=model_id,
        base_url=base_url,
        enable_thinking=False,
        temperature=0.0 if temperature is None else temperature,
    )


SMOKE_PAIRS = 8
"""`--smoke`: pairs per run (user decision 2026-11-20), each with one injection point."""
SMOKE_SUITES = ("banking", "travel")
"""`--smoke`: the suites it draws from in turn, the same two as the 64-pair pilot subset."""


def attacker_model_config(model_id: str, base_url: str, seed: int) -> AttackerModelConfig:
    """The local attacker model (decided 2026-11-17 by the user: no API), shared by both evolving attackers."""
    return AttackerModelConfig(
        model=model_id,
        base_url=base_url,
        api_key_env=None,
        seed=seed,
        enable_thinking=experiment_models.ATTACKER_ENABLE_THINKING,
    )


def muzzle_settings(model_id: str, base_url: str, seed: int) -> MuzzleSettings:
    return MuzzleSettings(attacker=attacker_model_config(model_id, base_url, seed))


def autodojo_settings(
    model_id: str,
    base_url: str,
    seed: int,
    external_checker: bool = False,
    memory: bool = True,
    strategy_bank: bool = True,
    temperature: float | None = None,
) -> AutoDojoSettings:
    settings = AutoDojoSettings(
        attacker=attacker_model_config(model_id, base_url, seed),
        target="filter" if external_checker else "agent",
        memory=memory,
        strategy_bank=strategy_bank,
    )
    if temperature is not None:
        settings.temperature = temperature
    return settings


def skill_agent_settings(
    model_id: str,
    base_url: str,
    seed: int,
    memory: bool = True,
    strategy_bank: bool = True,
    temperature: float | None = None,
    distill_every: int = 0,
    distill_mode: Literal["contrast", "plain"] = "contrast",
) -> SkillAgentSettings:
    """Item 7 (D2): the evolving attacker as a tool-using agent with its own skill files. Only the skills design
    (`target='agent'`), so it is never paired with --external-checker."""
    settings = SkillAgentSettings(
        attacker=attacker_model_config(model_id, base_url, seed),
        target="agent",
        memory=memory,
        strategy_bank=strategy_bank,
        distill_every=distill_every,
        distill_mode=distill_mode,
    )
    if temperature is not None:
        settings.temperature = temperature
    return settings


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
@click.option(
    "--subset",
    "subset_path",
    type=click.Path(path_type=Path, dir_okay=False, exists=True),
    default=None,
    help="Pilot cut: a file listing user tasks to keep, one suite/user_task per line (e.g. "
    "skillfile_evolution/subsets/banking_travel_64.txt). Every injection task is kept. Gets its own results directory.",
)
@click.option(
    "--single-injection-point",
    is_flag=True,
    default=False,
    help="Pilot cut: keep only pairs whose user task reads one injection vector. Gets its own results directory.",
)
@click.option(
    "--smoke",
    is_flag=True,
    default=False,
    help=f"Smoke run preset: single-injection-point pairs only, {SMOKE_PAIRS} pairs (or --limit) taken in turn from "
    f"each suite ({','.join(SMOKE_SUITES)} unless --suites is given). Gets its own '-smoke' results directory.",
)
@click.option("--trials-per-pair", type=int, default=3, show_default=True)
@click.option("--manifest", type=click.Path(path_type=Path), default=DEFAULT_MANIFEST_PATH, show_default=True)
@click.option("--victim-model-id", default=experiment_models.VICTIM_MODEL_ID, show_default=True)
@click.option("--victim-base-url", default=experiment_models.VICTIM_BASE_URL, show_default=True)
@click.option("--checker-model-id", default=experiment_models.CHECKER_MODEL_ID, show_default=True)
@click.option("--checker-base-url", default=experiment_models.CHECKER_BASE_URL, show_default=True)
@click.option("--min-span-chars", type=int, default=0, show_default=True, help="External checker only.")
@click.option(
    "--external-checker",
    is_flag=True,
    default=False,
    help="Ablation: the design before 2026-11-19. No skills; S0_checker.md goes to the external checker model.",
)
@click.option(
    "--skills-prompt",
    "skills_prompt_style",
    type=click.Choice(sorted(SKILLS_PROMPTS)),
    default="nudge",
    show_default=True,
    help="How the skill list is introduced: 'nudge' tells the agent to load the skills that apply before its first "
    "tool call; 'plain' (the first pilot's prompt) only offers them. 'plain' runs get a '-plain-skills' directory.",
)
@click.option(
    "--skills-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=experiment_models.AGENT_SKILLS_DIR,
    show_default=True,
    help="The victim's fixed skills (<name>/SKILL.md), listed next to S0.",
)
@click.option("--env-file", type=click.Path(path_type=Path), default=None, help="Loads API keys (judge, attacker).")
@click.option("--judge-model", default=DEFAULT_JUDGE_MODEL, show_default=True, help="C3/C4 only.")
@click.option(
    "--judge-edits-description/--judge-keeps-description",
    default=True,
    show_default=True,
    help="C3/C4 skills runs: whether the judge may also rewrite S0's one-line description, which decides how often "
    "the agent loads S0 (decided 2026-11-19 by user). Never with --external-checker.",
)
@click.option("--digest-size", type=int, default=20, show_default=True, help="Trials in the judge's history digest.")
@click.option("--attacker-model-id", default=experiment_models.ATTACKER_MODEL_ID, show_default=True, help="C2/C4.")
@click.option("--attacker-base-url", default=experiment_models.ATTACKER_BASE_URL, show_default=True, help="C2/C4.")
@click.option(
    "--attacker",
    "attacker_loop",
    type=click.Choice(["autodojo", "skill-agent", "muzzle"]),
    default="autodojo",
    show_default=True,
    help="C2/C4 evolving attacker: 'autodojo' scores each rewrite on the real pipeline (leaderboard of the pair's "
    "own trials); 'skill-agent' (item 7, D2) runs that optimizer as a tool-using agent that reads and writes its own "
    "strategy skill files across pairs, and gets a '-skill-agent' results directory; 'muzzle' is the earlier offline "
    "stand-in loop, kept for ablation.",
)
@click.option(
    "--attacker-memory/--no-attacker-memory",
    default=True,
    show_default=True,
    help="C2/C4/C0E with --attacker autodojo: the cross-pair playbook (winning texts and lessons from other pairs "
    "of the same run). On by default since 2026-11-20 (user decision); --no-attacker-memory is the ablation and gets "
    "a '-no-memory' results directory.",
)
@click.option(
    "--graded-score/--no-graded-score",
    default=True,
    show_default=True,
    help="Score each attacked trial 0-4 from the victim's trace (runner.score) as a finer signal for the evolving "
    "attacker's leaderboard; the attacker success rate still uses only the binary security result. On by default "
    "(user decision 2026-11-20); --no-graded-score is the ablation and gets a '-binary-score' results directory.",
)
@click.option(
    "--strategy-bank/--no-strategy-bank",
    default=True,
    show_default=True,
    help="C2/C4/C0E with --attacker autodojo: show the optimizer the named strategy bank (autodojo.strategies) and "
    "record which strategy it names, for a wins-by-strategy report. On by default (user decision 2026-11-20); "
    "--no-strategy-bank is the ablation and gets a '-no-strategies' results directory.",
)
@click.option(
    "--show-victim-trace/--no-victim-trace",
    default=True,
    show_default=True,
    help="C2/C4/C0E with --attacker autodojo: show the evolving attacker an S0-redacted digest of the victim's run "
    "(its thinking, what it said, the tools it called) on the leaderboard, so it sees why an injection was ignored. "
    "On by default (user decision 2026-11-20); --no-victim-trace gets a '-no-victim-trace' results directory.",
)
@click.option(
    "--stop-on-success",
    is_flag=True,
    default=False,
    help="C2/C4/C0E (evolving pairs): stop a pair at its first success and skip its later trials. A pair whose trial 0 "
    "(the original AgentDojo text) already succeeds is never evolved and is reported as 'baseline already works'; the "
    "evolving ASR counts only the other pairs. Gets a '-first-success' results directory.",
)
@click.option(
    "--victim-temperature",
    type=float,
    default=None,
    help="Victim sampling temperature (default: Magistral's model-card 0.7). Gets a '-vtemp<T>' results directory.",
)
@click.option(
    "--attacker-temperature",
    type=float,
    default=None,
    help="C2/C4/C0E: the evolving attacker's sampling temperature (default 1.0, as AutoDojo). Gets an '-atemp<T>' "
    "results directory.",
)
@click.option(
    "--distill-every",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="With --attacker skill-agent: every K finished pairs, the attacker model reads those pairs' attempts (wins "
    "and failures) and distils them into general attack-pattern notes in its skill library, which it can load in "
    "later trials. Notes carrying a pair's values or copying an attempt are rejected. 0 = off. Gets a "
    "'-distill<K>' results directory.",
)
@click.option(
    "--distill-mode",
    type=click.Choice(["contrast", "plain"]),
    default="contrast",
    show_default=True,
    help="With --distill-every: 'contrast' (default) matches a failed and a winning attempt (same item first, then "
    "same strategy on different items) and asks for do/avoid rules from the difference, falling back to 'plain' "
    "until a matched pair exists; 'plain' is the PR #53 distillation over the raw attempts and gets a "
    "'-distill<K>-plain' results directory.",
)
def main(
    condition: str,
    seed: int,
    results_root: Path,
    limit: int | None,
    suites: str | None,
    subset_path: Path | None,
    single_injection_point: bool,
    smoke: bool,
    trials_per_pair: int,
    manifest: Path,
    victim_model_id: str,
    victim_base_url: str,
    checker_model_id: str,
    checker_base_url: str,
    min_span_chars: int,
    external_checker: bool,
    skills_prompt_style: str,
    skills_dir: Path,
    env_file: Path | None,
    judge_model: str,
    judge_edits_description: bool,
    digest_size: int,
    attacker_model_id: str,
    attacker_base_url: str,
    attacker_loop: str,
    attacker_memory: bool,
    graded_score: bool,
    strategy_bank: bool,
    show_victim_trace: bool,
    stop_on_success: bool,
    victim_temperature: float | None,
    attacker_temperature: float | None,
    distill_every: int,
    distill_mode: str,
) -> None:
    if env_file is not None:
        load_dotenv(env_file, override=True)
    cond = CONDITIONS[condition]
    if smoke:
        single_injection_point = True
        limit = SMOKE_PAIRS if limit is None else limit
        suites = ",".join(SMOKE_SUITES) if suites is None else suites

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

    subset_tasks: frozenset[str] | None = None
    if subset_path is not None:
        try:
            subset_tasks = load_subset(subset_path, manifest_data)
        except ValueError as e:
            raise click.ClickException(str(e)) from e
    pair_filter = PairFilter(
        user_tasks=subset_tasks, single_injection_point=single_injection_point, balance_suites=smoke
    )
    units = build_units(cond, manifest_data, trials_per_pair, limit, suites_tuple, pair_filter)
    if not any(u.pair is not None for u in units):
        raise click.ClickException("the suite/subset/single-injection-point cuts leave no pairs to run")

    use_checker = external_checker and cond.s0
    defense = "checker" if external_checker else "skills"
    agent_skills = [] if external_checker else load_skills_dir(skills_dir)
    if any(s.name == experiment_models.S0_SKILL_NAME for s in agent_skills):
        raise click.ClickException(f"{skills_dir} already has a skill named {experiment_models.S0_SKILL_NAME}")

    try:
        victim_config = victim_deepagents_config(victim_model_id, victim_base_url, temperature=victim_temperature)
    except OSError as e:
        raise click.ClickException(
            f"could not download {experiment_models.REASONING_PROMPT_FILE} of {victim_model_id}: {e}"
        ) from e
    try:
        served = {"victim": check_served("victim", victim_base_url, victim_model_id)}
        if use_checker:
            served["checker"] = check_served("checker", checker_base_url, checker_model_id)
        if cond.evolving_skill:
            check_judge(judge_model)
        evolving_attack = cond.attack_mode == "evolving"
        if evolving_attack:
            served["attacker"] = check_served("attacker", attacker_base_url, attacker_model_id)
            check_attacker_not_thinking(attacker_base_url, attacker_model_id)
    except PreflightError as e:
        raise click.ClickException(str(e)) from e

    token_model_id, token_base_url = (
        (checker_model_id, checker_base_url) if external_checker else (victim_model_id, victim_base_url)
    )
    s0_path = experiment_models.CHECKER_SKILL_FILE if external_checker else Path(manifest_data["s0_path"])
    s0_text = s0_path.read_text(encoding="utf-8")
    policy = EditPolicy(description_editable=judge_edits_description and not external_checker)
    static = StaticAttacker(manifest_data)
    attacker: Attacker = static
    if attacker_temperature is not None and not (evolving_attack and attacker_loop in AUTODOJO_LOOPS):
        raise click.ClickException("--attacker-temperature applies to the autodojo/skill-agent evolving attacker only")
    if distill_every and not (evolving_attack and attacker_loop == "skill-agent"):
        raise click.ClickException("--distill-every needs an evolving-attack condition with --attacker skill-agent")
    if distill_mode != "contrast" and not distill_every:
        raise click.ClickException("--distill-mode only applies together with --distill-every K")
    if stop_on_success and not evolving_attack:
        raise click.ClickException("--stop-on-success applies to evolving-attack conditions only (C0E, C2, C4)")
    if evolving_attack:
        if attacker_loop == "muzzle":
            attacker = MuzzleAttacker(manifest_data, muzzle_settings(attacker_model_id, attacker_base_url, seed))
        elif attacker_loop == "skill-agent":
            if external_checker:
                raise click.ClickException("--attacker skill-agent is a skills-design loop; drop --external-checker")
            attacker = SkillAgentAttacker(
                manifest_data,
                skill_agent_settings(
                    attacker_model_id,
                    attacker_base_url,
                    seed,
                    attacker_memory,
                    strategy_bank,
                    attacker_temperature,
                    distill_every,
                    "plain" if distill_mode == "plain" else "contrast",
                ),
            )
        else:
            attacker = AutoDojoAttacker(
                manifest_data,
                autodojo_settings(
                    attacker_model_id,
                    attacker_base_url,
                    seed,
                    external_checker,
                    attacker_memory,
                    strategy_bank,
                    attacker_temperature,
                ),
            )
    run_dir = run_dir_for(
        results_root,
        cond,
        seed,
        limit,
        suites_tuple,
        external_checker,
        skills_prompt_style,
        subset=subset_path.stem if subset_path is not None else None,
        single_injection_point=single_injection_point,
        smoke=smoke,
        no_attacker_memory=evolving_attack and attacker_loop in AUTODOJO_LOOPS and not attacker_memory,
        no_graded_score=evolving_attack and not graded_score,
        no_strategy_bank=evolving_attack and attacker_loop in AUTODOJO_LOOPS and not strategy_bank,
        no_victim_trace=evolving_attack and attacker_loop in AUTODOJO_LOOPS and not show_victim_trace,
        skill_agent=evolving_attack and attacker_loop == "skill-agent",
        stop_on_success=stop_on_success,
        victim_temperature=victim_temperature,
        attacker_temperature=attacker_temperature,
        distill_every=distill_every,
        distill_plain=bool(distill_every) and distill_mode == "plain",
    )
    config = RunConfig(
        condition=cond.name,
        run_id=cond.run_id,
        seed=seed,
        checker=use_checker,
        defense=defense,
        skills=(
            None
            if external_checker
            else {
                "dir": str(skills_dir),
                "fixed": {s.name: s.sha256 for s in agent_skills},
                "security_skill": (
                    {
                        "name": experiment_models.S0_SKILL_NAME,
                        "description": experiment_models.S0_SKILL_DESCRIPTION,
                    }
                    if cond.s0
                    else None
                ),
                "load_tool": "load_skill",
                "prompt_style": skills_prompt_style,
                "prompt_sha256": sha256_text(SKILLS_PROMPTS[skills_prompt_style]),  # type: ignore[index]
            }
        ),
        skill_mode=cond.skill_mode,
        attack_mode=cond.attack_mode,
        trials_per_pair=trials_per_pair,
        limit=limit,
        suites=list(suites_tuple) if suites_tuple else None,
        subset=(
            {
                "path": str(subset_path),
                "sha256": sha256_text(subset_path.read_text(encoding="utf-8")),
                "user_tasks": sorted(subset_tasks),
            }
            if subset_path is not None and subset_tasks is not None
            else None
        ),
        single_injection_point=single_injection_point,
        smoke=smoke or None,
        graded_score=graded_score if evolving_attack else None,
        show_victim_trace=show_victim_trace if evolving_attack else None,
        victim_trace_thinking=True if evolving_attack and show_victim_trace else None,
        stop_on_success=stop_on_success or None,
        benchmark_version=manifest_data["benchmark_version"],
        manifest_path=str(manifest),
        manifest_sha256=manifest_sha256(manifest),
        s0_path=str(s0_path),
        s0_sha256=sha256_text(s0_text) if external_checker else manifest_data["s0_sha256"],
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
            if use_checker
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
                "system_prompt_sha256": sha256_text(JUDGE_SYSTEM_PROMPTS[defense]),
                "digest_size": digest_size,
                "policy": {k: v for k, v in asdict(policy).items() if k != "common_words"},
                "common_words": len(policy.common_words),
                "token_counter": f"vLLM /tokenize of {token_model_id}",
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
        s0_file = load_skill_file(s0_path)
        if not external_checker:
            s0_file.description = experiment_models.S0_SKILL_DESCRIPTION
        # File length is counted with the tokenizer of the model that reads the file: the victim, or the checker.
        evolver = SkillFileEvolver(
            s0_file,
            openrouter_judge_fn(judge_model, seed=seed),
            run_dir / "skill_files",
            vllm_token_counter(token_base_url, token_model_id),
            policy=policy,
            digest_size=digest_size,
            defense=defense,
        )
        skill_file = EvolvingSkillFile(evolver, manifest_data["benchmark_version"])
    elif cond.s0:
        skill_file = FixedSkillFile(s0_text)
    else:
        skill_file = NoChecker()
    checker_fn = (
        local_checker_fn(checker_model_id, checker_base_url, enable_thinking=experiment_models.CHECKER_ENABLE_THINKING)
        if use_checker
        else None
    )

    def victim_factory(unit: Unit, factory: ToolMiddlewareFactory | None) -> DeepAgentsPipeline:
        return DeepAgentsPipeline.from_config(victim_config, tool_middleware_factory=factory)

    runner = Runner(
        config=config,
        condition=cond,
        run_dir=run_dir,
        units=units,
        attacker=attacker,
        static_attacker=static,
        skill_file=skill_file,
        victim_factory=victim_factory,
        checker_fn=checker_fn,
        min_span_chars=min_span_chars,
        echo=click.echo,
        defense=defense,
        agent_skills=agent_skills,
        skills_prompt_style=skills_prompt_style,  # type: ignore[arg-type]
        graded_score=graded_score,
        show_victim_trace=show_victim_trace,
        stop_on_success=stop_on_success,
    )
    try:
        records = runner.run()
    except ConfigMismatchError as e:
        raise click.ClickException(str(e)) from e
    except KeyboardInterrupt:
        click.echo(f"Interrupted. Partial report: {write_report(runner.run_dir)}", err=True)
        click.echo("Run the same command again to resume from the last finished unit.", err=True)
        sys.exit(130)
    click.echo(json.dumps(quick_summary(records, trials_per_pair), indent=2))
    click.echo(f"report: {write_report(runner.run_dir)}")


if __name__ == "__main__":
    main()
