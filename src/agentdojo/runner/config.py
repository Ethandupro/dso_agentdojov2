"""A run's `config.json`: everything that defines the run, written before any model call. A run resumes only under an
identical config (DESIGN.md, "Conditions and result segmentation"). Secrets are never stored."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

CONFIG_FORMAT = 1
RESUME_IGNORED_KEYS = frozenset({"created_at"})
"""Keys that may differ when resuming."""

VERSIONED_PACKAGES = ("agentdojo", "deepagents", "langchain", "langchain-openai", "langgraph", "openai")


class RunConfig(BaseModel):
    format: int = CONFIG_FORMAT
    condition: str
    run_id: str
    seed: int
    checker: bool
    """Whether the external checker runs (only with `--external-checker`, in C1-C4)."""
    defense: str = "skills"
    """`skills`: S0 is one of the victim's skills (default since 2026-11-19). `checker`: the external-checker ablation."""
    skills: dict[str, Any] | None = None
    """The victim's skill library: the fixed skills' hashes and S0's listed name and description."""
    skill_mode: str
    attack_mode: str
    trials_per_pair: int
    limit: int | None
    """Smoke runs: only the first `limit` pairs of the run order. Such runs get their own directory."""
    suites: list[str] | None = None
    """Pilot cut: only pairs in these suites (e.g. banking, travel). Such runs get their own directory."""
    subset: dict[str, Any] | None = None
    """Pilot cut from a `--subset` file: its path, sha256 and the user tasks it lists. Own directory."""
    single_injection_point: bool = False
    """Pilot cut: only pairs whose user task reads one injection vector. Own directory."""
    smoke: bool | None = None
    """`--smoke` preset: single injection point, `limit` pairs taken in turn from each suite. Own directory. None
    (not False) when off, so runs made before the flag existed still resume."""
    benchmark_version: str
    manifest_path: str
    manifest_sha256: str
    s0_path: str
    s0_sha256: str
    victim: dict[str, Any]
    """`DeepAgentsConfig` as dumped by pydantic."""
    checker_model: dict[str, Any] | None
    judge: dict[str, Any] | None = None
    attacker: dict[str, Any] | None = None
    served_models: dict[str, list[str]]
    """What each local server reported serving at preflight."""
    code: dict[str, Any]
    versions: dict[str, str | None]
    created_at: str


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def git_state(repo: Path = Path(".")) -> dict[str, Any]:
    """The commit the run's code came from, and whether the working tree had uncommitted changes."""

    def git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=repo, capture_output=True, text=True, check=True, timeout=10
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": commit, "git_dirty": None if status is None else bool(status)}


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in VERSIONED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def config_diff(old: dict[str, Any], new: dict[str, Any], prefix: str = "") -> list[str]:
    """Dotted keys whose values differ, ignoring `RESUME_IGNORED_KEYS` at the top level."""
    diffs: list[str] = []
    for key in sorted(set(old) | set(new)):
        if not prefix and key in RESUME_IGNORED_KEYS:
            continue
        a, b = old.get(key), new.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            diffs.extend(config_diff(a, b, f"{prefix}{key}."))
        elif a != b:
            diffs.append(f"{prefix}{key}: {json.dumps(a, default=str)} -> {json.dumps(b, default=str)}")
    return diffs
