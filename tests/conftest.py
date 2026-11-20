"""Shared test setup."""

import importlib

import pytest

FAKE_REASONING_PROMPT = "Think first.[THINK]Draft your thoughts here.[/THINK]Then answer."
"""Stands in for Magistral's `SYSTEM_PROMPT.txt`, so no test downloads it."""


@pytest.fixture(autouse=True)
def _no_hf_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Magistral victim's reasoning prompt is downloaded at run start; tests get a fixed stand-in instead."""

    def fake_fetch(repo_id: str, filename: str, *args: object, **kwargs: object) -> str:
        return FAKE_REASONING_PROMPT

    for module in ("agentdojo.agent_pipeline.deepagents_pipeline", "agentdojo.runner.run_condition"):
        try:
            loaded = importlib.import_module(module)
        except ImportError:  # the deepagents extra is not installed
            continue
        monkeypatch.setattr(loaded, "fetch_hf_text", fake_fetch)
