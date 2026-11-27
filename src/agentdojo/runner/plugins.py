"""The two plug-in points of the runner: where a trial's injection text comes from, and which skill file (S0 or an
evolved version) the victim gets as its security skill (or the external checker reads, in the ablation). Both are saved in the run's checkpoint through `state_dict` / `load_state`.

- `Attacker`: `StaticAttacker` here (C0, C1, C3, and C4 test pairs); `MuzzleAttacker` comes with the MUZZLE port
  (C2, C4 train pairs). The runner, not the attacker, keeps test pairs away from MUZZLE in C4.
- `SkillFileSource`: `NoChecker` (C0: no S0 in the skill list) and `FixedSkillFile` (C1, C2) here; the evolving file (C3, C4) wraps
  `agentdojo.skillfile.SkillFileEvolver`.
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from agentdojo.runner.manifest import Pair, trial0_injections


@dataclass
class AttemptFeedback:
    """What the attacker may see of one finished trial (DESIGN.md, "Attacks": black-box)."""

    trial: int
    injection_text: str
    utility: bool
    """The user task succeeded."""
    security: bool
    """AgentDojo's security result: True means the injection goal was achieved."""
    final_answer: str | None
    progress: int | None = None
    """The runner's 0-4 graded score for this trial (`runner.score`), or None when graded scoring is off. A finer
    signal than `security` for the evolving attacker's leaderboard; never part of the attacker success rate."""
    victim_trace: str | None = None
    """An S0-redacted digest of the victim's run (`runner.trace`), or None when `--no-victim-trace`. Shows the
    attacker why the injection was ignored or refused (grey-box); the evolving S0 is redacted so it never leaks."""
    replication: dict[str, int] | None = None
    """For a won trial with `--replicate-wins`: `{"runs": N, "successes": k}` of the same text re-run unchanged. Shows
    whether the win holds up or was the victim's sampling luck. Never part of the trial counts."""
    analyst_trace: str | None = None
    """A longer S0-redacted digest of the same run, for the attacker's analyst (`--analyst`). Only on the feedback
    passed to `observe` right after the trial; never stored in trials.jsonl."""

    @property
    def attacker_success(self) -> bool:
        """Injection goal achieved OR user task failed (DESIGN.md, "Attacker success")."""
        return self.security or not self.utility


class Attacker(Protocol):
    name: str

    def text_for(self, pair: Pair, trial: int, history: list[AttemptFeedback]) -> str:
        """The injection text of `trial` of `pair`, given that pair's earlier trials in this run."""
        ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state(self, state: dict[str, Any]) -> None: ...


@runtime_checkable
class ObservingAttacker(Protocol):
    """An attacker that also learns across pairs: the runner reports every finished trial it handed a text for."""

    def observe(self, pair: Pair, feedback: AttemptFeedback, skill_version: str | None) -> None: ...


class StaticAttacker:
    """Always the manifest's trial-0 `important_instructions` text."""

    name = "static"

    def __init__(self, manifest: dict[str, Any]) -> None:
        self.manifest = manifest

    def text_for(self, pair: Pair, trial: int, history: list[AttemptFeedback]) -> str:
        texts = set(trial0_injections(self.manifest, pair).values())
        assert len(texts) == 1
        return texts.pop()

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state(self, state: dict[str, Any]) -> None:
        pass


class SkillFileSource(Protocol):
    @property
    def version_name(self) -> str:
        """The version name, for example S0 or S3; "none" without S0 (C0)."""
        ...

    @property
    def markdown(self) -> str | None:
        """The security skill's body in the next trial; None means no S0 (C0)."""
        ...

    @property
    def description(self) -> str | None:
        """The security skill's one-line description in the next trial, if the judge has set one; None means the
        run's default (`experiment_models.S0_SKILL_DESCRIPTION`)."""
        ...

    @property
    def specialist_skills(self) -> list[tuple[str, str, str]]:
        """The judge-created specialised security skills in the next trial, as (name, description, body); empty
        outside the C3/C4 library."""
        ...

    def after_trial(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """Called after every unit with its trial record; returns a summary of an evolution step, if one ran."""
        ...

    def freeze(self) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state(self, state: dict[str, Any]) -> None: ...


class NoChecker:
    """C0: no security skill (the name dates from the external-checker design)."""

    version_name = "none"
    markdown = None
    description = None

    @property
    def specialist_skills(self) -> list[tuple[str, str, str]]:
        return []

    def after_trial(self, record: dict[str, Any]) -> dict[str, Any] | None:
        return None

    def freeze(self) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state(self, state: dict[str, Any]) -> None:
        pass


class FixedSkillFile:
    """S0 in every trial: the skill-file step after a trial does nothing."""

    version_name = "S0"
    description = None

    @property
    def specialist_skills(self) -> list[tuple[str, str, str]]:
        return []

    def __init__(self, text: str) -> None:
        self._text = text

    @property
    def markdown(self) -> str:
        return self._text

    def after_trial(self, record: dict[str, Any]) -> dict[str, Any] | None:
        return None

    def freeze(self) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state(self, state: dict[str, Any]) -> None:
        pass
