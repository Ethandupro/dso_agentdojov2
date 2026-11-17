"""Port of SuperRed's MUZZLE attacker: its iterative part only (DESIGN.md, "Evolving attacker").

- `vendor/`: upstream's Judge and offline PAIR, byte-identical apart from import lines (pinned in `UPSTREAM.json`).
- `llm.py`: the two SuperRed core types those files use, on the OpenAI SDK (OpenRouter by default).
- `attacker.py`: `MuzzleAttacker`, the runner's evolving attacker for C2 and C4.
- `upstream_licenses/`: upstream's LICENSE (MIT), NOTICE, bundled Apache-2.0 (MUZZLE) and MIT (PAIR) licences, and
  its ASSUMPTIONS.md (the SuperRed port's own deviations from MUZZLE).
"""

from agentdojo.attacks.adaptive.muzzle.attacker import DEVIATIONS, MuzzleAttacker, MuzzleSettings
from agentdojo.attacks.adaptive.muzzle.llm import AttackerModelConfig, BudgetExhaustedError, OpenAICompatibleLLM

__all__ = [
    "DEVIATIONS",
    "AttackerModelConfig",
    "BudgetExhaustedError",
    "MuzzleAttacker",
    "MuzzleSettings",
    "OpenAICompatibleLLM",
]
