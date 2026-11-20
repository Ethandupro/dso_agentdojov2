"""AutoDojo's optimization loop as a runner `Attacker` for C2 and C4 (arXiv:2606.15057, MIT).

Unlike the MUZZLE port (`agentdojo.attacks.adaptive.muzzle`), which optimises offline against a Mistral stand-in
checker and stand-in agent, this attacker evolves a pair's injection across the pair's own real trials: trial k's
text is written by an optimizer LLM that sees a leaderboard of this pair's earlier trials (0..k-1) scored by their
**real** result through the real checker and victim. It stays black-box -- the optimizer sees only success or
failure, never the checker's removed spans or the victim's reasoning -- matching AutoDojo's "outcome feedback".
"""

from agentdojo.attacks.adaptive.autodojo.attacker import DEVIATIONS, AutoDojoAttacker, AutoDojoSettings
from agentdojo.attacks.adaptive.autodojo.skill_agent import SkillAgentAttacker, SkillAgentSettings

__all__ = [
    "DEVIATIONS",
    "AutoDojoAttacker",
    "AutoDojoSettings",
    "SkillAgentAttacker",
    "SkillAgentSettings",
]
