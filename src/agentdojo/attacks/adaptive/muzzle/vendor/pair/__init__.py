"""Vendored PAIR primitives (byte-faithful to patrickrchao/JailbreakingLLMs).

Copied from the reviewed superred ``pair_optimizer`` port; the only change to
``attacker.py``/``evaluator.py`` is the internal import path (``pair_optimizer`` ->
``agentdojo.attacks.adaptive.muzzle.vendor.pair``). ``json_utils.py`` and ``prompts.py`` are byte-identical.
MUZZLE's offline-surrogate augmentation lives in :mod:`agentdojo.attacks.adaptive.muzzle.vendor.pair.bridge`.
See ASSUMPTIONS.md.
"""

from __future__ import annotations

from agentdojo.attacks.adaptive.muzzle.vendor.pair.attacker import PairAttacker, PairStream
from agentdojo.attacks.adaptive.muzzle.vendor.pair.evaluator import GCG_REFUSAL_KEYWORDS, PairEvaluator, PairScore
from agentdojo.attacks.adaptive.muzzle.vendor.pair.json_utils import (
    JsonExtractionError,
    PairProposal,
    extract_attack_json,
)
from agentdojo.attacks.adaptive.muzzle.vendor.pair.prompts import (
    append_superred_context,
    get_attacker_system_prompts,
    get_init_msg,
    get_judge_system_prompt,
    process_target_response,
)

__all__ = [
    "PairAttacker",
    "PairStream",
    "PairEvaluator",
    "PairScore",
    "GCG_REFUSAL_KEYWORDS",
    "JsonExtractionError",
    "PairProposal",
    "extract_attack_json",
    "get_attacker_system_prompts",
    "get_judge_system_prompt",
    "get_init_msg",
    "process_target_response",
    "append_superred_context",
]
