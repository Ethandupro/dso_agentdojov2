"""The models of the skill-file experiments and where they are served (decided 2026-11-16 by the user).

The victim, the checker and the MUZZLE attacker run on the project's GPU VM behind vLLM's OpenAI-compatible server
(see `local_victim/README.md` and `local_victim/serve_all.sh`): the victim on GPU 0 (port 8000), the checker on GPU 1
(port 8002; 8001 is taken on the shared VM) and the attacker on GPU 1 next to it (port 8003, FP8). Each is overridable
from the CLI. The skill-file judge runs on OpenRouter.
"""

from pathlib import Path

VICTIM_MODEL_ID = "Qwen/Qwen3.5-27B"
VICTIM_BASE_URL = "http://localhost:8000/v1"
VICTIM_ENABLE_THINKING = False
"""Published AgentDojo numbers for Qwen3.5-27B are non-thinking."""

CHECKER_MODEL_ID = "Qwen/Qwen3.5-9B"
CHECKER_BASE_URL = "http://localhost:8002/v1"
CHECKER_ENABLE_THINKING = False

ATTACKER_MODEL_ID = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
"""The MUZZLE attacker of C2/C4 (decided 2026-11-17 by the user: Mistral Small 24B, local, no API). It plays MUZZLE's
judge and all three PAIR roles. Non-reasoning, so upstream's PAIR token caps work as they are."""
ATTACKER_BASE_URL = "http://localhost:8003/v1"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

JUDGE_MODEL_ID = "deepseek/deepseek-v4.1-flash"
"""The C3/C4 skill-file judge, on OpenRouter (decided 2026-11-17 by the user)."""
JUDGE_BASE_URL = OPENROUTER_BASE_URL
JUDGE_API_KEY_ENV = "OPENROUTER_API_KEY"

DEFAULT_SKILL_FILE = Path("skillfile_evolution/skill_files/S0.md")
"""The starting skill file S0, relative to the repository root."""
