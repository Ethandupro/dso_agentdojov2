"""The models of the skill-file experiments and where they are served (decided 2026-11-16 by the user).

Both run on the project's GPU VM behind vLLM's OpenAI-compatible server (see `local_victim/README.md`):
the victim on GPU 0 (port 8000) and the checker on GPU 1 (port 8001). Each is overridable from the benchmark CLI.
"""

from pathlib import Path

VICTIM_MODEL_ID = "Qwen/Qwen3.5-27B"
VICTIM_BASE_URL = "http://localhost:8000/v1"
VICTIM_ENABLE_THINKING = False
"""Published AgentDojo numbers for Qwen3.5-27B are non-thinking."""

CHECKER_MODEL_ID = "Qwen/Qwen3.5-9B"
CHECKER_BASE_URL = "http://localhost:8001/v1"
CHECKER_ENABLE_THINKING = False

DEFAULT_SKILL_FILE = Path("skillfile_evolution/skill_files/S0.md")
"""The starting skill file S0, relative to the repository root."""
