"""Token counting for the skill-file caps, with the checker model's own tokenizer.

The 1500-token cap exists for the checker's context, so it is counted with the checker's tokenizer. The checker is
served by vLLM (`experiment_models.CHECKER_BASE_URL`), whose `/tokenize` endpoint counts with the served model's
tokenizer, so no tokenizer is installed here.
"""

import functools

import httpx

from agentdojo.experiment_models import CHECKER_BASE_URL, CHECKER_MODEL_ID
from agentdojo.skillfile.checks import TokenCounter


def vllm_token_counter(base_url: str = CHECKER_BASE_URL, model: str = CHECKER_MODEL_ID) -> TokenCounter:
    server = base_url.rstrip("/").removesuffix("/v1")

    @functools.lru_cache(maxsize=4096)
    def count(text: str) -> int:
        response = httpx.post(
            f"{server}/tokenize", json={"model": model, "prompt": text, "add_special_tokens": False}, timeout=30
        )
        response.raise_for_status()
        return int(response.json()["count"])

    return count
