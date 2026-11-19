# Local victim model on a GPU VM (vLLM)

How to run the DeepAgents victim against a self-hosted model instead of OpenRouter. Written for a VM with
2x H100 NVL (96 GB each), NVIDIA driver 580, no CUDA toolkit, no PyTorch and no admin rights.

Nothing here needs `nvcc` or `apt`: the vLLM wheel pulls a PyTorch wheel that bundles its own CUDA runtime, and the
driver (580, CUDA 13.0) runs CUDA 12.x and 13.x wheels. `uv` installs its own Python, so the system Python 3.10 is
not used (the `deepagents` extra needs Python >= 3.11).

## Model choice

| Model | BF16 weights | Fits one H100 NVL | Tool parser | Published AgentDojo ASR (no defense) |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen3.5-27B` (default) | ~54 GB | yes, ~30 GB left for KV cache | `qwen3_coder` | ~26% (banking ~52%) |
| `Qwen/Qwen3-32B` | ~65 GB | yes, ~20 GB left | `hermes` | ~30% (important_instructions) |
| `google/gemma-4-26B-A4B-it` | ~52 GB (MoE, ~4B active) | yes | `gemma4` | none published for AgentDojo |

Both clear the C0 rule (ASR >= 20%) in the literature, but only the pilot on this setup counts. Both models think by
default; the published numbers are non-thinking, so the server turns thinking off by default and the benchmark is
run with `--no-thinking` (sent per request, so it is in every request even if the server default changes).

Serving the victim alone (`serve_vllm.sh` defaults): one full BF16 copy per GPU behind one endpoint
(`--data-parallel-size 2`). The project setup below instead gives GPU 1 to the checker. That is simpler and faster than
splitting one copy over both GPUs (tensor parallel 2), which only pays off for models above ~80 GB.

Gemma 4 needs vLLM >= 0.25 and vLLM's Gemma 4 tool chat template: download `examples/tool_chat_template_gemma4.jinja`
from the vLLM GitHub repository at the tag of your installed version and start the server with
`MODEL=google/gemma-4-26B-A4B-it EXTRA_ARGS="--chat-template /path/to/tool_chat_template_gemma4.jinja"`. Gemma 4 does
not think unless asked, so `--no-thinking` only makes that explicit.

## 1. Install (once, as your normal user)

```bash
# uv, into ~/.local/bin (no root)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

# vLLM in its own environment; --torch-backend=auto picks the PyTorch CUDA build for the installed driver
uv venv --python 3.12 ~/venvs/vllm
uv pip install --python ~/venvs/vllm/bin/python vllm --torch-backend=auto
~/venvs/vllm/bin/python -c "import torch, vllm; print(vllm.__version__, torch.version.cuda, torch.cuda.device_count())"
# expect a vLLM version >= 0.17 (needed for Qwen3.5) and 2 GPUs

# The repo, with the DeepAgents extra, in a separate environment
git clone https://github.com/Ethandupro/dso_agentdojov2.git ~/dso_agentdojov2
cd ~/dso_agentdojov2 && uv sync --dev --extra deepagents
```

Check disk space for the weights first (`df -h ~`): ~55 GB for Qwen3.5-27B, ~65 GB for Qwen3-32B. Put the cache on
a large disk with `export HF_HOME=/path/to/big/disk/hf` if the home directory is small.

## 2. Start the server

```bash
cd ~/dso_agentdojov2
tmux new -s vllm            # or screen / nohup; the server must outlive your SSH session
bash local_victim/serve_vllm.sh                      # Qwen/Qwen3.5-27B
# MODEL=Qwen/Qwen3-32B bash local_victim/serve_vllm.sh
```

The first start downloads the weights. It is ready when the log shows `Application startup complete`. The script
writes the model, vLLM, PyTorch and GPU versions to `~/vllm_logs/server_<time>.txt`; keep that file with the results.

Smoke test, in another shell (checks that a tool call comes back parsed, not as text):

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3.5-27B", "temperature": 0,
  "chat_template_kwargs": {"enable_thinking": false},
  "messages": [{"role": "user", "content": "What is my balance?"}],
  "tools": [{"type": "function", "function": {"name": "get_balance", "description": "Get the account balance",
            "parameters": {"type": "object", "properties": {}}}}]
}' | python3 -m json.tool | grep -A3 tool_calls
```

## 3. Run the benchmark against it

`--base-url` switches the DeepAgents harness from OpenRouter to the local server. `--model-id` must be the served
model name (the Hugging Face id unless you pass `--served-model-name`). `--provider` cannot be combined with it.

```bash
cd ~/dso_agentdojov2
uv run python -m agentdojo.scripts.benchmark \
  --harness deepagents --model-id Qwen/Qwen3.5-27B \
  --base-url http://localhost:8000/v1 --no-thinking \
  -s banking -ut user_task_0 -it injection_task_0 \
  --attack important_instructions --logdir runs/local-qwen3.5-27b-smoke
```

Traces go under `<logdir>/deepagents-local-Qwen_Qwen3.5-27B/`, so they never mix with OpenRouter runs of the same
model. Results are still cached by that name only: use a new `--logdir` for every change of server flags, vLLM
version or thinking setting.

To use both GPUs, run the suites in parallel (`--max-workers 4`, one process per suite). Each suite runs its pairs
one after another, so a single suite keeps only one request in flight.

## The project setup: victim (with DFlash) and attacker

Since the victim redesign of 2026-11-19 there is no external checker: the security skill S0 is one of the victim's
skills (`skillfile_evolution/DESIGN.md`, "Victim skills"). The project's models (`src/agentdojo/experiment_models.py`)
are the victim `Qwen/Qwen3.5-27B` on GPU 0 (port 8000) and, for C2 and C4, the attacker
`mistralai/Mistral-Small-3.2-24B-Instruct-2506` on GPU 1 (port 8003, BF16, the GPU to itself). None of them thinks.

The victim is served with the speculative-decoding draft model `z-lab/Qwen3.5-27B-DFlash` (~2B). The draft guesses
several tokens ahead and the 27B model checks them in one pass, so the answers are the same as plain 27B and only the
speed changes. The runner still talks to `Qwen/Qwen3.5-27B`; the draft is a server setting
(`--speculative-config '{"method": "dflash", "model": "z-lab/Qwen3.5-27B-DFlash", "num_speculative_tokens": 7}'`).
If your vLLM build does not know the `dflash` method, start without it (`VICTIM_DRAFT=none`) or upgrade vLLM.

```bash
cd ~/dso_agentdojov2
bash local_victim/serve_all.sh               # victim + attacker; logs in ~/vllm_logs
ATTACKER=0 bash local_victim/serve_all.sh    # victim only (C0, C1, C3)
VICTIM_DRAFT=none bash local_victim/serve_all.sh   # plain 27B, no DFlash
CHECKER=1 bash local_victim/serve_all.sh     # also the old checker (port 8002), for --external-checker runs
bash local_victim/serve_all.sh stop          # stop all
# the no-nvcc workaround, if the servers fail at "Engine core initialization":
VLLM_ATTENTION_BACKEND=FLASH_ATTN EXTRA_ARGS=--enforce-eager bash local_victim/serve_all.sh
```

Victim options: `VICTIM_DRAFT` (`z-lab/Qwen3.5-27B-DFlash`; `none` turns it off), `SPEC_TOKENS` (7: tokens drafted per
step). Attacker options: `ATTACKER_MODEL`, `ATTACKER_PORT`, `ATTACKER_GPU_MEM` (0.90, or 0.42 with `CHECKER=1`),
`ATTACKER_QUANT` (empty = BF16; `fp8` with `CHECKER=1`). The attacker needs no API key.

Experiment runs use `python -m agentdojo.runner.run_condition` (see `src/agentdojo/runner/run_condition.py`). The
benchmark CLI below still has the old checker flags, for quick single-task checks.

With `--harness deepagents` and no `--model-id`, the benchmark uses that victim; `--checker` adds that checker with
the skill file S0. So the pilot's two conditions are:

```bash
# C0: no checker
uv run python -m agentdojo.scripts.benchmark --harness deepagents \
  -s banking -ut user_task_0 --attack important_instructions --logdir runs/pilot-c0
# C1: checker with S0 (checker calls logged to runs/pilot-c1/checker_calls.jsonl)
uv run python -m agentdojo.scripts.benchmark --harness deepagents --checker \
  -s banking -ut user_task_0 --attack important_instructions --logdir runs/pilot-c1
```

Runs with the checker are saved under a name that includes the checker model and a hash of the skill file, so they
never reuse results from runs without it or with another skill file. Other choices: `--checker-model-id`,
`--checker-base-url` (`openrouter` for OpenRouter), `--skill-file`. The CLI log has no per-trial injection texts,
so its checker log does not record `injection_survival`; the experiment runner will.

## Things to know

- Determinism: temperature 0 on vLLM is greedy, but batching several requests can still change outputs slightly.
  Treat reruns as close, not identical, and keep the server log with the results.
- If a tool call arrives as plain text (`<tool_call>` or `<function=...>` in the answer) instead of a parsed call,
  the tool parser is wrong for the model: set `TOOL_PARSER=` (e.g. `qwen3_xml` for Qwen3.5) and restart.
- Context: the script sets 128k tokens for Qwen3.5 and 32k for Qwen3-32B (its native window). A conversation that
  overflows is ended and scored by the harness, as with OpenRouter.
- The server listens on 127.0.0.1 only. To reach it from another machine, use an SSH tunnel
  (`ssh -L 8000:localhost:8000 vm`) rather than opening the port; with `--api-key`, set `LOCAL_LLM_API_KEY` for the
  benchmark.
- The benchmark CLI's `--checker` uses `skillfile_evolution/skill_files/S0_checker.md` (S0 as written for the old
  external checker).
