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

Both clear the C0 rule (ASR >= 20%) in the literature, but only the pilot on this setup counts. Both models think by
default; the published numbers are non-thinking, so the server turns thinking off by default and the benchmark is
run with `--no-thinking` (sent per request, so it is in every request even if the server default changes).

Serving: one full BF16 copy per GPU behind one endpoint (`--data-parallel-size 2`). That is simpler and faster than
splitting one copy over both GPUs (tensor parallel 2), which only pays off for models above ~80 GB.

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
- The checker still runs on OpenRouter (`OPENROUTER_API_KEY`); only the victim moves to the VM.
