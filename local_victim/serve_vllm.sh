#!/usr/bin/env bash
# Serve a local victim model with vLLM's OpenAI-compatible server (no admin rights, no nvcc needed).
#
#   bash local_victim/serve_vllm.sh            # Qwen/Qwen3.5-27B, one copy per GPU, port 8000
#   MODEL=Qwen/Qwen3-32B bash local_victim/serve_vllm.sh
#
# Then run the benchmark with:
#   --harness deepagents --model-id "$MODEL" --base-url http://localhost:8000/v1 --no-thinking
# See local_victim/README.md.
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3.5-27B}
PORT=${PORT:-8000}
VENV=${VENV:-$HOME/venvs/vllm}
# Two H100s: by default one full BF16 copy per GPU behind one endpoint (data parallel). Set DP=1 TP=2 to split one
# copy over both GPUs instead (only needed for models that do not fit in one GPU).
DP=${DP:-2}
TP=${TP:-1}
GPU_MEM=${GPU_MEM:-0.90}

case "$MODEL" in
  *Qwen3.5*|*Qwen3.6*) TOOL_PARSER=${TOOL_PARSER:-qwen3_coder}; REASONING_PARSER=${REASONING_PARSER:-qwen3}; MAX_LEN=${MAX_LEN:-131072} ;;
  *Qwen3-*)            TOOL_PARSER=${TOOL_PARSER:-hermes};      REASONING_PARSER=${REASONING_PARSER:-qwen3}; MAX_LEN=${MAX_LEN:-32768} ;;
  # Gemma 4 needs vLLM >= 0.25 and vLLM's Gemma 4 tool chat template (pass it with
  # EXTRA_ARGS="--chat-template /path/to/tool_chat_template_gemma4.jinja", see local_victim/README.md).
  *gemma-4*)           TOOL_PARSER=${TOOL_PARSER:-gemma4};      REASONING_PARSER=${REASONING_PARSER:-gemma4}; MAX_LEN=${MAX_LEN:-131072} ;;
  # Mistral Small (the MUZZLE attacker): no reasoning, loaded in Mistral's own format as its model card recommends.
  # Its chat template is mistral_common's, so no enable_thinking switch is passed.
  *Mistral-Small*)     TOOL_PARSER=${TOOL_PARSER:-mistral};     REASONING_PARSER=${REASONING_PARSER:-none}; MAX_LEN=${MAX_LEN:-32768}
                       THINKING_KWARGS=${THINKING_KWARGS:-none}
                       MODEL_ARGS="--tokenizer-mode mistral --config-format mistral --load-format mistral --limit-mm-per-prompt {\"image\":0}" ;;
  *) : "${TOOL_PARSER:?set TOOL_PARSER for $MODEL}"; : "${REASONING_PARSER:?set REASONING_PARSER for $MODEL (none: no parser)}"; MAX_LEN=${MAX_LEN:-32768} ;;
esac
MODEL_ARGS=${MODEL_ARGS:-}
REASONING_ARGS=()
if [ "$REASONING_PARSER" != "none" ]; then REASONING_ARGS=(--reasoning-parser "$REASONING_PARSER"); fi
THINKING_ARGS=()
if [ "${THINKING_KWARGS:-}" != "none" ]; then THINKING_ARGS=(--default-chat-template-kwargs '{"enable_thinking": false}'); fi

if [ ! -x "$VENV/bin/vllm" ]; then
  echo "vLLM not found in $VENV. Install it first (local_victim/README.md, step 1)." >&2
  exit 1
fi

# Record exactly what served the run (put this file next to the run's results).
mkdir -p "${LOG_DIR:-$HOME/vllm_logs}"
INFO="${LOG_DIR:-$HOME/vllm_logs}/server_$(date -u +%Y%m%dT%H%M%SZ).txt"
{
  echo "model=$MODEL port=$PORT dp=$DP tp=$TP max_len=$MAX_LEN tool_parser=$TOOL_PARSER reasoning_parser=$REASONING_PARSER model_args=$MODEL_ARGS extra=${EXTRA_ARGS:-}"
  "$VENV/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} | tee "$INFO"

"$VENV/bin/vllm" serve "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --data-parallel-size "$DP" --tensor-parallel-size "$TP" \
  --dtype bfloat16 \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_MEM" \
  --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" \
  "${REASONING_ARGS[@]}" \
  "${THINKING_ARGS[@]}" \
  --generation-config vllm \
  $MODEL_ARGS ${EXTRA_ARGS:-} 2>&1 | tee -a "$INFO"
