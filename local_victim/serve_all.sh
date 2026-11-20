#!/usr/bin/env bash
# Start the project's model servers on the GPU VM and wait until they answer (victim redesign of 2026-11-19):
#   victim   Qwen/Qwen3.8-27B + DFlash2 draft z-lab/Qwen3.8-27B-DFlash2 on GPU 0, port 8000
#   attacker mistralai/Mistral-Small-3.2-24B-Instruct-2506                on GPU 1, port 8003 (C2/C4 only)
# There is no external checker any more: S0 is one of the victim's skills. CHECKER=1 also starts the old checker
# (Qwen/Qwen3.5-9B, GPU 1, port 8002) for the `run_condition --external-checker` ablation; the attacker then shares
# GPU 1 with it and is loaded in FP8, as before.
# The DFlash2 draft (~2B) makes the victim faster without changing its answers. VICTIM_DRAFT=none serves plain 27B
# (use it if this vLLM build cannot load the draft). The DFlash2 model card asks for a vLLM build from
# vllm-project/vllm PR #52816; install that in its own venv and point the victim at it with VICTIM_VENV=<venv>, so the
# attacker keeps the venv that works (local_victim/README.md).
# Ports: VICTIM_PORT / CHECKER_PORT / ATTACKER_PORT; pass the same port to run_condition if you change one.
# These match src/agentdojo/experiment_models.py, which the runner uses by default.
# The VM's no-nvcc workaround applies to all of them: VLLM_ATTENTION_BACKEND=FLASH_ATTN EXTRA_ARGS=--enforce-eager.
#
#   bash local_victim/serve_all.sh          # victim + attacker (logs in ~/vllm_logs)
#   ATTACKER=0 bash local_victim/serve_all.sh   # victim only (C0, C1, C3)
#   CHECKER=1 bash local_victim/serve_all.sh    # also the old external checker (ablation)
#   ONE_GPU=1 bash local_victim/serve_all.sh    # victim AND attacker both on GPU 1, both in FP8 (frees GPU 0)
#   bash local_victim/serve_all.sh stop     # stop all
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LOG_DIR=${LOG_DIR:-$HOME/vllm_logs}
mkdir -p "$LOG_DIR"

if [ "${1:-}" = "stop" ]; then
  for name in victim checker attacker; do
    if [ -f "$LOG_DIR/$name.pid" ]; then
      kill "$(cat "$LOG_DIR/$name.pid")" 2>/dev/null || true
      rm -f "$LOG_DIR/$name.pid"
    fi
  done
  pkill -f "vllm serve" 2>/dev/null || true
  echo "stopped"
  exit 0
fi

start() {  # name gpu port model gpu_mem [extra vLLM args for this server] [draft model or none] [vLLM venv]
  local name=$1 gpu=$2 port=$3 model=$4 mem=$5 extra=${6:-} draft=${7:-none} venv=${8:-${VENV:-$HOME/venvs/vllm}}
  if curl -sf "localhost:$port/v1/models" >/dev/null 2>&1; then
    echo "$name already running on port $port"
    return
  fi
  if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
    echo "port $port is already used by another program (not a vLLM server); pick a free one, e.g." >&2
    echo "  ${name^^}_PORT=8003 bash local_victim/serve_all.sh   (and pass the same port to run_condition)" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES=$gpu DP=1 TP=1 PORT=$port MODEL=$model GPU_MEM=$mem LOG_DIR=$LOG_DIR DRAFT=$draft VENV=$venv \
    EXTRA_ARGS="$extra ${EXTRA_ARGS:-}" \
    nohup bash "$HERE/serve_vllm.sh" >"$LOG_DIR/$name.log" 2>&1 &
  echo $! >"$LOG_DIR/$name.pid"
  echo "$name: $model on GPU $gpu, port $port, draft $draft, vLLM $venv (log: $LOG_DIR/$name.log)"
}

wait_for() {  # name port
  local name=$1 port=$2
  until curl -sf "localhost:$port/v1/models" >/dev/null 2>&1; do
    if [ -f "$LOG_DIR/$name.pid" ] && ! kill -0 "$(cat "$LOG_DIR/$name.pid")" 2>/dev/null; then
      echo "the $name server exited; see $LOG_DIR/$name.log" >&2
      exit 1
    fi
    sleep ${WAIT_SECONDS:-10}
  done
  echo "port $port ready: $(curl -s "localhost:$port/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
}

# ONE_GPU=<n> (added 2026-11-20 by the user's request): put the victim and the attacker on the same GPU n. Both are then
# loaded in FP8 so they fit in one 96 GB H100 together: victim ~28 GB + DFlash2 draft ~4 GB in 0.50 of the GPU,
# attacker ~24 GB in 0.40, the rest for their KV caches. The attacker starts only once the victim is up, so each
# server sees the other's memory as used. Use the same setting for every condition you compare: FP8 changes the
# victim's answers slightly against BF16, and the run's config.json does not record the server's quantization.
ONE_GPU=${ONE_GPU:-}
if [ -n "$ONE_GPU" ]; then
  if [ "${CHECKER:-0}" = "1" ]; then
    echo "ONE_GPU and CHECKER=1 can't be combined (three models don't fit on one GPU)" >&2
    exit 1
  fi
  VICTIM_GPU=${VICTIM_GPU:-$ONE_GPU}
  ATTACKER_GPU=${ATTACKER_GPU:-$ONE_GPU}
  VICTIM_QUANT=${VICTIM_QUANT-fp8}
  VICTIM_GPU_MEM=${VICTIM_GPU_MEM:-0.50}
  ATTACKER_QUANT=${ATTACKER_QUANT-fp8}
  ATTACKER_GPU_MEM=${ATTACKER_GPU_MEM:-0.40}
fi
VICTIM_GPU=${VICTIM_GPU:-0}
ATTACKER_GPU=${ATTACKER_GPU:-1}
VICTIM_QUANT=${VICTIM_QUANT-}

VICTIM_PORT=${VICTIM_PORT:-8000}
CHECKER_PORT=${CHECKER_PORT:-8002}
ATTACKER_PORT=${ATTACKER_PORT:-8003}
start victim "$VICTIM_GPU" "$VICTIM_PORT" "${VICTIM_MODEL:-Qwen/Qwen3.8-27B}" "${VICTIM_GPU_MEM:-0.90}" \
  "${VICTIM_QUANT:+--quantization $VICTIM_QUANT}" \
  "${VICTIM_DRAFT-z-lab/Qwen3.8-27B-DFlash2}" "${VICTIM_VENV:-${VENV:-$HOME/venvs/vllm}}"
if [ "${CHECKER:-0}" = "1" ]; then
  start checker 1 "$CHECKER_PORT" "${CHECKER_MODEL:-Qwen/Qwen3.5-9B}" "${CHECKER_GPU_MEM:-0.50}"
fi

echo "waiting for the servers (the first start downloads the weights: ~55 GB for the victim and its draft)..."
wait_for victim "$VICTIM_PORT"
if [ "${CHECKER:-0}" = "1" ]; then
  wait_for checker "$CHECKER_PORT"
fi
if [ "${ATTACKER:-1}" != "0" ]; then
  if [ -n "$ONE_GPU" ]; then
    : # set above: FP8 in 0.40 of the GPU it shares with the victim
  elif [ "${CHECKER:-0}" = "1" ]; then
    # Shares GPU 1 with the checker: FP8 weights (~24 GB) in 0.42 of the GPU, started once the checker is up.
    ATTACKER_QUANT=${ATTACKER_QUANT-fp8}
    ATTACKER_GPU_MEM=${ATTACKER_GPU_MEM:-0.42}
  else
    # GPU 1 to itself: BF16 weights (~48 GB).
    ATTACKER_QUANT=${ATTACKER_QUANT-}
    ATTACKER_GPU_MEM=${ATTACKER_GPU_MEM:-0.90}
  fi
  start attacker "$ATTACKER_GPU" "$ATTACKER_PORT" "${ATTACKER_MODEL:-mistralai/Mistral-Small-3.2-24B-Instruct-2506}" \
    "$ATTACKER_GPU_MEM" "${ATTACKER_QUANT:+--quantization $ATTACKER_QUANT}"
  wait_for attacker "$ATTACKER_PORT"
fi
