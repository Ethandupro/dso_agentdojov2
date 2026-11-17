#!/usr/bin/env bash
# Start the project's three model servers on the GPU VM and wait until they answer:
#   victim   Qwen/Qwen3.5-27B                          on GPU 0, port 8000
#   checker  Qwen/Qwen3.5-9B                           on GPU 1, port 8002 (half of GPU 1)
#   attacker mistralai/Mistral-Small-3.2-24B-Instruct-2506 on GPU 1, port 8003 (the MUZZLE attacker, C2/C4 only)
# The attacker shares GPU 1 with the checker: its BF16 weights (~48 GB) don't fit next to the checker's half, so it is
# quantized to FP8 when loaded (~24 GB of weights, 0.42 of the GPU), and it starts only once the checker is up, so the
# two don't measure free memory at the same time. ATTACKER=0 skips it (C0, C1, C3 don't need it).
# Ports: VICTIM_PORT / CHECKER_PORT / ATTACKER_PORT (8001 is taken by another user on the shared VM, so the checker
# uses 8002); pass the same port to run_condition if you change one.
# These match src/agentdojo/experiment_models.py, which the runner uses by default.
# The VM's no-nvcc workaround applies to all three: VLLM_ATTENTION_BACKEND=FLASH_ATTN EXTRA_ARGS=--enforce-eager.
#
#   bash local_victim/serve_all.sh          # start all three (logs in ~/vllm_logs)
#   ATTACKER=0 bash local_victim/serve_all.sh
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

start() {  # name gpu port model gpu_mem [extra vLLM args for this server]
  local name=$1 gpu=$2 port=$3 model=$4 mem=$5 extra=${6:-}
  if curl -sf "localhost:$port/v1/models" >/dev/null 2>&1; then
    echo "$name already running on port $port"
    return
  fi
  if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
    echo "port $port is already used by another program (not a vLLM server); pick a free one, e.g." >&2
    echo "  ${name^^}_PORT=8003 bash local_victim/serve_all.sh   (and pass the same port to run_condition)" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES=$gpu DP=1 TP=1 PORT=$port MODEL=$model GPU_MEM=$mem LOG_DIR=$LOG_DIR \
    EXTRA_ARGS="$extra ${EXTRA_ARGS:-}" \
    nohup bash "$HERE/serve_vllm.sh" >"$LOG_DIR/$name.log" 2>&1 &
  echo $! >"$LOG_DIR/$name.pid"
  echo "$name: $model on GPU $gpu, port $port (log: $LOG_DIR/$name.log)"
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

VICTIM_PORT=${VICTIM_PORT:-8000}
CHECKER_PORT=${CHECKER_PORT:-8002}
ATTACKER_PORT=${ATTACKER_PORT:-8003}
start victim 0 "$VICTIM_PORT" "${VICTIM_MODEL:-Qwen/Qwen3.5-27B}" "${VICTIM_GPU_MEM:-0.90}"
start checker 1 "$CHECKER_PORT" "${CHECKER_MODEL:-Qwen/Qwen3.5-9B}" "${CHECKER_GPU_MEM:-0.50}"

echo "waiting for the servers (the first start downloads the weights: ~75 GB, plus ~48 GB for the attacker)..."
wait_for victim "$VICTIM_PORT"
wait_for checker "$CHECKER_PORT"
if [ "${ATTACKER:-1}" != "0" ]; then
  ATTACKER_QUANT=${ATTACKER_QUANT-fp8}
  start attacker 1 "$ATTACKER_PORT" "${ATTACKER_MODEL:-mistralai/Mistral-Small-3.2-24B-Instruct-2506}" \
    "${ATTACKER_GPU_MEM:-0.42}" "${ATTACKER_QUANT:+--quantization $ATTACKER_QUANT}"
  wait_for attacker "$ATTACKER_PORT"
fi
