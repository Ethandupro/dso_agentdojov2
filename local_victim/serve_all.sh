#!/usr/bin/env bash
# Start the project's two model servers on the GPU VM and wait until both answer:
#   victim  Qwen/Qwen3.5-27B on GPU 0, port 8000
#   checker Qwen/Qwen3.5-9B  on GPU 1, port 8002 (half of GPU 1, so the attacker can use the rest later)
# Ports: VICTIM_PORT / CHECKER_PORT (8001 is taken by another user on the shared VM, so the checker uses 8002);
# pass the same port to run_condition with --checker-base-url if you change it.
# These match src/agentdojo/experiment_models.py, which the benchmark uses by default.
#
#   bash local_victim/serve_all.sh          # start both (logs in ~/vllm_logs)
#   bash local_victim/serve_all.sh stop     # stop both
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LOG_DIR=${LOG_DIR:-$HOME/vllm_logs}
mkdir -p "$LOG_DIR"

if [ "${1:-}" = "stop" ]; then
  for name in victim checker; do
    if [ -f "$LOG_DIR/$name.pid" ]; then
      kill "$(cat "$LOG_DIR/$name.pid")" 2>/dev/null || true
      rm -f "$LOG_DIR/$name.pid"
    fi
  done
  pkill -f "vllm serve" 2>/dev/null || true
  echo "stopped"
  exit 0
fi

start() {  # name gpu port model gpu_mem
  local name=$1 gpu=$2 port=$3 model=$4 mem=$5
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
    nohup bash "$HERE/serve_vllm.sh" >"$LOG_DIR/$name.log" 2>&1 &
  echo $! >"$LOG_DIR/$name.pid"
  echo "$name: $model on GPU $gpu, port $port (log: $LOG_DIR/$name.log)"
}

VICTIM_PORT=${VICTIM_PORT:-8000}
CHECKER_PORT=${CHECKER_PORT:-8002}
start victim 0 "$VICTIM_PORT" "${VICTIM_MODEL:-Qwen/Qwen3.5-27B}" "${VICTIM_GPU_MEM:-0.90}"
start checker 1 "$CHECKER_PORT" "${CHECKER_MODEL:-Qwen/Qwen3.5-9B}" "${CHECKER_GPU_MEM:-0.50}"

echo "waiting for both servers (the first start downloads ~75 GB of weights)..."
for pair in "victim:$VICTIM_PORT" "checker:$CHECKER_PORT"; do
  name=${pair%%:*} port=${pair##*:}
  until curl -sf "localhost:$port/v1/models" >/dev/null 2>&1; do
    if [ -f "$LOG_DIR/$name.pid" ] && ! kill -0 "$(cat "$LOG_DIR/$name.pid")" 2>/dev/null; then
      echo "the $name server exited; see $LOG_DIR/$name.log" >&2
      exit 1
    fi
    sleep 10
  done
  echo "port $port ready: $(curl -s "localhost:$port/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
done
