#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N smoke_qwen3_14b
#$ -o logs/
#$ -e logs/

set -euo pipefail

WORK="${HOME/home/work}"
PROJECT="$WORK/SILR-WISE26"
ENV_DIR="$WORK/envs/silr-vllm"
MODEL_DIR="$WORK/models/Qwen3-14B"
LOG_DIR="$PROJECT/logs"
JOB_TAG="${JOB_ID:-manual}"
PORT="${SILR_VLLM_PORT:-8001}"
SERVER_LOG="$LOG_DIR/tsubame_vllm_smoke_server_${JOB_TAG}.log"
RUN_LOG="$LOG_DIR/eval_tsubame_smoke_${JOB_TAG}.log"
OUT_JSON="$PROJECT/eval_tsubame_smoke_${JOB_TAG}.json"

mkdir -p "$LOG_DIR" "$WORK/hf"
exec > >(tee -a "$LOG_DIR/smoke_qwen3_14b_${JOB_TAG}.inner.log") 2>&1

echo "[start] $(date)"
echo "[host] $(hostname)"
echo "[project] $PROJECT"
echo "[env] $ENV_DIR"
echo "[model] $MODEL_DIR"
echo "[port] $PORT"

module purge
module load cuda/13.1.1

export HF_HOME="$WORK/hf"
export HF_HUB_CACHE="$WORK/hf/hub"
export TRANSFORMERS_CACHE="$WORK/hf/transformers"
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPATH="$PROJECT"
export PATH="$ENV_DIR/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SILR_MAX_TOKENS=2048
export SILR_SCALAR_PROGRESS_RELATIVE_SLACK=0.20
export SILR_SYNC_ID="tsubame_smoke_${JOB_TAG}"

cd "$PROJECT"

echo "[nvidia-smi]"
nvidia-smi

echo "[python-import-check]"
"$ENV_DIR/bin/python" -c '
import sys
print("python", sys.version)
for name in ["torch", "vllm", "transformers", "openai", "gym_anm", "cvxpy", "numpy"]:
    mod = __import__(name)
    print(name, getattr(mod, "__version__", "unknown"))
import torch
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_name", torch.cuda.get_device_name(0))
'

echo "[serve] $(date)"
"$ENV_DIR/bin/vllm" serve "$MODEL_DIR" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --served-model-name qwen3-14b \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[serve-pid] $SERVER_PID"

cleanup() {
  if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "[cleanup] stopping vllm pid=$SERVER_PID"
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "[wait-ready]"
for i in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[ready] attempt=$i $(date)"
    break
  fi
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "[serve-exited]"
    tail -120 "$SERVER_LOG" || true
    exit 1
  fi
  if [ "$i" -eq 120 ]; then
    echo "[ready-timeout]"
    tail -160 "$SERVER_LOG" || true
    exit 1
  fi
  sleep 5
done

echo "[eval] $(date)"
"$ENV_DIR/bin/python" -u scripts/anm_eval_sweep.py \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --model qwen3-14b \
  --scenarios mined_multi_action_1_l0p25g1p0_s5 \
  --policies progress_mag \
  --reps 1 \
  --rep-start-seed 1000 \
  --max-steps 6 \
  --max-proposals 3 \
  --request-timeout-s 240 \
  --max-retries 0 \
  --output "$OUT_JSON" \
  --log-file "$RUN_LOG"

echo "[output]"
"$ENV_DIR/bin/python" - <<PY
import json
from pathlib import Path
path = Path("$OUT_JSON")
data = json.loads(path.read_text())
print("json", path, data.get("status"))
for ep in data.get("episodes", []):
    print(ep["scenario"], ep["policy"], "recovered", ep["recovered"], "penalty", ep["final_penalty"], "props", ep["total_proposals"], "rejs", ep["total_rejections"])
PY

echo "[done] $(date)"
