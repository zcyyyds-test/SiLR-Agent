#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N cl_reward_sep
#$ -o logs/
#$ -e logs/

# === CityLearn GRPO reward-separability smoke (pre-training go/no-go) ===
# Falsify (cheaply, no training) whether arm D (product-order geometry) separates
# from arm E (count projection) on the hardened N=4 multi-type CityLearn band.
# Serves Qwen3-8B, runs the cl_mined_* band under progress_mag (thinking off),
# computes per-step r_C/r_D/r_E from the persisted per-branch Phi=(S,sigma),
# reports Spearman rho(D,E) + sigma-heterogeneity.
# GATE: rho(D,E) >= 0.9 or sigma-heterogeneity ~ 1 -> do NOT spend GPU-days;
#   else SEPARABLE -> proceed to train_grpo_citylearn.py.
# Multi-type prior: unlike single-type ANM, the band carries incomparable
# families at once (soc/export/import), so D and E *should* separate.
set -euo pipefail
WORK="${HOME/home/work}"
PROJECT="$WORK/SILR-WISE26"
ENV_DIR="$WORK/envs/silr-vllm"
MODEL_DIR="$WORK/models/Qwen_Qwen3-8B"
LOG_DIR="$PROJECT/logs"
JOB_TAG="${JOB_ID:-manual}"
PORT="${SILR_VLLM_PORT:-$((8100 + RANDOM % 800))}"  # randomize: gpu_1 is a shared node
SERVER_LOG="$LOG_DIR/cl_reward_sep_server_${JOB_TAG}.log"
RUN_LOG="$LOG_DIR/cl_reward_sep_${JOB_TAG}.log"
OUT_JSON="$PROJECT/reward_separability_smoke_citylearn.json"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/cl_reward_sep_${JOB_TAG}.inner.log") 2>&1
echo "[start] $(date)"
module purge; module load cuda/13.1.1
export HF_HOME="$WORK/hf"; export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJECT"; export PATH="$ENV_DIR/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SILR_MAX_TOKENS=2048
export SILR_CITYLEARN_N_BUILDINGS=4   # pillar-2 hardened band (default 3 keeps pillar-1 RQ5)
export SILR_SYNC_ID="cl_reward_sep_${JOB_TAG}"
cd "$PROJECT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$ENV_DIR/bin/vllm" --version 2>&1 | sed 's/^/[vllm-version] /' || true

# Sanity: confirm the domain-agnostic three-arm reward + Phi-persistence code is
# present (inline, dependency-free -- silr-vllm has no pytest).
"$ENV_DIR/bin/python" - <<'PYCHK' || { echo "[sanity-failed] three-arm reward code missing/broken -- aborting before serve"; exit 1; }
from silr.verifier.types import VerificationResult, Verdict
from silr.training.reward import (compute_grpo_reward as D, compute_scalar_reward as E,
                                  compute_binary_reward as C)
k1=("bl","line","0-1","load"); k2=("bl","line","1-2","load")
def sp(pre,post): return VerificationResult(verdict=Verdict.SAFE_PROGRESS,
        action={"tool_name":"t","params":{}}, baseline_branches=pre, post_branches=post)
pas=VerificationResult(verdict=Verdict.PASS, action={"tool_name":"t","params":{}})
assert D(sp({k1:1.0},{})) > 0.0, "SAFE_PROGRESS must be positive"
assert D(sp({k1:1.0},{})) < D(pas), "SAFE_PROGRESS must be < PASS"
pre={k1:8.0,k2:1.0}
assert D(sp(pre,{k2:1.0})) > D(sp(pre,{k1:8.0})), "arm D must be severity-weighted"
assert E(sp(pre,{k2:1.0})) == E(sp(pre,{k1:8.0})), "arm E must be count-blind"
assert C(pas)==0.5 and C(sp({k1:1.0},{}))==0.5, "arm C binary"
print("[sanity-ok] three-arm reward + Phi persistence present")
PYCHK

"$ENV_DIR/bin/vllm" serve "$MODEL_DIR" --host 127.0.0.1 --port "$PORT" \
  --served-model-name qwen3-8b --gpu-memory-utilization 0.85 \
  --max-model-len 16384 --enforce-eager \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --trust-remote-code >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT TERM INT
for i in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then echo "[ready] $(date)"; break; fi
  ! kill -0 "$SERVER_PID" 2>/dev/null && { echo "[serve-exited]"; tail -120 "$SERVER_LOG"; exit 1; }
  [ "$i" -eq 180 ] && { echo "[timeout]"; tail -160 "$SERVER_LOG"; exit 1; }
  sleep 5
done
# Thinking-off smoke: abort in seconds if the chat_template_kwargs path is ignored.
SMOKE_RESP=$(curl -fsS "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-8b","temperature":0,"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"Reason step by step about a battery SoC and feeder export violation."}]}') \
  || { echo "[inference-smoke-failed]"; tail -80 "$SERVER_LOG"; exit 1; }
printf '%s' "$SMOKE_RESP" | grep -q '<think>' \
  && { echo "[thinking-off-FAILED] <think> present; aborting"; exit 1; } \
  || echo "[thinking-off OK]"

echo "[separability smoke: hardened N=4 cl_mined band x N=3 @ step=8] $(date)"
"$ENV_DIR/bin/python" -u scripts/citylearn_reward_separability_smoke.py \
  --base-url "http://127.0.0.1:${PORT}/v1" --model qwen3-8b \
  --reps 2 --rep-start-seed 1000 \
  --max-steps 8 --max-proposals 3 --temperature 0.0 \
  --request-timeout-s 360 --max-retries 0 \
  --output "$OUT_JSON" --log-file "$RUN_LOG"
echo "[done] $(date)"
