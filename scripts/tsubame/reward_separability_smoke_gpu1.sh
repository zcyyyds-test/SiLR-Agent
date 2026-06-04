#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N reward_sep_smoke
#$ -o logs/
#$ -e logs/

# === GRPO reward-separability smoke (panel 2026-06-03 gating experiment) ===
# Falsify (cheaply, no training) whether arm D (geometry) separates from arm E
# (count projection) on single-type ANM. Serves Qwen3-8B, runs the 9 Step-0
# sub-saturated scenarios under progress_mag, computes per-step r_C/r_D/r_E from
# the persisted per-branch Φ=(S,σ), reports Spearman ρ(D,E) + σ-heterogeneity.
# Same serving config as the Step-0 job (step8 / tok2048 / temp0).
# GATE: ρ(D,E) ≥ 0.9 or σ-heterogeneity ≈ 1 → do NOT spend GPU-weeks on training.
set -euo pipefail
WORK="${HOME/home/work}"
PROJECT="$WORK/SILR-WISE26"
ENV_DIR="$WORK/envs/silr-vllm"
MODEL_DIR="$WORK/models/Qwen_Qwen3-8B"
LOG_DIR="$PROJECT/logs"
JOB_TAG="${JOB_ID:-manual}"
PORT="${SILR_VLLM_PORT:-8005}"
SERVER_LOG="$LOG_DIR/reward_sep_server_${JOB_TAG}.log"
RUN_LOG="$LOG_DIR/reward_sep_smoke_${JOB_TAG}.log"
OUT_JSON="$PROJECT/reward_separability_smoke.json"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/reward_sep_smoke_${JOB_TAG}.inner.log") 2>&1
echo "[start] $(date)"
module purge; module load cuda/13.1.1
export HF_HOME="$WORK/hf"; export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPATH="$PROJECT"; export PATH="$ENV_DIR/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SILR_MAX_TOKENS=2048
export SILR_SYNC_ID="reward_sep_smoke_${JOB_TAG}"
cd "$PROJECT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# Sanity: confirm the new three-arm reward + Φ-persistence code is present
# (inline, dependency-free — silr-vllm has no pytest).
"$ENV_DIR/bin/python" - <<'PYCHK' || { echo "[sanity-failed] three-arm reward code missing/broken — aborting before serve"; exit 1; }
from silr.verifier.types import VerificationResult, Verdict
from silr.training.reward import (compute_grpo_reward as D, compute_scalar_reward as E,
                                  compute_binary_reward as C)
k1=("bl","line","0-1","load"); k2=("bl","line","1-2","load")
def sp(pre,post): return VerificationResult(verdict=Verdict.SAFE_PROGRESS,
        action={"tool_name":"t","params":{}}, baseline_branches=pre, post_branches=post)
pas=VerificationResult(verdict=Verdict.PASS, action={"tool_name":"t","params":{}})
assert D(sp({k1:1.0},{})) > 0.0, "SAFE_PROGRESS must be positive (bugfix)"
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
trap cleanup EXIT TERM
for i in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then echo "[ready] $(date)"; break; fi
  ! kill -0 "$SERVER_PID" 2>/dev/null && { echo "[serve-exited]"; tail -120 "$SERVER_LOG"; exit 1; }
  [ "$i" -eq 180 ] && { echo "[timeout]"; tail -160 "$SERVER_LOG"; exit 1; }
  sleep 5
done
echo "[separability smoke: 9 sub-saturated scenarios x N=3 @ step=8] $(date)"
"$ENV_DIR/bin/python" -u scripts/anm_reward_separability_smoke.py \
  --base-url "http://127.0.0.1:${PORT}/v1" --model qwen3-8b \
  --reps 3 --rep-start-seed 1000 \
  --max-steps 8 --max-proposals 3 --temperature 0.0 \
  --request-timeout-s 360 --max-retries 0 \
  --output "$OUT_JSON" --log-file "$RUN_LOG"
echo "[done] $(date)"
