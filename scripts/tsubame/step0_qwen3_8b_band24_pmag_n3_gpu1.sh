#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N qwen8b_band24_pmag_n3
#$ -o logs/
#$ -e logs/

# === Step 0 go/no-go gate (direction-gaai-verifier-as-reward.md §4 Step 0) ===
# Qwen3-8B + progress_mag over the FULL 24-scenario multi-action band, N=3,
# greedy (temp 0). Question: is 8B near-saturated on band24?
#   - If failures remain (recovery < 3/3 on some scenarios) -> training signal
#     exists -> train on the sub-saturated scenarios, main DV = recovery rate.
#   - If near-saturated -> recovery has no signal -> switch main DV to ungated
#     metrics for the GRPO experiment.
# This is the decisive gate BEFORE investing GPU-weeks in GRPO wiring.
#
# Config matches the 8B canonical exactly (= 14B band canonical: step8 / tok2048
# / temp0 / max-proposals 3 / seeds 1000-1002), only the scenario set is the full
# 24-band and the policy is progress_mag (the structured headline gate).
set -euo pipefail
WORK="${HOME/home/work}"
PROJECT="$WORK/SILR-WISE26"
ENV_DIR="$WORK/envs/silr-vllm"
MODEL_DIR="$WORK/models/Qwen_Qwen3-8B"
LOG_DIR="$PROJECT/logs"
JOB_TAG="${JOB_ID:-manual}"
PORT="${SILR_VLLM_PORT:-8005}"
SERVER_LOG="$LOG_DIR/step0_qwen8b_band24_server_${JOB_TAG}.log"
RUN_LOG="$LOG_DIR/step0_qwen8b_band24_pmag_n3_${JOB_TAG}.log"
OUT_JSON="$PROJECT/eval_step0_qwen3_8b_band24_n3.json"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/step0_qwen8b_band24_pmag_n3_${JOB_TAG}.inner.log") 2>&1
echo "[start] $(date)"
module purge; module load cuda/13.1.1
export HF_HOME="$WORK/hf"; export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPATH="$PROJECT"; export PATH="$ENV_DIR/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SILR_MAX_TOKENS=2048
export SILR_SYNC_ID="step0_qwen8b_band24_pmag_n3_${JOB_TAG}"
cd "$PROJECT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
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
echo "[eval progress_mag 24-band x N=3 = 72 ep @ step=8] $(date)"
"$ENV_DIR/bin/python" -u scripts/anm_eval_sweep.py \
  --base-url "http://127.0.0.1:${PORT}/v1" --model qwen3-8b \
  --scenarios \
    mined_multi_action_1_l0p25g1p0_s5 \
    mined_multi_action_2_l1p0g1p0_s5 \
    mined_multi_action_3_l0p25g1p0_s12 \
    mined_multi_action_4_l1p0g1p0_s12 \
    mined_multi_action_5_l0p25g1p0_s16 \
    mined_multi_action_6_l1p0g1p0_s16 \
    mined_multi_action_7_l0p25g1p0_s19 \
    mined_multi_action_8_l1p0g1p0_s19 \
    mined_multi_action_1_l0p25g1p0_s5_socnear_min \
    mined_multi_action_1_l0p25g1p0_s5_socnear_max \
    mined_multi_action_2_l1p0g1p0_s5_socnear_min \
    mined_multi_action_2_l1p0g1p0_s5_socnear_max \
    mined_multi_action_3_l0p25g1p0_s12_socnear_min \
    mined_multi_action_3_l0p25g1p0_s12_socnear_max \
    mined_multi_action_4_l1p0g1p0_s12_socnear_min \
    mined_multi_action_4_l1p0g1p0_s12_socnear_max \
    mined_multi_action_5_l0p25g1p0_s16_socnear_min \
    mined_multi_action_5_l0p25g1p0_s16_socnear_max \
    mined_multi_action_6_l1p0g1p0_s16_socnear_min \
    mined_multi_action_6_l1p0g1p0_s16_socnear_max \
    mined_multi_action_7_l0p25g1p0_s19_socnear_min \
    mined_multi_action_7_l0p25g1p0_s19_socnear_max \
    mined_multi_action_8_l1p0g1p0_s19_socnear_min \
    mined_multi_action_8_l1p0g1p0_s19_socnear_max \
  --policies progress_mag --reps 3 --rep-start-seed 1000 \
  --max-steps 8 --max-proposals 3 --temperature 0.0 \
  --request-timeout-s 360 --max-retries 0 \
  --output "$OUT_JSON" --log-file "$RUN_LOG"
echo "[done] $(date)"
