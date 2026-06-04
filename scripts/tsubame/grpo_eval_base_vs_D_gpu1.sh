#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N grpo_eval_AvD
#$ -o logs/
#$ -e logs/

# === Decisive greedy eval: untrained base (arm A) vs GRPO arm-D adapter ===
# Both evaluated greedy (temp 0), thinking-off, gated + ungated, on the 9
# Step-0 sub-saturated scenarios. Tells us (a) did training help gated recovery,
# (b) did arm D internalise geometry (ungated recovery + lower residual penalty
# vs base). Cheap (~36 greedy episodes). h_rt 02:00:00.
set -euo pipefail
WORK="${HOME/home/work}"
PROJECT="$WORK/SILR-WISE26"
ENV_DIR="$WORK/envs/silr-vllm"
BASE_MODEL="$WORK/models/Qwen_Qwen3-8B"
ADAPTER_D="${ADAPTER_D:-$PROJECT/outputs/anm_grpo_D_7846091/final}"
LOG_DIR="$PROJECT/logs"
JOB_TAG="${JOB_ID:-manual}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/grpo_eval_AvD_${JOB_TAG}.inner.log") 2>&1
echo "[start] $(date)"
module purge; module load cuda/13.1.1
export HF_HOME="$WORK/hf"; export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPATH="$PROJECT"; export PATH="$ENV_DIR/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
cd "$PROJECT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
test -d "$ADAPTER_D" || { echo "[error] adapter not found: $ADAPTER_D"; exit 1; }

echo "[eval arm A — untrained base] $(date)"
"$ENV_DIR/bin/python" -u scripts/anm_grpo_eval.py \
  --base-model "$BASE_MODEL" --label baseA \
  --output "$PROJECT/eval_grpo_baseA.json" \
  --log-file "$LOG_DIR/eval_grpo_baseA_${JOB_TAG}.log"

echo "[eval arm D — GRPO-verifier adapter] $(date)"
"$ENV_DIR/bin/python" -u scripts/anm_grpo_eval.py \
  --base-model "$BASE_MODEL" --adapter "$ADAPTER_D" --label armD \
  --output "$PROJECT/eval_grpo_armD.json" \
  --log-file "$LOG_DIR/eval_grpo_armD_${JOB_TAG}.log"
echo "[done] $(date)"
