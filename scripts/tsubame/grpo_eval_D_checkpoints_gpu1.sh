#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N grpo_eval_Dck
#$ -o logs/
#$ -e logs/
# Diagnose arm-D degradation trajectory: greedy eval of iter_2 and iter_4
# checkpoints (base=77%, final=33% already known). Monotonic decline →
# over-optimisation (early-stop/lower-lr fixable); iter_2 already low → fast collapse.
set -euo pipefail
WORK="${HOME/home/work}"; PROJECT="$WORK/SILR-WISE26"; ENV_DIR="$WORK/envs/silr-vllm"
BASE_MODEL="$WORK/models/Qwen_Qwen3-8B"; RUN="$PROJECT/outputs/anm_grpo_D_7846091"
LOG_DIR="$PROJECT/logs"; JOB_TAG="${JOB_ID:-manual}"; mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/grpo_eval_Dck_${JOB_TAG}.inner.log") 2>&1
echo "[start] $(date)"; module purge; module load cuda/13.1.1
export HF_HOME="$WORK/hf"; export PYTHONPATH="$PROJECT"; export PATH="$ENV_DIR/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0; export TOKENIZERS_PARALLELISM=false; cd "$PROJECT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for CK in iter_2 iter_4; do
  test -d "$RUN/$CK" || { echo "[skip] $CK not found"; continue; }
  echo "[eval $CK] $(date)"
  "$ENV_DIR/bin/python" -u scripts/anm_grpo_eval.py --base-model "$BASE_MODEL" \
    --adapter "$RUN/$CK" --label "armD_$CK" \
    --output "$PROJECT/eval_grpo_armD_${CK}.json" \
    --log-file "$LOG_DIR/eval_grpo_armD_${CK}_${JOB_TAG}.log"
done
echo "[done] $(date)"
