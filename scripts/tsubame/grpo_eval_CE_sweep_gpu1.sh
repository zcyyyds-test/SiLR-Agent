#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N grpo_eval_CE
#$ -o logs/
#$ -e logs/
# Matched 3-arm comparison: greedy eval (gated+ungated) of every checkpoint of
# arm C and arm E, plus the missing D iter_1/iter_3, to pick each arm's best.
set -euo pipefail
WORK="${HOME/home/work}"; PROJECT="$WORK/SILR-WISE26"; ENV_DIR="$WORK/envs/silr-vllm"
BASE="$WORK/models/Qwen_Qwen3-8B"; LOG_DIR="$PROJECT/logs"; JT="${JOB_ID:-manual}"; mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/grpo_eval_CE_${JT}.inner.log") 2>&1
echo "[start] $(date)"; module purge; module load cuda/13.1.1
export HF_HOME="$WORK/hf"; export PYTHONPATH="$PROJECT"; export PATH="$ENV_DIR/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0; export TOKENIZERS_PARALLELISM=false; cd "$PROJECT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
evalck () { # arm run_dir ck
  local arm="$1" run="$2" ck="$3"
  [ -d "$run/$ck" ] || { echo "[skip] $arm $ck"; return; }
  echo "[eval $arm $ck] $(date)"
  "$ENV_DIR/bin/python" -u scripts/anm_grpo_eval.py --base-model "$BASE" \
    --adapter "$run/$ck" --label "${arm}_${ck}" \
    --output "$PROJECT/eval_grpo_${arm}_${ck}.json" \
    --log-file "$LOG_DIR/eval_grpo_${arm}_${ck}_${JT}.log"
}
for ck in iter_1 iter_2 iter_3 iter_4 iter_5; do evalck C "$PROJECT/outputs/anm_grpo_C_7849254" "$ck"; done
for ck in iter_1 iter_2 iter_3 iter_4 iter_5; do evalck E "$PROJECT/outputs/anm_grpo_E_7849255" "$ck"; done
evalck D "$PROJECT/outputs/anm_grpo_D_7846091" iter_1
evalck D "$PROJECT/outputs/anm_grpo_D_7846091" iter_3
echo "[done] $(date)"
