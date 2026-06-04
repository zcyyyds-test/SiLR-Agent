#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N grpo_eval_run
#$ -o logs/
#$ -e logs/
# Eval all iter_* checkpoints of one training run (greedy, gated+ungated).
# Submit with -v RUNDIR=<abs path>,LABEL=<arm_sN>
set -euo pipefail
WORK="${HOME/home/work}"; PROJECT="$WORK/SILR-WISE26"; ENV_DIR="$WORK/envs/silr-vllm"
BASE="$WORK/models/Qwen_Qwen3-8B"; LOG_DIR="$PROJECT/logs"; JT="${JOB_ID:-manual}"; mkdir -p "$LOG_DIR"
: "${RUNDIR:?need RUNDIR}"; : "${LABEL:?need LABEL}"
exec > >(tee -a "$LOG_DIR/grpo_eval_run_${LABEL}_${JT}.inner.log") 2>&1
echo "[start] $LABEL $RUNDIR $(date)"; module purge; module load cuda/13.1.1
export HF_HOME="$WORK/hf"; export PYTHONPATH="$PROJECT"; export PATH="$ENV_DIR/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0; export TOKENIZERS_PARALLELISM=false; cd "$PROJECT"
nvidia-smi --query-gpu=name --format=csv,noheader
for ck in iter_1 iter_2 iter_3 iter_4 iter_5; do
  [ -d "$RUNDIR/$ck" ] || continue
  echo "[eval $LABEL $ck] $(date)"
  "$ENV_DIR/bin/python" -u scripts/anm_grpo_eval.py --base-model "$BASE" \
    --adapter "$RUNDIR/$ck" --label "${LABEL}_${ck}" \
    --output "$PROJECT/eval_grpo_${LABEL}_${ck}.json" \
    --log-file "$LOG_DIR/eval_grpo_${LABEL}_${ck}_${JT}.log"
done
echo "[done] $LABEL $(date)"
