#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N grpo_mech
#$ -o logs/
#$ -e logs/
# Mechanism re-eval: eval iter_2 of all seeds of ARM on the 9 training scenarios
# with per-step Φ geometry trace (for first-admissible-step, worst-branch-reduction,
# drift). Submit with -v ARM=D|E|Dflat.
set -euo pipefail
ARM="${ARM:?need ARM}"
WORK="${HOME/home/work}"; PROJECT="$WORK/SILR-WISE26"; ENV_DIR="$WORK/envs/silr-vllm"
BASE="$WORK/models/Qwen_Qwen3-8B"; LOG_DIR="$PROJECT/logs"; JT="${JOB_ID:-manual}"; mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/grpo_mech_${ARM}_${JT}.inner.log") 2>&1
echo "[start] ARM=$ARM $(date)"; module purge; module load cuda/13.1.1
export HF_HOME="$WORK/hf"; export PYTHONPATH="$PROJECT"; export PATH="$ENV_DIR/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0; export TOKENIZERS_PARALLELISM=false; cd "$PROJECT"
nvidia-smi --query-gpu=name --format=csv,noheader
for d in outputs/anm_grpo_${ARM}_*/iter_2; do
  [ -d "$d" ] || continue
  tag=$(echo "$d" | sed -E "s#outputs/anm_grpo_(.*)_[0-9]+/iter_2#\1#")  # e.g. D_s1 or D (seed0) or Dflat_s0
  echo "[mech-eval $tag] $(date)"
  "$ENV_DIR/bin/python" -u scripts/anm_grpo_eval.py --base-model "$BASE" \
    --adapter "$PROJECT/$d" --label "mech_${tag}" \
    --output "$PROJECT/eval_grpo_mech_${tag}.json" \
    --log-file "$LOG_DIR/eval_grpo_mech_${tag}_${JT}.log"
done
echo "[done] ARM=$ARM $(date)"
