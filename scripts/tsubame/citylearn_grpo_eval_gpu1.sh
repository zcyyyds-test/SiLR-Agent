#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N cl_grpo_eval
#$ -o logs/
#$ -e logs/

# === Greedy gated+ungated eval of a CityLearn GRPO adapter (or base) ===
# The decisive (paper) measure -- temp 0 greedy, thinking off, in-process HF.
# Submit per adapter, e.g.
#   qsub -g tga-zhou-spring -l gpu_1=1 -l h_rt=03:00:00 \
#        -v ADAPTER=outputs/citylearn_grpo_D_s0_JOBID/iter_2,LABEL=D_s0_iter2 \
#        scripts/tsubame/citylearn_grpo_eval_gpu1.sh
# Base (untrained) baseline: omit ADAPTER (LABEL=base).
# Evaluates ALL 24 mined scenarios so the 12 TRAINED (in-distribution) and the
# 12 held-out can be split post-hoc -- the held-out ungated lift is the
# generalization claim (mirrors ANM 06). UNGATED recovery is the primary DV.
set -euo pipefail
ADAPTER="${ADAPTER:-}"
LABEL="${LABEL:-base}"
WORK="${HOME/home/work}"
PROJECT="$WORK/SILR-WISE26"
ENV_DIR="$WORK/envs/silr-vllm"
BASE_MODEL="$WORK/models/Qwen_Qwen3-8B"
LOG_DIR="$PROJECT/logs"
JOB_TAG="${JOB_ID:-manual}"
OUT_JSON="$PROJECT/eval_citylearn_grpo_${LABEL}.json"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/cl_grpo_eval_${LABEL}_${JOB_TAG}.inner.log") 2>&1
echo "[start] label=$LABEL adapter=${ADAPTER:-none} $(date)"
module purge; module load cuda/13.1.1
export HF_HOME="$WORK/hf"; export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJECT"; export PATH="$ENV_DIR/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export SILR_CITYLEARN_N_BUILDINGS=4   # hardened pillar-2 band
cd "$PROJECT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -n "$ADAPTER" ] && [ ! -d "$PROJECT/$ADAPTER" ] && [ ! -d "$ADAPTER" ] && { echo "[error] adapter dir not found: $ADAPTER"; exit 1; }
ADAPTER_ARG=""
[ -n "$ADAPTER" ] && ADAPTER_ARG="--adapter $ADAPTER"
echo "[greedy gated+ungated eval, 24 scen x 2 regimes @ N=4] $(date)"
"$ENV_DIR/bin/python" -u scripts/citylearn_grpo_eval.py \
  --base-model "$BASE_MODEL" $ADAPTER_ARG --label "$LABEL" \
  --max-steps 8 --max-proposals 3 --max-new-tokens 512 \
  --output "$OUT_JSON" --log-file "$LOG_DIR/cl_grpo_eval_${LABEL}_run.log"
echo "[done] label=$LABEL $(date)"
