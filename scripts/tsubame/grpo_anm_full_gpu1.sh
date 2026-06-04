#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N grpo_anm_full
#$ -o logs/
#$ -e logs/

# === GRPO ANM full training run (one arm) ===
# Submit per arm with -v ARM=C|D|E, e.g.
#   qsub -g tga-zhou-spring -l gpu_1=1 -l h_rt=08:00:00 -v ARM=D scripts/tsubame/grpo_anm_full_gpu1.sh
# Trains the 9 Step-0 sub-saturated band scenarios, bf16 LoRA, thinking-off
# rollouts (fast + parseable), per-step verdict→reward (arm-specific), 5 iters.
# Chain validated by the smoke (job 7846062: null-var groups 0, gradient alive).
# Recovery here is the SAMPLING-rollout figure; the paper number comes from a
# SEPARATE greedy eval of the saved adapter.
set -euo pipefail
ARM="${ARM:-D}"
SEED="${SEED:-0}"
LABEL="${LABEL:-$ARM}"          # output/log tag; D-flat ablation uses LABEL=Dflat
WORK="${HOME/home/work}"
PROJECT="$WORK/SILR-WISE26"
ENV_DIR="$WORK/envs/silr-vllm"
BASE_MODEL="$WORK/models/Qwen_Qwen3-8B"
LOG_DIR="$PROJECT/logs"
JOB_TAG="${JOB_ID:-manual}"
OUT_DIR="$PROJECT/outputs/anm_grpo_${LABEL}_s${SEED}_${JOB_TAG}"
# D-flat ablation: flat SAFE_PROGRESS constant (no Φ geometry) via reward.py hook.
[ -n "${SP_FLAT:-}" ] && export SILR_SP_FLAT="$SP_FLAT"
mkdir -p "$LOG_DIR" "$OUT_DIR"
exec > >(tee -a "$LOG_DIR/grpo_anm_full_${ARM}_${JOB_TAG}.inner.log") 2>&1
echo "[start] arm=$ARM $(date)"
module purge; module load cuda/13.1.1
export HF_HOME="$WORK/hf"; export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPATH="$PROJECT"; export PATH="$ENV_DIR/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
cd "$PROJECT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$ENV_DIR/bin/python" - <<'PYCHK' || { echo "[deps-failed]"; exit 1; }
import torch, peft, transformers
print("torch", torch.__version__, "| peft", peft.__version__,
      "| transformers", transformers.__version__, "| cuda", torch.cuda.is_available())
PYCHK
echo "[full train arm $ARM seed $SEED: 9 scen x 6 rollout x 5 iter] $(date)"
"$ENV_DIR/bin/python" -u scripts/train_grpo_anm.py \
  --arm "$ARM" --seed "$SEED" \
  --base-model "$BASE_MODEL" \
  --iterations 5 --rollouts-per-scenario 6 \
  --max-steps 8 --max-proposals 3 --temperature 0.7 \
  --max-new-tokens 512 \
  --output "$OUT_DIR"
echo "[done] arm=$ARM seed=$SEED $(date)"
