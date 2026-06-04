#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N grpo_anm_smoke_D
#$ -o logs/
#$ -e logs/

# === GRPO ANM smoke train (arm D) — full-chain silent-bug check ===
# Tiny config (3 scenarios x 2 rollouts x 2 iters) to verify the chain
# verdict → Φ-reward → advantage → log-prob → PPO loss → LoRA adapter runs end
# to end without a silent bug, BEFORE the 5-arm main experiment. Watch for:
#   - zero-variance groups (null advantage) in the Phase-2 log
#   - non-NaN loss, log-ratio not all-clamped
#   - the bare-text ANM rollout actually parses actions (accept count > 0)
# Training is in-process HF (no vLLM serve). Greedy eval of the trained adapter
# is a SEPARATE later step (per GRPO lessons: judge recovery with greedy, not
# this sampling rollout).
set -euo pipefail
WORK="${HOME/home/work}"
PROJECT="$WORK/SILR-WISE26"
ENV_DIR="$WORK/envs/silr-vllm"
BASE_MODEL="$WORK/models/Qwen_Qwen3-8B"
LOG_DIR="$PROJECT/logs"
JOB_TAG="${JOB_ID:-manual}"
OUT_DIR="$PROJECT/outputs/anm_grpo_smoke_D_${JOB_TAG}"
mkdir -p "$LOG_DIR" "$OUT_DIR"
exec > >(tee -a "$LOG_DIR/grpo_anm_smoke_D_${JOB_TAG}.inner.log") 2>&1
echo "[start] $(date)"
module purge; module load cuda/13.1.1
export HF_HOME="$WORK/hf"; export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPATH="$PROJECT"; export PATH="$ENV_DIR/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
cd "$PROJECT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# Confirm the training stack imports before the run (bf16 path — no bitsandbytes).
"$ENV_DIR/bin/python" - <<'PYCHK' || { echo "[deps-failed] training stack missing on compute node"; exit 1; }
import torch, peft, transformers
print("torch", torch.__version__, "| peft", peft.__version__,
      "| transformers", transformers.__version__, "| cuda", torch.cuda.is_available())
PYCHK
echo "[smoke train arm D: 3 scen x 2 rollout x 2 iter] $(date)"
"$ENV_DIR/bin/python" -u scripts/train_grpo_anm.py \
  --arm D \
  --base-model "$BASE_MODEL" \
  --scenarios \
    mined_multi_action_6_l1p0g1p0_s16_socnear_min \
    mined_multi_action_1_l0p25g1p0_s5_socnear_min \
    mined_multi_action_8_l1p0g1p0_s19 \
  --iterations 2 --rollouts-per-scenario 2 \
  --max-steps 8 --max-proposals 3 --temperature 0.7 \
  --max-new-tokens 512 \
  --output "$OUT_DIR"
echo "[done] $(date)"
