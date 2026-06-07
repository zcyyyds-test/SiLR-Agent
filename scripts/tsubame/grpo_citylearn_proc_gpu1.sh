#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N grpo_cl_proc
#$ -o logs/
#$ -e logs/

# === GRPO CityLearn PROCESS-REWARD-DOMINANT training (pillar-2 sigma-het trap) ===
# Same 12 multi-type N=4 scenarios as the full campaign, but with
#   recovery_bonus = 0  (the terminal outcome bonus is what diluted the per-step
#                        geometric advantage; with it off the per-family process
#                        reward drives the GRPO gradient -- see the tabular sims
#                        grpo_train_coupled / grpo_advantage_dilution)
#   max_steps = 6, step_cost = 0.02  (tighter budget -> prioritisation pressure)
# so the geometric (per-family) reward D should train a better policy than the
# scalar count reward E in the real LLM RL, not just at eval of old adapters.
# Submit per (arm, seed):
#   qsub -g tga-zhou-spring -l gpu_1=1 -l h_rt=14:00:00 -v ARM=D,SEED=0 \
#        scripts/tsubame/grpo_citylearn_proc_gpu1.sh
set -euo pipefail
ARM="${ARM:-D}"
SEED="${SEED:-0}"
LABEL="${LABEL:-${ARM}proc}"
RBONUS="${RBONUS:-0.0}"
MAXSTEPS="${MAXSTEPS:-6}"
STEPCOST="${STEPCOST:-0.02}"
ITERS="${ITERS:-4}"
WORK="${HOME/home/work}"
PROJECT="$WORK/SILR-WISE26"
ENV_DIR="$WORK/envs/silr-vllm"
BASE_MODEL="$WORK/models/Qwen_Qwen3-8B"
LOG_DIR="$PROJECT/logs"
JOB_TAG="${JOB_ID:-manual}"
OUT_DIR="/gs/fs/tga-zhou-spring/silr-outputs/citylearn_grpo_${LABEL}_s${SEED}_${JOB_TAG}"
mkdir -p "$LOG_DIR" "$OUT_DIR"
exec > >(tee -a "$LOG_DIR/grpo_cl_proc_${LABEL}_s${SEED}_${JOB_TAG}.inner.log") 2>&1
echo "[start] arm=$ARM seed=$SEED rbonus=$RBONUS maxsteps=$MAXSTEPS $(date)"
module purge; module load cuda/13.1.1
export HF_HOME="$WORK/hf"; export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJECT"; export PATH="$ENV_DIR/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export SILR_CITYLEARN_N_BUILDINGS=4
cd "$PROJECT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$ENV_DIR/bin/python" -c "import domains.citylearn.simulator as s; assert s.N_BUILDINGS==4; print('[band-ok] N=4')"

SCEN12="cl_mined_000_t11_smax-exp cl_mined_005_t12_smax-exp cl_mined_010_t13_smax-exp \
cl_mined_002_t11_smax-smin-exp cl_mined_007_t12_smax-smin-exp cl_mined_012_t13_smax-smin-exp \
cl_mined_003_t11_smin-exp cl_mined_008_t12_smin-exp cl_mined_013_t13_smin-exp \
cl_mined_004_t11_smax-smin cl_mined_009_t12_smax-smin cl_mined_014_t13_smax-smin"

echo "[proc-reward train arm $ARM seed $SEED: rbonus=$RBONUS maxsteps=$MAXSTEPS iters=$ITERS] $(date)"
"$ENV_DIR/bin/python" -u scripts/train_grpo_citylearn.py \
  --arm "$ARM" --seed "$SEED" \
  --base-model "$BASE_MODEL" \
  --scenarios $SCEN12 \
  --iterations "$ITERS" --rollouts-per-scenario 6 \
  --max-steps "$MAXSTEPS" --max-proposals 3 --temperature 0.7 \
  --step-cost "$STEPCOST" --recovery-bonus "$RBONUS" \
  --max-new-tokens 512 \
  --output "$OUT_DIR"
echo "[done] arm=$ARM seed=$SEED $(date)"
