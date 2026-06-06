#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N cl_grpo_obs
#$ -o logs/
#$ -e logs/

# === Observer eval: ungated execution + PASSIVE Phi trace (no gating) ===
# Measures the policy's INTRINSIC product-order geometry use (pre-reg H1b:
# worst-branch elimination, magnitude-drift avoidance) which the verifier-off
# primary-DV run cannot record (no Phi computed when the gate is off). Recovery
# numbers match the plain ungated regime; the new payload is the per-step Phi.
# Submit per adapter, e.g.
#   qsub -g tga-zhou-spring -l gpu_1=1 -l h_rt=02:00:00 \
#        -v ADAPTER=/gs/.../citylearn_grpo_D_s0_7877588/iter_2,LABEL=D_s0_obs \
#        scripts/tsubame/citylearn_grpo_observer_gpu1.sh
set -euo pipefail
ADAPTER="${ADAPTER:-}"
LABEL="${LABEL:-base_obs}"
WORK="${HOME/home/work}"
PROJECT="$WORK/SILR-WISE26"
ENV_DIR="$WORK/envs/silr-vllm"
BASE_MODEL="$WORK/models/Qwen_Qwen3-8B"
LOG_DIR="$PROJECT/logs"
JOB_TAG="${JOB_ID:-manual}"
OUT_JSON="$PROJECT/eval_citylearn_grpo_${LABEL}.json"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/cl_grpo_obs_${LABEL}_${JOB_TAG}.inner.log") 2>&1
echo "[start] OBSERVER label=$LABEL adapter=${ADAPTER:-none} $(date)"
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
echo "[observer eval: ungated + passive Phi, 24 scen @ N=4] $(date)"
"$ENV_DIR/bin/python" -u scripts/citylearn_grpo_eval.py \
  --base-model "$BASE_MODEL" $ADAPTER_ARG --label "$LABEL" --observer-trace \
  --max-steps 8 --max-proposals 3 --max-new-tokens 512 \
  --output "$OUT_JSON" --log-file "$LOG_DIR/cl_grpo_obs_${LABEL}_run.log"
echo "[done] OBSERVER label=$LABEL $(date)"
