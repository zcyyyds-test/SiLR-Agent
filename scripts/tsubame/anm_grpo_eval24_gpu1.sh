#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N anm_eval24
#$ -o logs/
#$ -e logs/

# === Greedy gated+ungated eval of an ANM GRPO adapter over ALL 24 mined
# multi_action scenarios (8 base traps x up to 3 SoC conditions) ===
# The original ANM eval covered only the 9 DEFAULT_SCENARIOS -> the outcome-level
# D-vs-E test was underpowered (power 0.42 at N=9; ~27 scenarios needed for 0.80,
# see decisions power analysis). This widens the eval N to 24 so the paired
# significance test gets more observations (8 base-trap clusters x conditions);
# UNGATED recovery is the primary DV. In-process HF, temp-0 greedy, thinking off.
# Submit per adapter:
#   qsub -g tga-zhou-spring -l gpu_1=1 -l h_rt=03:00:00 \
#        -v ADAPTER=outputs/anm_grpo_D_7846091/iter_2,LABEL=D24_s0 \
#        scripts/tsubame/anm_grpo_eval24_gpu1.sh
set -euo pipefail
ADAPTER="${ADAPTER:-}"
LABEL="${LABEL:-base}"
WORK="${HOME/home/work}"
PROJECT="$WORK/SILR-WISE26"
ENV_DIR="$WORK/envs/silr-vllm"
BASE_MODEL="$WORK/models/Qwen_Qwen3-8B"
LOG_DIR="$PROJECT/logs"
JOB_TAG="${JOB_ID:-manual}"
OUT_JSON="$PROJECT/eval_anm24_grpo_${LABEL}.json"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/anm_eval24_${LABEL}_${JOB_TAG}.inner.log") 2>&1
echo "[start] label=$LABEL adapter=${ADAPTER:-none} $(date)"
module purge; module load cuda/13.1.1
export HF_HOME="$WORK/hf"; export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJECT"; export PATH="$ENV_DIR/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
cd "$PROJECT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -n "$ADAPTER" ] && [ ! -d "$PROJECT/$ADAPTER" ] && [ ! -d "$ADAPTER" ] && { echo "[error] adapter dir not found: $ADAPTER"; exit 1; }
ADAPTER_ARG=""
[ -n "$ADAPTER" ] && ADAPTER_ARG="--adapter $ADAPTER"

SCENS="mined_multi_action_1_l0p25g1p0_s5 mined_multi_action_2_l1p0g1p0_s5 \
mined_multi_action_3_l0p25g1p0_s12 mined_multi_action_4_l1p0g1p0_s12 \
mined_multi_action_5_l0p25g1p0_s16 mined_multi_action_6_l1p0g1p0_s16 \
mined_multi_action_7_l0p25g1p0_s19 mined_multi_action_8_l1p0g1p0_s19 \
mined_multi_action_1_l0p25g1p0_s5_socnear_min mined_multi_action_1_l0p25g1p0_s5_socnear_max \
mined_multi_action_2_l1p0g1p0_s5_socnear_min mined_multi_action_2_l1p0g1p0_s5_socnear_max \
mined_multi_action_3_l0p25g1p0_s12_socnear_min mined_multi_action_3_l0p25g1p0_s12_socnear_max \
mined_multi_action_4_l1p0g1p0_s12_socnear_min mined_multi_action_4_l1p0g1p0_s12_socnear_max \
mined_multi_action_5_l0p25g1p0_s16_socnear_min mined_multi_action_5_l0p25g1p0_s16_socnear_max \
mined_multi_action_6_l1p0g1p0_s16_socnear_min mined_multi_action_6_l1p0g1p0_s16_socnear_max \
mined_multi_action_7_l0p25g1p0_s19_socnear_min mined_multi_action_7_l0p25g1p0_s19_socnear_max \
mined_multi_action_8_l1p0g1p0_s19_socnear_min mined_multi_action_8_l1p0g1p0_s19_socnear_max"

echo "[greedy gated+ungated eval, 24 scen x 2 regimes] $(date)"
"$ENV_DIR/bin/python" -u scripts/anm_grpo_eval.py \
  --base-model "$BASE_MODEL" $ADAPTER_ARG --label "$LABEL" \
  --scenarios $SCENS \
  --max-steps 8 --max-proposals 3 --max-new-tokens 512 \
  --output "$OUT_JSON" --log-file "$LOG_DIR/anm_eval24_${LABEL}_run.log"
echo "[done] label=$LABEL $(date)"
