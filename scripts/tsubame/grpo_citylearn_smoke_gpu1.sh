#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N grpo_cl_smoke
#$ -o logs/
#$ -e logs/

# === GRPO CityLearn training SMOKE (validate the N=4 loop before the campaign) ===
#   qsub -g tga-zhou-spring -l gpu_1=1 -l h_rt=01:00:00 scripts/tsubame/grpo_citylearn_smoke_gpu1.sh
# 1 iteration, 2 hard multi-type scenarios, 4 rollouts each: confirm the
# rollout -> per-step verdict reward -> group-relative advantage -> PPO update
# loop runs end-to-end on the hardened N=4 band, and that group std is NONZERO
# (the null-advantage / dead-gradient failure the ANM smoke caught). This is the
# cheap gate before the multi-day 3-arm campaign -- NOT a paper number.
set -euo pipefail
ARM="${ARM:-D}"
SEED="${SEED:-0}"
WORK="${HOME/home/work}"
PROJECT="$WORK/SILR-WISE26"
ENV_DIR="$WORK/envs/silr-vllm"
BASE_MODEL="$WORK/models/Qwen_Qwen3-8B"
LOG_DIR="$PROJECT/logs"
JOB_TAG="${JOB_ID:-manual}"
OUT_DIR="$PROJECT/outputs/citylearn_grpo_smoke_${ARM}_${JOB_TAG}"
mkdir -p "$LOG_DIR" "$OUT_DIR"
exec > >(tee -a "$LOG_DIR/grpo_cl_smoke_${ARM}_${JOB_TAG}.inner.log") 2>&1
echo "[start] arm=$ARM $(date)"
module purge; module load cuda/13.1.1
export HF_HOME="$WORK/hf"; export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJECT"; export PATH="$ENV_DIR/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export SILR_CITYLEARN_N_BUILDINGS=4   # hardened pillar-2 band
cd "$PROJECT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$ENV_DIR/bin/python" - <<'PYCHK' || { echo "[deps-failed]"; exit 1; }
import torch, peft, transformers
print("torch", torch.__version__, "| peft", peft.__version__,
      "| transformers", transformers.__version__, "| cuda", torch.cuda.is_available())
PYCHK
# Confirm the hardened band loads at N=4 (24 mined scenarios) before training.
"$ENV_DIR/bin/python" -c "
from domains.citylearn.scenarios import SCENARIOS
import domains.citylearn.simulator as s
mined=[x.id for x in SCENARIOS if x.id.startswith('cl_mined_')]
assert s.N_BUILDINGS==4, f'expected N=4, got {s.N_BUILDINGS}'
assert len(mined)==24, f'expected 24 mined, got {len(mined)}'
print(f'[band-ok] N={s.N_BUILDINGS}, {len(mined)} mined scenarios')
"
# Two hard multi-type scenarios with headroom: a 3-family + a cross-family pair.
SMOKE_SCEN="cl_mined_002_t11_smax-smin-exp cl_mined_000_t11_smax-exp"
echo "[grpo smoke arm $ARM: 2 scen x 4 rollout x 1 iter @ N=4] $(date)"
"$ENV_DIR/bin/python" -u scripts/train_grpo_citylearn.py \
  --arm "$ARM" --seed "$SEED" \
  --base-model "$BASE_MODEL" \
  --scenarios $SMOKE_SCEN \
  --iterations 1 --rollouts-per-scenario 4 \
  --max-steps 8 --max-proposals 3 --temperature 0.7 \
  --max-new-tokens 512 \
  --output "$OUT_DIR"
echo "[done] arm=$ARM seed=$SEED $(date)"
