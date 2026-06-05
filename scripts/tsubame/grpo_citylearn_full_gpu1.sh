#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N grpo_cl_full
#$ -o logs/
#$ -e logs/

# === GRPO CityLearn full training (one arm-seed) -- pillar-2 multi-type campaign ===
# Submit per (arm, seed) as an independent gpu_1 job, e.g.
#   qsub -g tga-zhou-spring -l gpu_1=1 -l h_rt=14:00:00 -v ARM=D,SEED=0 \
#        scripts/tsubame/grpo_citylearn_full_gpu1.sh
# Scope (panel + checkpoint 2026-06-04): arms C/D/E x seeds {0,1,2} = 9 parallel
# jobs. 12 headroom multi-type scenarios (3 each of smax-exp / smax-smin-exp /
# smin-exp / smax-smin; the saturated smax-imp set is dropped). bf16 LoRA,
# thinking-off rollouts, per-step verdict->reward (arm-specific), 4 iters x 4
# rollouts. Loop validated by the N=4 smoke (job 7866067: null-var groups 0/2,
# gradient alive). Recovery here is the SAMPLING-rollout figure; the paper number
# comes from a SEPARATE greedy ungated eval of the saved adapters.
set -euo pipefail
ARM="${ARM:-D}"
SEED="${SEED:-0}"
LABEL="${LABEL:-$ARM}"          # D-flat ablation would use LABEL=Dflat + SP_FLAT=1
WORK="${HOME/home/work}"
PROJECT="$WORK/SILR-WISE26"
ENV_DIR="$WORK/envs/silr-vllm"
BASE_MODEL="$WORK/models/Qwen_Qwen3-8B"
LOG_DIR="$PROJECT/logs"
JOB_TAG="${JOB_ID:-manual}"
OUT_DIR="/gs/fs/tga-zhou-spring/silr-outputs/citylearn_grpo_${LABEL}_s${SEED}_${JOB_TAG}"   # SSD: checkpoints belong on group SSD, not /work (100G quota)
[ -n "${SP_FLAT:-}" ] && export SILR_SP_FLAT="$SP_FLAT"
mkdir -p "$LOG_DIR" "$OUT_DIR"
exec > >(tee -a "$LOG_DIR/grpo_cl_full_${LABEL}_s${SEED}_${JOB_TAG}.inner.log") 2>&1
echo "[start] arm=$ARM seed=$SEED $(date)"
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
"$ENV_DIR/bin/python" -c "
from domains.citylearn.scenarios import SCENARIOS
import domains.citylearn.simulator as s
assert s.N_BUILDINGS==4, f'expected N=4, got {s.N_BUILDINGS}'
print(f'[band-ok] N={s.N_BUILDINGS}, {len([x for x in SCENARIOS if x.id.startswith(\"cl_mined_\")])} mined')
"

# 12 headroom multi-type scenarios (3 per family signature; smax-imp dropped).
SCEN12="cl_mined_000_t11_smax-exp cl_mined_005_t12_smax-exp cl_mined_010_t13_smax-exp \
cl_mined_002_t11_smax-smin-exp cl_mined_007_t12_smax-smin-exp cl_mined_012_t13_smax-smin-exp \
cl_mined_003_t11_smin-exp cl_mined_008_t12_smin-exp cl_mined_013_t13_smin-exp \
cl_mined_004_t11_smax-smin cl_mined_009_t12_smax-smin cl_mined_014_t13_smax-smin"

echo "[full train arm $ARM seed $SEED: 12 scen x 6 rollout x 4 iter @ N=4] $(date)"
"$ENV_DIR/bin/python" -u scripts/train_grpo_citylearn.py \
  --arm "$ARM" --seed "$SEED" \
  --base-model "$BASE_MODEL" \
  --scenarios $SCEN12 \
  --iterations 4 --rollouts-per-scenario 6 \
  --max-steps 8 --max-proposals 3 --temperature 0.7 \
  --max-new-tokens 512 \
  --output "$OUT_DIR"
echo "[done] arm=$ARM seed=$SEED $(date)"
