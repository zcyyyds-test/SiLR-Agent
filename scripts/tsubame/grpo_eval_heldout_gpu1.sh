#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N grpo_eval_ho
#$ -o logs/
#$ -e logs/
# Held-out generalization eval: base + D + E (seed0 iter_2) on the 15 NON-training
# band24 scenarios. Tests whether D's ungated-preservation generalizes (kills the
# in-distribution attack). Greedy, gated+ungated, thinking-off.
set -euo pipefail
WORK="${HOME/home/work}"; PROJECT="$WORK/SILR-WISE26"; ENV_DIR="$WORK/envs/silr-vllm"
BASE="$WORK/models/Qwen_Qwen3-8B"; LOG_DIR="$PROJECT/logs"; JT="${JOB_ID:-manual}"; mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/grpo_eval_ho_${JT}.inner.log") 2>&1
echo "[start] $(date)"; module purge; module load cuda/13.1.1
export HF_HOME="$WORK/hf"; export PYTHONPATH="$PROJECT"; export PATH="$ENV_DIR/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0; export TOKENIZERS_PARALLELISM=false; cd "$PROJECT"
HO="mined_multi_action_1_l0p25g1p0_s5 mined_multi_action_2_l1p0g1p0_s5 mined_multi_action_3_l0p25g1p0_s12 mined_multi_action_6_l1p0g1p0_s16 mined_multi_action_7_l0p25g1p0_s19 mined_multi_action_1_l0p25g1p0_s5_socnear_max mined_multi_action_2_l1p0g1p0_s5_socnear_max mined_multi_action_3_l0p25g1p0_s12_socnear_min mined_multi_action_3_l0p25g1p0_s12_socnear_max mined_multi_action_4_l1p0g1p0_s12_socnear_min mined_multi_action_4_l1p0g1p0_s12_socnear_max mined_multi_action_6_l1p0g1p0_s16_socnear_max mined_multi_action_7_l0p25g1p0_s19_socnear_min mined_multi_action_8_l1p0g1p0_s19_socnear_min mined_multi_action_8_l1p0g1p0_s19_socnear_max"
run () { # label adapterflag
  echo "[heldout eval $1] $(date)"
  "$ENV_DIR/bin/python" -u scripts/anm_grpo_eval.py --base-model "$BASE" $2 \
    --label "ho_$1" --scenarios $HO \
    --output "$PROJECT/eval_grpo_heldout_$1.json" \
    --log-file "$LOG_DIR/eval_grpo_heldout_$1_${JT}.log"
}
run base ""
run D "--adapter $PROJECT/outputs/anm_grpo_D_7846091/iter_2"
run E "--adapter $PROJECT/outputs/anm_grpo_E_7849255/iter_2"
echo "[done] $(date)"
