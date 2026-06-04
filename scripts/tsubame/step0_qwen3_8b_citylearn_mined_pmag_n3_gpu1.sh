#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N qwen8b_cl_mined_pmag_n3
#$ -o logs/
#$ -e logs/

# === CityLearn Step-0 go/no-go gate (pillar-2 multi-type amplification) ===
# Qwen3-8B + progress_mag over the mined CityLearn multi-action band, N=3,
# greedy (temp 0). Mirrors the ANM band24 Step-0 gate exactly; only the domain
# and scenario source differ. Question: is 8B sub-saturated on the multi-type
# CityLearn band (so a GRPO training signal exists)?
#
# QUANTIFIED go/no-go (7-way panel 2026-06-04, panel-log/2026-06-04-1027):
#   Per scenario, count gated recovery rate over N=3 reps.
#   - GO (train): >=6 of 24 scenarios are SUB-SATURATED, i.e. gated recovery
#     in {0/3,1/3,2/3} (8B partially recovers -> learnable signal). This
#     mirrors ANM, where 9 sub-saturated scenarios carried the GRPO signal.
#   - NO-GO (re-mine): nearly all scenarios at 3/3 (saturated) -> no signal;
#     re-mine harder snapshots (other hours / lower n_feasible) before training.
#   - The band is stratified by family (ids encode the tag): report recovery
#     SEPARATELY for soc_min x export_limit (cross-family, *_smin-exp) and
#     soc_min x soc_max (cross-device antichain, *_smax-smin / *_smin-smax).
#     The cross-family subset is the regime the geometric reward should help
#     most; if only the cross-device subset is sub-saturated, narrow the claim.
# Decisive gate BEFORE forking train_grpo_anm.py -> train_grpo_citylearn.py.
# DO NOT freeze the multi-type narrative in the paper before this returns.
#
# Prereq: scripts/citylearn_scenario_mine.py + citylearn_select_multi_action.py
# have been run, so domains/citylearn/scenarios_mined.json exists.
set -euo pipefail
WORK="${HOME/home/work}"
PROJECT="$WORK/SILR-WISE26"
ENV_DIR="$WORK/envs/silr-vllm"
MODEL_DIR="$WORK/models/Qwen_Qwen3-8B"
LOG_DIR="$PROJECT/logs"
JOB_TAG="${JOB_ID:-manual}"
PORT="${SILR_VLLM_PORT:-8006}"
SERVER_LOG="$LOG_DIR/step0_qwen8b_cl_mined_server_${JOB_TAG}.log"
RUN_LOG="$LOG_DIR/step0_qwen8b_cl_mined_pmag_n3_${JOB_TAG}.log"
OUT_JSON="$PROJECT/eval_step0_qwen3_8b_citylearn_mined_n3.json"
MINED_JSON="$PROJECT/domains/citylearn/scenarios_mined.json"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/step0_qwen8b_cl_mined_pmag_n3_${JOB_TAG}.inner.log") 2>&1
echo "[start] $(date)"
module purge; module load cuda/13.1.1
export HF_HOME="$WORK/hf"; export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPATH="$PROJECT"; export PATH="$ENV_DIR/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SILR_MAX_TOKENS=2048
export SILR_SYNC_ID="step0_qwen8b_cl_mined_pmag_n3_${JOB_TAG}"
cd "$PROJECT"

# Pull the mined scenario ids from the selector output (fail loudly if absent).
[ -f "$MINED_JSON" ] || { echo "[error] $MINED_JSON missing -- run mine+select first"; exit 1; }
SCENARIO_IDS=$("$ENV_DIR/bin/python" -c \
  "import json,sys; print(' '.join(s['id'] for s in json.load(open('$MINED_JSON'))['scenarios']))")
[ -n "$SCENARIO_IDS" ] || { echo "[error] no scenarios in $MINED_JSON"; exit 1; }
N_SCEN=$(echo "$SCENARIO_IDS" | wc -w)
echo "[scenarios] $N_SCEN mined CityLearn scenarios"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$ENV_DIR/bin/vllm" serve "$MODEL_DIR" --host 127.0.0.1 --port "$PORT" \
  --served-model-name qwen3-8b --gpu-memory-utilization 0.85 \
  --max-model-len 16384 --enforce-eager \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --trust-remote-code >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT TERM
for i in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then echo "[ready] $(date)"; break; fi
  ! kill -0 "$SERVER_PID" 2>/dev/null && { echo "[serve-exited]"; tail -120 "$SERVER_LOG"; exit 1; }
  [ "$i" -eq 180 ] && { echo "[timeout]"; tail -160 "$SERVER_LOG"; exit 1; }
  sleep 5
done
echo "[eval progress_mag CityLearn-mined x N=3 @ step=8] $(date)"
"$ENV_DIR/bin/python" -u scripts/citylearn_eval_sweep.py \
  --base-url "http://127.0.0.1:${PORT}/v1" --model qwen3-8b \
  --scenarios $SCENARIO_IDS \
  --policies progress_mag --reps 3 --rep-start-seed 1000 \
  --max-steps 8 --max-proposals 3 --temperature 0.0 \
  --request-timeout-s 360 --max-retries 0 \
  --output "$OUT_JSON" --log-file "$RUN_LOG"
echo "[done] $(date)"
