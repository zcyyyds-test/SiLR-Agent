#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N qwen8b_cl_mined_pmag_n3
#$ -o logs/
#$ -e logs/

# === CityLearn Step-0 go/no-go gate (pillar-2 multi-type amplification) ===
# Qwen3-8B over the mined CityLearn multi-action band, N=3, greedy (temp 0),
# THINKING OFF. Runs BOTH the OFF (ungated, no admission gate) and progress_mag
# (gated) policies. Question: is there a GRPO training signal?
#
# Why OFF, not just gated (first run, job 7863959, found gated saturated):
#   The action set is small (5^3=125 joint actions) and every mined snapshot has
#   >=14 recovering actions, so the GATED policy trivially recovers (gated ~1.0,
#   structurally saturated) -- gated is NOT a useful signal here. The pillar-2
#   dependent variable is UNGATED recovery (internalization): with no gate the
#   model gets no reject feedback, so base ungated can be far below 1.0 even when
#   gated saturates. base ungated recovery is the real headroom check.
#
# QUANTIFIED go/no-go (7-way panel 2026-06-04, panel-log/2026-06-04-1027):
#   Per scenario, base 8B UNGATED (OFF policy) recovery over N=3 reps.
#   - GO (train): base ungated has headroom -- mean ungated recovery <~0.7 on
#     >=6 of 24 scenarios (room for training to internalize the geometry).
#   - NO-GO (defer): base ungated already high everywhere -> no headroom ->
#     defer the CityLearn campaign, keep pillar-2 as the single-type ANM result.
#   - Stratify by family (ids encode the tag): report ungated recovery SEPARATELY
#     for soc_min x export_limit (cross-family, *_smin-exp) and soc_min x soc_max
#     (cross-device antichain, *_smax-smin / *_smin-smax). The cross-family subset
#     is the regime the geometric reward should help most.
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
PORT="${SILR_VLLM_PORT:-$((8100 + RANDOM % 800))}"  # randomize: gpu_1 is a shared node
SERVER_LOG="$LOG_DIR/step0_qwen8b_cl_mined_server_${JOB_TAG}.log"
RUN_LOG="$LOG_DIR/step0_qwen8b_cl_mined_pmag_n3_${JOB_TAG}.log"
OUT_JSON="$PROJECT/eval_step0_qwen3_8b_citylearn_mined_n3.json"
MINED_JSON="$PROJECT/domains/citylearn/scenarios_mined.json"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/step0_qwen8b_cl_mined_pmag_n3_${JOB_TAG}.inner.log") 2>&1
echo "[start] $(date)"
module purge; module load cuda/13.1.1
export HF_HOME="$WORK/hf"; export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1   # model is local; no network probe
export PYTHONPATH="$PROJECT"; export PATH="$ENV_DIR/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SILR_MAX_TOKENS=2048
export SILR_SYNC_ID="step0_qwen8b_cl_mined_pmag_n3_${JOB_TAG}"
cd "$PROJECT"

# Pull the mined scenario ids THROUGH the loader (not the raw JSON) so the list
# is exactly what the eval will accept -- if the loader skips a malformed record
# the id never reaches eval (fail-fast consistency).
[ -f "$MINED_JSON" ] || { echo "[error] $MINED_JSON missing -- run mine+select first"; exit 1; }
SCENARIO_IDS=$("$ENV_DIR/bin/python" -c \
  "from domains.citylearn.scenarios import SCENARIOS; print(' '.join(s.id for s in SCENARIOS if s.id.startswith('cl_mined_')))")
[ -n "$SCENARIO_IDS" ] || { echo "[error] no scenarios in $MINED_JSON"; exit 1; }
N_SCEN=$(echo "$SCENARIO_IDS" | wc -w)
echo "[scenarios] $N_SCEN mined CityLearn scenarios"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$ENV_DIR/bin/vllm" --version 2>&1 | sed 's/^/[vllm-version] /' || true
"$ENV_DIR/bin/vllm" serve "$MODEL_DIR" --host 127.0.0.1 --port "$PORT" \
  --served-model-name qwen3-8b --gpu-memory-utilization 0.85 \
  --max-model-len 16384 --enforce-eager \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --trust-remote-code >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT TERM INT
for i in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then echo "[ready] $(date)"; break; fi
  ! kill -0 "$SERVER_PID" 2>/dev/null && { echo "[serve-exited]"; tail -120 "$SERVER_LOG"; exit 1; }
  [ "$i" -eq 180 ] && { echo "[timeout]"; tail -160 "$SERVER_LOG"; exit 1; }
  sleep 5
done
# Inference + thinking-off smoke: one real completion with the SAME
# chat_template_kwargs the eval injects (top-level body field = what the openai
# SDK extra_body produces). Two guards before the 144-episode sweep:
#   (1) server up but cannot infer (inference-time OOM) -> fail in seconds;
#   (2) the chat_template_kwargs.enable_thinking=false path is silently ignored
#       (wrong vLLM/template) -> a <think> block appears -> abort, do NOT repeat
#       the first run's 320s/episode thinking-on stall over 144 episodes.
echo "[inference + thinking-off smoke] $(date)"
SMOKE_RESP=$(curl -fsS "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-8b","temperature":0,"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"A district has battery SoC and feeder export violations. Reason step by step about which set-points to change."}]}') \
  || { echo "[inference-smoke-failed] server up but inference broken"; tail -80 "$SERVER_LOG"; exit 1; }
echo "[smoke resp head] $(printf '%s' "$SMOKE_RESP" | head -c 280)"
printf '%s' "$SMOKE_RESP" | grep -q '<think>' \
  && { echo "[thinking-off-FAILED] <think> present -> enable_thinking not honored by server; aborting before 144-ep run"; exit 1; } \
  || echo "[thinking-off OK] no <think> block"
echo "[eval OFF+progress_mag CityLearn-mined x N=3 @ step=8 = 144 ep] $(date)"
"$ENV_DIR/bin/python" -u scripts/citylearn_eval_sweep.py \
  --base-url "http://127.0.0.1:${PORT}/v1" --model qwen3-8b \
  --scenarios $SCENARIO_IDS \
  --policies OFF progress_mag --reps 3 --rep-start-seed 1000 \
  --max-steps 8 --max-proposals 3 --temperature 0.0 \
  --request-timeout-s 360 --max-retries 0 \
  --output "$OUT_JSON" --log-file "$RUN_LOG"
echo "[done] $(date)"
