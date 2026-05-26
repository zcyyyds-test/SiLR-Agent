#!/bin/bash
# Launch Qwen3-14B vLLM on AMD WSL GPU 0 and record a PID/log for monitoring.
set -euo pipefail

cd /mnt/d/zcy/SILR-WISE26
mkdir -p logs

pid_file="logs/qwen3_14b_gpu0_8003.pid"
log_file="logs/qwen3_14b_gpu0_8003.log"

if [[ -s "$pid_file" ]]; then
    old_pid="$(cat "$pid_file" || true)"
    if [[ -n "$old_pid" ]] && ps -p "$old_pid" >/dev/null 2>&1; then
        echo "already-running:$old_pid"
        exit 0
    fi
fi

nohup bash scripts/serve_qwen3_14b_gpu0_8003.sh >"$log_file" 2>&1 </dev/null &
pid="$!"
echo "$pid" >"$pid_file"
echo "started:$pid"
