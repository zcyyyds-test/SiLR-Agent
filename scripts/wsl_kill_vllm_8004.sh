#!/bin/bash
# Kill any vLLM serving port 8004 in WSL. No-op if none.
set +e
pids=$(ss -tlnp 2>/dev/null | awk '/:8004/{print $NF}' | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)
if [ -z "$pids" ]; then
    pids=$(ps -ef | grep -E 'vllm serve.*8004' | grep -v grep | awk '{print $2}')
fi
if [ -z "$pids" ]; then
    echo "no_vllm_8004"
    exit 0
fi
for p in $pids; do
    parent=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    echo "killing pid=$p ppid=$parent"
    kill -9 "$p" 2>/dev/null || true
done
sleep 2
echo "post_kill_pids:" $(ps -ef | grep -E 'vllm serve.*8004' | grep -v grep | awk '{print $2}')
