#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N dl_qwen3_14b
#$ -o logs/
#$ -e logs/

set -euo pipefail

WORK="${HOME/home/work}"
PROJECT="$WORK/SILR-WISE26"
ENV_DIR="$WORK/envs/hf"
HF_HOME_DIR="$WORK/hf"
MODEL_DIR="$WORK/models/Qwen3-14B"
LOG_DIR="$PROJECT/logs"

mkdir -p "$PROJECT" "$ENV_DIR" "$HF_HOME_DIR" "$MODEL_DIR" "$LOG_DIR"

exec > >(tee -a "$LOG_DIR/download_qwen3_14b_inner.log") 2>&1

echo "[start] $(date)"
echo "[host] $(hostname)"
echo "[cwd] $(pwd)"
echo "[work] $WORK"
echo "[model_dir] $MODEL_DIR"

export HF_HOME="$HF_HOME_DIR"
export HF_HUB_CACHE="$HF_HOME_DIR/hub"
export TRANSFORMERS_CACHE="$HF_HOME_DIR/transformers"
export HF_HUB_DISABLE_TELEMETRY=1
export MODEL_DIR

BASE_PY="/apps/t4/rhel9/free/miniconda/24.1.2/bin/python"
if [ ! -x "$ENV_DIR/bin/python" ]; then
  "$BASE_PY" -m venv "$ENV_DIR"
fi

source "$ENV_DIR/bin/activate"
python -m pip install -U pip huggingface_hub

python - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
import os

repo_id = "Qwen/Qwen3-14B"
target = Path(os.environ["MODEL_DIR"])
target.mkdir(parents=True, exist_ok=True)
print(f"[download] repo={repo_id} target={target}", flush=True)
path = snapshot_download(
    repo_id=repo_id,
    local_dir=str(target),
    local_dir_use_symlinks=False,
    resume_download=True,
)
print(f"[downloaded] {path}", flush=True)
PY

echo "[size]"
du -sh "$MODEL_DIR" "$HF_HOME_DIR" || true
find "$MODEL_DIR" -maxdepth 1 -type f | wc -l | awk '{print "files=" $1}'
echo "[done] $(date)"
