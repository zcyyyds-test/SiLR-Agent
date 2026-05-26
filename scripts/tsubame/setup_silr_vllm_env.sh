#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N setup_silr_vllm
#$ -o logs/
#$ -e logs/

set -euo pipefail

WORK="${HOME/home/work}"
PROJECT="$WORK/SILR-WISE26"
ENV_DIR="$WORK/envs/silr-vllm"
LOG_DIR="$PROJECT/logs"
CONDA="/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda"
PYTHON="$CONDA run -p $ENV_DIR python"

mkdir -p "$PROJECT" "$LOG_DIR" "$WORK/envs"
exec > >(tee -a "$LOG_DIR/setup_silr_vllm_env_inner.log") 2>&1

echo "[start] $(date)"
echo "[host] $(hostname)"
echo "[work] $WORK"
echo "[project] $PROJECT"
echo "[env] $ENV_DIR"

module purge
module load cuda/13.1.1

if [ ! -x "$ENV_DIR/bin/python" ]; then
  "$CONDA" create -y -p "$ENV_DIR" python=3.12 pip
fi

$PYTHON -m pip install -U pip wheel setuptools

# Match the working AMD vLLM stack closely enough for Qwen3-14B serving while
# keeping ANM runtime dependencies in the same environment.
$PYTHON -m pip install \
  "vllm==0.21.0" \
  "openai==2.37.0" \
  "gym-anm==2.0.1" \
  "cvxpy" \
  "matplotlib"

if [ -f "$PROJECT/pyproject.toml" ]; then
  $PYTHON -m pip install -e "$PROJECT[agent]"
fi

$PYTHON - <<'PY'
import sys
print("python", sys.version)
mods = ["torch", "vllm", "transformers", "openai", "gym_anm", "cvxpy", "numpy"]
for name in mods:
    try:
        mod = __import__(name)
        print(name, getattr(mod, "__version__", "unknown"))
    except Exception as exc:
        print(name, "IMPORT_ERROR", repr(exc))
        raise

import torch
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
PY

echo "[done] $(date)"
