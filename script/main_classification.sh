#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --array=0
#SBATCH -J cls
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
##SBATCH --gres=gpu:v100:1
##SBATCH --mem=30G

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

detect_python_bin() {
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        echo "$PYTHON_BIN"
        return 0
    fi

    if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
        echo "$REPO_ROOT/.venv/bin/python"
        return 0
    fi

    if command -v python >/dev/null 2>&1; then
        command -v python
        return 0
    fi

    echo "python"
}

maybe_load_modules() {
    if [[ "${USE_ENV_MODULES:-0}" != "1" ]]; then
        return 0
    fi

    if command -v module >/dev/null 2>&1; then
        module load cuda/11.1.1
        module load gcc
        echo "Loaded environment modules (cuda/11.1.1, gcc)"
    else
        echo "USE_ENV_MODULES=1 was requested, but \`module\` is unavailable; continuing with current environment"
    fi
}

maybe_load_modules

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

nvidia-smi
nvcc --version

hostname
NUM_GPU_AVAILABLE=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
echo $NUM_GPU_AVAILABLE


PYTHON_BIN=$(detect_python_bin)
echo "Using Python: $PYTHON_BIN"

cfg=$1
PY_ARGS=${@:2}
"$PYTHON_BIN" examples/classification/main.py --cfg $cfg ${PY_ARGS}

# how to run
# this script supports training using 1 GPU or multi-gpu,
# simply run jobs on multiple GPUs will launch distributed training by default.
# load different cfgs for training on different benchmarks (modelnet40 classification, or scanobjectnn classification) and using different models.

# For examples,
# if using a cluster with slurm, train PointNeXt-S on scanobjectnn classification using only 1 GPU, by 3 times:
# sbatch --array=0-2 --gres=gpu:1 --time=10:00:00 main_classification.sh cfgs/scaobjetnn/pointnext-s.yaml

# if using local machine with GPUs, train PointNeXt-S on scanobjectnn classification using all GPUs
# bash script/main_classification.sh cfgs/scaobjetnn/pointnext-s.yaml

# if using local machine with GPUs, train PointNeXt-S on scanobjectnn classification using only 1 GPU
# CUDA_VISIBLE_DEVICES=0 bash script/main_classification.sh cfgs/scaobjetnn/pointnext-s.yaml
