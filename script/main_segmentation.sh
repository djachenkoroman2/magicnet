#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --array=0
#SBATCH -J seg
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=10:00:00
##SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=40G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
CALLER_PWD=$PWD
cd "$REPO_ROOT"

resolve_cfg_path() {
    local input_path=$1
    local candidate
    local compat_candidate
    local task_dir
    local cfg_name
    local nested_candidate

    if [[ "$input_path" == /* ]]; then
        candidate=$input_path
    else
        candidate=$CALLER_PWD/$input_path
    fi
    if [[ -f "$candidate" ]]; then
        readlink -f "$candidate"
        return 0
    fi

    if [[ "$input_path" != /* && -f "$REPO_ROOT/$input_path" ]]; then
        readlink -f "$REPO_ROOT/$input_path"
        return 0
    fi

    if [[ "$input_path" == cfgs/* ]]; then
        task_dir=${input_path#cfgs/}
        task_dir=${task_dir%%/*}
        cfg_name=$(basename "$input_path")
        if [[ -d "$REPO_ROOT/cfgs/$task_dir" ]]; then
            nested_candidate=$(find "$REPO_ROOT/cfgs/$task_dir" -type f -name "$cfg_name" | head -n 1)
            if [[ -n "$nested_candidate" ]]; then
                echo "Resolved nested config fallback: $input_path -> ${nested_candidate#$REPO_ROOT/}" >&2
                readlink -f "$nested_candidate"
                return 0
            fi
        fi
    fi

    # Compatibility fallback for stale paths like cfgs/k3d_cfgs/k3d_xyz/... .
    if [[ "$input_path" == cfgs/*/* ]]; then
        compat_candidate="$REPO_ROOT/cfgs/${input_path#cfgs/*/}"
        if [[ -f "$compat_candidate" ]]; then
            echo "Resolved config path fallback: $input_path -> ${compat_candidate#$REPO_ROOT/}" >&2
            readlink -f "$compat_candidate"
            return 0
        fi
    fi

    echo "Config not found: $input_path" >&2
    return 1
}

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

if [[ $# -lt 1 ]]; then
    echo "Usage: bash script/main_segmentation.sh <config_path> [extra args...]" >&2
    exit 1
fi

[ ! -d "slurm_logs" ] && echo "Create a directory slurm_logs" && mkdir -p slurm_logs

maybe_load_modules

PYTHON_BIN=$(detect_python_bin)
echo "Using Python: $PYTHON_BIN"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || echo "nvidia-smi failed, continuing"
else
    echo "nvidia-smi command not found, skipping GPU summary"
fi

if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
else
    echo "nvcc command not found, skipping nvcc --version"
fi

hostname
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_QUERY=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)
    if [[ -n "$GPU_QUERY" ]]; then
        NUM_GPU_AVAILABLE=$(printf '%s\n' "$GPU_QUERY" | wc -l)
    else
        NUM_GPU_AVAILABLE=0
    fi
else
    NUM_GPU_AVAILABLE=0
fi
echo $NUM_GPU_AVAILABLE


cfg=$(resolve_cfg_path "$1")
PY_ARGS=${@:2}
"$PYTHON_BIN" examples/segmentation/main.py --cfg "$cfg" ${PY_ARGS}


# how to run
# using slurm, run with 1 GPU, by 3 times (array=0-2):
# sbatch --array=0-2 --gres=gpu:1 --time=12:00:00 script/main_segmentation.sh cfgs/s3dis/pointnext-s.yaml

# if using local machine with GPUs, run with ALL GPUs:
# bash script/main_segmentation.sh cfgs/s3dis/pointnext-s.yaml

# local machine, run with 1 GPU:
# CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointnext-s.yaml
