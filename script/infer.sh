#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash script/infer.sh <checkpoint_path> <config_path> <point_cloud_path> [output_dir]

Examples:
  bash script/infer.sh \
    log/k3d_xyz/<run_name>/checkpoint/<run_name>_ckpt_best.pth \
    cfgs/k3d_xyz/<model>.yaml \
    data/k3d_xyz/raw/scene.txt

  bash script/infer.sh \
    /abs/path/to/model.pth \
    /abs/path/to/config.yaml \
    /abs/path/to/raw_scene.txt \
    /abs/path/to/output_dir

Environment variables:
  PYTHON_BIN  Python executable to use. Default: python
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -lt 3 || $# -gt 4 ]]; then
    usage >&2
    exit 1
fi

CKPT_INPUT=$1
CFG_INPUT=$2
POINT_INPUT=$3
OUTPUT_INPUT=${4:-results}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
CALLER_PWD=$PWD
cd "$REPO_ROOT"

resolve_existing_file() {
    local label=$1
    local input_path=$2
    local candidate

    if [[ "$input_path" == /* ]]; then
        candidate=$input_path
    else
        candidate=$CALLER_PWD/$input_path
    fi

    if [[ ! -f "$candidate" ]]; then
        echo "$label not found: $input_path" >&2
        return 1
    fi

    readlink -f "$candidate"
}

resolve_directory() {
    local input_path=$1
    local candidate

    if [[ "$input_path" == /* ]]; then
        candidate=$input_path
    else
        candidate=$CALLER_PWD/$input_path
    fi

    mkdir -p "$candidate"
    readlink -f "$candidate"
}

CKPT_PATH=$(resolve_existing_file "Checkpoint" "$CKPT_INPUT")
CFG_PATH=$(resolve_existing_file "Config" "$CFG_INPUT")
POINT_CLOUD_PATH=$(resolve_existing_file "Point cloud" "$POINT_INPUT")
OUTPUT_DIR=$(resolve_directory "$OUTPUT_INPUT")

PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

POINT_STEM=$(basename "${POINT_CLOUD_PATH%.*}")
EXPECTED_OUTPUT="$OUTPUT_DIR/${POINT_STEM}_pred.txt"

echo "Using config: $CFG_PATH"
echo "Using checkpoint: $CKPT_PATH"
echo "Input point cloud: $POINT_CLOUD_PATH"
echo "Saving predictions to: $OUTPUT_DIR"

"$PYTHON_BIN" examples/segmentation/main.py \
    --cfg "$CFG_PATH" \
    mode=test \
    pretrained_path="$CKPT_PATH" \
    dataset.test.data_path="$POINT_CLOUD_PATH" \
    save_pred=True \
    save_path="$OUTPUT_DIR" \
    wandb.use_wandb=False

echo "Expected prediction file: $EXPECTED_OUTPUT"
