#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/S3DIS/
cd data/S3DIS

if [[ ! -f s3disfull.tar ]]; then
    if command -v gdown >/dev/null 2>&1; then
        gdown https://drive.google.com/uc?id=1MX3ZCnwqyRztG1vFRiHkKTz68ZJeHS4Y -O s3disfull.tar
    elif [[ -x ../../.venv/bin/python ]]; then
        ../../.venv/bin/python -m gdown https://drive.google.com/uc?id=1MX3ZCnwqyRztG1vFRiHkKTz68ZJeHS4Y -O s3disfull.tar
    elif python -c "import gdown" >/dev/null 2>&1; then
        python -m gdown https://drive.google.com/uc?id=1MX3ZCnwqyRztG1vFRiHkKTz68ZJeHS4Y -O s3disfull.tar
    else
        echo "gdown is not available. Activate .venv or install gdown, then rerun this script." >&2
        exit 1
    fi
fi

tar -xvf s3disfull.tar
