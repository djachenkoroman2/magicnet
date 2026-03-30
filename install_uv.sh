#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Run this script with: bash install_uv.sh"
  return 1 2>/dev/null || exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

VENV_DIR="${VENV_DIR:-.venv}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
PIP_TIMEOUT="${PIP_TIMEOUT:-300}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-6.1;6.2;7.0;7.5;8.0}"
INSTALL_DEEPSPEED="${INSTALL_DEEPSPEED:-0}"
USE_SYSTEM_TORCH="${USE_SYSTEM_TORCH:-0}"
VENV_PYTHON="${VENV_PYTHON:-python3}"
VENV_CLEAR="${VENV_CLEAR:-auto}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.5.1}"
INSTALL_TORCHAUDIO="${INSTALL_TORCHAUDIO:-0}"
PYTORCH_CUDA_FLAVOR="${PYTORCH_CUDA_FLAVOR:-cu118}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/${PYTORCH_CUDA_FLAVOR}}"
TORCH_SCATTER_INDEX_URL="${TORCH_SCATTER_INDEX_URL:-https://data.pyg.org/whl/torch-${TORCH_VERSION}+${PYTORCH_CUDA_FLAVOR}.html}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found in PATH." >&2
  exit 1
fi

if [[ ! -f "pyproject.toml" ]]; then
  echo "pyproject.toml is required but was not found in the repository root." >&2
  exit 1
fi

if [[ ! -f "openpoints/cpp/pointnet2_batch/setup.py" ]]; then
  echo "openpoints sources are missing. Populate the submodule before running install_uv.sh." >&2
  exit 1
fi

mkdir -p "${UV_CACHE_DIR}"
desired_site_packages="false"
if [[ "${USE_SYSTEM_TORCH}" == "1" ]]; then
  desired_site_packages="true"
  echo "USE_SYSTEM_TORCH=1 keeps access to system packages. The default is isolated .venv installs." >&2
fi

venv_args=(--python "${VENV_PYTHON}" --seed)
if [[ "${desired_site_packages}" == "true" ]]; then
  venv_args+=(--system-site-packages)
fi

if [[ -d "${VENV_DIR}" ]]; then
  existing_site_packages=""
  if [[ -f "${VENV_DIR}/pyvenv.cfg" ]]; then
    existing_site_packages="$(awk -F' = ' '/^include-system-site-packages = / {print $2}' "${VENV_DIR}/pyvenv.cfg" | tr -d '\r')"
  fi

  if [[ "${VENV_CLEAR}" == "1" ]]; then
    venv_args+=(--clear)
  elif [[ "${VENV_CLEAR}" == "auto" && "${existing_site_packages}" != "${desired_site_packages}" ]]; then
    echo "Recreating ${VENV_DIR} to match include-system-site-packages=${desired_site_packages}." >&2
    venv_args+=(--clear)
  else
    venv_args+=(--allow-existing)
  fi
fi

UV_CACHE_DIR="${UV_CACHE_DIR}" uv venv "${venv_args[@]}" "${VENV_DIR}"

PYTHON_BIN="${ROOT_DIR}/${VENV_DIR}/bin/python"
VENV_ABS_DIR="${ROOT_DIR}/${VENV_DIR}"
export PATH="${ROOT_DIR}/${VENV_DIR}/bin:${PATH}"
"${PYTHON_BIN}" -m ensurepip --upgrade >/dev/null

pip_install() {
  "${PYTHON_BIN}" -m pip install --timeout "${PIP_TIMEOUT}" --prefer-binary "$@"
}

assert_local_module() {
  local module="$1"
  local resolved

  resolved="$("${PYTHON_BIN}" -c "import importlib.util, pathlib; spec = importlib.util.find_spec('${module}'); assert spec and spec.origin, '${module} is not installed'; print(pathlib.Path(spec.origin).resolve())")"
  case "${resolved}" in
    "${VENV_ABS_DIR}"/*)
      ;;
    *)
      echo "${module} resolved outside ${VENV_DIR}: ${resolved}" >&2
      return 1
      ;;
  esac
}

parse_toml_array() {
  local key="$1"
  local section="${2:-}"

  awk -v key="${key}" -v section="${section}" '
    function emit_strings(s, tmp) {
      tmp = s
      while (match(tmp, /"[^"]+"/)) {
        print substr(tmp, RSTART + 1, RLENGTH - 2)
        tmp = substr(tmp, RSTART + RLENGTH)
      }
    }

    /^\[/ {
      if (section != "" && $0 == "[" section "]") {
        in_section = 1
        next
      }
      if (section != "" && in_section) {
        exit
      }
    }

    {
      if (section != "" && !in_section) {
        next
      }

      if (!capturing && $0 ~ "^" key "[[:space:]]*=[[:space:]]*\\[") {
        capturing = 1
        line = $0
        sub("^" key "[[:space:]]*=[[:space:]]*\\[", "", line)
        emit_strings(line)
        if (line ~ /\]/) {
          exit
        }
        next
      }

      if (capturing) {
        if ($0 ~ /^[[:space:]]*\]/) {
          exit
        }
        emit_strings($0)
      }
    }
  ' pyproject.toml
}

install_pyproject_group() {
  local key="$1"
  local section="${2:-}"
  local deps=()
  local failed=()

  mapfile -t deps < <(parse_toml_array "${key}" "${section}")
  if [[ "${#deps[@]}" -gt 0 ]]; then
    if ! pip_install "${deps[@]}"; then
      echo "Bulk install for ${key} failed; retrying packages one by one for clearer diagnostics." >&2
      for dep in "${deps[@]}"; do
        if ! pip_install "${dep}"; then
          failed+=("${dep}")
        fi
      done
      if [[ "${#failed[@]}" -gt 0 ]]; then
        echo "Failed to install packages: ${failed[*]}" >&2
        return 1
      fi
    fi
  fi
}

run_with_cuda() {
  env \
    CUDA_HOME="${CUDA_HOME}" \
    PATH="${CUDA_HOME}/bin:${PATH}" \
    LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    "$@"
}

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "nvcc was not found under CUDA_HOME=${CUDA_HOME}." >&2
  exit 1
fi

if [[ "${USE_SYSTEM_TORCH}" == "1" ]]; then
  pip_install torch-scatter -f "${TORCH_SCATTER_INDEX_URL}"
else
  torch_packages=("torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}")
  if [[ "${INSTALL_TORCHAUDIO}" == "1" ]]; then
    torch_packages+=("torchaudio==${TORCHAUDIO_VERSION}")
  fi

  pip_install --index-url "${PYTORCH_INDEX_URL}" "${torch_packages[@]}"
  pip_install torch-scatter -f "${TORCH_SCATTER_INDEX_URL}"
  assert_local_module torch
  assert_local_module torchvision
  assert_local_module torch_scatter
  if [[ "${INSTALL_TORCHAUDIO}" == "1" ]]; then
    assert_local_module torchaudio
  fi
fi

install_pyproject_group "dependencies"

if [[ "${INSTALL_DEEPSPEED}" == "1" ]]; then
  install_pyproject_group "distributed" "project.optional-dependencies"
fi

pushd openpoints/cpp/pointnet2_batch >/dev/null
run_with_cuda "${PYTHON_BIN}" setup.py install
popd >/dev/null

pushd openpoints/cpp/subsampling >/dev/null
"${PYTHON_BIN}" setup.py build_ext --inplace
popd >/dev/null

pushd openpoints/cpp/pointops >/dev/null
run_with_cuda "${PYTHON_BIN}" setup.py install
popd >/dev/null

pushd openpoints/cpp/chamfer_dist >/dev/null
run_with_cuda "${PYTHON_BIN}" setup.py install
popd >/dev/null

pushd openpoints/cpp/emd >/dev/null
run_with_cuda "${PYTHON_BIN}" setup.py install
popd >/dev/null

echo
echo "Environment ready:"
echo "  source ${VENV_DIR}/bin/activate"
echo "Optional packages:"
echo "  INSTALL_DEEPSPEED=1 bash install_uv.sh"
echo "Modes:"
echo "  USE_SYSTEM_TORCH=0 bash install_uv.sh   # default, installs torch/cu118 into ${VENV_DIR}"
echo "  USE_SYSTEM_TORCH=1 bash install_uv.sh   # legacy, keeps access to system torch"
