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
UV_PYTHON_STAGE_DIR="${UV_PYTHON_STAGE_DIR:-${UV_CACHE_DIR}/python-standalone}"
UV_PYTHON_VERSION="${UV_PYTHON_VERSION:-3.12}"
PIP_TIMEOUT="${PIP_TIMEOUT:-300}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-6.1;6.2;7.0;7.5;8.0}"
INSTALL_DEEPSPEED="${INSTALL_DEEPSPEED:-0}"
USE_SYSTEM_TORCH="${USE_SYSTEM_TORCH:-0}"
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

mkdir -p "${UV_CACHE_DIR}" "${UV_PYTHON_STAGE_DIR}"
desired_site_packages="false"
if [[ "${USE_SYSTEM_TORCH}" == "1" ]]; then
  desired_site_packages="true"
  echo "USE_SYSTEM_TORCH=1 is a legacy mode and adds external site-packages via .pth. The default remains a fully self-contained ${VENV_DIR}." >&2
fi

has_symlinks() {
  local root="$1"
  find "${root}" -type l -print -quit | grep -q .
}

copy_tree_with_copies() {
  local src="$1"
  local dst="$2"
  local tmp_dir="${dst}.nosymlink.$$"

  rm -rf "${tmp_dir}"
  mkdir -p "${tmp_dir}"
  cp -aL "${src}/." "${tmp_dir}/"
  rm -rf "${dst}"
  mv "${tmp_dir}" "${dst}"
}

flatten_tree_with_copies() {
  local tree="$1"
  copy_tree_with_copies "${tree}" "${tree}"
}

replace_symlink_with_copy() {
  local link_path="$1"
  local tmp_path="${link_path}.nosymlink.$$"

  rm -rf "${tmp_path}"
  if [[ -d "${link_path}" ]]; then
    mkdir -p "${tmp_path}"
    cp -aL "${link_path}/." "${tmp_path}/"
  else
    cp -aL "${link_path}" "${tmp_path}"
  fi
  rm -rf "${link_path}"
  mv "${tmp_path}" "${link_path}"
}

replace_runtime_symlinks() {
  local root="$1"
  local link_path

  while IFS= read -r -d '' link_path; do
    replace_symlink_with_copy "${link_path}"
  done < <(find "${root}" -depth -type l -print0)
}

ensure_managed_python_root() {
  local version="$1"
  local stage_dir="$2"
  local python_name="python${version}"
  local python_bin
  local python_root

  mkdir -p "${stage_dir}"
  UV_CACHE_DIR="${UV_CACHE_DIR}" uv python install --managed-python --install-dir "${stage_dir}" "${version}" >&2

  python_bin="$(find "${stage_dir}" -maxdepth 3 -type f -path "*/bin/${python_name}" | sort -V | tail -n 1)"
  if [[ -z "${python_bin}" ]]; then
    echo "Failed to find ${python_name} under ${stage_dir}." >&2
    return 1
  fi

  python_root="$(cd "$(dirname "${python_bin}")/.." && pwd)"
  if has_symlinks "${python_root}"; then
    echo "Flattening symlinks under ${python_root}." >&2
    flatten_tree_with_copies "${python_root}"
  fi

  printf '%s\n' "${python_root}"
}

install_activation_scripts() {
  local target_dir="$1"
  local bootstrap_python="$2"
  local temp_venv

  temp_venv="$(mktemp -d "${UV_CACHE_DIR}/activate.XXXXXX")"
  "${bootstrap_python}" -m venv --copies "${temp_venv}" >/dev/null
  cp "${temp_venv}/bin/activate" "${target_dir}/bin/"
  cp "${temp_venv}/bin/activate.csh" "${target_dir}/bin/"
  cp "${temp_venv}/bin/activate.fish" "${target_dir}/bin/"
  cp "${temp_venv}/bin/Activate.ps1" "${target_dir}/bin/"
  sed -i "s|${temp_venv}|${target_dir}|g" \
    "${target_dir}/bin/activate" \
    "${target_dir}/bin/activate.csh" \
    "${target_dir}/bin/activate.fish"
  rm -rf "${temp_venv}"
}

write_external_site_packages_pth() {
  local venv_dir="$1"
  local python_mm="$2"
  local pth_path="${venv_dir}/lib/python${python_mm}/site-packages/_magicnet_external_site_packages.pth"
  local site_paths=()

  mapfile -t site_paths < <(
    python3 - <<'PY'
import pathlib
import site
import sysconfig

paths = []
for candidate in [*site.getsitepackages(), site.getusersitepackages(), sysconfig.get_path("purelib"), sysconfig.get_path("platlib")]:
    if candidate:
        path = pathlib.Path(candidate).resolve()
        if path.exists():
            paths.append(str(path))

seen = set()
for path in paths:
    if path not in seen:
        print(path)
        seen.add(path)
PY
  )

  mkdir -p "$(dirname "${pth_path}")"
  : > "${pth_path}"
  if [[ "${#site_paths[@]}" -gt 0 ]]; then
    printf '%s\n' "${site_paths[@]}" > "${pth_path}"
  fi
}

remove_external_site_packages_pth() {
  local venv_dir="$1"
  local python_mm="$2"
  rm -f "${venv_dir}/lib/python${python_mm}/site-packages/_magicnet_external_site_packages.pth"
}

env_is_self_contained() {
  local env_dir="$1"
  local expected_mm="$2"

  if [[ ! -x "${env_dir}/bin/python" ]]; then
    return 1
  fi

  ENV_DIR_CHECK="$(cd "${env_dir}" && pwd)" EXPECTED_MM="${expected_mm}" "${env_dir}/bin/python" - <<'PY' >/dev/null
import os
import pathlib
import sys
import sysconfig

env_dir = pathlib.Path(os.environ["ENV_DIR_CHECK"]).resolve()
expected_mm = os.environ["EXPECTED_MM"]

if f"{sys.version_info[0]}.{sys.version_info[1]}" != expected_mm:
    raise SystemExit(1)
if pathlib.Path(sys.prefix).resolve() != env_dir:
    raise SystemExit(1)
if pathlib.Path(sys.base_prefix).resolve() != env_dir:
    raise SystemExit(1)

for key in ("stdlib", "purelib", "platlib", "scripts"):
    path = pathlib.Path(sysconfig.get_path(key)).resolve()
    if path != env_dir and not str(path).startswith(f"{env_dir}/"):
        raise SystemExit(1)
PY
}

assemble_self_contained_env() {
  local python_root="$1"
  local python_bin="$2"
  local target_dir="$3"
  local target_abs_dir

  target_abs_dir="$(cd "$(dirname "${target_dir}")" && pwd)/$(basename "${target_dir}")"
  copy_tree_with_copies "${python_root}" "${target_dir}"
  find "${target_dir}" -name 'EXTERNALLY-MANAGED' -delete
  replace_runtime_symlinks "${target_dir}"
  install_activation_scripts "${target_abs_dir}" "${python_bin}"
}

BASE_PYTHON_ROOT="$(ensure_managed_python_root "${UV_PYTHON_VERSION}" "${UV_PYTHON_STAGE_DIR}")"
BASE_PYTHON_BIN="${BASE_PYTHON_ROOT}/bin/python${UV_PYTHON_VERSION}"
PYTHON_MM="$("${BASE_PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"

recreate_venv="0"
if [[ -d "${VENV_DIR}" ]]; then
  if [[ "${VENV_CLEAR}" == "1" ]]; then
    recreate_venv="1"
  elif [[ "${VENV_CLEAR}" == "auto" ]]; then
    if has_symlinks "${VENV_DIR}"; then
      echo "Recreating ${VENV_DIR} to remove symlinks." >&2
      recreate_venv="1"
    elif ! env_is_self_contained "${VENV_DIR}" "${PYTHON_MM}"; then
      echo "Recreating ${VENV_DIR} as a fully self-contained Python ${PYTHON_MM} environment." >&2
      recreate_venv="1"
    fi
  fi
fi

if [[ ! -d "${VENV_DIR}" || "${recreate_venv}" == "1" ]]; then
  assemble_self_contained_env "${BASE_PYTHON_ROOT}" "${BASE_PYTHON_BIN}" "${VENV_DIR}"
fi

PYTHON_BIN="${ROOT_DIR}/${VENV_DIR}/bin/python"
VENV_ABS_DIR="${ROOT_DIR}/${VENV_DIR}"
if [[ "${desired_site_packages}" == "true" ]]; then
  write_external_site_packages_pth "${VENV_ABS_DIR}" "${PYTHON_MM}"
else
  remove_external_site_packages_pth "${VENV_ABS_DIR}" "${PYTHON_MM}"
fi
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

replace_runtime_symlinks "${VENV_DIR}"

if has_symlinks "${BASE_PYTHON_ROOT}"; then
  echo "Symlinks remain under ${BASE_PYTHON_ROOT}." >&2
  exit 1
fi

if has_symlinks "${VENV_DIR}"; then
  echo "Symlinks remain under ${VENV_DIR}." >&2
  exit 1
fi

if ! env_is_self_contained "${VENV_DIR}" "${PYTHON_MM}"; then
  echo "${VENV_DIR} is not fully self-contained after installation." >&2
  exit 1
fi

echo
echo "Environment ready:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  Python: ${PYTHON_BIN} ($(\"${PYTHON_BIN}\" --version 2>&1))"
echo "Optional packages:"
echo "  INSTALL_DEEPSPEED=1 bash install_uv.sh"
echo "Modes:"
echo "  USE_SYSTEM_TORCH=0 bash install_uv.sh   # default, installs torch/cu118 into ${VENV_DIR}"
echo "  USE_SYSTEM_TORCH=1 bash install_uv.sh   # legacy, keeps access to system torch"
