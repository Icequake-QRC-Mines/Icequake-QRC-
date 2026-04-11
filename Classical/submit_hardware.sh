#!/bin/bash
#SBATCH --job-name=hardware-qrc
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/hardware_qrc_%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"

if [[ -z "${CODE_DIR:-}" ]]; then
  if [[ -d "${REPO_ROOT}/Code" ]]; then
    CODE_DIR="${REPO_ROOT}/Code"
  elif [[ -d "${REPO_ROOT}/Classical" ]]; then
    CODE_DIR="${REPO_ROOT}/Classical"
  else
    echo "ERROR: Could not find Code/ or Classical/ under ${REPO_ROOT}"
    echo "Set CODE_DIR explicitly."
    exit 1
  fi
fi

CONFIG_PATH="${CONFIG_PATH:-${CODE_DIR}/results/noisy_qrc_run/hardware_config.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-${CODE_DIR}/results/hardware_qrc_run}"
BACKEND="${BACKEND:-ibm_sherbrooke}"
SHOTS="${SHOTS:-4096}"
MODULES_TO_LOAD="${MODULES_TO_LOAD:-uv}"
UV_ENV_PATH="${UV_ENV_PATH:-${UV_ENVS:-/projects/${USER}/software/uv/envs}/qrc}"
UV_REQUIREMENTS_FILE="${UV_REQUIREMENTS_FILE:-${CODE_DIR}/requirements-curc.txt}"
UV_BOOTSTRAP="${UV_BOOTSTRAP:-0}"

mkdir -p logs

# CURC module-first environment setup for non-interactive SLURM jobs.
# Override with e.g.: MODULES_TO_LOAD="uv cuda/12.2"
if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  for mod in ${MODULES_TO_LOAD}; do
    module load "${mod}"
  done
fi

# CURC uv workflow.
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not available after module load."
  echo "Set MODULES_TO_LOAD to include the CURC uv module."
  exit 1
fi
if [[ ! -d "${UV_ENV_PATH}" ]]; then
  if [[ "${UV_BOOTSTRAP}" == "1" ]]; then
    echo "Creating uv environment at ${UV_ENV_PATH}"
    uv venv "${UV_ENV_PATH}"
    if [[ -n "${UV_REQUIREMENTS_FILE}" && -f "${UV_REQUIREMENTS_FILE}" ]]; then
      echo "Installing dependencies from ${UV_REQUIREMENTS_FILE}"
      uv pip install --python "${UV_ENV_PATH}/bin/python" -r "${UV_REQUIREMENTS_FILE}"
    else
      echo "ERROR: UV_REQUIREMENTS_FILE not found: ${UV_REQUIREMENTS_FILE}"
      exit 1
    fi
  else
    echo "ERROR: uv environment not found at ${UV_ENV_PATH}"
    echo "Create it first (or set UV_BOOTSTRAP=1)."
    exit 1
  fi
fi
if [[ ! -x "${UV_ENV_PATH}/bin/python" ]]; then
  echo "ERROR: ${UV_ENV_PATH}/bin/python not executable."
  exit 1
fi

PYTHON_BIN="${UV_ENV_PATH}/bin/python"
echo "Using python: ${PYTHON_BIN}"
"${PYTHON_BIN}" -V

"${PYTHON_BIN}" "${CODE_DIR}/hardware_qrc.py" \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --backend "${BACKEND}" \
  --shots "${SHOTS}" \
  --no-confirm
