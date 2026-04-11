#!/bin/bash
#SBATCH --job-name=noisy-qrc
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/noisy_qrc_%A_%a.out
#SBATCH --partition=amilan
#SBATCH --qos=normal

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

OUTPUT_DIR="${OUTPUT_DIR:-${CODE_DIR}/results/noisy_qrc_run}"
SHOTS="${SHOTS:-1024}"
N_ITERATIONS="${N_ITERATIONS:-5}"
TOP_K="${TOP_K:-3}"
OPTUNA_TRIALS="${OPTUNA_TRIALS:-30}"
SEED="${SEED:-42}"
SUBSET_FRAC="${SUBSET_FRAC:-1.0}"
DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_MEMORY_MB="${MAX_MEMORY_MB:-24576}"
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
    echo "Create it first (or set UV_BOOTSTRAP=1). Example:"
    echo "  module load uv"
    echo "  uv venv \"${UV_ENV_PATH}\""
    echo "  source \"${UV_ENV_PATH}/bin/activate\""
    echo "  uv pip install qiskit qiskit-aer qiskit-ibm-runtime optuna xgboost pandas numpy scikit-learn"
    exit 1
  fi
fi
if [[ ! -x "${UV_ENV_PATH}/bin/python" ]]; then
  echo "ERROR: ${UV_ENV_PATH}/bin/python not executable."
  exit 1
fi

# Keep thread usage aligned with SLURM allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

PYTHON_BIN="${UV_ENV_PATH}/bin/python"
echo "Using python: ${PYTHON_BIN}"
"${PYTHON_BIN}" -V

"${PYTHON_BIN}" "${CODE_DIR}/noisy_qrc_slurm.py" \
  --output-dir "${OUTPUT_DIR}" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --num-tasks 10 \
  --shots "${SHOTS}" \
  --n-iterations "${N_ITERATIONS}" \
  --top-k "${TOP_K}" \
  --optuna-trials "${OPTUNA_TRIALS}" \
  --seed "${SEED}" \
  --subset-frac "${SUBSET_FRAC}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --max-memory-mb "${MAX_MEMORY_MB}" \
  --resume

# Aggregate after the final array task completes.
if [[ "${SLURM_ARRAY_TASK_ID}" -eq 9 ]]; then
  "${PYTHON_BIN}" "${CODE_DIR}/noisy_qrc_slurm.py" \
    --output-dir "${OUTPUT_DIR}" \
    --aggregate \
    --shots "${SHOTS}" \
    --n-iterations "${N_ITERATIONS}" \
    --top-k "${TOP_K}" \
    --optuna-trials "${OPTUNA_TRIALS}" \
    --seed "${SEED}" \
    --subset-frac "${SUBSET_FRAC}"
fi
