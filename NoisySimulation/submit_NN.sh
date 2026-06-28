#!/bin/bash
# Opt-NN-TFI evolution-time-sweep submission script for CURC Alpine.
#
#
# Array size = N_ITERATIONS * 2 * len(T_SWEEP)
#   sbatch --array=0-49 --account=<your_allocation> submit_t_sweep.sh
# Override the sweep with e.g.:
#   sbatch --array=0-39 --account=<acct> --export=ALL,T_SWEEP="0.1,0.25,0.5,1.0" submit_t_sweep.sh
#
# Directives below follow the CURC Alpine docs (curc.readthedocs.io):
# amilan is the default CPU partition (3.8 GB RAM/core), qos=normal allows
# up to 1 day walltime. #SBATCH lines are parsed before the script body
# runs, so override them on the sbatch command line, not via env vars.
#SBATCH --job-name=tsweep-qrc
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/tsweep_qrc_%x_%A_%a.out

set -euo pipefail

# Evolution times to sweep. Each value gets its own partials/t<value>/ tree
# and aggregate summary; the cross-point comparison lands in
# t_sweep_summary.json at aggregate time.
T_SWEEP="${T_SWEEP:-0.5}"

# Pipeline knobs.
SHOTS="${SHOTS:-4096}"
N_ITERATIONS="${N_ITERATIONS:-5}"
TOP_K="${TOP_K:-3}"
OPTUNA_TRIALS="${OPTUNA_TRIALS:-30}"
SEED="${SEED:-42}"
SUBSET_FRAC="${SUBSET_FRAC:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-8}"

# Hamiltonian architecture (defaults match the python CLI defaults).
TOPOLOGY="${TOPOLOGY:-nn}"
NUM_LAYERS="${NUM_LAYERS:-1}"
N_TROTTER_STEPS="${N_TROTTER_STEPS:-3}"
H_FIELD="${H_FIELD:-0.5}"
NOISE_MODEL="${NOISE_MODEL:-fakefez}"
# Ideal 6q simulation: let Aer pick statevector; density_matrix only pays
# off when a noise model is attached.
SIM_METHOD="${SIM_METHOD:-automatic}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"

# Locate the directory that actually holds the pipeline script. Check for
# the file itself, not just a Classical/ dir -- a stray Classical/Classical/
# artifact directory exists in the repo and matching on the dir name alone
# sends python to a path with no script (silent-looking failure).
if [[ -z "${CODE_DIR:-}" ]]; then
  for candidate in "${REPO_ROOT}/Classical" "${REPO_ROOT}" "${REPO_ROOT}/.."; do
    if [[ -f "${candidate}/NN-or-FC-Noisy-SLURM.py" ]]; then
      CODE_DIR="${candidate}"
      break
    fi
  done
fi
if [[ -z "${CODE_DIR:-}" || ! -f "${CODE_DIR}/NN-or-FC-NOISY-SLURM.py" ]]; then
  echo "FATAL: could not find python file to run near ${REPO_ROOT}; set CODE_DIR." >&2
  exit 1
fi
echo "CODE_DIR=${CODE_DIR}"

OUTPUT_DIR="${OUTPUT_DIR:-${CODE_DIR}/results/exp_nn}"
mkdir -p logs "${OUTPUT_DIR}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: submit with --array (e.g. sbatch --array=0-49 ...)." >&2
  exit 1
fi

# Per CURC docs: `module load miniforge` then `mamba activate <env>`
# (or `module load anaconda` + `conda activate` -- set CONDA_MODULE).
CONDA_MODULE="${CONDA_MODULE:-miniforge}"
MAMBA_ENV_NAME="${MAMBA_ENV_NAME:-icequake_env}"

# Lmod may not be initialized in a non-interactive batch shell.
if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  elif [[ -f /etc/profile.d/lmod.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/lmod.sh
  fi
fi
if ! command -v module >/dev/null 2>&1; then
  echo "FATAL: 'module' command not available in this shell." >&2
  exit 1
fi

module purge >/dev/null 2>&1 || true
echo "Loading module ${CONDA_MODULE}..."
module load "${CONDA_MODULE}"
module list 2>&1 || true

# `mamba activate`/`conda activate` are shell functions, absent in fresh
# non-interactive shells until the init scripts are sourced. Those init
# scripts touch unset variables, so relax nounset around activation.
set +u
CONDA_BASE="$(mamba info --base 2>/dev/null || conda info --base 2>/dev/null || true)"
echo "Conda base: ${CONDA_BASE:-<not found>}"
if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi
if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/mamba.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/mamba.sh"
fi
if ! mamba activate "${MAMBA_ENV_NAME}" && ! conda activate "${MAMBA_ENV_NAME}"; then
  echo "FATAL: failed to activate env '${MAMBA_ENV_NAME}'. Available envs:" >&2
  mamba env list 2>&1 || conda env list 2>&1 || true
  exit 1
fi
set -u

if ! command -v python >/dev/null 2>&1; then
  echo "FATAL: no 'python' on PATH after activating '${MAMBA_ENV_NAME}'." >&2
  echo "PATH=${PATH}" >&2
  exit 1
fi
echo "Activated env ${MAMBA_ENV_NAME}: $(which python) ($(python -V 2>&1))"
python -c "import qiskit, qiskit_aer, optuna, xgboost" || {
  echo "FATAL: required packages missing from '${MAMBA_ENV_NAME}'." >&2
  exit 1
}

# Match thread pools to the SLURM allocation; keeps numpy/XGBoost from
# oversubscribing during Optuna tuning.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

NUM_SWEEP_POINTS=$(awk -F',' '{print NF}' <<<"${T_SWEEP}")
EXPECTED_TASKS=$((N_ITERATIONS * 2 * NUM_SWEEP_POINTS))
if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" && "${SLURM_ARRAY_TASK_COUNT}" -ne "${EXPECTED_TASKS}" ]]; then
  echo "WARNING: --array size ${SLURM_ARRAY_TASK_COUNT} != expected ${EXPECTED_TASKS}" >&2
  echo "         (n_iter * 2 * |t_sweep| = ${N_ITERATIONS} * 2 * ${NUM_SWEEP_POINTS})" >&2
fi

PYTHON_BIN="$(which python)"
echo "==============================================================="
echo "Experiment        : NN-TFI evolution-time sweep"
echo "t sweep           : ${T_SWEEP}"
echo "Topology          : ${TOPOLOGY}  layers=${NUM_LAYERS}  trotter=${N_TROTTER_STEPS}  h=${H_FIELD}"
echo "Noise model       : ${NOISE_MODEL}  Aer method=${SIM_METHOD}"
echo "Shots             : ${SHOTS}  N_iter=${N_ITERATIONS}  Top-K=${TOP_K}"
echo "Env               : ${MAMBA_ENV_NAME} (python: ${PYTHON_BIN})"
echo "Output dir        : ${OUTPUT_DIR}"
echo "Task id / array   : ${SLURM_ARRAY_TASK_ID} / ${SLURM_ARRAY_TASK_COUNT:-N/A}"
echo "==============================================================="
"${PYTHON_BIN}" -V

"${PYTHON_BIN}" "${CODE_DIR}/NN-or-FC-Noisy-SLURM.py" \
  --output-dir "${OUTPUT_DIR}" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --num-tasks "${EXPECTED_TASKS}" \
  --shots "${SHOTS}" \
  --n-iterations "${N_ITERATIONS}" \
  --top-k "${TOP_K}" \
  --optuna-trials "${OPTUNA_TRIALS}" \
  --seed "${SEED}" \
  --subset-frac "${SUBSET_FRAC}" \
  --device cpu \
  --batch-size "${BATCH_SIZE}" \
  --topology "${TOPOLOGY}" \
  --num-layers "${NUM_LAYERS}" \
  --n-trotter-steps "${N_TROTTER_STEPS}" \
  --h-field "${H_FIELD}" \
  --t-sweep "${T_SWEEP}" \
  --noise-model "${NOISE_MODEL}" \
  --sim-method "${SIM_METHOD}" \
  --resume

# Aggregate on the last array index. Array tasks can finish out of order, so
# if any earlier task is still running when this fires, rerun the aggregate
# manually (or submit it with --dependency=afterok:<array job id>):
#   sbatch --dependency=afterok:<jobid> --array=0 \
#     --export=ALL,SKIP_INLINE_AGGREGATE=0 ... (or just run --aggregate locally)
LAST_TASK_ID=$((EXPECTED_TASKS - 1))
if [[ "${SLURM_ARRAY_TASK_ID}" -eq "${LAST_TASK_ID}" && "${SKIP_INLINE_AGGREGATE:-0}" -ne 1 ]]; then
  echo "Last array task; running aggregate step inline."
  "${PYTHON_BIN}" "${CODE_DIR}/NN-or-FC-Noisy-SLURM.py" \
    --output-dir "${OUTPUT_DIR}" \
    --aggregate \
    --shots "${SHOTS}" \
    --n-iterations "${N_ITERATIONS}" \
    --top-k "${TOP_K}" \
    --optuna-trials "${OPTUNA_TRIALS}" \
    --seed "${SEED}" \
    --subset-frac "${SUBSET_FRAC}" \
    --topology "${TOPOLOGY}" \
    --num-layers "${NUM_LAYERS}" \
    --n-trotter-steps "${N_TROTTER_STEPS}" \
    --h-field "${H_FIELD}" \
    --t-sweep "${T_SWEEP}" \
    --noise-model "${NOISE_MODEL}"
fi
