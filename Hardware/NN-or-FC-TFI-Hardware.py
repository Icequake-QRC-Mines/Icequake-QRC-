#!/usr/bin/env python3
"""
Hardware QRC migration for Fez (Heron r2) with packed event execution.

Pipeline in this file:
1) Build 21-event-history classifier (routing model).
2) Build 6-feature reservoir input stream:
   - current event: form factor, slip distance, time since
   - previous event: tide height, tide derivative, high tide event
3) Run 5 noiseless reservoir candidates, rank by validation MAE, keep top 2.
4) Submit all hardware pubs for those top-2 candidates as one Runtime job.
5) Extract Z/X/Y/ZZ features directly from shot counts post-readout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, qpy
from qiskit.circuit import ClassicalRegister, Parameter
from qiskit_aer import AerSimulator
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.transpiler.passes import RemoveBarriers
from qiskit_ibm_runtime import Batch, QiskitRuntimeService, SamplerV2 as Sampler
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from Preprocess import preprocess_data_window

CLASSIFIER_N_PREVIOUS_EVENTS = 20
RESERVOIR_N_QUBITS = 6
# Exponential decay applied to previously encoded events: the encoded angles of
# the event k steps back in the history are scaled by DEFAULT_EVENT_DECAY**k, so
# older events rotate the reservoir qubits less than recent ones.
DEFAULT_EVENT_DECAY = 0.3
# Simulation candidate scoring must stay tractable: always run one 6-qubit block
# per circuit, regardless of hardware backend width.
SIMULATION_MAX_BLOCKS_PER_CIRCUIT = 1
DEFAULT_REP_DELAY_SECONDS = 250e-6
DEFAULT_SUB_JOB_OVERHEAD_SECONDS = 2.0
DEFAULT_RUNTIME_PUBS_PER_JOB = 500
DEFAULT_HARDWARE_QUBITS = 156
KNOWN_BACKEND_QUBITS = {
    "ibm_fez": 156,
}
TRANSPILE_CACHE_VERSION = 1
# Cache files written by the first cache-enabled run used the full source-file
# hash. Keep that lookup so runtime-estimator-only fixes do not waste them.
LEGACY_TRANSPILE_CACHE_CODE_HASHES = (
    "8076826db71d8f74dcec14d4fc4918376930744ead11c4656f2c88bba678753a",
)
# Asynchronous / resumable hardware submission artifacts. All of these live in a
# dedicated subdirectory of the run output dir so they never collide with the
# (large, expensive) transpilation cache.
HARDWARE_JOBS_DIRNAME = "hardware_jobs"
SUBMISSION_MANIFEST_NAME = "submission_manifest.json"
FEATURE_CHECKPOINT_NAME = "feature_store_checkpoint.npz"
SUBMISSION_MANIFEST_VERSION = 1
# IBM caps the service-calculated per-job timeout at 3 hours of quantum time.
IBM_MAX_JOB_SECONDS = 3 * 60 * 60
DEFAULT_NUM_BATCHES = 3
DEFAULT_COLLECT_POLL_SECONDS = 60.0
# Terminal Runtime job states.
_JOB_STATUS_DONE = "DONE"
_JOB_STATUS_ERROR_STATES = ("ERROR", "CANCELLED", "FAILED")
RESERVOIR_FEATURE_COLUMNS = [
    "form_fac-0",
    "slip_size-0",
    "time_since-0",
    "tide_height-1",
    "tide_deriv-1",
    "high_t_evt-1",
]


@dataclass
class ReservoirParams:
    J: np.ndarray
    h: float
    t: float


@dataclass
class CandidateConfig:
    candidate_id: int
    short_params: ReservoirParams
    long_params: ReservoirParams
    seed: int


@dataclass
class PackingConfig:
    n_qubits_per_block: int
    max_blocks_per_circuit: int


@dataclass
class ChunkRequest:
    chunk_id: int
    matrix_key: Tuple[int, str]  # (candidate_id, dataset_key)
    start_row: int
    end_row: int
    n_blocks: int
    event_offset: int


@dataclass
class PubMeta:
    chunk_id: int
    basis: str  # "z" | "x" | "y"


@dataclass
class PubSubmissionMeta:
    chunk_id: int
    basis: str
    matrix_key: Tuple[int, str]
    start_row: int
    end_row: int
    n_blocks: int
    event_offset: int


@dataclass
class HardwareJobRecord:
    job: object
    pub_metas: List[PubSubmissionMeta]
    num_pubs: int


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(contiguous.shape).encode("utf-8"))
    h.update(str(contiguous.dtype).encode("utf-8"))
    h.update(contiguous.tobytes())
    return h.hexdigest()


def reservoir_params_digest(params: ReservoirParams) -> str:
    h = hashlib.sha256()
    h.update(_array_digest(params.J).encode("utf-8"))
    h.update(np.asarray([params.h, params.t], dtype=float).tobytes())
    return h.hexdigest()


def make_transpile_cache_key(
    *,
    backend_name: str,
    optimization_level: int,
    transpile_strategy: str,
    candidate_id: int,
    regime: str,
    dataset_key: str,
    event_idx: int,
    start: int,
    end: int,
    n_blocks: int,
    num_layers: int,
    n_qubits_per_block: int,
    code_hash: str,
    basis: str,
    chunk: np.ndarray,
    reservoir_params_digest: str,
) -> str:
    payload = {
        "version": TRANSPILE_CACHE_VERSION,
        "backend_name": backend_name,
        "optimization_level": int(optimization_level),
        "transpile_strategy": transpile_strategy,
        "candidate_id": int(candidate_id),
        "regime": regime,
        "dataset_key": dataset_key,
        "event_idx": int(event_idx),
        "start": int(start),
        "end": int(end),
        "n_blocks": int(n_blocks),
        "num_layers": int(num_layers),
        "n_qubits_per_block": int(n_qubits_per_block),
        "code_hash": code_hash,
        "basis": basis,
        "chunk_digest": _array_digest(chunk),
        "reservoir_params_digest": reservoir_params_digest,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def _transpile_cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.qpy"


def load_cached_transpiled_circuit(
    cache_dir: Path,
    key: str,
) -> Optional[QuantumCircuit]:
    path = _transpile_cache_path(cache_dir, key)
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            circuits = qpy.load(f)
    except Exception:
        return None
    if len(circuits) != 1:
        return None
    return circuits[0]


def save_cached_transpiled_circuit(
    cache_dir: Path,
    key: str,
    circuit: QuantumCircuit,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _transpile_cache_path(cache_dir, key)
    tmp_path = path.with_suffix(".qpy.tmp")
    with tmp_path.open("wb") as f:
        qpy.dump([circuit], f)
    tmp_path.replace(path)


def _collect_measure_pairs(circuit: QuantumCircuit) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for instruction in circuit.data:
        if instruction.operation.name != "measure":
            continue
        if len(instruction.qubits) != 1 or len(instruction.clbits) != 1:
            raise ValueError("Expected one-to-one measure operations in transpiled circuit.")
        qubit_idx = circuit.find_bit(instruction.qubits[0]).index
        cbit_idx = circuit.find_bit(instruction.clbits[0]).index
        pairs.append((qubit_idx, cbit_idx))
    return pairs


def _append_basis_rotations(
    circuit: QuantumCircuit,
    measured_qubits: Sequence[int],
    basis: str,
) -> None:
    if basis == "z":
        return
    if basis == "x":
        for qubit in measured_qubits:
            # H = RZ(pi/2) SX RZ(pi/2), fully native for IBM backends.
            circuit.rz(np.pi / 2.0, qubit)
            circuit.sx(qubit)
            circuit.rz(np.pi / 2.0, qubit)
        return
    if basis == "y":
        for qubit in measured_qubits:
            # Sdg then H, rewritten to native gates up to global phase.
            circuit.sx(qubit)
            circuit.rz(np.pi / 2.0, qubit)
        return
    raise ValueError(f"Unsupported measurement basis: {basis}")


def derive_transpiled_basis_circuit(
    z_transpiled_circuit: QuantumCircuit,
    basis: str,
) -> QuantumCircuit:
    if basis == "z":
        return z_transpiled_circuit

    measure_pairs = _collect_measure_pairs(z_transpiled_circuit)
    if len(measure_pairs) == 0:
        raise ValueError("Expected transpiled Z-basis circuit to include measurements.")

    base = z_transpiled_circuit.remove_final_measurements(inplace=False)
    measured_qubits = [qubit for qubit, _ in measure_pairs]
    derived = base.copy()
    n_clbits_required = max(cbit for _, cbit in measure_pairs) + 1
    if len(derived.clbits) < n_clbits_required:
        derived.add_register(
            ClassicalRegister(n_clbits_required - len(derived.clbits), "c_basis")
        )
    _append_basis_rotations(derived, measured_qubits, basis)
    for qubit, cbit in measure_pairs:
        derived.measure(qubit, cbit)
    return derived


def parse_args() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(
        description="Run migrated Fez hardware QRC pipeline without hardware_config.pkl."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("Classical/results/hardware_qrc_run")
    )
    parser.add_argument("--summary-name", type=str, default="hardware_summary.json")
    parser.add_argument("--backend", type=str, default="ibm_fez")
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--subset-frac", type=float, default=0.25)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--n-candidates", type=int, default=5)
    parser.add_argument(
        "--reservoir-model",
        type=str,
        default="all-to-all",
        choices=["all-to-all", "nn-tfim"],
        help=(
            "Reservoir Hamiltonian coupling topology. 'all-to-all' samples a "
            "random coupling for every qubit pair (original behavior). "
            "'nn-tfim' is a nearest-neighbor transverse-field Ising chain: "
            "only J[i, i+1] bonds are kept."
        ),
    )
    parser.add_argument(
        "--event-decay",
        type=float,
        default=DEFAULT_EVENT_DECAY,
        help=(
            "Exponential decay for previously encoded events: the encoded "
            "angles of the event k steps back are scaled by decay**k, so older "
            "events are weighted less. Set to 1.0 to disable. Default: 0.3."
        ),
    )
    parser.add_argument("--ensemble-size", type=int, default=2)
    parser.add_argument("--resilience-level", type=int, default=0)
    parser.add_argument("--optimization-level", type=int, default=2)
    parser.add_argument(
        "--transpile-strategy",
        type=str,
        default="reuse-z-basis",
        choices=["reuse-z-basis", "independent"],
        help=(
            "Transpilation strategy for X/Y/Z pubs. "
            "'reuse-z-basis' transpiles Z once and derives X/Y via native basis "
            "rotations (much faster at high optimization levels). "
            "'independent' transpiles each basis separately."
        ),
    )
    parser.add_argument(
        "--transpile-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for cached transpiled QPY circuits. "
            "Default: <output-dir>/transpile_cache."
        ),
    )
    parser.add_argument(
        "--no-transpile-cache",
        action="store_true",
        help="Disable file-backed transpilation cache.",
    )
    parser.add_argument(
        "--transpile-log-every",
        type=int,
        default=25,
        help=(
            "Print aggregate transpilation progress every N transpile units. "
            "Set to 1 for very verbose progress."
        ),
    )
    parser.add_argument(
        "--runtime-pubs-per-job",
        type=int,
        default=DEFAULT_RUNTIME_PUBS_PER_JOB,
        help=(
            "Maximum number of sampler pubs per IBM Runtime job when using "
            "batched hardware submission. Rounded down to a whole number of "
            "z/x/y chunks so each job is independently decodable."
        ),
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=DEFAULT_NUM_BATCHES,
        help=(
            "Number of sequential IBM Runtime batches to spread jobs across. "
            "Splitting a long workload keeps each batch's total work under the "
            "batch max-TTL ceiling. Default: 3."
        ),
    )
    parser.add_argument(
        "--batch-max-time",
        type=int,
        default=None,
        help=(
            "Optional per-batch max TTL in seconds passed to Batch(max_time=...). "
            "Default: unset (use the IBM system-assigned maximum)."
        ),
    )
    parser.add_argument(
        "--collect-poll-seconds",
        type=float,
        default=DEFAULT_COLLECT_POLL_SECONDS,
        help=(
            "Polling interval in seconds while waiting for outstanding Runtime "
            "jobs during the collect phase. Default: 60."
        ),
    )
    parser.add_argument(
        "--fresh-submit",
        action="store_true",
        help=(
            "Ignore and overwrite any existing submission manifest/checkpoint in "
            "the output dir and submit a new set of jobs. Without this flag a "
            "matching manifest is resumed instead of resubmitted."
        ),
    )
    parser.add_argument(
        "--allow-partial-results",
        action="store_true",
        help=(
            "Continue to training/finalization even if some Runtime jobs ended in "
            "an error state (their feature rows stay zero). Default: abort."
        ),
    )
    parser.add_argument(
        "--allow-runtime-estimate-extrapolation",
        action="store_true",
        help=(
            "Allow aggregate pre-submit runtime estimation to extrapolate missing "
            "circuit durations instead of failing."
        ),
    )
    parser.add_argument("--load-selected-features", action="store_true", default=False)
    parser.add_argument(
        "--classifier-optuna-trials",
        type=int,
        default=0,
        help=(
            "Number of Optuna trials for classifier tuning on fixed classical data. "
            "0 disables classifier Optuna tuning."
        ),
    )
    parser.add_argument(
        "--regressor-optuna-trials",
        type=int,
        default=0,
        help=(
            "Number of Optuna trials per regime for regressor tuning on fixed "
            "reservoir features. 0 disables regressor Optuna tuning."
        ),
    )
    parser.add_argument(
        "--regressor-optuna-cv-folds",
        type=int,
        default=3,
        help="CV folds for regressor Optuna objective (minimum 2).",
    )
    parser.add_argument(
        "--parallel-circuits",
        type=int,
        default=None,
        help=(
            "How many reservoir blocks to pack per hardware circuit. "
            "Default: auto (maximum supported by backend width)."
        ),
    )
    parser.add_argument(
        "--simulate-only",
        action="store_true",
        help="Stop after 5-candidate noiseless ranking (no hardware submission).",
    )
    parser.add_argument(
        "--transpile-only",
        action="store_true",
        help=(
            "Transpile all hardware pubs at --optimization-level (populating "
            "the transpile cache) and exit before any IBM Runtime job is "
            "submitted. Still requires --token: transpilation targets the "
            "resolved backend. A later full run reuses the cached circuits."
        ),
    )
    parser.add_argument(
        "--execution-limit",
        type=float,
        default=2 * 60 * 60,
        help=(
            "Maximum allowed estimated quantum time PER JOB in seconds before "
            "submission. Clamped to IBM's 3-hour per-job cap. The total workload "
            "may exceed this since it is spread across many jobs. Default: 7200 "
            "(2 hours)."
        ),
    )
    parser.add_argument("--no-confirm", action="store_true")
    return parser.parse_args(), parser


def save_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def maybe_subset_arrays(
    X_hist: np.ndarray,
    X_res: np.ndarray,
    y: np.ndarray,
    frac: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if frac >= 1.0 or len(X_hist) == 0:
        return X_hist, X_res, y
    keep = max(1, int(len(X_hist) * frac))
    idx = np.sort(rng.choice(len(X_hist), size=keep, replace=False))
    return X_hist[idx], X_res[idx], y[idx]


def scale_to_pi(
    X: np.ndarray,
    train_min: Optional[np.ndarray] = None,
    train_max: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train_min is None or train_max is None:
        train_min = X.min(axis=0)
        train_max = X.max(axis=0)
    denom = train_max - train_min
    denom[denom == 0.0] = 1.0
    scaled = np.clip((X - train_min) / denom, 0.0, 1.0) * np.pi
    return scaled, train_min, train_max


def load_split_data(
    subset_frac: float,
    subset_seed: int,
    load_selected_features: bool = False,
    event_decay: float = 1.0,
) -> Dict[str, np.ndarray]:
    repo_root = Path(__file__).resolve().parent.parent
    data_csv = repo_root / "Whillians-GPS-Data-and-Features.csv"
    filtered_csv = repo_root / "filtered_time_to_next_event.csv"
    if not data_csv.exists() or not filtered_csv.exists():
        raise FileNotFoundError(
            f"Expected {data_csv} and {filtered_csv} at repository root."
        )

    data_orig = pd.read_csv(data_csv)
    filtered_time = pd.read_csv(filtered_csv)
    X_train_df, X_val_df, X_test_df, y_train_s, y_val_s, y_test_s, _ = preprocess_data_window(
        filtered_time, data_orig, CLASSIFIER_N_PREVIOUS_EVENTS
    )

    if load_selected_features:
        for col in RESERVOIR_FEATURE_COLUMNS:
            if col not in X_train_df.columns:
                raise KeyError(
                    f"Missing required reservoir feature column '{col}'. "
                    "Ensure preprocess_data_window uses 21-event history."
                )

    X_train_hist = X_train_df.to_numpy(dtype=float)
    X_val_hist = X_val_df.to_numpy(dtype=float)
    X_test_hist = X_test_df.to_numpy(dtype=float)

    def _ordered_window_cols(df: pd.DataFrame) -> List[str]:
        pattern = re.compile(r"^(.+)-(\d+)$") # feature_name-event_index
        parsed: List[Tuple[int, str, str]] = []
        for col in df.columns:
            m = pattern.match(str(col))
            if not m: # not a valid feature column
                continue
            base = m.group(1) # feature_name
            event_idx = int(m.group(2)) # event_index
            parsed.append((event_idx, base, col)) # we want to use this to create the reservoir encoding
        if not parsed:
            raise ValueError("No sequential '-k' feature columns found for reservoir encoding.")

        # Chronological order: current event first (0), then 1, 2, ...
        # Within each event, preserve the source DataFrame column order.
        col_position = {name: i for i, name in enumerate(df.columns)}
        parsed.sort(key=lambda row: (row[0], col_position[row[2]]))
        ordered = [col for _, _, col in parsed]
        return ordered

    if load_selected_features:
        reservoir_cols = list(RESERVOIR_FEATURE_COLUMNS)
        n_events = 1
        n_features_per_event = RESERVOIR_N_QUBITS
    else:
        reservoir_cols = _ordered_window_cols(X_train_df)
        event_suffixes = sorted({int(c.rsplit("-", 1)[1]) for c in reservoir_cols})
        if event_suffixes[0] != 0:
            raise ValueError("Reservoir sequential columns must start at event index 0.")
        if event_suffixes[-1] != len(event_suffixes) - 1:
            raise ValueError(
                "Reservoir sequential event indices must be contiguous (0..N)."
            )
        n_events = len(event_suffixes)
        if len(reservoir_cols) % n_events != 0:
            raise ValueError(
                "Reservoir sequential columns are not evenly divisible by event count."
            )
        n_features_per_event = len(reservoir_cols) // n_events
        if n_features_per_event != RESERVOIR_N_QUBITS:
            raise ValueError(
                "Reservoir requires 6 features per event. "
                f"Found {n_features_per_event} features/event from selected columns."
            )

    X_train_res_raw = X_train_df[reservoir_cols].to_numpy(dtype=float)
    X_val_res_raw = X_val_df[reservoir_cols].to_numpy(dtype=float)
    X_test_res_raw = X_test_df[reservoir_cols].to_numpy(dtype=float)

    X_train_res, train_min, train_max = scale_to_pi(X_train_res_raw)
    X_val_res, _, _ = scale_to_pi(X_val_res_raw, train_min, train_max)
    X_test_res, _, _ = scale_to_pi(X_test_res_raw, train_min, train_max)

    # Exponentially decay previously encoded events: each column's encoded
    # angle is weighted by decay**event_index (event 0 = current event, so it
    # is unweighted). Applied after [0, pi] scaling — a pre-scaling weight
    # would be cancelled by the per-column min/max normalization.
    if not (0.0 < event_decay <= 1.0):
        raise ValueError(f"event_decay must satisfy 0 < decay <= 1, got {event_decay}.")
    decay_weights = np.power(
        float(event_decay),
        np.array([int(str(c).rsplit("-", 1)[1]) for c in reservoir_cols], dtype=float),
    )
    X_train_res = X_train_res * decay_weights
    X_val_res = X_val_res * decay_weights
    X_test_res = X_test_res * decay_weights

    y_train = y_train_s.to_numpy(dtype=float)
    y_val = y_val_s.to_numpy(dtype=float)
    y_test = y_test_s.to_numpy(dtype=float)

    rng = np.random.default_rng(subset_seed)
    X_train_hist, X_train_res, y_train = maybe_subset_arrays(
        X_train_hist, X_train_res, y_train, subset_frac, rng
    )
    X_val_hist, X_val_res, y_val = maybe_subset_arrays(
        X_val_hist, X_val_res, y_val, subset_frac, rng
    )
    X_test_hist, X_test_res, y_test = maybe_subset_arrays(
        X_test_hist, X_test_res, y_test, subset_frac, rng
    )

    return {
        "X_train_hist": X_train_hist,
        "X_val_hist": X_val_hist,
        "X_test_hist": X_test_hist,
        "X_train_res": X_train_res,
        "X_val_res": X_val_res,
        "X_test_res": X_test_res,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "reservoir_train_min": train_min,
        "reservoir_train_max": train_max,
        "reservoir_n_events": n_events,
        "reservoir_n_features_per_event": n_features_per_event,
        "reservoir_sequence_cols": reservoir_cols,
        "reservoir_event_decay": float(event_decay),
    }


def _kmeans_regime_labels(y_train: np.ndarray, seed: int) -> np.ndarray:
    """Match notebook regime creation: KMeans on y_train only, then remap by center order."""
    km = KMeans(n_clusters=2, n_init=50, random_state=seed)
    km.fit(y_train.reshape(-1, 1))
    centers = km.cluster_centers_.ravel()
    order = np.argsort(centers)
    remap = {int(old): int(new) for new, old in enumerate(order)}
    raw = km.predict(y_train.reshape(-1, 1))
    return np.array([remap[int(label)] for label in raw], dtype=int)


def _sample_xgb_classifier_params(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 3.0),
    }


def _sample_xgb_regressor_params(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
    }


def tune_regime_classifier_params(
    X_train_hist: np.ndarray,
    labels: np.ndarray,
    seed: int,
    n_trials: int,
    n_splits: int = 3,
) -> Dict[str, float | int]:
    if n_trials <= 0:
        return {}
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "optuna is required for --classifier-optuna-trials > 0. Install optuna or set trials to 0."
        ) from exc

    sample_weights = compute_sample_weight("balanced", labels)

    def objective(trial):
        tuned = _sample_xgb_classifier_params(trial)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_accs: List[float] = []
        for train_idx, val_idx in skf.split(X_train_hist, labels):
            clf = XGBClassifier(
                objective="binary:logistic",
                n_estimators=500,
                eval_metric="logloss",
                random_state=seed,
                n_jobs=-1,
                **tuned,
            )
            X_f, X_v = X_train_hist[train_idx], X_train_hist[val_idx]
            y_f, y_v = labels[train_idx], labels[val_idx]
            w_f = sample_weights[train_idx]
            clf.fit(X_f, y_f, sample_weight=w_f, verbose=False)
            pred = clf.predict(X_v)
            fold_accs.append(float(np.mean(pred == y_v)))
        return 1.0 - float(np.mean(fold_accs))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)
    return {**_sample_xgb_classifier_params(study.best_trial)}


def tune_regime_regressor_params(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    n_trials: int,
    n_splits: int = 3,
) -> Dict[str, float | int]:
    if n_trials <= 0:
        return {}
    if len(y_train) < n_splits:
        return {}
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "optuna is required for --regressor-optuna-trials > 0. Install optuna or set trials to 0."
        ) from exc

    def objective(trial):
        tuned = _sample_xgb_regressor_params(trial)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        maes: List[float] = []
        for tr_idx, vl_idx in kf.split(X_train):
            model = XGBRegressor(
                objective="reg:squarederror",
                n_estimators=450,
                random_state=seed,
                n_jobs=-1,
                **tuned,
            )
            model.fit(X_train[tr_idx], y_train[tr_idx], verbose=False)
            pred = model.predict(X_train[vl_idx])
            maes.append(float(mean_absolute_error(y_train[vl_idx], pred)))
        return float(np.mean(maes))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)
    return {**_sample_xgb_regressor_params(study.best_trial)}


def train_regime_classifier(
    X_train_hist: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    optuna_trials: int = 0,
) -> XGBClassifier:
    labels = _kmeans_regime_labels(y_train, seed)
    if labels.min() == labels.max():
        raise ValueError(
            "KMeans regime labels collapse to one class after sampling. "
            "Increase --subset-frac or change --subset-seed/random-seed."
        )
    clf_kwargs: Dict[str, float | int | str] = {
        "n_estimators": 350,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": seed,
        "n_jobs": -1,
    }
    if optuna_trials > 0:
        tuned = tune_regime_classifier_params(
            X_train_hist,
            labels,
            seed=seed,
            n_trials=optuna_trials,
            n_splits=3,
        )
        clf_kwargs.update(tuned)
    clf = XGBClassifier(
        **clf_kwargs,
    )
    if optuna_trials > 0:
        sample_weights = compute_sample_weight("balanced", labels)
        clf.fit(X_train_hist, labels, sample_weight=sample_weights, verbose=False)
    else:
        clf.fit(X_train_hist, labels)
    return clf


def sample_reservoir_params(
    rng: np.random.Generator,
    n_qubits: int,
    model: str = "all-to-all",
) -> ReservoirParams:
    # The full upper triangle is always sampled so the rng stream (and hence
    # candidate reproducibility per seed) is identical across models.
    J = rng.uniform(-1.0, 1.0, size=(n_qubits, n_qubits))
    J = np.triu(J, 1)
    if model == "nn-tfim":
        # Nearest-neighbor TFIM chain: keep only the J[i, i+1] bonds.
        J = J - np.triu(J, 2)
    elif model != "all-to-all":
        raise ValueError(f"Unsupported reservoir model: {model}")
    J = J + J.T
    np.fill_diagonal(J, 0.0)
    h = float(rng.uniform(0.1, 1.0))
    t = float(rng.uniform(0.5, 2.5))
    return ReservoirParams(J=J, h=h, t=t)


def create_candidates(
    n_candidates: int,
    n_qubits: int,
    seed: int,
    model: str = "all-to-all",
) -> List[CandidateConfig]:
    rng = np.random.default_rng(seed)
    candidates: List[CandidateConfig] = []
    for cidx in range(n_candidates):
        candidates.append(
            CandidateConfig(
                candidate_id=cidx,
                short_params=sample_reservoir_params(rng, n_qubits, model),
                long_params=sample_reservoir_params(rng, n_qubits, model),
                seed=seed + cidx,
            )
        )
    return candidates


def trotter_ising_layer(
    qc: QuantumCircuit,
    n_qubits: int,
    J: np.ndarray,
    h: float,
    t: float,
    n_trotter_steps: int = 3,
) -> None:
    dt = t / n_trotter_steps
    for _ in range(n_trotter_steps):
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                if abs(J[i, j]) < 1e-12:
                    continue
                qc.cx(i, j)
                qc.rz(2.0 * J[i, j] * dt, j)
                qc.cx(i, j)
        for i in range(n_qubits):
            qc.rx(2.0 * h * dt, i)


def build_parametric_block_circuit(
    params: ReservoirParams,
    num_layers: int,
    n_qubits: int,
) -> Tuple[QuantumCircuit, List[Parameter]]:
    theta = [Parameter(f"theta_{i}") for i in range(n_qubits)]
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.h(i)
    for i in range(n_qubits):
        qc.ry(theta[i], i)
    for _ in range(num_layers):
        trotter_ising_layer(qc, n_qubits, params.J, params.h, params.t)
        for i in range(n_qubits):
            qc.ry(theta[i], i)
        trotter_ising_layer(qc, n_qubits, params.J, params.h, params.t)
    qc = RemoveBarriers()(qc)
    return qc, theta


def build_packed_state_circuit(
    chunk_inputs: np.ndarray,
    block_template: QuantumCircuit,
    block_params: Sequence[Parameter],
) -> QuantumCircuit:
    n_blocks, n_qubits = chunk_inputs.shape
    packed = QuantumCircuit(n_blocks * n_qubits)
    for block_idx in range(n_blocks):
        values = chunk_inputs[block_idx]
        bound = block_template.assign_parameters(
            {block_params[i]: float(values[i]) for i in range(n_qubits)},
            inplace=False,
        )
        start = block_idx * n_qubits
        packed.compose(bound, qubits=list(range(start, start + n_qubits)), inplace=True)
    return packed


def with_measurement_basis(
    state_circuit: QuantumCircuit,
    basis: str,
    n_qubits_per_block: int,
    n_blocks: int,
) -> QuantumCircuit:
    if basis not in {"z", "x", "y"}:
        raise ValueError(f"Unsupported measurement basis: {basis}")
    total_qubits = n_qubits_per_block * n_blocks
    qc = QuantumCircuit(total_qubits, total_qubits)
    qc.compose(state_circuit, qubits=list(range(total_qubits)), inplace=True)
    for block_idx in range(n_blocks):
        base = block_idx * n_qubits_per_block
        for q in range(n_qubits_per_block):
            tgt = base + q
            if basis == "x":
                qc.h(tgt)
            elif basis == "y":
                qc.sdg(tgt)
                qc.h(tgt)
    qc.measure(list(range(total_qubits)), list(range(total_qubits)))
    return qc


def _clean_bitstring(bits: str) -> str:
    return bits.replace(" ", "")


def _bit_at_cbit(bitstring: str, cbit: int) -> str:
    clean = _clean_bitstring(bitstring)
    return clean[-1 - cbit]


def counts_to_expectations_block(
    z_counts: Mapping[str, int],
    x_counts: Mapping[str, int],
    y_counts: Mapping[str, int],
    block_idx: int,
    n_qubits: int,
) -> np.ndarray:
    block = np.zeros(4 * n_qubits, dtype=float)
    cbase = block_idx * n_qubits

    z_total = float(sum(z_counts.values()))
    x_total = float(sum(x_counts.values()))
    y_total = float(sum(y_counts.values()))
    if z_total <= 0 or x_total <= 0 or y_total <= 0:
        return block

    for i in range(n_qubits):
        cbit = cbase + i
        z_exp = 0.0
        x_exp = 0.0
        y_exp = 0.0
        for bits, count in z_counts.items():
            z_exp += (1.0 if _bit_at_cbit(bits, cbit) == "0" else -1.0) * count / z_total
        for bits, count in x_counts.items():
            x_exp += (1.0 if _bit_at_cbit(bits, cbit) == "0" else -1.0) * count / x_total
        for bits, count in y_counts.items():
            y_exp += (1.0 if _bit_at_cbit(bits, cbit) == "0" else -1.0) * count / y_total
        block[i] = z_exp
        block[n_qubits + i] = x_exp
        block[2 * n_qubits + i] = y_exp

    # ZZ is derived from Z-basis counts post-readout (no separate pub).
    for i in range(n_qubits):
        j = (i + 1) % n_qubits
        cbit_i = cbase + i
        cbit_j = cbase + j
        zz_exp = 0.0
        for bits, count in z_counts.items():
            bi = _bit_at_cbit(bits, cbit_i)
            bj = _bit_at_cbit(bits, cbit_j)
            zz_exp += (1.0 if bi == bj else -1.0) * count / z_total
        block[3 * n_qubits + i] = zz_exp
    return block


def _counts_from_sampler_pub(pub_result) -> Dict[str, int]:
    data = pub_result.data
    if hasattr(pub_result, "join_data"):
        try:
            joined = pub_result.join_data()
            if hasattr(joined, "get_counts"):
                return dict(joined.get_counts())
        except Exception:
            pass
    if hasattr(data, "meas"):
        return dict(data.meas.get_counts())
    if hasattr(data, "c"):
        return dict(data.c.get_counts())
    if hasattr(data, "items"):
        for _name, value in data.items():
            if hasattr(value, "get_counts"):
                return dict(value.get_counts())
    available = list(data.keys()) if hasattr(data, "keys") else []
    raise ValueError(
        "Sampler result has no recognized classical count register "
        f"(available={available})."
    )


def _split_indices(n_rows: int, chunk_size: int) -> List[Tuple[int, int]]:
    return [(s, min(s + chunk_size, n_rows)) for s in range(0, n_rows, chunk_size)]


def _reservoir_shape_info(X_data: np.ndarray, n_qubits_per_block: int) -> Tuple[int, int]:
    if X_data.ndim != 2:
        raise ValueError("Reservoir input must be a 2D array (n_samples, n_features).")
    total_features = int(X_data.shape[1])
    if total_features == 0:
        return 0, 0
    if total_features % n_qubits_per_block != 0:
        raise ValueError(
            f"Reservoir feature dimension {total_features} is not divisible by "
            f"n_qubits_per_block={n_qubits_per_block}."
        )
    n_events = total_features // n_qubits_per_block
    return n_events, 4 * n_qubits_per_block


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    if seconds < 3600.0:
        return f"{seconds / 60.0:.1f}m"
    return f"{seconds / 3600.0:.2f}h"


def _is_unset_runtime_option(value) -> bool:
    if value is None:
        return True
    return value.__class__.__name__ == "UnsetType" or str(value) == "Unset"


def _resolve_runtime_option(value, default):
    return default if _is_unset_runtime_option(value) else value


def estimate_quantum_runtime_seconds(
    circuits: Sequence[QuantumCircuit],
    backend,
    shots: int,
    rep_delay_seconds: Optional[float] = None,
    init_qubits: Optional[bool] = None,
    per_sub_job_overhead_seconds: float = DEFAULT_SUB_JOB_OVERHEAD_SECONDS,
    scheduling_optimization_level: int = 0,
) -> Dict[str, Optional[float]]:
    # IBM baseline:
    #   per_sub_job_overhead + (rep_delay + circuit_length) * num_executions
    # where num_executions = broadcasted_circuits * shots.
    if len(circuits) == 0:
        return {
            "quantum_seconds_estimated": 0.0,
            "circuit_seconds_total": 0.0,
            "duration_coverage_ratio": 1.0,
            "used_duration_extrapolation": 0.0,
            "rep_delay_seconds": float(
                _resolve_runtime_option(
                    rep_delay_seconds,
                    getattr(backend, "default_rep_delay", DEFAULT_REP_DELAY_SECONDS),
                )
            ),
            "init_qubits_assumed": bool(_resolve_runtime_option(init_qubits, False)),
            "init_seconds_total": 0.0,
            "per_sub_job_overhead_seconds": float(per_sub_job_overhead_seconds),
            "num_sub_jobs_assumed": 0,
            "num_executions": 0,
            "quick_formula_seconds": 0.0,
        }

    resolved_rep_delay = float(
        _resolve_runtime_option(
            rep_delay_seconds,
            getattr(backend, "default_rep_delay", DEFAULT_REP_DELAY_SECONDS),
        )
    )
    # Runtime initializes qubits by default unless explicitly disabled.
    init_qubits = bool(_resolve_runtime_option(init_qubits, True))

    reset_duration = 0.0
    target = getattr(backend, "target", None)
    if target is not None:
        try:
            reset_duration = float(target["reset"][(0,)].duration)
        except Exception:
            reset_duration = 0.0

    circuits_for_timing: Sequence[QuantumCircuit] = circuits
    try:
        schedule_pm = generate_preset_pass_manager(
            target=backend.target,
            optimization_level=scheduling_optimization_level,
            scheduling_method="alap",
        )
        scheduled = schedule_pm.run(list(circuits))
        circuits_for_timing = [scheduled] if isinstance(scheduled, QuantumCircuit) else scheduled
    except Exception:
        circuits_for_timing = circuits

    durations: List[float] = []
    for circuit in circuits_for_timing:
        if not hasattr(circuit, "estimate_duration"):
            continue
        try:
            duration = circuit.estimate_duration(target=backend.target, unit="s")
        except Exception:
            continue
        if duration is None:
            continue
        durations.append(float(duration))

    if not durations:
        return {
            "quantum_seconds_estimated": None,
            "circuit_seconds_total": None,
            "duration_coverage_ratio": 0.0,
            "used_duration_extrapolation": None,
            "rep_delay_seconds": resolved_rep_delay,
            "init_qubits_assumed": bool(init_qubits),
            "init_seconds_total": None,
            "per_sub_job_overhead_seconds": float(per_sub_job_overhead_seconds),
            "num_sub_jobs_assumed": len(circuits),
            "num_executions": len(circuits) * int(shots),
            "scheduling_optimization_level": int(scheduling_optimization_level),
            "quick_formula_seconds": float(
                per_sub_job_overhead_seconds + 0.00035 * len(circuits) * int(shots)
            ),
        }

    n_circuits = len(circuits)
    n_covered = len(durations)
    covered_total = float(np.sum(durations))
    used_extrapolation = 0.0
    if n_covered < n_circuits:
        mean_duration = covered_total / n_covered
        covered_total += mean_duration * (n_circuits - n_covered)
        used_extrapolation = 1.0

    init_seconds_total = reset_duration * n_circuits if init_qubits else 0.0
    total_circuit_seconds = covered_total + resolved_rep_delay * n_circuits + init_seconds_total
    num_executions = n_circuits * int(shots)
    # Runtime may split payload into multiple sub-jobs internally. For local
    # pre-submit planning, approximate one sub-job per pub.
    num_sub_jobs_assumed = n_circuits
    quantum_seconds = (
        per_sub_job_overhead_seconds * num_sub_jobs_assumed
        + total_circuit_seconds * shots
    )
    return {
        "quantum_seconds_estimated": float(quantum_seconds),
        "circuit_seconds_total": float(total_circuit_seconds),
        "duration_coverage_ratio": float(n_covered / n_circuits),
        "used_duration_extrapolation": used_extrapolation,
        "rep_delay_seconds": resolved_rep_delay,
        "init_qubits_assumed": bool(init_qubits),
        "init_seconds_total": float(init_seconds_total),
        "per_sub_job_overhead_seconds": float(per_sub_job_overhead_seconds),
        "num_sub_jobs_assumed": int(num_sub_jobs_assumed),
        "num_executions": int(num_executions),
        "scheduling_optimization_level": int(scheduling_optimization_level),
        "quick_formula_seconds": float(
            per_sub_job_overhead_seconds + 0.00035 * num_executions
        ),
    }


def extract_features_noiseless(
    X_data: np.ndarray,
    params: ReservoirParams,
    num_layers: int,
    shots: int,
    packing: PackingConfig,
) -> np.ndarray:
    n_rows = len(X_data)
    n_events, n_obs = _reservoir_shape_info(X_data, packing.n_qubits_per_block)
    out = np.zeros((n_rows, n_events * n_obs), dtype=float)
    if n_rows == 0:
        return out

    backend = AerSimulator(seed_simulator=42)
    block_template, block_params = build_parametric_block_circuit(
        params, num_layers, packing.n_qubits_per_block
    )

    # For simulation ranking we intentionally keep chunking separate from hardware
    # packing: max_blocks_per_circuit can be forced to 1 so each simulation circuit
    # remains a strict 6-qubit execution.
    for event_idx in range(n_events):
        start_col = event_idx * packing.n_qubits_per_block
        end_col = start_col + packing.n_qubits_per_block
        X_event = X_data[:, start_col:end_col]
        obs_offset = event_idx * n_obs
        for start, end in _split_indices(n_rows, packing.max_blocks_per_circuit):
            chunk = X_event[start:end]
            n_blocks = len(chunk)
            state = build_packed_state_circuit(chunk, block_template, block_params)
            circuits = [
                with_measurement_basis(state, "z", packing.n_qubits_per_block, n_blocks),
                with_measurement_basis(state, "x", packing.n_qubits_per_block, n_blocks),
                with_measurement_basis(state, "y", packing.n_qubits_per_block, n_blocks),
            ]
            result = backend.run(circuits, shots=shots).result()
            z_counts = dict(result.get_counts(0))
            x_counts = dict(result.get_counts(1))
            y_counts = dict(result.get_counts(2))
            for b in range(n_blocks):
                out[start + b, obs_offset : obs_offset + n_obs] = counts_to_expectations_block(
                    z_counts, x_counts, y_counts, b, packing.n_qubits_per_block
                )
    return out


def build_routing_data(
    clf: XGBClassifier,
    X_train_hist: np.ndarray,
    X_val_hist: np.ndarray,
    X_test_hist: np.ndarray,
    X_train_res: np.ndarray,
    X_val_res: np.ndarray,
    X_test_res: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, np.ndarray]:
    train_labels = clf.predict(X_train_hist)
    val_labels = clf.predict(X_val_hist)
    test_labels = clf.predict(X_test_hist)

    short_train_idx = np.where(train_labels == 0)[0]
    long_train_idx = np.where(train_labels == 1)[0]
    if len(short_train_idx) == 0 or len(long_train_idx) == 0:
        raise ValueError(
            "Classifier routing produced empty short/long train split. "
            "Use a larger subset or different seed."
        )

    return {
        "X_train_res": X_train_res,
        "X_val_res": X_val_res,
        "X_test_res": X_test_res,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "train_labels": train_labels,
        "val_labels": val_labels,
        "test_labels": test_labels,
        "short_train_idx": short_train_idx,
        "long_train_idx": long_train_idx,
        "short_val_idx": np.where(val_labels == 0)[0],
        "long_val_idx": np.where(val_labels == 1)[0],
        "short_test_idx": np.where(test_labels == 0)[0],
        "long_test_idx": np.where(test_labels == 1)[0],
    }


def evaluate_candidate(
    candidate: CandidateConfig,
    routing: Dict[str, np.ndarray],
    num_layers: int,
    shots: int,
    packing: PackingConfig,
) -> Dict:
    short_train = routing["short_train_idx"]
    long_train = routing["long_train_idx"]
    short_val = routing["short_val_idx"]
    long_val = routing["long_val_idx"]
    short_test = routing["short_test_idx"]
    long_test = routing["long_test_idx"]

    X_train_res = routing["X_train_res"]
    X_val_res = routing["X_val_res"]
    X_test_res = routing["X_test_res"]
    y_train = routing["y_train"]
    y_val = routing["y_val"]
    y_test = routing["y_test"]

    P_tr_short = extract_features_noiseless(
        X_train_res[short_train], candidate.short_params, num_layers, shots, packing
    )
    P_vl_short = extract_features_noiseless(
        X_val_res[short_val], candidate.short_params, num_layers, shots, packing
    )
    P_te_short = extract_features_noiseless(
        X_test_res[short_test], candidate.short_params, num_layers, shots, packing
    )
    P_tr_long = extract_features_noiseless(
        X_train_res[long_train], candidate.long_params, num_layers, shots, packing
    )
    P_vl_long = extract_features_noiseless(
        X_val_res[long_val], candidate.long_params, num_layers, shots, packing
    )
    P_te_long = extract_features_noiseless(
        X_test_res[long_test], candidate.long_params, num_layers, shots, packing
    )

    model_short = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=450,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=candidate.seed,
        n_jobs=-1,
    )
    model_long = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=450,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=candidate.seed + 1000,
        n_jobs=-1,
    )
    eval_short = [(P_vl_short, y_val[short_val])] if len(short_val) > 0 else None
    eval_long = [(P_vl_long, y_val[long_val])] if len(long_val) > 0 else None
    model_short.fit(
        P_tr_short,
        y_train[short_train],
        eval_set=eval_short,
        verbose=False,
    )
    model_long.fit(
        P_tr_long,
        y_train[long_train],
        eval_set=eval_long,
        verbose=False,
    )

    val_pred = np.empty_like(y_val)
    test_pred = np.empty_like(y_test)
    if len(short_val) > 0:
        val_pred[short_val] = model_short.predict(P_vl_short)
    if len(long_val) > 0:
        val_pred[long_val] = model_long.predict(P_vl_long)
    if len(short_test) > 0:
        test_pred[short_test] = model_short.predict(P_te_short)
    if len(long_test) > 0:
        test_pred[long_test] = model_long.predict(P_te_long)

    return {
        "candidate_id": candidate.candidate_id,
        "candidate": candidate,
        "model_short": model_short,
        "model_long": model_long,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "val_mae": float(mean_absolute_error(y_val, val_pred)),
        "val_rmse": float(root_mean_squared_error(y_val, val_pred)),
        "test_mae": float(mean_absolute_error(y_test, test_pred)),
        "test_rmse": float(root_mean_squared_error(y_test, test_pred)),
    }


def _hardware_dataset_specs(
    routing: Dict[str, np.ndarray],
) -> List[Tuple[str, np.ndarray, str]]:
    return [
        ("short_train", routing["X_train_res"][routing["short_train_idx"]], "short"),
        ("short_val", routing["X_val_res"][routing["short_val_idx"]], "short"),
        ("short_test", routing["X_test_res"][routing["short_test_idx"]], "short"),
        ("long_train", routing["X_train_res"][routing["long_train_idx"]], "long"),
        ("long_val", routing["X_val_res"][routing["long_val_idx"]], "long"),
        ("long_test", routing["X_test_res"][routing["long_test_idx"]], "long"),
    ]


def build_empty_feature_store(
    selected: Sequence[Dict],
    routing: Dict[str, np.ndarray],
    packing: PackingConfig,
) -> Dict[Tuple[int, str], np.ndarray]:
    feature_store: Dict[Tuple[int, str], np.ndarray] = {}
    for item in selected:
        cand: CandidateConfig = item["candidate"]
        for dataset_key, X_ds, _ in _hardware_dataset_specs(routing):
            n_events, n_obs = _reservoir_shape_info(X_ds, packing.n_qubits_per_block)
            feature_store[(cand.candidate_id, dataset_key)] = np.zeros(
                (len(X_ds), n_events * n_obs), dtype=float
            )
    return feature_store


def estimate_offline_hardware_runtime(
    *,
    selected: Sequence[Dict],
    routing: Dict[str, np.ndarray],
    packing: PackingConfig,
    shots: int,
    pubs_per_job: int = DEFAULT_RUNTIME_PUBS_PER_JOB,
    transpile_strategy: str = "reuse-z-basis",
    per_sub_job_overhead_seconds: float = DEFAULT_SUB_JOB_OVERHEAD_SECONDS,
    quick_seconds_per_execution: float = 0.00035,
) -> Dict[str, Optional[float]]:
    num_transpile_units = count_hardware_transpile_units(
        selected, routing, packing, transpile_strategy
    )
    if transpile_strategy == "independent":
        num_pubs_total = num_transpile_units
    else:
        num_pubs_total = num_transpile_units * 3

    num_executions = num_pubs_total * int(shots)
    num_sub_jobs = int(np.ceil(num_pubs_total / max(1, pubs_per_job)))
    quick_formula_seconds = float(quick_seconds_per_execution * num_executions)
    quantum_seconds = float(
        per_sub_job_overhead_seconds * num_sub_jobs + quick_formula_seconds
    )
    return {
        "estimate_type": "offline_quick_baseline",
        "quantum_seconds_estimated": quantum_seconds,
        "circuit_seconds_total": None,
        "duration_coverage_ratio": 0.0 if num_pubs_total else 1.0,
        "used_duration_extrapolation": None,
        "rep_delay_seconds": DEFAULT_REP_DELAY_SECONDS,
        "init_qubits_assumed": True,
        "init_seconds_total": None,
        "per_sub_job_overhead_seconds": float(per_sub_job_overhead_seconds),
        "num_sub_jobs_assumed": int(num_sub_jobs),
        "num_executions": int(num_executions),
        "num_pubs_total": int(num_pubs_total),
        "num_transpile_units": int(num_transpile_units),
        "missing_duration_pubs": int(num_pubs_total),
        "scheduling_optimization_level": None,
        "quick_formula_seconds": float(quick_formula_seconds),
    }


def _estimate_single_circuit_duration_seconds(
    circuit: QuantumCircuit,
    backend,
    scheduling_optimization_level: int,
) -> Optional[float]:
    if not hasattr(circuit, "estimate_duration"):
        return None
    target = getattr(backend, "target", None)
    if target is None:
        return None
    try:
        scheduled_pm = generate_preset_pass_manager(
            target=target,
            optimization_level=scheduling_optimization_level,
            scheduling_method="alap",
        )
        scheduled = scheduled_pm.run(circuit)
    except Exception:
        scheduled = circuit
    try:
        duration = scheduled.estimate_duration(target=target, unit="s")
    except Exception:
        return None
    if duration is None:
        return None
    return float(duration)


def estimate_aggregate_hardware_runtime(
    pub_plan_iterator,
    *,
    backend,
    shots: int,
    rep_delay_seconds: Optional[float] = None,
    init_qubits: Optional[bool] = None,
    per_sub_job_overhead_seconds: float = DEFAULT_SUB_JOB_OVERHEAD_SECONDS,
    scheduling_optimization_level: int = 0,
    pubs_per_job: int = DEFAULT_RUNTIME_PUBS_PER_JOB,
    allow_extrapolation: bool = False,
) -> Dict[str, Optional[float]]:
    resolved_rep_delay = float(
        _resolve_runtime_option(
            rep_delay_seconds,
            getattr(backend, "default_rep_delay", DEFAULT_REP_DELAY_SECONDS),
        )
    )
    init_qubits = bool(_resolve_runtime_option(init_qubits, True))

    reset_duration = 0.0
    target = getattr(backend, "target", None)
    if target is not None:
        try:
            reset_duration = float(target["reset"][(0,)].duration)
        except Exception:
            reset_duration = 0.0

    total_pubs = 0
    covered_pubs = 0
    covered_duration_total = 0.0
    missing_duration_pubs = 0

    for circuit, _meta in pub_plan_iterator:
        total_pubs += 1
        duration = _estimate_single_circuit_duration_seconds(
            circuit, backend, scheduling_optimization_level
        )
        if duration is None:
            missing_duration_pubs += 1
            continue
        covered_pubs += 1
        covered_duration_total += duration

    if total_pubs == 0:
        return {
            "quantum_seconds_estimated": 0.0,
            "circuit_seconds_total": 0.0,
            "duration_coverage_ratio": 1.0,
            "used_duration_extrapolation": 0.0,
            "rep_delay_seconds": resolved_rep_delay,
            "init_qubits_assumed": init_qubits,
            "init_seconds_total": 0.0,
            "per_sub_job_overhead_seconds": float(per_sub_job_overhead_seconds),
            "num_sub_jobs_assumed": 0,
            "num_executions": 0,
            "num_pubs_total": 0,
            "missing_duration_pubs": 0,
            "scheduling_optimization_level": int(scheduling_optimization_level),
            "quick_formula_seconds": 0.0,
        }

    used_extrapolation = 0.0
    if covered_pubs < total_pubs:
        if not allow_extrapolation:
            raise RuntimeError(
                "Aggregate runtime estimate is incomplete: "
                f"covered {covered_pubs}/{total_pubs} pubs. "
                "Re-run with --allow-runtime-estimate-extrapolation to extrapolate."
            )
        if covered_pubs > 0:
            mean_duration = covered_duration_total / covered_pubs
            covered_duration_total += mean_duration * (total_pubs - covered_pubs)
        used_extrapolation = 1.0

    init_seconds_total = reset_duration * total_pubs if init_qubits else 0.0
    total_circuit_seconds = (
        covered_duration_total + resolved_rep_delay * total_pubs + init_seconds_total
    )
    num_executions = total_pubs * int(shots)
    num_sub_jobs = int(np.ceil(total_pubs / max(1, pubs_per_job)))
    quantum_seconds = (
        per_sub_job_overhead_seconds * num_sub_jobs + total_circuit_seconds * shots
    )
    return {
        "quantum_seconds_estimated": float(quantum_seconds),
        "circuit_seconds_total": float(total_circuit_seconds),
        "duration_coverage_ratio": float(covered_pubs / total_pubs),
        "used_duration_extrapolation": used_extrapolation,
        "rep_delay_seconds": resolved_rep_delay,
        "init_qubits_assumed": init_qubits,
        "init_seconds_total": float(init_seconds_total),
        "per_sub_job_overhead_seconds": float(per_sub_job_overhead_seconds),
        "num_sub_jobs_assumed": int(num_sub_jobs),
        "num_executions": int(num_executions),
        "num_pubs_total": int(total_pubs),
        "missing_duration_pubs": int(missing_duration_pubs),
        "scheduling_optimization_level": int(scheduling_optimization_level),
        "quick_formula_seconds": float(
            per_sub_job_overhead_seconds + 0.00035 * num_executions
        ),
    }


def count_hardware_transpile_units(
    selected: Sequence[Dict],
    routing: Dict[str, np.ndarray],
    packing: PackingConfig,
    transpile_strategy: str,
) -> int:
    basis_multiplier = 3 if transpile_strategy == "independent" else 1
    total = 0
    for _ in selected:
        for _, X_ds, _ in _hardware_dataset_specs(routing):
            n_events, _ = _reservoir_shape_info(X_ds, packing.n_qubits_per_block)
            if len(X_ds) == 0:
                continue
            total += n_events * len(
                _split_indices(len(X_ds), packing.max_blocks_per_circuit)
            )
    return total * basis_multiplier


def _maybe_print_transpile_progress(
    *,
    completed: int,
    total: int,
    hits: int,
    misses: int,
    elapsed_seconds: float,
    log_every: int,
    force: bool = False,
) -> None:
    if log_every <= 0:
        return
    if not force and completed % log_every != 0:
        return
    rate = completed / elapsed_seconds if elapsed_seconds > 0 else 0.0
    remaining = (total - completed) / rate if rate > 0 else None
    remaining_text = _format_duration(remaining) if remaining is not None else "unknown"
    print(
        "Transpile progress: "
        f"{completed}/{total} units | cache hits={hits} misses={misses} | "
        f"elapsed={_format_duration(elapsed_seconds)} | eta={remaining_text}",
        flush=True,
    )


def _load_or_transpile_circuit(
    *,
    circuit: QuantumCircuit,
    pm,
    cache_dir: Optional[Path],
    cache_key: Optional[str],
    label: str,
    progress_completed: int,
    progress_total: int,
    log_cache_events: bool,
    fallback_cache_keys: Sequence[str] = (),
) -> Tuple[QuantumCircuit, bool, float]:
    if cache_dir is not None and cache_key is not None:
        for candidate_key in (cache_key, *fallback_cache_keys):
            cached = load_cached_transpiled_circuit(cache_dir, candidate_key)
            if cached is not None:
                if candidate_key != cache_key:
                    save_cached_transpiled_circuit(cache_dir, cache_key, cached)
                if log_cache_events:
                    print(
                        f"Transpile cache hit [{progress_completed + 1}/{progress_total}] {label}",
                        flush=True,
                    )
                return cached, True, 0.0

    print(
        f"Transpiling [{progress_completed + 1}/{progress_total}] {label}...",
        flush=True,
    )
    started = time.perf_counter()
    transpiled = pm.run(circuit)
    elapsed = time.perf_counter() - started
    print(
        f"Transpiled [{progress_completed + 1}/{progress_total}] {label} "
        f"in {_format_duration(elapsed)}",
        flush=True,
    )
    if cache_dir is not None and cache_key is not None:
        save_cached_transpiled_circuit(cache_dir, cache_key, transpiled)
    return transpiled, False, elapsed


def iter_hardware_pub_plans(
    selected: Sequence[Dict],
    routing: Dict[str, np.ndarray],
    packing: PackingConfig,
    num_layers: int,
    pm,
    transpile_strategy: str,
    backend_name: str,
    optimization_level: int,
    cache_dir: Optional[Path],
    code_hash: str,
    transpile_log_every: int,
) -> Iterator[Tuple[QuantumCircuit, PubSubmissionMeta]]:
    chunk_id = 0
    progress_completed = 0
    progress_hits = 0
    progress_misses = 0
    progress_started = time.perf_counter()
    progress_total = count_hardware_transpile_units(
        selected, routing, packing, transpile_strategy
    )

    dataset_specs = _hardware_dataset_specs(routing)
    print(
        "Hardware transpilation bundle: "
        f"{progress_total} transpile units | strategy={transpile_strategy} | "
        f"cache={'enabled' if cache_dir is not None else 'disabled'}",
        flush=True,
    )

    for item in selected:
        cand: CandidateConfig = item["candidate"]
        for dataset_key, X_ds, regime in dataset_specs:
            n_events, n_obs = _reservoir_shape_info(X_ds, packing.n_qubits_per_block)
            store_key = (cand.candidate_id, dataset_key)
            if len(X_ds) == 0:
                continue
            regime_params = cand.short_params if regime == "short" else cand.long_params
            params_digest = reservoir_params_digest(regime_params)
            block_template, block_params = build_parametric_block_circuit(
                regime_params, num_layers, packing.n_qubits_per_block
            )
            for event_idx in range(n_events):
                start_col = event_idx * packing.n_qubits_per_block
                end_col = start_col + packing.n_qubits_per_block
                X_event = X_ds[:, start_col:end_col]
                event_offset = event_idx * n_obs
                for start, end in _split_indices(len(X_event), packing.max_blocks_per_circuit):
                    chunk = X_event[start:end]
                    n_blocks = len(chunk)

                    def _cache_keys_for_basis(basis: str) -> List[str]:
                        return [
                            make_transpile_cache_key(
                                backend_name=backend_name,
                                optimization_level=optimization_level,
                                transpile_strategy=transpile_strategy,
                                candidate_id=cand.candidate_id,
                                regime=regime,
                                dataset_key=dataset_key,
                                event_idx=event_idx,
                                start=start,
                                end=end,
                                n_blocks=n_blocks,
                                num_layers=num_layers,
                                n_qubits_per_block=packing.n_qubits_per_block,
                                code_hash=cache_code_hash,
                                basis=basis,
                                chunk=chunk,
                                reservoir_params_digest=params_digest,
                            )
                            for cache_code_hash in (
                                code_hash,
                                *LEGACY_TRANSPILE_CACHE_CODE_HASHES,
                            )
                        ]

                    state = build_packed_state_circuit(chunk, block_template, block_params)
                    z_measured = with_measurement_basis(
                        state, "z", packing.n_qubits_per_block, n_blocks
                    )
                    z_keys = _cache_keys_for_basis("z")
                    label = (
                        f"candidate={cand.candidate_id} dataset={dataset_key} "
                        f"event={event_idx + 1}/{n_events} rows={start}:{end} basis=z"
                    )
                    z_circuit, cache_hit, _ = _load_or_transpile_circuit(
                        circuit=z_measured,
                        pm=pm,
                        cache_dir=cache_dir,
                        cache_key=z_keys[0],
                        label=label,
                        progress_completed=progress_completed,
                        progress_total=progress_total,
                        log_cache_events=transpile_log_every == 1,
                        fallback_cache_keys=z_keys[1:],
                    )
                    progress_completed += 1
                    progress_hits += int(cache_hit)
                    progress_misses += int(not cache_hit)
                    _maybe_print_transpile_progress(
                        completed=progress_completed,
                        total=progress_total,
                        hits=progress_hits,
                        misses=progress_misses,
                        elapsed_seconds=time.perf_counter() - progress_started,
                        log_every=transpile_log_every,
                    )
                    if transpile_strategy == "reuse-z-basis":
                        x_circuit = derive_transpiled_basis_circuit(z_circuit, "x")
                        y_circuit = derive_transpiled_basis_circuit(z_circuit, "y")
                    elif transpile_strategy == "independent":
                        x_measured = with_measurement_basis(
                            state, "x", packing.n_qubits_per_block, n_blocks
                        )
                        x_keys = _cache_keys_for_basis("x")
                        x_circuit, cache_hit, _ = _load_or_transpile_circuit(
                            circuit=x_measured,
                            pm=pm,
                            cache_dir=cache_dir,
                            cache_key=x_keys[0],
                            label=(
                                f"candidate={cand.candidate_id} dataset={dataset_key} "
                                f"event={event_idx + 1}/{n_events} rows={start}:{end} basis=x"
                            ),
                            progress_completed=progress_completed,
                            progress_total=progress_total,
                            log_cache_events=transpile_log_every == 1,
                            fallback_cache_keys=x_keys[1:],
                        )
                        progress_completed += 1
                        progress_hits += int(cache_hit)
                        progress_misses += int(not cache_hit)
                        _maybe_print_transpile_progress(
                            completed=progress_completed,
                            total=progress_total,
                            hits=progress_hits,
                            misses=progress_misses,
                            elapsed_seconds=time.perf_counter() - progress_started,
                            log_every=transpile_log_every,
                        )
                        y_measured = with_measurement_basis(
                            state, "y", packing.n_qubits_per_block, n_blocks
                        )
                        y_keys = _cache_keys_for_basis("y")
                        y_circuit, cache_hit, _ = _load_or_transpile_circuit(
                            circuit=y_measured,
                            pm=pm,
                            cache_dir=cache_dir,
                            cache_key=y_keys[0],
                            label=(
                                f"candidate={cand.candidate_id} dataset={dataset_key} "
                                f"event={event_idx + 1}/{n_events} rows={start}:{end} basis=y"
                            ),
                            progress_completed=progress_completed,
                            progress_total=progress_total,
                            log_cache_events=transpile_log_every == 1,
                            fallback_cache_keys=y_keys[1:],
                        )
                        progress_completed += 1
                        progress_hits += int(cache_hit)
                        progress_misses += int(not cache_hit)
                        _maybe_print_transpile_progress(
                            completed=progress_completed,
                            total=progress_total,
                            hits=progress_hits,
                            misses=progress_misses,
                            elapsed_seconds=time.perf_counter() - progress_started,
                            log_every=transpile_log_every,
                        )
                    else:
                        raise ValueError(
                            f"Unsupported transpile strategy: {transpile_strategy}"
                        )

                    for basis, circuit in (
                        ("z", z_circuit),
                        ("x", x_circuit),
                        ("y", y_circuit),
                    ):
                        yield circuit, PubSubmissionMeta(
                            chunk_id=chunk_id,
                            basis=basis,
                            matrix_key=store_key,
                            start_row=start,
                            end_row=end,
                            n_blocks=n_blocks,
                            event_offset=event_offset,
                        )
                    chunk_id += 1

    _maybe_print_transpile_progress(
        completed=progress_completed,
        total=progress_total,
        hits=progress_hits,
        misses=progress_misses,
        elapsed_seconds=time.perf_counter() - progress_started,
        log_every=max(1, transpile_log_every),
        force=True,
    )


def group_pubs_for_runtime_jobs(
    pub_plan_iterator: Iterator[Tuple[QuantumCircuit, PubSubmissionMeta]],
    pubs_per_job: int,
) -> Iterator[Tuple[List[QuantumCircuit], List[PubSubmissionMeta]]]:
    if pubs_per_job < 1:
        raise ValueError("pubs_per_job must be >= 1")
    batch_circuits: List[QuantumCircuit] = []
    batch_metas: List[PubSubmissionMeta] = []
    for circuit, meta in pub_plan_iterator:
        batch_circuits.append(circuit)
        batch_metas.append(meta)
        if len(batch_circuits) >= pubs_per_job:
            yield batch_circuits, batch_metas
            batch_circuits = []
            batch_metas = []
    if batch_circuits:
        yield batch_circuits, batch_metas


def decode_pub_results_into_feature_store(
    pub_results: Sequence,
    pub_metas: Sequence[PubSubmissionMeta],
    feature_store: Dict[Tuple[int, str], np.ndarray],
    n_qubits_per_block: int,
) -> int:
    if len(pub_results) != len(pub_metas):
        raise RuntimeError(
            "Pub result count does not match metadata: "
            f"{len(pub_results)} != {len(pub_metas)}"
        )

    chunk_counts: Dict[int, Dict[str, Dict[str, int]]] = {}
    chunk_meta: Dict[int, PubSubmissionMeta] = {}
    for i, meta in enumerate(pub_metas):
        if meta.chunk_id not in chunk_counts:
            chunk_counts[meta.chunk_id] = {}
            chunk_meta[meta.chunk_id] = meta
        chunk_counts[meta.chunk_id][meta.basis] = _counts_from_sampler_pub(pub_results[i])

    n_obs = 4 * n_qubits_per_block
    decoded_chunks = 0
    for chunk_id, counts_map in chunk_counts.items():
        if {"z", "x", "y"} - set(counts_map):
            raise RuntimeError(f"Missing basis counts for chunk_id={chunk_id}")
        meta = chunk_meta[chunk_id]
        matrix = feature_store[meta.matrix_key]
        for block_idx in range(meta.n_blocks):
            matrix[
                meta.start_row + block_idx,
                meta.event_offset : meta.event_offset + n_obs,
            ] = counts_to_expectations_block(
                counts_map["z"],
                counts_map["x"],
                counts_map["y"],
                block_idx,
                n_qubits_per_block,
            )
        decoded_chunks += 1
    return decoded_chunks


def submit_hardware_batch_jobs(
    pub_plan_iterator: Iterator[Tuple[QuantumCircuit, PubSubmissionMeta]],
    *,
    backend,
    shots: int,
    pubs_per_job: int,
) -> List[HardwareJobRecord]:
    job_records: List[HardwareJobRecord] = []
    submitted_jobs = 0
    submitted_pubs = 0

    with Batch(backend=backend) as batch:
        sampler = Sampler(mode=batch)
        sampler.options.default_shots = shots
        for batch_circuits, batch_metas in group_pubs_for_runtime_jobs(
            pub_plan_iterator, pubs_per_job
        ):
            job = sampler.run(batch_circuits)
            job_records.append(
                HardwareJobRecord(
                    job=job,
                    pub_metas=list(batch_metas),
                    num_pubs=len(batch_metas),
                )
            )
            submitted_jobs += 1
            submitted_pubs += len(batch_metas)
            print(
                f"Submitted Runtime job {submitted_jobs} "
                f"({len(batch_metas)} pubs, {submitted_pubs} pubs total)",
                flush=True,
            )
            del batch_circuits

    print(
        f"Batch submission complete: {submitted_jobs} jobs, {submitted_pubs} pubs",
        flush=True,
    )
    return job_records


def decode_all_hardware_job_records(
    job_records: Sequence[HardwareJobRecord],
    feature_store: Dict[Tuple[int, str], np.ndarray],
    n_qubits_per_block: int,
) -> Dict[str, int]:
    decoded_jobs = 0
    decoded_pubs = 0
    decoded_chunks = 0
    for record in job_records:
        pub_results = record.job.result()
        decoded_chunks += decode_pub_results_into_feature_store(
            pub_results,
            record.pub_metas,
            feature_store,
            n_qubits_per_block,
        )
        decoded_jobs += 1
        decoded_pubs += record.num_pubs
        print(
            f"Decoded Runtime job {decoded_jobs}/{len(job_records)} "
            f"({record.num_pubs} pubs, {decoded_pubs} pubs total, "
            f"{decoded_chunks} chunks)",
            flush=True,
        )
    return {
        "decoded_jobs": decoded_jobs,
        "decoded_pubs": decoded_pubs,
        "decoded_chunks": decoded_chunks,
    }


# ---------------------------------------------------------------------------
# Asynchronous / resumable submission (chunked batches + incremental decode).
#
# The flow is split into a submit phase (chunked Runtime batches whose job ids
# are persisted to a manifest as soon as each job is created) and a collect
# phase (reconnects to each job via ``service.job``, decodes completed jobs into
# the feature store, and checkpoints the feature store to disk after every job).
# Both phases are idempotent so a crashed run resumes by re-reading the manifest
# and checkpoint instead of re-submitting work or losing decoded results.
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def _pub_meta_to_dict(meta: PubSubmissionMeta) -> Dict:
    return {
        "chunk_id": int(meta.chunk_id),
        "basis": meta.basis,
        "matrix_key": [int(meta.matrix_key[0]), str(meta.matrix_key[1])],
        "start_row": int(meta.start_row),
        "end_row": int(meta.end_row),
        "n_blocks": int(meta.n_blocks),
        "event_offset": int(meta.event_offset),
    }


def _pub_meta_from_dict(payload: Mapping) -> PubSubmissionMeta:
    matrix_key = payload["matrix_key"]
    return PubSubmissionMeta(
        chunk_id=int(payload["chunk_id"]),
        basis=str(payload["basis"]),
        matrix_key=(int(matrix_key[0]), str(matrix_key[1])),
        start_row=int(payload["start_row"]),
        end_row=int(payload["end_row"]),
        n_blocks=int(payload["n_blocks"]),
        event_offset=int(payload["event_offset"]),
    )


def compute_resume_fingerprint(
    *,
    subset_frac: float,
    subset_seed: int,
    random_seed: int,
    n_candidates: int,
    ensemble_size: int,
    num_layers: int,
    shots: int,
    blocks_per_circuit: int,
    transpile_strategy: str,
    backend: str,
    load_selected_features: bool,
    optimization_level: int,
    reservoir_model: str,
    event_decay: float,
) -> str:
    """Hash the args that determine candidate selection, routing and the pub
    plan. A resumed run must match this fingerprint, otherwise the persisted job
    ids would no longer line up with the rebuilt feature store layout."""
    payload = {
        "subset_frac": float(subset_frac),
        "subset_seed": int(subset_seed),
        "random_seed": int(random_seed),
        "n_candidates": int(n_candidates),
        "ensemble_size": int(ensemble_size),
        "num_layers": int(num_layers),
        "shots": int(shots),
        "blocks_per_circuit": int(blocks_per_circuit),
        "transpile_strategy": str(transpile_strategy),
        "backend": str(backend),
        "load_selected_features": bool(load_selected_features),
        "optimization_level": int(optimization_level),
        "reservoir_model": str(reservoir_model),
        "event_decay": float(event_decay),
    }
    return _sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))


def load_submission_manifest(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_submission_manifest(path: Path, manifest: Mapping) -> None:
    _atomic_write_json(path, manifest)


def _feature_store_checkpoint_arrays(
    feature_store: Mapping[Tuple[int, str], np.ndarray]
) -> Dict[str, np.ndarray]:
    return {
        f"{int(cand_id)}::{dataset_key}": np.asarray(array)
        for (cand_id, dataset_key), array in feature_store.items()
    }


def save_feature_store_checkpoint(
    path: Path, feature_store: Mapping[Tuple[int, str], np.ndarray]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez(f, **_feature_store_checkpoint_arrays(feature_store))
    tmp.replace(path)


def load_feature_store_checkpoint(path: Path) -> Dict[Tuple[int, str], np.ndarray]:
    if not path.exists():
        return {}
    loaded: Dict[Tuple[int, str], np.ndarray] = {}
    with np.load(path) as data:
        for key in data.files:
            cand_str, dataset_key = key.split("::", 1)
            loaded[(int(cand_str), dataset_key)] = data[key]
    return loaded


def apply_feature_store_checkpoint(
    feature_store: Dict[Tuple[int, str], np.ndarray],
    loaded: Mapping[Tuple[int, str], np.ndarray],
) -> int:
    applied = 0
    for key, array in loaded.items():
        target = feature_store.get(key)
        if target is not None and target.shape == array.shape:
            target[...] = array
            applied += 1
    return applied


def iter_chunk_groups(
    pub_plan_iterator: Iterator[Tuple[QuantumCircuit, PubSubmissionMeta]],
) -> Iterator[List[Tuple[QuantumCircuit, PubSubmissionMeta]]]:
    """Group consecutive pubs that share a ``chunk_id`` (the z/x/y triple for one
    packed circuit). Keeping a chunk's bases together guarantees every Runtime
    job is independently decodable."""
    current_id: Optional[int] = None
    group: List[Tuple[QuantumCircuit, PubSubmissionMeta]] = []
    for circuit, meta in pub_plan_iterator:
        if current_id is None:
            current_id = meta.chunk_id
        if meta.chunk_id != current_id:
            yield group
            group = []
            current_id = meta.chunk_id
        group.append((circuit, meta))
    if group:
        yield group


def group_chunks_for_runtime_jobs(
    pub_plan_iterator: Iterator[Tuple[QuantumCircuit, PubSubmissionMeta]],
    chunks_per_job: int,
) -> Iterator[Tuple[List[QuantumCircuit], List[PubSubmissionMeta]]]:
    if chunks_per_job < 1:
        raise ValueError("chunks_per_job must be >= 1")
    job_circuits: List[QuantumCircuit] = []
    job_metas: List[PubSubmissionMeta] = []
    chunks_in_job = 0
    for chunk_group in iter_chunk_groups(pub_plan_iterator):
        for circuit, meta in chunk_group:
            job_circuits.append(circuit)
            job_metas.append(meta)
        chunks_in_job += 1
        if chunks_in_job >= chunks_per_job:
            yield job_circuits, job_metas
            job_circuits = []
            job_metas = []
            chunks_in_job = 0
    if job_circuits:
        yield job_circuits, job_metas


def chunks_per_job_from_pubs(pubs_per_job: int) -> int:
    return max(1, pubs_per_job // 3)


def plan_chunked_batches(
    *, num_pubs_total: int, pubs_per_job: int, num_batches: int
) -> Dict[str, int]:
    if num_batches < 1:
        raise ValueError("num_batches must be >= 1")
    chunks_per_job = chunks_per_job_from_pubs(pubs_per_job)
    total_chunks = int(num_pubs_total) // 3
    total_jobs = int(np.ceil(total_chunks / chunks_per_job)) if total_chunks else 0
    effective_batches = max(1, min(int(num_batches), total_jobs)) if total_jobs else 0
    jobs_per_batch = (
        int(np.ceil(total_jobs / effective_batches)) if effective_batches else 0
    )
    return {
        "chunks_per_job": int(chunks_per_job),
        "total_chunks": int(total_chunks),
        "total_jobs": int(total_jobs),
        "num_batches": int(effective_batches),
        "jobs_per_batch": int(jobs_per_batch),
    }


def estimate_per_job_quantum_seconds(
    *,
    pubs_per_job: int,
    shots: int,
    per_sub_job_overhead_seconds: float = DEFAULT_SUB_JOB_OVERHEAD_SECONDS,
    quick_seconds_per_execution: float = 0.00035,
) -> float:
    chunks_per_job = chunks_per_job_from_pubs(pubs_per_job)
    pubs = chunks_per_job * 3
    return float(
        per_sub_job_overhead_seconds + quick_seconds_per_execution * pubs * int(shots)
    )


def _extract_job_id(job) -> Optional[str]:
    raw = getattr(job, "job_id", None)
    if callable(raw):
        try:
            value = raw()
        except Exception:
            return None
        return None if value is None else str(value)
    if raw is not None:
        return str(raw)
    return None


def _job_status_str(job) -> str:
    status = job.status()
    # Modern qiskit-ibm-runtime returns a plain string (e.g. "DONE"); older
    # versions return a JobStatus enum whose ``str()`` is "JobStatus.DONE".
    name = getattr(status, "name", None)
    if isinstance(name, str):
        return name.upper()
    return str(status).upper()


def submit_hardware_chunked_batches(
    pub_plan_iterator: Iterator[Tuple[QuantumCircuit, PubSubmissionMeta]],
    *,
    backend,
    shots: int,
    chunks_per_job: int,
    jobs_per_batch: int,
    skip_jobs: int = 0,
    batch_max_time: Optional[int],
    on_job_submitted,
) -> int:
    """Submit whole-chunk jobs across a small number of sequential Runtime
    batches. ``on_job_submitted(job, batch_index, job_metas, job_index)`` is
    invoked immediately after each ``sampler.run`` so the caller can persist the
    job id to disk before any result is awaited. ``skip_jobs`` lets a resumed run
    re-iterate the (deterministic) pub plan and skip jobs that were already
    submitted, without re-submitting them."""
    if jobs_per_batch < 1:
        raise ValueError("jobs_per_batch must be >= 1")

    current_batch_cm = None
    current_batch_index = -1
    sampler = None
    job_index = 0
    submitted = 0

    def _close_batch() -> None:
        nonlocal current_batch_cm
        if current_batch_cm is not None:
            current_batch_cm.__exit__(None, None, None)
            current_batch_cm = None

    try:
        for job_circuits, job_metas in group_chunks_for_runtime_jobs(
            pub_plan_iterator, chunks_per_job
        ):
            if job_index < skip_jobs:
                job_index += 1
                continue
            target_batch_index = job_index // jobs_per_batch
            if target_batch_index != current_batch_index:
                _close_batch()
                batch_kwargs = {"backend": backend}
                if batch_max_time is not None:
                    batch_kwargs["max_time"] = int(batch_max_time)
                current_batch_cm = Batch(**batch_kwargs)
                batch = current_batch_cm.__enter__()
                sampler = Sampler(mode=batch)
                sampler.options.default_shots = shots
                current_batch_index = target_batch_index
            job = sampler.run(job_circuits)
            on_job_submitted(job, current_batch_index, job_metas, job_index)
            job_index += 1
            submitted += 1
    finally:
        _close_batch()
    return submitted


def collect_and_decode_hardware_jobs(
    *,
    service,
    manifest: Dict,
    manifest_path: Path,
    feature_store: Dict[Tuple[int, str], np.ndarray],
    checkpoint_path: Path,
    n_qubits_per_block: int,
    poll_interval_seconds: float,
) -> Dict[str, int]:
    """Reconnect to every submitted job and decode the completed ones into the
    feature store, checkpointing after each job. Safe to call repeatedly: jobs
    already marked ``done`` are skipped (their values were restored from the
    checkpoint)."""
    jobs = manifest["jobs"]
    decoded_jobs = sum(1 for entry in jobs if entry.get("status") == "done")
    decoded_pubs = sum(
        int(entry.get("num_pubs", 0)) for entry in jobs if entry.get("status") == "done"
    )
    decoded_chunks = 0
    errored: List[str] = [
        str(entry.get("job_id")) for entry in jobs if entry.get("status") == "error"
    ]

    total_jobs = len(jobs)
    while True:
        progressed = False
        outstanding = 0
        for entry in jobs:
            status = entry.get("status")
            if status in ("done", "error"):
                continue
            job_id = entry.get("job_id")
            if not job_id:
                entry["status"] = "error"
                entry["error_message"] = "missing job id in manifest"
                errored.append(str(job_id))
                save_submission_manifest(manifest_path, manifest)
                progressed = True
                continue
            job = service.job(job_id)
            job_status = _job_status_str(job)
            if job_status == _JOB_STATUS_DONE:
                pub_results = job.result()
                pub_metas = [_pub_meta_from_dict(m) for m in entry["pub_metas"]]
                decoded_chunks += decode_pub_results_into_feature_store(
                    pub_results,
                    pub_metas,
                    feature_store,
                    n_qubits_per_block,
                )
                save_feature_store_checkpoint(checkpoint_path, feature_store)
                entry["status"] = "done"
                save_submission_manifest(manifest_path, manifest)
                decoded_jobs += 1
                decoded_pubs += int(entry.get("num_pubs", 0))
                progressed = True
                print(
                    f"Decoded Runtime job {decoded_jobs}/{total_jobs} "
                    f"(batch={entry.get('batch_index')}, "
                    f"{entry.get('num_pubs')} pubs, job_id={job_id})",
                    flush=True,
                )
            elif job_status in _JOB_STATUS_ERROR_STATES:
                entry["status"] = "error"
                entry["error_message"] = job_status
                errored.append(str(job_id))
                save_submission_manifest(manifest_path, manifest)
                progressed = True
                print(
                    f"Runtime job failed (status={job_status}, job_id={job_id}, "
                    f"batch={entry.get('batch_index')})",
                    flush=True,
                )
            else:
                outstanding += 1
        if outstanding == 0:
            break
        if not progressed:
            print(
                f"Waiting on {outstanding} outstanding Runtime job(s); "
                f"{decoded_jobs}/{total_jobs} decoded. "
                f"Sleeping {poll_interval_seconds:.0f}s...",
                flush=True,
            )
            time.sleep(poll_interval_seconds)
    return {
        "decoded_jobs": int(decoded_jobs),
        "decoded_pubs": int(decoded_pubs),
        "decoded_chunks": int(decoded_chunks),
        "errored_jobs": list(errored),
        "num_jobs": int(total_jobs),
    }


def save_final_hardware_pipeline(
    output_path: Path,
    *,
    classifier,
    selected: Sequence[Dict],
    split: Dict,
    routing: Dict[str, np.ndarray],
    feature_store: Dict[Tuple[int, str], np.ndarray],
    hw_results: Sequence[Dict],
    ensemble_pred: np.ndarray,
    summary: Dict,
    ranking_payload: Sequence[Dict],
) -> None:
    selected_candidates = [
        {
            "candidate_id": int(item["candidate"].candidate_id),
            "short_params": item["candidate"].short_params,
            "long_params": item["candidate"].long_params,
            "seed": int(item["candidate"].seed),
        }
        for item in selected
    ]
    hardware_regressors = [
        {
            "candidate_id": int(r["candidate_id"]),
            "model_short": r["model_short"],
            "model_long": r["model_long"],
            "regressor_optuna_best_params": r["regressor_optuna_best_params"],
        }
        for r in hw_results
    ]
    payload = {
        "classifier": classifier,
        "selected_candidates": selected_candidates,
        "reservoir_scaler": {
            "train_min": split["reservoir_train_min"],
            "train_max": split["reservoir_train_max"],
        },
        "reservoir_feature_metadata": {
            "sequence_cols": split["reservoir_sequence_cols"],
            "n_events": int(split["reservoir_n_events"]),
            "n_features_per_event": int(split["reservoir_n_features_per_event"]),
            "encoding": "sequential_chronological",
        },
        "routing_indices": {
            key: routing[key]
            for key in (
                "short_train_idx",
                "short_val_idx",
                "short_test_idx",
                "long_train_idx",
                "long_val_idx",
                "long_test_idx",
            )
        },
        "feature_store_keys": list(feature_store.keys()),
        "hardware_regressors": hardware_regressors,
        "hardware_results_metrics": [
            {k: v for k, v in r.items() if k not in {"model_short", "model_long", "test_pred"}}
            for r in hw_results
        ],
        "ensemble_test_pred": ensemble_pred,
        "summary": summary,
        "simulation_ranking": list(ranking_payload),
    }
    with output_path.open("wb") as f:
        pickle.dump(payload, f)


def build_hardware_submission_bundle(
    selected: Sequence[Dict],
    routing: Dict[str, np.ndarray],
    packing: PackingConfig,
    num_layers: int,
    pm,
    transpile_strategy: str,
    backend_name: str,
    optimization_level: int,
    cache_dir: Optional[Path],
    code_hash: str,
    transpile_log_every: int,
) -> Tuple[List[QuantumCircuit], List[PubMeta], Dict[Tuple[int, str], np.ndarray], Dict[int, ChunkRequest]]:
    circuits: List[QuantumCircuit] = []
    metas: List[PubMeta] = []
    chunk_requests: Dict[int, ChunkRequest] = {}
    feature_store = build_empty_feature_store(selected, routing, packing)
    for circuit, pub_meta in iter_hardware_pub_plans(
        selected=selected,
        routing=routing,
        packing=packing,
        num_layers=num_layers,
        pm=pm,
        transpile_strategy=transpile_strategy,
        backend_name=backend_name,
        optimization_level=optimization_level,
        cache_dir=cache_dir,
        code_hash=code_hash,
        transpile_log_every=transpile_log_every,
    ):
        circuits.append(circuit)
        metas.append(PubMeta(chunk_id=pub_meta.chunk_id, basis=pub_meta.basis))
        if pub_meta.basis == "z":
            chunk_requests[pub_meta.chunk_id] = ChunkRequest(
                chunk_id=pub_meta.chunk_id,
                matrix_key=pub_meta.matrix_key,
                start_row=pub_meta.start_row,
                end_row=pub_meta.end_row,
                n_blocks=pub_meta.n_blocks,
                event_offset=pub_meta.event_offset,
            )
    return circuits, metas, feature_store, chunk_requests


def train_and_score_from_hardware_features(
    selected: Sequence[Dict],
    routing: Dict[str, np.ndarray],
    feature_store: Dict[Tuple[int, str], np.ndarray],
    regressor_optuna_trials: int = 0,
    regressor_optuna_cv_folds: int = 3,
) -> Tuple[List[Dict], np.ndarray]:
    y_train = routing["y_train"]
    y_val = routing["y_val"]
    y_test = routing["y_test"]

    results: List[Dict] = []
    for item in selected:
        cand: CandidateConfig = item["candidate"]
        cid = cand.candidate_id
        P_tr_short = feature_store[(cid, "short_train")]
        P_vl_short = feature_store[(cid, "short_val")]
        P_te_short = feature_store[(cid, "short_test")]
        P_tr_long = feature_store[(cid, "long_train")]
        P_vl_long = feature_store[(cid, "long_val")]
        P_te_long = feature_store[(cid, "long_test")]

        short_base_params = item["model_short"].get_params()
        long_base_params = item["model_long"].get_params()
        short_train_idx = routing["short_train_idx"]
        long_train_idx = routing["long_train_idx"]
        tuned_short = tune_regime_regressor_params(
            P_tr_short,
            y_train[short_train_idx],
            seed=int(short_base_params.get("random_state", cand.seed)),
            n_trials=regressor_optuna_trials,
            n_splits=regressor_optuna_cv_folds,
        )
        tuned_long = tune_regime_regressor_params(
            P_tr_long,
            y_train[long_train_idx],
            seed=int(long_base_params.get("random_state", cand.seed + 1000)),
            n_trials=regressor_optuna_trials,
            n_splits=regressor_optuna_cv_folds,
        )
        short_base_params.update(tuned_short)
        long_base_params.update(tuned_long)
        model_short = XGBRegressor(**short_base_params)
        model_long = XGBRegressor(**long_base_params)
        short_val_idx = routing["short_val_idx"]
        long_val_idx = routing["long_val_idx"]
        eval_short = [(P_vl_short, y_val[short_val_idx])] if len(short_val_idx) > 0 else None
        eval_long = [(P_vl_long, y_val[long_val_idx])] if len(long_val_idx) > 0 else None
        model_short.fit(
            P_tr_short,
            y_train[short_train_idx],
            eval_set=eval_short,
            verbose=False,
        )
        model_long.fit(
            P_tr_long,
            y_train[long_train_idx],
            eval_set=eval_long,
            verbose=False,
        )

        val_pred = np.empty_like(y_val)
        test_pred = np.empty_like(y_test)
        if len(routing["short_val_idx"]) > 0:
            val_pred[routing["short_val_idx"]] = model_short.predict(P_vl_short)
        if len(routing["long_val_idx"]) > 0:
            val_pred[routing["long_val_idx"]] = model_long.predict(P_vl_long)
        if len(routing["short_test_idx"]) > 0:
            test_pred[routing["short_test_idx"]] = model_short.predict(P_te_short)
        if len(routing["long_test_idx"]) > 0:
            test_pred[routing["long_test_idx"]] = model_long.predict(P_te_long)

        results.append(
            {
                "candidate_id": cid,
                "val_mae": float(mean_absolute_error(y_val, val_pred)),
                "val_rmse": float(root_mean_squared_error(y_val, val_pred)),
                "val_r2": float(r2_score(y_val, val_pred)),
                "test_mae": float(mean_absolute_error(y_test, test_pred)),
                "test_rmse": float(root_mean_squared_error(y_test, test_pred)),
                "test_r2": float(r2_score(y_test, test_pred)),
                "test_pred": test_pred,
                "model_short": model_short,
                "model_long": model_long,
                "regressor_optuna_best_params": {
                    "short": tuned_short,
                    "long": tuned_long,
                },
            }
        )
    ensemble_pred = np.mean([r["test_pred"] for r in results], axis=0)
    return results, ensemble_pred


def main() -> None:
    args, parser = parse_args()
    if not (0.0 < args.subset_frac <= 1.0):
        parser.error("--subset-frac must satisfy 0 < subset-frac <= 1.")
        # raise ValueError("--subset-frac must satisfy 0 < subset-frac <= 1.")
    # if args.ensemble_size != 2:
    #     parser.error("Ensemble size is locked to 2 for this migration.")
    #     # raise ValueError("Ensemble size is locked to 2 for this migration.")
    if args.n_candidates != 5:
        parser.error("Candidate search is locked to 5 noiseless candidates.")
        # raise ValueError("Candidate search is locked to 5 noiseless candidates.")
    if args.classifier_optuna_trials < 0:
        parser.error("--classifier-optuna-trials must be >= 0.")
    if args.regressor_optuna_trials < 0:
        parser.error("--regressor-optuna-trials must be >= 0.")
    if args.regressor_optuna_cv_folds < 2:
        parser.error("--regressor-optuna-cv-folds must be >= 2.")
    if args.execution_limit <= 0:
        parser.error("--execution-limit must be > 0 seconds.")
    if not (0.0 < args.event_decay <= 1.0):
        parser.error("--event-decay must satisfy 0 < decay <= 1.")
    if args.transpile_log_every < 0:
        parser.error("--transpile-log-every must be >= 0.")
    if args.runtime_pubs_per_job < 1:
        parser.error("--runtime-pubs-per-job must be >= 1.")
    if args.transpile_only and args.simulate_only:
        parser.error("--transpile-only and --simulate-only are mutually exclusive.")
    if args.transpile_only and args.no_transpile_cache:
        parser.error(
            "--transpile-only without the transpile cache would discard all "
            "work; remove --no-transpile-cache."
        )
    if not args.simulate_only and not args.token:
        parser.error("--token is required when it is not a simulate-only run")
        # raise ValueError("--token is required when using --hardware")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = load_split_data(
        args.subset_frac,
        args.subset_seed,
        args.load_selected_features,
        event_decay=args.event_decay,
    )

    clf = train_regime_classifier(
        split["X_train_hist"],
        split["y_train"],
        args.random_seed,
        optuna_trials=args.classifier_optuna_trials,
    )
    routing = build_routing_data(
        clf,
        split["X_train_hist"],
        split["X_val_hist"],
        split["X_test_hist"],
        split["X_train_res"],
        split["X_val_res"],
        split["X_test_res"],
        split["y_train"],
        split["y_val"],
        split["y_test"],
    )

    # Candidate search uses noiseless simulation only.
    # Keep simulation strictly at one 6-qubit block per circuit; do not couple this
    # path to backend qubit capacity (e.g., 156-qubit Fez packing).
    sim_packing = PackingConfig(
        n_qubits_per_block=RESERVOIR_N_QUBITS,
        max_blocks_per_circuit=SIMULATION_MAX_BLOCKS_PER_CIRCUIT,
    )
    candidates = create_candidates(
        n_candidates=args.n_candidates,
        n_qubits=RESERVOIR_N_QUBITS,
        seed=args.random_seed,
        model=args.reservoir_model,
    )
    candidate_scores = [
        evaluate_candidate(c, routing, args.num_layers, args.shots, sim_packing)
        for c in candidates
    ]
    candidate_scores.sort(key=lambda row: row["val_mae"])
    selected = candidate_scores[: args.ensemble_size]

    ranking_payload = [
        {
            "candidate_id": int(row["candidate_id"]),
            "val_mae": float(row["val_mae"]),
            "val_rmse": float(row["val_rmse"]),
            "test_mae": float(row["test_mae"]),
            "test_rmse": float(row["test_rmse"]),
        }
        for row in candidate_scores
    ]
    save_json(args.output_dir / "simulation_candidate_ranking.json", ranking_payload)
    print("Noiseless candidate ranking complete.")
    for row in ranking_payload:
        marker = "*" if row["candidate_id"] in {x["candidate_id"] for x in selected} else " "
        print(
            f"{marker} candidate={row['candidate_id']} "
            f"val_mae={row['val_mae']:.2f} test_mae={row['test_mae']:.2f}"
        )

    if args.simulate_only:
        print("simulate-only enabled; skipping hardware submission.")
        return

    backend_qubits = KNOWN_BACKEND_QUBITS.get(args.backend, DEFAULT_HARDWARE_QUBITS)
    max_blocks_supported = max(1, backend_qubits // RESERVOIR_N_QUBITS)
    # Hardware execution can either auto-pack across backend width or be
    # constrained by --parallel-circuits for tighter runtime control.
    if args.parallel_circuits is None:
        blocks_per_circuit = max_blocks_supported
    else:
        if args.parallel_circuits < 1:
            parser.error("--parallel-circuits must be >= 1.")
        if args.parallel_circuits > max_blocks_supported:
            parser.error(
                "--parallel-circuits exceeds backend capacity: "
                f"requested {args.parallel_circuits}, max {max_blocks_supported} "
                f"for backend={args.backend} ({backend_qubits} qubits)."
            )
        blocks_per_circuit = args.parallel_circuits
    hw_packing = PackingConfig(
        n_qubits_per_block=RESERVOIR_N_QUBITS,
        max_blocks_per_circuit=blocks_per_circuit,
    )

    transpile_cache_dir = None
    if not args.no_transpile_cache:
        transpile_cache_dir = args.transpile_cache_dir or (args.output_dir / "transpile_cache")
    code_hash = f"transpile-cache-v{TRANSPILE_CACHE_VERSION}"

    if args.transpile_only:
        print(
            "Connecting to IBM Runtime for transpile-only run "
            "(no jobs will be submitted)...",
            flush=True,
        )
        service = QiskitRuntimeService(token=args.token)
        backend = service.backend(args.backend)
        actual_backend_qubits = getattr(backend, "num_qubits", backend_qubits)
        actual_max_blocks_supported = max(1, actual_backend_qubits // RESERVOIR_N_QUBITS)
        if blocks_per_circuit > actual_max_blocks_supported:
            parser.error(
                "--parallel-circuits exceeds resolved backend capacity: "
                f"using {blocks_per_circuit}, max {actual_max_blocks_supported} "
                f"for backend={args.backend} ({actual_backend_qubits} qubits)."
            )
        pm = generate_preset_pass_manager(
            backend=backend, optimization_level=args.optimization_level
        )
        started = time.perf_counter()
        num_pubs = 0
        for _circuit, _meta in iter_hardware_pub_plans(
            selected=selected,
            routing=routing,
            packing=hw_packing,
            num_layers=args.num_layers,
            pm=pm,
            transpile_strategy=args.transpile_strategy,
            backend_name=args.backend,
            optimization_level=args.optimization_level,
            cache_dir=transpile_cache_dir,
            code_hash=code_hash,
            transpile_log_every=args.transpile_log_every,
        ):
            num_pubs += 1
        print(
            f"Transpile-only complete: {num_pubs} pubs prepared at "
            f"optimization_level={args.optimization_level} in "
            f"{_format_duration(time.perf_counter() - started)}. "
            f"Cache dir: {transpile_cache_dir}",
            flush=True,
        )
        print("Exiting before IBM submission (--transpile-only).")
        return

    feature_store = build_empty_feature_store(selected, routing, hw_packing)

    hardware_jobs_dir = args.output_dir / HARDWARE_JOBS_DIRNAME
    manifest_path = hardware_jobs_dir / SUBMISSION_MANIFEST_NAME
    checkpoint_path = hardware_jobs_dir / FEATURE_CHECKPOINT_NAME
    resume_fingerprint = compute_resume_fingerprint(
        subset_frac=args.subset_frac,
        subset_seed=args.subset_seed,
        random_seed=args.random_seed,
        n_candidates=args.n_candidates,
        ensemble_size=args.ensemble_size,
        num_layers=args.num_layers,
        shots=args.shots,
        blocks_per_circuit=blocks_per_circuit,
        transpile_strategy=args.transpile_strategy,
        backend=args.backend,
        load_selected_features=args.load_selected_features,
        optimization_level=args.optimization_level,
        reservoir_model=args.reservoir_model,
        event_decay=args.event_decay,
    )

    print(
        "Computing offline quick runtime estimate before IBM Runtime connection...",
        flush=True,
    )
    runtime_estimate = estimate_offline_hardware_runtime(
        selected=selected,
        routing=routing,
        packing=hw_packing,
        shots=args.shots,
        pubs_per_job=args.runtime_pubs_per_job,
        transpile_strategy=args.transpile_strategy,
    )
    num_pubs_total = int(runtime_estimate.get("num_pubs_total", 0))
    if num_pubs_total == 0:
        raise ValueError("No hardware pubs were built; cannot submit an empty workload.")

    # Each individual Runtime job must stay under IBM's per-job quantum-time cap
    # (3h). The TOTAL workload may legitimately exceed this: it is spread across
    # many jobs and several sequential batches.
    per_job_seconds = estimate_per_job_quantum_seconds(
        pubs_per_job=args.runtime_pubs_per_job, shots=args.shots
    )
    per_job_cap = min(float(args.execution_limit), float(IBM_MAX_JOB_SECONDS))
    if per_job_seconds > per_job_cap:
        raise RuntimeError(
            "Estimated per-job quantum time exceeds the per-job limit: "
            f"per_job={_format_duration(per_job_seconds)} "
            f"({per_job_seconds:.1f}s) > limit={_format_duration(per_job_cap)} "
            f"({per_job_cap:.1f}s). Lower --runtime-pubs-per-job or --shots."
        )

    batch_plan = plan_chunked_batches(
        num_pubs_total=num_pubs_total,
        pubs_per_job=args.runtime_pubs_per_job,
        num_batches=args.num_batches,
    )

    existing_manifest = None
    if not args.fresh_submit:
        existing_manifest = load_submission_manifest(manifest_path)
    resume_mode = existing_manifest is not None
    if resume_mode:
        if existing_manifest.get("resume_fingerprint") != resume_fingerprint:
            raise RuntimeError(
                "Existing submission manifest does not match the current "
                f"configuration (fingerprint mismatch) at {manifest_path}. "
                "Use a fresh --output-dir, or pass --fresh-submit to discard the "
                "previous jobs and resubmit."
            )
        manifest = existing_manifest
        loaded_checkpoint = load_feature_store_checkpoint(checkpoint_path)
        applied = apply_feature_store_checkpoint(feature_store, loaded_checkpoint)
        done_jobs = sum(1 for e in manifest["jobs"] if e.get("status") == "done")
        print(
            f"Resuming from existing manifest at {manifest_path}: "
            f"{len(manifest['jobs'])} submitted job(s), {done_jobs} already "
            f"decoded, {applied} feature block(s) restored from checkpoint.",
            flush=True,
        )
    else:
        if not args.no_confirm:
            print(
                f"Preparing chunked-batch hardware Runtime submission | "
                f"backend={args.backend} | shots={args.shots} | "
                f"blocks/circuit={blocks_per_circuit} | "
                f"pubs/job={args.runtime_pubs_per_job} | "
                f"selected={ [int(r['candidate_id']) for r in selected] }"
            )
            print(
                "Estimated TOTAL quantum runtime: "
                f"{_format_duration(runtime_estimate['quantum_seconds_estimated'])} "
                f"(pubs={num_pubs_total})"
            )
            print(
                "Per-job estimate: "
                f"{_format_duration(per_job_seconds)} "
                f"(cap {_format_duration(per_job_cap)})"
            )
            print(
                f"Batches: {batch_plan['num_batches']} | "
                f"jobs: {batch_plan['total_jobs']} | "
                f"jobs/batch: {batch_plan['jobs_per_batch']} | "
                f"chunks/job: {batch_plan['chunks_per_job']}"
            )
            if input("Continue? [y/N]: ").strip().lower() not in {"y", "yes"}:
                print("Aborted.")
                return
        manifest = {
            "version": SUBMISSION_MANIFEST_VERSION,
            "resume_fingerprint": resume_fingerprint,
            "backend": args.backend,
            "shots": int(args.shots),
            "runtime_pubs_per_job": int(args.runtime_pubs_per_job),
            "n_qubits_per_block": int(hw_packing.n_qubits_per_block),
            "num_pubs_total": num_pubs_total,
            "chunks_per_job": batch_plan["chunks_per_job"],
            "jobs_per_batch": batch_plan["jobs_per_batch"],
            "num_batches": batch_plan["num_batches"],
            "total_jobs": batch_plan["total_jobs"],
            "batch_max_time": args.batch_max_time,
            "selected_candidate_ids": [int(r["candidate_id"]) for r in selected],
            "submission_complete": False,
            "jobs": [],
        }
        save_submission_manifest(manifest_path, manifest)

    print("Connecting to IBM Runtime...", flush=True)
    service = QiskitRuntimeService(token=args.token)  # , instance="Icequakes-QRC-run1")
    backend = service.backend(args.backend)
    actual_backend_qubits = getattr(backend, "num_qubits", backend_qubits)
    actual_max_blocks_supported = max(1, actual_backend_qubits // RESERVOIR_N_QUBITS)
    if blocks_per_circuit > actual_max_blocks_supported:
        parser.error(
            "--parallel-circuits exceeds resolved backend capacity: "
            f"using {blocks_per_circuit}, max {actual_max_blocks_supported} "
            f"for backend={args.backend} ({actual_backend_qubits} qubits)."
        )

    if not manifest.get("submission_complete", False):
        pm = generate_preset_pass_manager(
            backend=backend, optimization_level=args.optimization_level
        )

        def make_pub_plan_iterator() -> Iterator[Tuple[QuantumCircuit, PubSubmissionMeta]]:
            return iter_hardware_pub_plans(
                selected=selected,
                routing=routing,
                packing=hw_packing,
                num_layers=args.num_layers,
                pm=pm,
                transpile_strategy=args.transpile_strategy,
                backend_name=args.backend,
                optimization_level=args.optimization_level,
                cache_dir=transpile_cache_dir,
                code_hash=code_hash,
                transpile_log_every=args.transpile_log_every,
            )

        skip_jobs = len(manifest["jobs"])
        if skip_jobs:
            print(
                f"Resuming submission; skipping {skip_jobs} already-submitted "
                "job(s) without resubmitting them.",
                flush=True,
            )

        def on_job_submitted(job, batch_index, job_metas, job_index) -> None:
            job_id = _extract_job_id(job)
            manifest["jobs"].append(
                {
                    "job_index": int(job_index),
                    "batch_index": int(batch_index),
                    "job_id": job_id,
                    "num_pubs": len(job_metas),
                    "status": "submitted",
                    "pub_metas": [_pub_meta_to_dict(m) for m in job_metas],
                }
            )
            save_submission_manifest(manifest_path, manifest)
            print(
                f"Submitted Runtime job {len(manifest['jobs'])}/"
                f"{manifest['total_jobs']} (batch={batch_index}, "
                f"{len(job_metas)} pubs, job_id={job_id})",
                flush=True,
            )

        submit_hardware_chunked_batches(
            make_pub_plan_iterator(),
            backend=backend,
            shots=args.shots,
            chunks_per_job=int(manifest["chunks_per_job"]),
            jobs_per_batch=int(manifest["jobs_per_batch"]),
            skip_jobs=skip_jobs,
            batch_max_time=manifest.get("batch_max_time"),
            on_job_submitted=on_job_submitted,
        )
        manifest["submission_complete"] = True
        save_submission_manifest(manifest_path, manifest)
        print(
            f"Submission complete: {len(manifest['jobs'])} job(s) across "
            f"{manifest['num_batches']} batch(es).",
            flush=True,
        )

    decode_stats = collect_and_decode_hardware_jobs(
        service=service,
        manifest=manifest,
        manifest_path=manifest_path,
        feature_store=feature_store,
        checkpoint_path=checkpoint_path,
        n_qubits_per_block=hw_packing.n_qubits_per_block,
        poll_interval_seconds=args.collect_poll_seconds,
    )
    errored_jobs = decode_stats.get("errored_jobs", [])
    if errored_jobs and not args.allow_partial_results:
        raise RuntimeError(
            f"{len(errored_jobs)} Runtime job(s) ended in an error state: "
            f"{errored_jobs}. Re-run the same command to retry collection, or "
            "pass --allow-partial-results to finalize with zero-filled feature "
            "rows for the failed jobs."
        )

    usage_estimation = None
    done_job_ids = [
        e.get("job_id") for e in manifest["jobs"] if e.get("status") == "done"
    ]
    if done_job_ids:
        try:
            usage_estimation = getattr(
                service.job(done_job_ids[-1]), "usage_estimation", None
            )
        except Exception:
            usage_estimation = None

    hw_results, ensemble_pred = train_and_score_from_hardware_features(
        selected=selected,
        routing=routing,
        feature_store=feature_store,
        regressor_optuna_trials=args.regressor_optuna_trials,
        regressor_optuna_cv_folds=args.regressor_optuna_cv_folds,
    )
    summary = {
        "backend": args.backend,
        "shots": args.shots,
        "subset_frac": args.subset_frac,
        "subset_seed": args.subset_seed,
        "classifier_history_events": CLASSIFIER_N_PREVIOUS_EVENTS + 1,
        "classifier_regime_labeling": "kmeans_on_y_train",
        "classifier_optuna_trials": args.classifier_optuna_trials,
        "regressor_optuna_trials": args.regressor_optuna_trials,
        "regressor_optuna_cv_folds": args.regressor_optuna_cv_folds,
        "reservoir_features": (
            "selected_handcrafted_single_event"
            if args.load_selected_features
            else "all_sequential"
        ),
        "reservoir_feature_count_per_event": int(split["reservoir_n_features_per_event"]),
        "reservoir_event_count": int(split["reservoir_n_events"]),
        "reservoir_model": args.reservoir_model,
        "reservoir_event_decay": float(args.event_decay),
        "reservoir_encoding": (
            "handcrafted_single_event"
            if args.load_selected_features
            else "sequential_chronological"
        ),
        "candidate_search_count": args.n_candidates,
        "ensemble_size": args.ensemble_size,
        "parallel_circuits_requested": args.parallel_circuits,
        "parallel_circuits_used": blocks_per_circuit,
        "parallel_circuits_backend_max": max_blocks_supported,
        "transpile_strategy": args.transpile_strategy,
        "optimization_level": int(args.optimization_level),
        "transpile_cache_enabled": transpile_cache_dir is not None,
        "transpile_cache_dir": str(transpile_cache_dir) if transpile_cache_dir else None,
        "transpile_cache_version": TRANSPILE_CACHE_VERSION,
        "runtime_pubs_per_job": int(args.runtime_pubs_per_job),
        "runtime_batch_decode_stats": decode_stats,
        "runtime_num_batches": int(manifest.get("num_batches", 0)),
        "runtime_jobs_per_batch": int(manifest.get("jobs_per_batch", 0)),
        "runtime_chunks_per_job": int(manifest.get("chunks_per_job", 0)),
        "runtime_errored_job_ids": list(decode_stats.get("errored_jobs", [])),
        "submission_manifest_path": str(manifest_path),
        "feature_store_checkpoint_path": str(checkpoint_path),
        "execution_limit_seconds": float(args.execution_limit),
        "selected_candidate_ids": [int(row["candidate_id"]) for row in selected],
        "results_per_candidate": [
            {
                "candidate_id": int(r["candidate_id"]),
                "val_mae": float(r["val_mae"]),
                "val_rmse": float(r["val_rmse"]),
                "val_r2": float(r["val_r2"]),
                "test_mae": float(r["test_mae"]),
                "test_rmse": float(r["test_rmse"]),
                "test_r2": float(r["test_r2"]),
            }
            for r in hw_results
        ],
        "ensemble_test_mae": float(mean_absolute_error(routing["y_test"], ensemble_pred)),
        "ensemble_test_rmse": float(root_mean_squared_error(routing["y_test"], ensemble_pred)),
        "ensemble_test_r2": float(r2_score(routing["y_test"], ensemble_pred)),
        "pre_submit_runtime_estimate": runtime_estimate,
        "runtime_per_job_seconds_estimated": float(per_job_seconds),
        "runtime_usage_estimation": usage_estimation,
        "runtime_job_ids": [e.get("job_id") for e in manifest["jobs"]],
        "final_hardware_pipeline_path": str(
            args.output_dir / "final_hardware_pipeline.pkl"
        ),
    }
    save_json(args.output_dir / args.summary_name, summary)
    pipeline_path = args.output_dir / "final_hardware_pipeline.pkl"
    save_final_hardware_pipeline(
        pipeline_path,
        classifier=clf,
        selected=selected,
        split=split,
        routing=routing,
        feature_store=feature_store,
        hw_results=hw_results,
        ensemble_pred=ensemble_pred,
        summary=summary,
        ranking_payload=ranking_payload,
    )
    with (args.output_dir / "hardware_results.pkl").open("wb") as f:
        pickle.dump(
            {
                "summary": summary,
                "simulation_ranking": ranking_payload,
                "hardware_results": [
                    {k: v for k, v in r.items() if k not in {"model_short", "model_long"}}
                    for r in hw_results
                ],
                "ensemble_pred": ensemble_pred,
                "final_hardware_pipeline_path": str(pipeline_path),
            },
            f,
        )
    print(f"Saved summary -> {args.output_dir / args.summary_name}")
    print(f"Saved final pipeline -> {pipeline_path}")


if __name__ == "__main__":
    main()
