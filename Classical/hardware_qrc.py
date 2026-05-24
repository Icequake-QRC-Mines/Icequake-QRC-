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
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.transpiler.passes import RemoveBarriers
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from Preprocess import preprocess_data_window

CLASSIFIER_N_PREVIOUS_EVENTS = 20
RESERVOIR_N_QUBITS = 6
# Simulation candidate scoring must stay tractable: always run one 6-qubit block
# per circuit, regardless of hardware backend width.
SIMULATION_MAX_BLOCKS_PER_CIRCUIT = 1
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


@dataclass
class PubMeta:
    chunk_id: int
    basis: str  # "z" | "x" | "y"


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
    parser.add_argument("--ensemble-size", type=int, default=2)
    parser.add_argument("--resilience-level", type=int, default=0)
    parser.add_argument("--optimization-level", type=int, default=2)
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

    for col in RESERVOIR_FEATURE_COLUMNS:
        if col not in X_train_df.columns:
            raise KeyError(
                f"Missing required reservoir feature column '{col}'. "
                "Ensure preprocess_data_window uses 21-event history."
            )

    X_train_hist = X_train_df.to_numpy(dtype=float)
    X_val_hist = X_val_df.to_numpy(dtype=float)
    X_test_hist = X_test_df.to_numpy(dtype=float)

    X_train_res_raw = X_train_df[RESERVOIR_FEATURE_COLUMNS].to_numpy(dtype=float)
    X_val_res_raw = X_val_df[RESERVOIR_FEATURE_COLUMNS].to_numpy(dtype=float)
    X_test_res_raw = X_test_df[RESERVOIR_FEATURE_COLUMNS].to_numpy(dtype=float)

    X_train_res, train_min, train_max = scale_to_pi(X_train_res_raw)
    X_val_res, _, _ = scale_to_pi(X_val_res_raw, train_min, train_max)
    X_test_res, _, _ = scale_to_pi(X_test_res_raw, train_min, train_max)

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


def sample_reservoir_params(rng: np.random.Generator, n_qubits: int) -> ReservoirParams:
    J = rng.uniform(-1.0, 1.0, size=(n_qubits, n_qubits))
    J = np.triu(J, 1)
    J = J + J.T
    np.fill_diagonal(J, 0.0)
    h = float(rng.uniform(0.1, 1.0))
    t = float(rng.uniform(0.5, 2.5))
    return ReservoirParams(J=J, h=h, t=t)


def create_candidates(
    n_candidates: int,
    n_qubits: int,
    seed: int,
) -> List[CandidateConfig]:
    rng = np.random.default_rng(seed)
    candidates: List[CandidateConfig] = []
    for cidx in range(n_candidates):
        candidates.append(
            CandidateConfig(
                candidate_id=cidx,
                short_params=sample_reservoir_params(rng, n_qubits),
                long_params=sample_reservoir_params(rng, n_qubits),
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
    if hasattr(data, "meas"):
        return dict(data.meas.get_counts())
    if hasattr(data, "c"):
        return dict(data.c.get_counts())
    raise ValueError("Sampler result has no recognized classical register (meas/c).")


def _split_indices(n_rows: int, chunk_size: int) -> List[Tuple[int, int]]:
    return [(s, min(s + chunk_size, n_rows)) for s in range(0, n_rows, chunk_size)]


def extract_features_noiseless(
    X_data: np.ndarray,
    params: ReservoirParams,
    num_layers: int,
    shots: int,
    packing: PackingConfig,
) -> np.ndarray:
    n_rows = len(X_data)
    out = np.zeros((n_rows, 4 * packing.n_qubits_per_block), dtype=float)
    if n_rows == 0:
        return out

    backend = BasicSimulator()
    block_template, block_params = build_parametric_block_circuit(
        params, num_layers, packing.n_qubits_per_block
    )

    # For simulation ranking we intentionally keep chunking separate from hardware
    # packing: max_blocks_per_circuit can be forced to 1 so each simulation circuit
    # remains a strict 6-qubit execution.
    for start, end in _split_indices(n_rows, packing.max_blocks_per_circuit):
        chunk = X_data[start:end]
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
            out[start + b] = counts_to_expectations_block(
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


def build_hardware_submission_bundle(
    selected: Sequence[Dict],
    routing: Dict[str, np.ndarray],
    packing: PackingConfig,
    num_layers: int,
    pm,
) -> Tuple[List[QuantumCircuit], List[PubMeta], Dict[Tuple[int, str], np.ndarray], Dict[int, ChunkRequest]]:
    circuits: List[QuantumCircuit] = []
    metas: List[PubMeta] = []
    chunk_requests: Dict[int, ChunkRequest] = {}
    feature_store: Dict[Tuple[int, str], np.ndarray] = {}
    chunk_id = 0

    dataset_specs = [
        ("short_train", routing["X_train_res"][routing["short_train_idx"]], "short"),
        ("short_val", routing["X_val_res"][routing["short_val_idx"]], "short"),
        ("short_test", routing["X_test_res"][routing["short_test_idx"]], "short"),
        ("long_train", routing["X_train_res"][routing["long_train_idx"]], "long"),
        ("long_val", routing["X_val_res"][routing["long_val_idx"]], "long"),
        ("long_test", routing["X_test_res"][routing["long_test_idx"]], "long"),
    ]

    for item in selected:
        cand: CandidateConfig = item["candidate"]
        for dataset_key, X_ds, regime in dataset_specs:
            store_key = (cand.candidate_id, dataset_key)
            feature_store[store_key] = np.zeros(
                (len(X_ds), 4 * packing.n_qubits_per_block), dtype=float
            )
            if len(X_ds) == 0:
                continue
            regime_params = cand.short_params if regime == "short" else cand.long_params
            block_template, block_params = build_parametric_block_circuit(
                regime_params, num_layers, packing.n_qubits_per_block
            )
            for start, end in _split_indices(len(X_ds), packing.max_blocks_per_circuit):
                chunk = X_ds[start:end]
                n_blocks = len(chunk)
                state = build_packed_state_circuit(chunk, block_template, block_params)
                z_circuit = pm.run(
                    with_measurement_basis(
                        state, "z", packing.n_qubits_per_block, n_blocks
                    )
                )
                x_circuit = pm.run(
                    with_measurement_basis(
                        state, "x", packing.n_qubits_per_block, n_blocks
                    )
                )
                y_circuit = pm.run(
                    with_measurement_basis(
                        state, "y", packing.n_qubits_per_block, n_blocks
                    )
                )
                circuits.extend([z_circuit, x_circuit, y_circuit])
                metas.extend(
                    [
                        PubMeta(chunk_id=chunk_id, basis="z"),
                        PubMeta(chunk_id=chunk_id, basis="x"),
                        PubMeta(chunk_id=chunk_id, basis="y"),
                    ]
                )
                chunk_requests[chunk_id] = ChunkRequest(
                    chunk_id=chunk_id,
                    matrix_key=store_key,
                    start_row=start,
                    end_row=end,
                    n_blocks=n_blocks,
                )
                chunk_id += 1

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
    if args.ensemble_size != 2:
        parser.error("Ensemble size is locked to 2 for this migration.")
        # raise ValueError("Ensemble size is locked to 2 for this migration.")
    if args.n_candidates != 5:
        parser.error("Candidate search is locked to 5 noiseless candidates.")
        # raise ValueError("Candidate search is locked to 5 noiseless candidates.")
    if args.classifier_optuna_trials < 0:
        parser.error("--classifier-optuna-trials must be >= 0.")
    if args.regressor_optuna_trials < 0:
        parser.error("--regressor-optuna-trials must be >= 0.")
    if args.regressor_optuna_cv_folds < 2:
        parser.error("--regressor-optuna-cv-folds must be >= 2.")
    if not args.simulate_only and not args.token:
        parser.error("--token is required when it is not a simulate-only run")
        # raise ValueError("--token is required when using --hardware")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = load_split_data(args.subset_frac, args.subset_seed)

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

    service = QiskitRuntimeService(token=args.token)
    backend = service.backend(args.backend)
    backend_qubits = getattr(backend, "num_qubits", 156)
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

    if not args.no_confirm:
        print(
            f"Submitting one hardware Runtime job | backend={args.backend} "
            f"| shots={args.shots} | blocks/circuit={blocks_per_circuit} "
            f"| selected={ [int(r['candidate_id']) for r in selected] }"
        )
        if input("Continue? [y/N]: ").strip().lower() not in {"y", "yes"}:
            print("Aborted.")
            return

    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=args.optimization_level
    )
    circuits, metas, feature_store, chunk_requests = build_hardware_submission_bundle(
        selected=selected,
        routing=routing,
        packing=hw_packing,
        num_layers=args.num_layers,
        pm=pm,
    )

    if len(circuits) == 0:
        raise ValueError("No hardware circuits were built; cannot submit an empty job.")

    sampler = Sampler(mode=backend)
    sampler.options.default_shots = args.shots
    # sampler.options.resilience_level = args.resilience_level
    print(f"Submitting one Runtime Sampler job with {len(circuits)} pubs...")
    job = sampler.run(circuits)
    pub_results = job.result()
    usage_estimation = getattr(job, "usage_estimation", None)

    chunk_counts: Dict[int, Dict[str, Dict[str, int]]] = {}
    for i, meta in enumerate(metas):
        if meta.chunk_id not in chunk_counts:
            chunk_counts[meta.chunk_id] = {}
        chunk_counts[meta.chunk_id][meta.basis] = _counts_from_sampler_pub(pub_results[i])

    for chunk_id, req in chunk_requests.items():
        counts_map = chunk_counts.get(chunk_id, {})
        if {"z", "x", "y"} - set(counts_map):
            raise RuntimeError(f"Missing basis counts for chunk_id={chunk_id}")
        matrix = feature_store[req.matrix_key]
        for block_idx in range(req.n_blocks):
            matrix[req.start_row + block_idx] = counts_to_expectations_block(
                counts_map["z"],
                counts_map["x"],
                counts_map["y"],
                block_idx,
                hw_packing.n_qubits_per_block,
            )

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
        "reservoir_features": RESERVOIR_FEATURE_COLUMNS,
        "candidate_search_count": args.n_candidates,
        "ensemble_size": args.ensemble_size,
        "parallel_circuits_requested": args.parallel_circuits,
        "parallel_circuits_used": blocks_per_circuit,
        "parallel_circuits_backend_max": max_blocks_supported,
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
        "runtime_usage_estimation": usage_estimation,
        "runtime_job_id": getattr(job, "job_id", lambda: None)(),
    }
    save_json(args.output_dir / args.summary_name, summary)
    with (args.output_dir / "hardware_results.pkl").open("wb") as f:
        pickle.dump(
            {
                "summary": summary,
                "simulation_ranking": ranking_payload,
                "hardware_results": hw_results,
                "ensemble_pred": ensemble_pred,
            },
            f,
        )
    print(f"Saved summary -> {args.output_dir / 'hardware_summary.json'}")


if __name__ == "__main__":
    main()
