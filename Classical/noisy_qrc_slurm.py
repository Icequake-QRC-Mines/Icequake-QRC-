#!/usr/bin/env python3
"""
Noisy FC-TFI QRC pipeline for SLURM clusters.

Modes:
1) Full run (single process): compute quantum features + tune/train + export hardware config
2) Array task mode: compute one (iteration, regime) partial artifact
3) Aggregate mode: consume partial artifacts and run classical tuning/selection only
4) Estimate mode: print transpiled circuit resources and exit
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.transpiler import PassManager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from Preprocess import preprocess_data_window


optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    # Built-in transpiler analysis bundle for depth/size/count_ops.
    from qiskit.transpiler.passes import ResourceEstimation
except Exception:
    ResourceEstimation = None


@dataclass
class QRCConfig:
    num_layers_per_event: int = 2
    shots: int = 4096
    n_iterations: int = 5
    top_k: int = 3
    random_seed: int = 42
    optuna_trials: int = 30
    short_threshold: int = 65_000
    n_previous_events: int = 20
    n_qubits: int = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Noisy QRC pipeline with SLURM support.")
    parser.add_argument("--output-dir", type=Path, default=Path("Classical/results/noisy_qrc_run"))
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--n-iterations", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--optuna-trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset-frac", type=float, default=1.0)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=8, help="Max circuits per Aer run() batch")
    parser.add_argument("--max-memory-mb", type=int, default=None, help="Aer max_memory_mb override")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def scale_to_pi_range(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_min = X_train.min(axis=0)
    train_max = X_train.max(axis=0)
    denom = train_max - train_min
    denom[denom == 0] = 1.0

    def transform(X: np.ndarray) -> np.ndarray:
        scaled = (X - train_min) / denom
        scaled = np.clip(scaled, 0.0, 1.0)
        return scaled * np.pi

    return transform(X_train), transform(X_val), transform(X_test), train_min, train_max


def generate_ising_params(n_qubits: int, rng: np.random.Generator, J_std: float = 0.5, h: float = 1.0, t: float = 0.5):
    J = np.zeros((n_qubits, n_qubits))
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            J[i, j] = rng.normal(0, J_std)
    return J, h, t


def trotter_ising_layer(qc: QuantumCircuit, n_qubits: int, J: np.ndarray, h: float, t: float, n_trotter_steps: int = 3) -> None:
    dt = t / n_trotter_steps
    for _ in range(n_trotter_steps):
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                if abs(J[i, j]) > 1e-10:
                    qc.cx(i, j)
                    qc.rz(2 * J[i, j] * dt, j)
                    qc.cx(i, j)
        for i in range(n_qubits):
            qc.rx(2 * h * dt, i)


def build_parametric_reservoir_circuit(ising_params, num_layers: int, n_qubits: int) -> Tuple[QuantumCircuit, List[Parameter]]:
    J, h, t = ising_params
    thetas = [Parameter(f"theta_{i}") for i in range(n_qubits)]
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.h(i)
    for i in range(n_qubits):
        qc.ry(thetas[i], i)
    qc.barrier()
    for _ in range(num_layers):
        trotter_ising_layer(qc, n_qubits, J, h, t)
        qc.barrier()
        for i in range(n_qubits):
            qc.ry(thetas[i], i)
        qc.barrier()
        trotter_ising_layer(qc, n_qubits, J, h, t)
        qc.barrier()
    return qc, thetas


def build_noisy_simulator(device: str, max_memory_mb: int | None):
    fake_backend = FakeSherbrooke()
    noise_model = NoiseModel.from_backend(fake_backend)
    sim_kwargs = {
        "noise_model": noise_model,
        # Prevent Aer from launching too many concurrent experiments on laptops.
        "max_parallel_experiments": 1,
    }
    if max_memory_mb is not None:
        sim_kwargs["max_memory_mb"] = max_memory_mb
    if device == "gpu":
        sim_kwargs.update({"device": "GPU"})
    simulator = AerSimulator(**sim_kwargs)
    # IMPORTANT: do not transpile against full fake backend target for noisy local sim.
    # Keep transpilation local to avoid backend-wide qubit inflation and memory blowups.
    basis_gates = sorted(set(noise_model.basis_gates) - {"measure", "reset", "delay"})

    def local_transpile(circuit: QuantumCircuit) -> QuantumCircuit:
        return transpile(circuit, basis_gates=basis_gates, optimization_level=1)

    return fake_backend, simulator, local_transpile


def add_measurement_basis(circuit: QuantumCircuit, basis: str) -> QuantumCircuit:
    qc = circuit.copy()
    n_qubits = qc.num_qubits
    if basis == "X":
        for i in range(n_qubits):
            qc.h(i)
    elif basis == "Y":
        for i in range(n_qubits):
            qc.sdg(i)
            qc.h(i)
    qc.measure_all()
    return qc


def _parse_bit(bitstring: str, n_qubits: int, q: int) -> int:
    bits = bitstring.replace(" ", "")
    return int(bits[n_qubits - 1 - q])


def _counts_to_exp_and_zz(counts, n_qubits: int, shots: int):
    zexp = np.zeros(n_qubits)
    zz = np.zeros(n_qubits)
    for bitstring, count in counts.items():
        for q in range(n_qubits):
            bi = _parse_bit(bitstring, n_qubits, q)
            zexp[q] += (1 - 2 * bi) * count / shots
            q2 = (q + 1) % n_qubits
            bj = _parse_bit(bitstring, n_qubits, q2)
            zz[q] += (1 - 2 * bi) * (1 - 2 * bj) * count / shots
    return zexp, zz


def _counts_to_basis_exp(counts, n_qubits: int, shots: int):
    exp = np.zeros(n_qubits)
    for bitstring, count in counts.items():
        for q in range(n_qubits):
            b = _parse_bit(bitstring, n_qubits, q)
            exp[q] += (1 - 2 * b) * count / shots
    return exp


def estimate_resources(isa_circuit: QuantumCircuit, backend, shots: int, n_bindings: int) -> Dict[str, float]:
    ops = isa_circuit.count_ops()
    depth = float(isa_circuit.depth())
    size = float(sum(ops.values()))

    # Use built-in Qiskit transpiler resource analysis when available.
    if ResourceEstimation is not None:
        try:
            pm = PassManager([ResourceEstimation()])
            pm.run(isa_circuit)
            pset = pm.property_set
            depth = float(pset.get("depth", depth))
            size = float(pset.get("size", size))
            count_ops = pset.get("count_ops")
            if count_ops:
                ops = count_ops
        except Exception:
            pass

    # Use built-in circuit duration estimator if available.
    total_duration = None
    if hasattr(isa_circuit, "estimate_duration"):
        try:
            total_duration = float(isa_circuit.estimate_duration(target=backend.target, unit="s"))
        except Exception:
            total_duration = None

    # Fallback: derive duration from backend target instruction durations.
    if total_duration is None:
        total_duration = 0.0
        target = backend.target
        for inst in isa_circuit.data:
            gate_name = inst.operation.name
            qubits = tuple(isa_circuit.find_bit(q).index for q in inst.qubits)
            props = target[gate_name].get(qubits) if gate_name in target else None
            if props and props.duration:
                total_duration += props.duration

    rep_delay = 250e-6
    qpu_seconds = (total_duration + rep_delay) * shots * n_bindings
    return {
        "depth": depth,
        "total_gates": size,
        "ecr_gates": float(ops.get("ecr", 0)),
        "sx_gates": float(ops.get("sx", 0)),
        "rz_gates": float(ops.get("rz", 0)),
        "x_gates": float(ops.get("x", 0)),
        "estimated_circuit_seconds": float(total_duration),
        "est_qpu_seconds": qpu_seconds,
    }


def run_quantum_reservoir_pauli(
    X_data: np.ndarray,
    angle_bank,
    cfg: QRCConfig,
    simulator: AerSimulator,
    local_transpile,
    backend,
    checkpoint_prefix: Path | None = None,
    resume: bool = False,
    batch_size: int = 8,
):
    m = X_data.shape[0]
    n_obs = 4 * cfg.n_qubits
    n_total_events = cfg.n_previous_events + 1
    pauli_matrix = np.zeros((m, n_total_events * n_obs))
    resources = []

    for event_idx in range(n_total_events):
        ckpt = None if checkpoint_prefix is None else checkpoint_prefix.with_name(f"{checkpoint_prefix.name}_event{event_idx}.npy")
        if ckpt and resume and ckpt.exists():
            block = np.load(ckpt)
            pauli_matrix[:, event_idx * n_obs : (event_idx + 1) * n_obs] = block
            continue

        start_col = event_idx * cfg.n_qubits
        end_col = start_col + cfg.n_qubits
        X_event = X_data[:, start_col:end_col]
        template, params = build_parametric_reservoir_circuit(angle_bank[event_idx], cfg.num_layers_per_event, cfg.n_qubits)
        isa_template = local_transpile(template)
        resources.append(estimate_resources(isa_template, backend, cfg.shots, len(X_event)))

        bound = [isa_template.assign_parameters(dict(zip(params, row))) for row in X_event]
        batch_z = [add_measurement_basis(c, "Z") for c in bound]
        batch_x = [add_measurement_basis(c, "X") for c in bound]
        batch_y = [add_measurement_basis(c, "Y") for c in bound]

        z_counts = []
        x_counts = []
        y_counts = []
        aer_time_taken = 0.0
        for start in range(0, len(bound), batch_size):
            end = min(start + batch_size, len(bound))
            result_z = simulator.run(batch_z[start:end], shots=cfg.shots).result()
            result_x = simulator.run(batch_x[start:end], shots=cfg.shots).result()
            result_y = simulator.run(batch_y[start:end], shots=cfg.shots).result()
            aer_time_taken += float(getattr(result_z, "time_taken", 0.0))
            aer_time_taken += float(getattr(result_x, "time_taken", 0.0))
            aer_time_taken += float(getattr(result_y, "time_taken", 0.0))
            for idx in range(end - start):
                z_counts.append(result_z.get_counts(idx))
                x_counts.append(result_x.get_counts(idx))
                y_counts.append(result_y.get_counts(idx))

        event_block = np.zeros((m, n_obs))
        for sample_idx in range(m):
            counts_z = z_counts[sample_idx]
            counts_x = x_counts[sample_idx]
            counts_y = y_counts[sample_idx]
            zexp, zz = _counts_to_exp_and_zz(counts_z, cfg.n_qubits, cfg.shots)
            xexp = _counts_to_basis_exp(counts_x, cfg.n_qubits, cfg.shots)
            yexp = _counts_to_basis_exp(counts_y, cfg.n_qubits, cfg.shots)
            event_block[sample_idx] = np.concatenate([zexp, xexp, yexp, zz])

        pauli_matrix[:, event_idx * n_obs : (event_idx + 1) * n_obs] = event_block
        if ckpt:
            np.save(ckpt, event_block)
        resources[-1]["aer_time_taken_seconds"] = aer_time_taken
        print(f"  Event {event_idx + 1}/{n_total_events} complete | aer_time={aer_time_taken:.2f}s")

    return pauli_matrix, resources


def make_hybrid_features_decay(P_matrix: np.ndarray, n_total_events: int, n_obs: int, decay: float = 0.3):
    weights = np.array([np.exp(-decay * (n_total_events - 1 - i)) for i in range(n_total_events)])
    weights /= weights.sum()
    weighted = P_matrix.copy()
    for event_idx in range(n_total_events):
        s = event_idx * n_obs
        e = s + n_obs
        weighted[:, s:e] *= weights[event_idx]
    return weighted


def tune_and_train_regressor(X_train, y_train, X_val, y_val, seed: int, n_trials: int):
    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "n_estimators": 1200,
            "random_state": 42,
            "early_stopping_rounds": 50,
            "tree_method": "hist",
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        pred = model.predict(X_val)
        return mean_absolute_error(y_val, pred)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_trial.params
    full_params = {
        "objective": "reg:squarederror",
        "n_estimators": 1200,
        "random_state": 42,
        "early_stopping_rounds": 50,
        "tree_method": "hist",
        **best,
    }
    model = XGBRegressor(**full_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model, full_params


def task_to_iter_regime(task_id: int, n_iterations: int):
    total = n_iterations * 2
    if task_id < 0 or task_id >= total:
        raise ValueError(f"task_id must be in [0, {total - 1}]")
    iter_idx = task_id // 2
    regime = "short" if (task_id % 2 == 0) else "long"
    return iter_idx, regime


def train_classifier(X_train_q, X_val_q, X_test_q, y_train, y_val, y_test, threshold: int):
    y_clf_train = (y_train >= threshold).astype(int)
    y_clf_val = (y_val >= threshold).astype(int)
    y_clf_test = (y_test >= threshold).astype(int)
    sample_weights = compute_sample_weight("balanced", y_clf_train)
    clf = XGBClassifier(
        objective="binary:logistic",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )
    clf.fit(X_train_q, y_clf_train, sample_weight=sample_weights, eval_set=[(X_val_q, y_clf_val)], verbose=False)
    print(f"Classifier test acc: {accuracy_score(y_clf_test, clf.predict(X_test_q)):.4f}")
    return clf


def load_data(cfg: QRCConfig, subset_frac: float):
    repo_root = Path(__file__).resolve().parent.parent
    data_csv = repo_root / "Whillians-GPS-Data-and-Features.csv"
    filtered_csv = repo_root / "filtered_time_to_next_event.csv"
    if not data_csv.exists() or not filtered_csv.exists():
        raise FileNotFoundError(
            "Expected data files at repo root. Missing one of: "
            f"{data_csv} or {filtered_csv}"
        )
    data_orig = pd.read_csv(data_csv)
    filtered_time = pd.read_csv(filtered_csv)
    X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocess_data_window(filtered_time, data_orig, cfg.n_previous_events)
    if subset_frac < 1.0:
        rng = np.random.default_rng(cfg.random_seed)
        train_idx = rng.choice(len(X_train), size=max(20, int(len(X_train) * subset_frac)), replace=False)
        train_idx.sort()
        X_train = X_train.iloc[train_idx]
        y_train = y_train.iloc[train_idx]
    X_train_q, X_val_q, X_test_q, train_min, train_max = scale_to_pi_range(X_train.to_numpy(), X_val.to_numpy(), X_test.to_numpy())
    return X_train_q, X_val_q, X_test_q, y_train.to_numpy(), y_val.to_numpy(), y_test.to_numpy(), train_min, train_max


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_partial_task(args: argparse.Namespace, cfg: QRCConfig):
    iter_idx, regime = task_to_iter_regime(args.task_id, cfg.n_iterations)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = out_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)

    X_train_q, X_val_q, X_test_q, y_train, y_val, y_test, *_ = load_data(cfg, args.subset_frac)
    clf = train_classifier(X_train_q, X_val_q, X_test_q, y_train, y_val, y_test, cfg.short_threshold)
    clf_test_labels = clf.predict(X_test_q)

    short_mask_train = y_train < cfg.short_threshold
    long_mask_train = ~short_mask_train
    short_mask_val = y_val < cfg.short_threshold
    long_mask_val = ~short_mask_val
    short_test_idx = np.where(clf_test_labels == 0)[0]
    long_test_idx = np.where(clf_test_labels == 1)[0]

    rng = np.random.default_rng(cfg.random_seed + iter_idx + (0 if regime == "short" else 10_000))
    angle_bank = [generate_ising_params(cfg.n_qubits, rng) for _ in range(cfg.n_previous_events + 1)]
    _, sim, local_transpile = build_noisy_simulator(args.device, args.max_memory_mb)
    fake_backend = FakeSherbrooke()

    if regime == "short":
        Xtr, Xvl, Xte = X_train_q[short_mask_train], X_val_q[short_mask_val], X_test_q[short_test_idx]
    else:
        Xtr, Xvl, Xte = X_train_q[long_mask_train], X_val_q[long_mask_val], X_test_q[long_test_idx]

    print(f"Running partial: iteration={iter_idx} regime={regime} train={len(Xtr)} val={len(Xvl)} test={len(Xte)}")
    t0 = time.time()
    P_tr, resources = run_quantum_reservoir_pauli(
        Xtr,
        angle_bank,
        cfg,
        sim,
        local_transpile,
        fake_backend,
        checkpoint_prefix=partial_dir / f"iter{iter_idx}_{regime}_train",
        resume=args.resume,
        batch_size=args.batch_size,
    )
    P_vl, _ = run_quantum_reservoir_pauli(
        Xvl,
        angle_bank,
        cfg,
        sim,
        local_transpile,
        fake_backend,
        checkpoint_prefix=partial_dir / f"iter{iter_idx}_{regime}_val",
        resume=args.resume,
        batch_size=args.batch_size,
    )
    P_te, _ = run_quantum_reservoir_pauli(
        Xte,
        angle_bank,
        cfg,
        sim,
        local_transpile,
        fake_backend,
        checkpoint_prefix=partial_dir / f"iter{iter_idx}_{regime}_test",
        resume=args.resume,
        batch_size=args.batch_size,
    )
    elapsed = time.time() - t0

    payload = {
        "iteration": iter_idx,
        "regime": regime,
        "angle_bank": angle_bank,
        "P_train": P_tr,
        "P_val": P_vl,
        "P_test": P_te,
        "resources": resources,
        "elapsed_seconds": elapsed,
    }
    partial_path = partial_dir / f"partial_iter{iter_idx}_{regime}.pkl"
    with partial_path.open("wb") as f:
        pickle.dump(payload, f)
    print(f"Saved partial artifact -> {partial_path}")


def aggregate_partials(args: argparse.Namespace, cfg: QRCConfig):
    out_dir = args.output_dir
    partial_dir = out_dir / "partials"
    X_train_q, X_val_q, X_test_q, y_train, y_val, y_test, train_min, train_max = load_data(cfg, args.subset_frac)
    clf = train_classifier(X_train_q, X_val_q, X_test_q, y_train, y_val, y_test, cfg.short_threshold)
    clf_val_labels = clf.predict(X_val_q)
    clf_test_labels = clf.predict(X_test_q)
    short_mask_train = y_train < cfg.short_threshold
    long_mask_train = ~short_mask_train
    short_mask_val = y_val < cfg.short_threshold
    long_mask_val = ~short_mask_val
    short_test_idx = np.where(clf_test_labels == 0)[0]
    long_test_idx = np.where(clf_test_labels == 1)[0]
    short_val_idx = np.where(clf_val_labels == 0)[0]
    long_val_idx = np.where(clf_val_labels == 1)[0]

    all_results = []
    n_total_events = cfg.n_previous_events + 1
    n_obs = 4 * cfg.n_qubits
    for i in range(cfg.n_iterations):
        with (partial_dir / f"partial_iter{i}_short.pkl").open("rb") as f:
            ps = pickle.load(f)
        with (partial_dir / f"partial_iter{i}_long.pkl").open("rb") as f:
            pl = pickle.load(f)

        H_tr_short = make_hybrid_features_decay(ps["P_train"], n_total_events, n_obs)
        H_vl_short = make_hybrid_features_decay(ps["P_val"], n_total_events, n_obs)
        H_te_short = make_hybrid_features_decay(ps["P_test"], n_total_events, n_obs)
        H_tr_long = make_hybrid_features_decay(pl["P_train"], n_total_events, n_obs)
        H_vl_long = make_hybrid_features_decay(pl["P_val"], n_total_events, n_obs)
        H_te_long = make_hybrid_features_decay(pl["P_test"], n_total_events, n_obs)

        y_tr_short = y_train[short_mask_train]
        y_tr_long = y_train[long_mask_train]
        y_vl_short = y_val[short_mask_val]
        y_vl_long = y_val[long_mask_val]

        model_short, short_params = tune_and_train_regressor(H_tr_short, y_tr_short, H_vl_short, y_vl_short, cfg.random_seed + i, cfg.optuna_trials)
        model_long, long_params = tune_and_train_regressor(H_tr_long, y_tr_long, H_vl_long, y_vl_long, cfg.random_seed + i + 1, cfg.optuna_trials)

        test_pred = np.empty(len(X_test_q))
        test_pred[short_test_idx] = model_short.predict(H_te_short)
        test_pred[long_test_idx] = model_long.predict(H_te_long)

        val_pred = np.empty(len(X_val_q))
        short_val_positions = {idx: pos for pos, idx in enumerate(np.where(short_mask_val)[0])}
        long_val_positions = {idx: pos for pos, idx in enumerate(np.where(long_mask_val)[0])}
        for idx in short_val_idx:
            if idx in short_val_positions:
                val_pred[idx] = model_short.predict(H_vl_short[short_val_positions[idx] : short_val_positions[idx] + 1])[0]
            elif idx in long_val_positions:
                val_pred[idx] = model_short.predict(H_vl_long[long_val_positions[idx] : long_val_positions[idx] + 1])[0]
        for idx in long_val_idx:
            if idx in long_val_positions:
                val_pred[idx] = model_long.predict(H_vl_long[long_val_positions[idx] : long_val_positions[idx] + 1])[0]
            elif idx in short_val_positions:
                val_pred[idx] = model_long.predict(H_vl_short[short_val_positions[idx] : short_val_positions[idx] + 1])[0]

        all_results.append(
            {
                "iteration": i,
                "val_mae": float(mean_absolute_error(y_val, val_pred)),
                "val_rmse": float(root_mean_squared_error(y_val, val_pred)),
                "val_r2": float(r2_score(y_val, val_pred)),
                "test_mae": float(mean_absolute_error(y_test, test_pred)),
                "test_rmse": float(root_mean_squared_error(y_test, test_pred)),
                "test_pred": test_pred,
                "short_params": short_params,
                "long_params": long_params,
                "angle_bank_short": ps["angle_bank"],
                "angle_bank_long": pl["angle_bank"],
            }
        )
        print(f"Aggregated iteration {i + 1}/{cfg.n_iterations} | val_mae={all_results[-1]['val_mae']:.2f}")

    top_results = sorted(all_results, key=lambda r: r["val_mae"])[: cfg.top_k]
    top_indices = [r["iteration"] for r in top_results]
    ensemble_pred = np.mean([r["test_pred"] for r in top_results], axis=0)

    summary = {
        "top_indices": top_indices,
        "ensemble_test_mae": float(mean_absolute_error(y_test, ensemble_pred)),
        "ensemble_test_rmse": float(root_mean_squared_error(y_test, ensemble_pred)),
        "ensemble_test_r2": float(r2_score(y_test, ensemble_pred)),
    }
    save_json(out_dir / "aggregate_summary.json", summary)

    hardware_config = {
        "top_k_indices": top_indices,
        "top_k_seeds": [cfg.random_seed + i for i in top_indices],
        "ising_params_per_iteration": {r["iteration"]: {"short": r["angle_bank_short"], "long": r["angle_bank_long"]} for r in top_results},
        "xgb_params_per_iteration": {r["iteration"]: {"short": r["short_params"], "long": r["long_params"]} for r in top_results},
        "regime_classifier": clf,
        "pipeline_config": cfg.__dict__,
        "scaling_params": {"train_min": train_min, "train_max": train_max},
        "short_threshold": cfg.short_threshold,
    }
    with (out_dir / "hardware_config.pkl").open("wb") as f:
        pickle.dump(hardware_config, f)
    print(f"Wrote hardware config -> {out_dir / 'hardware_config.pkl'}")


def print_estimate(cfg: QRCConfig, shots: int, subset_frac: float, device: str):
    X_train_q, *_ = load_data(cfg, subset_frac)
    rng = np.random.default_rng(cfg.random_seed)
    angle_bank = [generate_ising_params(cfg.n_qubits, rng) for _ in range(cfg.n_previous_events + 1)]
    backend, _, local_transpile = build_noisy_simulator(device, None)
    rows = []
    for event_idx in range(cfg.n_previous_events + 1):
        template, _ = build_parametric_reservoir_circuit(angle_bank[event_idx], cfg.num_layers_per_event, cfg.n_qubits)
        isa = local_transpile(template)
        rows.append(estimate_resources(isa, backend, shots, len(X_train_q)))
    print("=== Noisy Simulation Resource Estimate ===")
    for i, r in enumerate(rows, start=1):
        print(
            f"Event {i:02d} | depth={int(r['depth'])} ecr={int(r['ecr_gates'])} total={int(r['total_gates'])} "
            f"est_qpu_equiv_s={r['est_qpu_seconds']:.2f}"
        )
    print(f"Total est seconds across events: {sum(r['est_qpu_seconds'] for r in rows):.2f}")


def main():
    args = parse_args()
    cfg = QRCConfig(
        shots=args.shots,
        n_iterations=args.n_iterations,
        top_k=args.top_k,
        random_seed=args.seed,
        optuna_trials=args.optuna_trials,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.estimate_only:
        print_estimate(cfg, args.shots, args.subset_frac, args.device)
        return

    if args.task_id is not None:
        run_partial_task(args, cfg)
        return

    if args.aggregate:
        aggregate_partials(args, cfg)
        return

    # Single-node full execution: compute partials locally then aggregate.
    for task_id in range(cfg.n_iterations * 2):
        args_task = argparse.Namespace(**vars(args))
        args_task.task_id = task_id
        run_partial_task(args_task, cfg)
    aggregate_partials(args, cfg)


if __name__ == "__main__":
    main()
