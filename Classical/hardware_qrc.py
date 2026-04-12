#!/usr/bin/env python3
"""
Hardware QRC execution using pre-tuned noisy-sim configuration.

Loads hardware_config.pkl and runs only top-k reservoir configurations on IBM hardware.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import PassManager, generate_preset_pass_manager
from qiskit_ibm_runtime import Batch, EstimatorV2, QiskitRuntimeService
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor

from Preprocess import preprocess_data_window

try:
    from qiskit.transpiler.passes import ResourceEstimation
except Exception:
    ResourceEstimation = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run top-k QRC reservoirs on IBM hardware.")
    parser.add_argument("--config", type=Path, required=True, help="Path to hardware_config.pkl")
    parser.add_argument("--output-dir", type=Path, default=Path("Classical/results/hardware_qrc_run"))
    parser.add_argument("--backend", type=str, default="ibm_sherbrooke")
    parser.add_argument("--shots", type=int, default=None, help="Override shots from config")
    parser.add_argument(
        "--subset-frac",
        type=float,
        default=1.0,
        help="Fraction of train/val/test rows to use for hardware run (0 < frac <= 1).",
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=None,
        help="Seed for subset sampling. Defaults to pipeline random_seed.",
    )
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--no-confirm", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    return parser.parse_args()


def scale_with_params(X: np.ndarray, train_min: np.ndarray, train_max: np.ndarray) -> np.ndarray:
    denom = train_max - train_min
    denom[denom == 0] = 1.0
    scaled = (X - train_min) / denom
    scaled = np.clip(scaled, 0.0, 1.0)
    return scaled * np.pi


def load_data_with_scaling(config_payload):
    cfg = config_payload["pipeline_config"]
    n_previous_events = cfg["n_previous_events"]
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
    X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocess_data_window(filtered_time, data_orig, n_previous_events)
    train_min = config_payload["scaling_params"]["train_min"]
    train_max = config_payload["scaling_params"]["train_max"]
    X_train_q = scale_with_params(X_train.to_numpy(), train_min, train_max)
    X_val_q = scale_with_params(X_val.to_numpy(), train_min, train_max)
    X_test_q = scale_with_params(X_test.to_numpy(), train_min, train_max)
    return X_train_q, X_val_q, X_test_q, y_train.to_numpy(), y_val.to_numpy(), y_test.to_numpy()


def maybe_subset_split(
    X: np.ndarray,
    y: np.ndarray,
    frac: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if frac >= 1.0:
        return X, y
    n = len(X)
    if n == 0:
        return X, y
    keep = max(1, int(n * frac))
    idx = rng.choice(n, size=keep, replace=False)
    idx.sort()
    return X[idx], y[idx]


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


def build_observables(n_qubits: int) -> List[SparsePauliOp]:
    observables = []
    for i in range(n_qubits):
        for pauli in ["Z", "X", "Y"]:
            label = ["I"] * n_qubits
            label[n_qubits - 1 - i] = pauli
            observables.append(SparsePauliOp("".join(label)))
        j = (i + 1) % n_qubits
        label = ["I"] * n_qubits
        label[n_qubits - 1 - i] = "Z"
        label[n_qubits - 1 - j] = "Z"
        observables.append(SparsePauliOp("".join(label)))
    return observables


def estimate_resources(isa_circuit: QuantumCircuit, backend, shots: int, n_bindings: int) -> Dict[str, float]:
    ops = isa_circuit.count_ops()
    depth = float(isa_circuit.depth())
    size = float(sum(ops.values()))
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

    total_duration = None
    if hasattr(isa_circuit, "estimate_duration"):
        try:
            total_duration = float(isa_circuit.estimate_duration(target=backend.target, unit="s"))
        except Exception:
            total_duration = None
    if total_duration is None:
        target = backend.target
        total_duration = 0.0
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
        "estimated_circuit_seconds": float(total_duration),
        "est_qpu_seconds": qpu_seconds,
    }


def print_resource_report(resource_rows, backend_name: str):
    print(f"=== Resource Estimation for {backend_name} ===")
    total = 0.0
    for idx, row in enumerate(resource_rows, start=1):
        total += row["est_qpu_seconds"]
        print(
            f"Event {idx:02d} | depth={int(row['depth'])} ecr={int(row['ecr_gates'])} "
            f"total={int(row['total_gates'])} est_qpu_s={row['est_qpu_seconds']:.2f}"
        )
    print(f"Estimated QPU-seconds for this circuit bundle: {total:.2f}")


def run_estimator_for_dataset(
    X_data: np.ndarray,
    angle_bank,
    estimator: EstimatorV2,
    pm,
    backend,
    n_qubits: int,
    num_layers: int,
    shots: int,
    checkpoint_prefix: Path | None = None,
):
    n_total_events = X_data.shape[1] // n_qubits
    n_obs = 4 * n_qubits
    pauli_matrix = np.zeros((len(X_data), n_total_events * n_obs))
    if len(X_data) == 0:
        return pauli_matrix, []
    observables = build_observables(n_qubits)
    resources = []

    for event_idx in range(n_total_events):
        ckpt = None if checkpoint_prefix is None else checkpoint_prefix.with_name(f"{checkpoint_prefix.name}_event{event_idx}.npy")
        if ckpt and ckpt.exists():
            block = np.load(ckpt)
            pauli_matrix[:, event_idx * n_obs : (event_idx + 1) * n_obs] = block
            continue

        template, params = build_parametric_reservoir_circuit(angle_bank[event_idx], num_layers, n_qubits)
        isa_circuit = pm.run(template)
        resources.append(estimate_resources(isa_circuit, backend, shots, len(X_data)))
        isa_observables = [obs.apply_layout(isa_circuit.layout) for obs in observables]

        start_col = event_idx * n_qubits
        X_event = X_data[:, start_col : start_col + n_qubits]
        # EstimatorV2 expects observables and parameter-values shapes to be broadcastable.
        # Submit one pub per observable to avoid (n_obs,) vs (n_bindings,) mismatch.
        param_values = np.asarray(X_event, dtype=float)
        pubs = [(isa_circuit, obs, param_values) for obs in isa_observables]
        job = estimator.run(pubs, precision=None)
        results = job.result()
        # Each pub yields shape (n_bindings,); stack to (n_bindings, n_observables).
        evs = np.column_stack([np.asarray(r.data.evs) for r in results])
        usage_estimation = getattr(job, "usage_estimation", None)
        if usage_estimation is not None:
            resources[-1]["runtime_usage_estimation"] = usage_estimation
        pauli_matrix[:, event_idx * n_obs : (event_idx + 1) * n_obs] = evs
        if ckpt:
            np.save(ckpt, evs)
        print(f"  Event {event_idx + 1}/{n_total_events} complete")

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


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    args = parse_args()
    if not (0.0 < args.subset_frac <= 1.0):
        raise ValueError("--subset-frac must satisfy 0 < subset-frac <= 1.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_dir is None:
        args.checkpoint_dir = args.output_dir / "checkpoints"
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with args.config.open("rb") as f:
        cfg_bundle = pickle.load(f)
    pipeline_cfg = cfg_bundle["pipeline_config"]
    shots = args.shots if args.shots is not None else pipeline_cfg["shots"]
    n_qubits = pipeline_cfg["n_qubits"]
    n_previous_events = pipeline_cfg["n_previous_events"]
    n_total_events = n_previous_events + 1
    n_obs = 4 * n_qubits
    short_threshold = cfg_bundle["short_threshold"]

    X_train_q, X_val_q, X_test_q, y_train, y_val, y_test = load_data_with_scaling(cfg_bundle)
    subset_seed = args.subset_seed if args.subset_seed is not None else pipeline_cfg["random_seed"]
    subset_rng = np.random.default_rng(subset_seed)
    X_train_q, y_train = maybe_subset_split(X_train_q, y_train, args.subset_frac, subset_rng)
    X_val_q, y_val = maybe_subset_split(X_val_q, y_val, args.subset_frac, subset_rng)
    X_test_q, y_test = maybe_subset_split(X_test_q, y_test, args.subset_frac, subset_rng)

    clf = cfg_bundle["regime_classifier"]
    clf_val_labels = clf.predict(X_val_q)
    clf_test_labels = clf.predict(X_test_q)
    short_mask_train = y_train < short_threshold
    long_mask_train = ~short_mask_train
    short_mask_val = y_val < short_threshold
    long_mask_val = ~short_mask_val
    short_val_idx = np.where(clf_val_labels == 0)[0]
    long_val_idx = np.where(clf_val_labels == 1)[0]
    short_test_idx = np.where(clf_test_labels == 0)[0]
    long_test_idx = np.where(clf_test_labels == 1)[0]
    if not np.any(short_mask_train) or not np.any(long_mask_train):
        raise ValueError(
            "Subset selection produced an empty short or long training regime. "
            "Increase --subset-frac or adjust --subset-seed."
        )

    service = QiskitRuntimeService()
    backend = service.backend(args.backend)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)

    # Estimate mode uses first top-k iteration only for fast reporting.
    if args.estimate_only:
        example_iter = cfg_bundle["top_k_indices"][0]
        angle_short = cfg_bundle["ising_params_per_iteration"][example_iter]["short"]
        rows = []
        for event_idx in range(n_total_events):
            template, _ = build_parametric_reservoir_circuit(
                angle_short[event_idx], pipeline_cfg["num_layers_per_event"], n_qubits
            )
            isa = pm.run(template)
            rows.append(estimate_resources(isa, backend, shots, len(X_train_q)))
        print_resource_report(rows, args.backend)
        return

    if not args.no_confirm:
        print("About to submit quantum jobs to IBM Runtime.")
        print(
            f"Backend={args.backend} | shots={shots} | top_k={len(cfg_bundle['top_k_indices'])} "
            f"| subset_frac={args.subset_frac}"
        )
        confirm = input("Continue? [y/N]: ").strip().lower()
        if confirm not in {"y", "yes"}:
            print("Aborted.")
            return

    iter_results = []
    batch_usage = None
    with Batch(backend=backend) as batch:
        estimator = EstimatorV2()
        estimator.options.default_shots = shots
        estimator.options.resilience_level = 1

        for iter_idx in cfg_bundle["top_k_indices"]:
            print(f"\nRunning hardware iteration={iter_idx}")
            ising_short = cfg_bundle["ising_params_per_iteration"][iter_idx]["short"]
            ising_long = cfg_bundle["ising_params_per_iteration"][iter_idx]["long"]

            P_tr_short, res_short = run_estimator_for_dataset(
                X_train_q[short_mask_train],
                ising_short,
                estimator,
                pm,
                backend,
                n_qubits,
                pipeline_cfg["num_layers_per_event"],
                shots,
                checkpoint_prefix=args.checkpoint_dir / f"iter{iter_idx}_short_train",
            )
            P_vl_short, _ = run_estimator_for_dataset(
                X_val_q[short_mask_val],
                ising_short,
                estimator,
                pm,
                backend,
                n_qubits,
                pipeline_cfg["num_layers_per_event"],
                shots,
                checkpoint_prefix=args.checkpoint_dir / f"iter{iter_idx}_short_val",
            )
            P_te_short, _ = run_estimator_for_dataset(
                X_test_q[short_test_idx],
                ising_short,
                estimator,
                pm,
                backend,
                n_qubits,
                pipeline_cfg["num_layers_per_event"],
                shots,
                checkpoint_prefix=args.checkpoint_dir / f"iter{iter_idx}_short_test",
            )

            P_tr_long, res_long = run_estimator_for_dataset(
                X_train_q[long_mask_train],
                ising_long,
                estimator,
                pm,
                backend,
                n_qubits,
                pipeline_cfg["num_layers_per_event"],
                shots,
                checkpoint_prefix=args.checkpoint_dir / f"iter{iter_idx}_long_train",
            )
            P_vl_long, _ = run_estimator_for_dataset(
                X_val_q[long_mask_val],
                ising_long,
                estimator,
                pm,
                backend,
                n_qubits,
                pipeline_cfg["num_layers_per_event"],
                shots,
                checkpoint_prefix=args.checkpoint_dir / f"iter{iter_idx}_long_val",
            )
            P_te_long, _ = run_estimator_for_dataset(
                X_test_q[long_test_idx],
                ising_long,
                estimator,
                pm,
                backend,
                n_qubits,
                pipeline_cfg["num_layers_per_event"],
                shots,
                checkpoint_prefix=args.checkpoint_dir / f"iter{iter_idx}_long_test",
            )

            H_tr_short = make_hybrid_features_decay(P_tr_short, n_total_events, n_obs)
            H_vl_short = make_hybrid_features_decay(P_vl_short, n_total_events, n_obs)
            H_te_short = make_hybrid_features_decay(P_te_short, n_total_events, n_obs)
            H_tr_long = make_hybrid_features_decay(P_tr_long, n_total_events, n_obs)
            H_vl_long = make_hybrid_features_decay(P_vl_long, n_total_events, n_obs)
            H_te_long = make_hybrid_features_decay(P_te_long, n_total_events, n_obs)

            params_short = cfg_bundle["xgb_params_per_iteration"][iter_idx]["short"]
            params_long = cfg_bundle["xgb_params_per_iteration"][iter_idx]["long"]
            model_short = XGBRegressor(**params_short)
            model_long = XGBRegressor(**params_long)

            y_tr_short = y_train[short_mask_train]
            y_tr_long = y_train[long_mask_train]
            y_vl_short = y_val[short_mask_val]
            y_vl_long = y_val[long_mask_val]
            model_short.fit(H_tr_short, y_tr_short, eval_set=[(H_vl_short, y_vl_short)], verbose=False)
            model_long.fit(H_tr_long, y_tr_long, eval_set=[(H_vl_long, y_vl_long)], verbose=False)

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

            iter_results.append(
                {
                    "iteration": int(iter_idx),
                    "val_mae": float(mean_absolute_error(y_val, val_pred)),
                    "val_rmse": float(root_mean_squared_error(y_val, val_pred)),
                    "val_r2": float(r2_score(y_val, val_pred)),
                    "test_mae": float(mean_absolute_error(y_test, test_pred)),
                    "test_rmse": float(root_mean_squared_error(y_test, test_pred)),
                    "test_pred": test_pred,
                    "resource_short": res_short,
                    "resource_long": res_long,
                }
            )
            print(f"Iteration {iter_idx} complete | test_mae={iter_results[-1]['test_mae']:.2f}")

        # Built-in IBM Runtime usage metric for the batch when available.
        try:
            batch_usage = batch.usage()
        except Exception:
            pass

    ensemble_pred = np.mean([r["test_pred"] for r in iter_results], axis=0)
    summary = {
        "backend": args.backend,
        "shots": shots,
        "subset_frac": args.subset_frac,
        "subset_seed": subset_seed,
        "iterations_run": [int(i) for i in cfg_bundle["top_k_indices"]],
        "ensemble_test_mae": float(mean_absolute_error(y_test, ensemble_pred)),
        "ensemble_test_rmse": float(root_mean_squared_error(y_test, ensemble_pred)),
        "ensemble_test_r2": float(r2_score(y_test, ensemble_pred)),
        "batch_usage": batch_usage,
    }
    save_json(args.output_dir / "hardware_summary.json", summary)
    with (args.output_dir / "hardware_results.pkl").open("wb") as f:
        pickle.dump({"iterations": iter_results, "ensemble_pred": ensemble_pred}, f)
    print(f"Saved summary -> {args.output_dir / 'hardware_summary.json'}")


if __name__ == "__main__":
    main()
