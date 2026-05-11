#!/usr/bin/env python3
"""
Hardware QRC execution using pre-tuned noisy-sim configuration.

Loads hardware_config.pkl and runs only top-k reservoir configurations on IBM hardware.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import PassManager, generate_preset_pass_manager
from qiskit.transpiler.passes import RemoveBarriers
from qiskit_ibm_runtime import Batch, EstimatorV2, QiskitRuntimeService
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor

from Preprocess import preprocess_data_window

try:
    from qiskit.transpiler.passes import ResourceEstimation
except Exception:
    ResourceEstimation = None

try:
    from qiskit.transpiler.passes import ALAPScheduleAnalysis
except Exception:
    ALAPScheduleAnalysis = None

try:
    from qiskit.transpiler.passes import PadDelay
except Exception:
    PadDelay = None


# Per-sub-job fixed overhead (IBM Runtime accounting). Applied once per pub.
# See: https://quantum.cloud.ibm.com/docs/en/guides/estimate-job-run-time
SUBJOB_OVERHEAD_S = 2.0
# Extra QPU time for TREX / measurement-mitigation calibrations (resilience_level >= 1).
RESILIENCE_CALIBRATION_S = 2.0
# Gap between shots on IBM hardware.
# Default is 250 us
REP_DELAY_S = 250e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run top-k QRC reservoirs on IBM hardware."
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to hardware_config.pkl"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("Classical/results/hardware_qrc_run")
    )
    parser.add_argument("--backend", type=str, default="ibm_sherbrooke")
    parser.add_argument(
        "--shots", type=int, default=None, help="Override shots from config"
    )
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
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Submit a single pub to IBM Runtime, print job.usage_estimation, cancel, then exit.",
    )
    parser.add_argument("--no-confirm", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    return parser.parse_args()


def scale_with_params(
    X: np.ndarray, train_min: np.ndarray, train_max: np.ndarray
) -> np.ndarray:
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
    X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocess_data_window(
        filtered_time, data_orig, n_previous_events
    )
    train_min = config_payload["scaling_params"]["train_min"]
    train_max = config_payload["scaling_params"]["train_max"]
    X_train_q = scale_with_params(X_train.to_numpy(), train_min, train_max)
    X_val_q = scale_with_params(X_val.to_numpy(), train_min, train_max)
    X_test_q = scale_with_params(X_test.to_numpy(), train_min, train_max)
    return (
        X_train_q,
        X_val_q,
        X_test_q,
        y_train.to_numpy(),
        y_val.to_numpy(),
        y_test.to_numpy(),
    )


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
                if abs(J[i, j]) > 1e-10:
                    qc.cx(i, j)
                    qc.rz(2 * J[i, j] * dt, j)
                    qc.cx(i, j)
        for i in range(n_qubits):
            qc.rx(2 * h * dt, i)


def build_parametric_reservoir_circuit(
    ising_params, num_layers: int, n_qubits: int, optimize=True
) -> Tuple[QuantumCircuit, List[Parameter]]:
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
    if optimize:
        return RemoveBarriers()(qc), thetas
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


def count_qwc_groups(observables: List[SparsePauliOp]) -> int:
    """Number of qubit-wise-commuting groups — equal to EstimatorV2's hardware sub-jobs per pub."""
    if not observables:
        return 0
    n_qubits = observables[0].num_qubits
    pauli_list = [(op.paulis[0].to_label(), 1.0) for op in observables]
    combined = SparsePauliOp.from_list(pauli_list, num_qubits=n_qubits)
    return len(combined.group_commuting(qubit_wise=True))


def estimate_resources(
    isa_circuit: QuantumCircuit,
    backend,
    shots: int,
    n_bindings: int,
    n_subjobs: int = 1,
    resilience_level: int = 0,
) -> Dict[str, float]:
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

    # Schedule the circuit before asking for a duration estimate so idle windows
    # (from gate parallelism on the coupling map) are counted, not just summed gate times.
    scheduled_circuit = isa_circuit
    if ALAPScheduleAnalysis is not None:
        try:
            passes = [ALAPScheduleAnalysis(target=backend.target)]
            if PadDelay is not None:
                try:
                    passes.append(PadDelay(target=backend.target))
                except Exception:
                    pass
            sched_pm = PassManager(passes)
            scheduled_circuit = sched_pm.run(isa_circuit)
        except Exception:
            scheduled_circuit = isa_circuit

    total_duration = None
    if hasattr(scheduled_circuit, "estimate_duration"):
        try:
            total_duration = float(
                scheduled_circuit.estimate_duration(target=backend.target, unit="s")
            )
        except Exception:
            total_duration = None
    if total_duration is None and hasattr(isa_circuit, "estimate_duration"):
        try:
            total_duration = float(
                isa_circuit.estimate_duration(target=backend.target, unit="s")
            )
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

    # IBM Runtime cost model: overhead + (rep_delay + circuit_len) * num_executions, per sub-job.
    per_subjob_wall = (
        SUBJOB_OVERHEAD_S + (total_duration + REP_DELAY_S) * shots * n_bindings
    )
    qpu_seconds = n_subjobs * per_subjob_wall
    if resilience_level >= 1:
        qpu_seconds += RESILIENCE_CALIBRATION_S * n_subjobs

    return {
        "depth": depth,
        "total_gates": size,
        "ecr_gates": float(ops.get("ecr", 0)),
        "estimated_circuit_seconds": float(total_duration),
        "n_subjobs": int(n_subjobs),
        "n_bindings": int(n_bindings),
        "shots": int(shots),
        "resilience_level": int(resilience_level),
        "est_qpu_seconds": float(qpu_seconds),
    }


def print_resource_report(resource_rows: List[Dict], backend_name: str) -> float:
    """Print per-(iteration, dataset) subtotals and a single grand total.

    Returns the aggregate QPU-seconds across all rows.
    """
    print(f"=== Resource Estimation for {backend_name} ===")
    grouped: Dict[Tuple, List[Dict]] = {}
    order: List[Tuple] = []
    for row in resource_rows:
        key = (row.get("iteration", "-"), row.get("dataset", "-"))
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(row)

    grand_total = 0.0
    for key in order:
        iter_id, dataset = key
        rows = grouped[key]
        subtotal = 0.0
        print(f"--- iter={iter_id} dataset={dataset} ---")
        for row in rows:
            ev = row.get("event", "?")
            subtotal += row["est_qpu_seconds"]
            print(
                f"  event={ev} depth={int(row['depth'])} ecr={int(row['ecr_gates'])} "
                f"total={int(row['total_gates'])} subjobs={int(row['n_subjobs'])} "
                f"bindings={int(row.get('n_bindings', 0))} est_qpu_s={row['est_qpu_seconds']:.2f}"
            )
        print(f"  subtotal: {subtotal:.2f} s")
        grand_total += subtotal

    print(f"Aggregate estimated QPU-seconds: {grand_total:.2f}")
    return grand_total


def build_pubs_for_dataset(
    X_data: np.ndarray,
    angle_bank,
    pm,
    backend,
    n_qubits: int,
    num_layers: int,
    shots: int,
    resilience_level: int,
    iter_idx: int,
    dataset_key: str,
    checkpoint_dir: Optional[Path],
):
    """Build pubs + metadata for one dataset without submitting.

    Returns:
        pauli_matrix   : (n_bindings, n_total_events * n_obs), pre-filled for
                         any events restored from checkpoint.
        pubs           : list of (isa_circuit, isa_obs, param_values) tuples.
        pub_index      : parallel list of (dataset_key, event_idx, obs_idx, n_bindings).
        resources      : per-event resource rows (with iteration/dataset/event keys).
        fresh_events   : set of event_idx values built this run (for post-submit checkpointing).
    """
    n_total_events = X_data.shape[1] // n_qubits
    n_obs = 4 * n_qubits
    pauli_matrix = np.zeros((len(X_data), n_total_events * n_obs))
    pubs: List = []
    pub_index: List = []
    resources: List = []
    fresh_events: set = set()

    if len(X_data) == 0:
        return pauli_matrix, pubs, pub_index, resources, fresh_events

    observables = build_observables(n_qubits)
    n_subjobs = count_qwc_groups(observables)

    for event_idx in range(n_total_events):
        ckpt = None
        if checkpoint_dir is not None:
            ckpt = checkpoint_dir / f"iter{iter_idx}_{dataset_key}_event{event_idx}.npy"
            if ckpt.exists():
                block = np.load(ckpt)
                pauli_matrix[:, event_idx * n_obs : (event_idx + 1) * n_obs] = block
                continue

        template, _params = build_parametric_reservoir_circuit(
            angle_bank[event_idx], num_layers, n_qubits
        )
        isa_circuit = pm.run(template)
        res_row = estimate_resources(
            isa_circuit,
            backend,
            shots,
            len(X_data),
            n_subjobs=n_subjobs,
            resilience_level=resilience_level,
        )
        res_row.update(
            {
                "iteration": int(iter_idx),
                "dataset": dataset_key,
                "event": int(event_idx),
            }
        )
        resources.append(res_row)

        isa_observables = [obs.apply_layout(isa_circuit.layout) for obs in observables]
        start_col = event_idx * n_qubits
        X_event = X_data[:, start_col : start_col + n_qubits]
        param_values = np.asarray(X_event, dtype=float)

        # Single pub per (dataset, event): observables array broadcasts against bindings.
        # Shapes: obs (n_obs, 1), params (n_bindings, n_params) -> EstimatorV2 output (n_obs, n_bindings).
        obs_array = np.array(isa_observables, dtype=object).reshape(-1, 1)
        pubs.append((isa_circuit, obs_array, param_values))
        pub_index.append((dataset_key, event_idx, len(X_data)))
        fresh_events.add(event_idx)

    return pauli_matrix, pubs, pub_index, resources, fresh_events


def make_hybrid_features_decay(
    P_matrix: np.ndarray, n_total_events: int, n_obs: int, decay: float = 0.3
):
    weights = np.array(
        [np.exp(-decay * (n_total_events - 1 - i)) for i in range(n_total_events)]
    )
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


def run_probe(
    cfg_bundle,
    pipeline_cfg,
    backend,
    pm,
    shots: int,
    n_qubits: int,
    X_sample: np.ndarray,
):
    """Submit one pub, read IBM's server-side job.usage_estimation, cancel, exit."""
    if X_sample.shape[0] < 1:
        raise ValueError("No data available for probe submission.")
    example_iter = cfg_bundle["top_k_indices"][0]
    angle_short = cfg_bundle["ising_params_per_iteration"][example_iter]["short"]
    template, _ = build_parametric_reservoir_circuit(
        angle_short[0], pipeline_cfg["num_layers_per_event"], n_qubits
    )
    isa_circuit = pm.run(template)
    observables = build_observables(n_qubits)
    isa_obs = observables[0].apply_layout(isa_circuit.layout)
    param_values = np.asarray(X_sample[:1, 0:n_qubits], dtype=float)

    print(
        "Submitting probe pub to IBM Runtime (will cancel after reading usage_estimation)..."
    )
    with Batch(backend=backend) as _batch:
        estimator = EstimatorV2()
        estimator.options.default_shots = shots
        estimator.options.resilience_level = 1
        job = estimator.run([(isa_circuit, isa_obs, param_values)], precision=None)
        usage = None
        for _ in range(10):
            usage = getattr(job, "usage_estimation", None)
            if usage:
                break
            time.sleep(1)
        print(f"IBM usage_estimation: {usage}")
        try:
            job.cancel()
            print("Probe job cancelled.")
        except Exception as exc:
            print(f"Probe cancel raised: {exc}")


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
    # Keep in sync with estimator.options.resilience_level below.
    resilience_level = 1

    X_train_q, X_val_q, X_test_q, y_train, y_val, y_test = load_data_with_scaling(
        cfg_bundle
    )
    subset_seed = (
        args.subset_seed
        if args.subset_seed is not None
        else pipeline_cfg["random_seed"]
    )
    subset_rng = np.random.default_rng(subset_seed)
    X_train_q, y_train = maybe_subset_split(
        X_train_q, y_train, args.subset_frac, subset_rng
    )
    X_val_q, y_val = maybe_subset_split(X_val_q, y_val, args.subset_frac, subset_rng)
    X_test_q, y_test = maybe_subset_split(
        X_test_q, y_test, args.subset_frac, subset_rng
    )

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

    if args.probe:
        run_probe(cfg_bundle, pipeline_cfg, backend, pm, shots, n_qubits, X_train_q)
        return

    n_subjobs_per_event = count_qwc_groups(build_observables(n_qubits))

    # Canonical (dataset_key, X, regime) specs used by both estimate-only and submission paths.
    dataset_specs = [
        ("short_train", X_train_q[short_mask_train], "short"),
        ("short_val", X_val_q[short_mask_val], "short"),
        ("short_test", X_test_q[short_test_idx], "short"),
        ("long_train", X_train_q[long_mask_train], "long"),
        ("long_val", X_val_q[long_mask_val], "long"),
        ("long_test", X_test_q[long_test_idx], "long"),
    ]

    if args.estimate_only:
        all_rows: List[Dict] = []
        for iter_idx in cfg_bundle["top_k_indices"]:
            for dataset_key, X_ds, regime in dataset_specs:
                if len(X_ds) == 0:
                    continue
                angle_bank = cfg_bundle["ising_params_per_iteration"][iter_idx][regime]
                for event_idx in range(n_total_events):
                    template, _ = build_parametric_reservoir_circuit(
                        angle_bank[event_idx],
                        pipeline_cfg["num_layers_per_event"],
                        n_qubits,
                    )
                    isa = pm.run(template)
                    row = estimate_resources(
                        isa,
                        backend,
                        shots,
                        len(X_ds),
                        n_subjobs=n_subjobs_per_event,
                        resilience_level=resilience_level,
                    )
                    row.update(
                        {
                            "iteration": int(iter_idx),
                            "dataset": dataset_key,
                            "event": int(event_idx),
                        }
                    )
                    all_rows.append(row)
        print_resource_report(all_rows, args.backend)
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

    iter_results: List[Dict] = []
    batch_usage = None
    all_resource_rows: List[Dict] = []

    with Batch(backend=backend) as batch:
        estimator = EstimatorV2()
        estimator.options.default_shots = shots
        estimator.options.resilience_level = resilience_level

        for iter_idx in cfg_bundle["top_k_indices"]:
            print(f"\nRunning hardware iteration={iter_idx}")
            ising_short = cfg_bundle["ising_params_per_iteration"][iter_idx]["short"]
            ising_long = cfg_bundle["ising_params_per_iteration"][iter_idx]["long"]
            angle_banks = {"short": ising_short, "long": ising_long}

            # Collect pubs from all six datasets for this iteration into one submission.
            pauli_matrices: Dict[str, np.ndarray] = {}
            all_pubs: List = []
            all_pub_index: List = []
            fresh_by_dataset: Dict[str, set] = {}
            iter_res_by_dataset: Dict[str, List[Dict]] = {}

            for dataset_key, X_ds, regime in dataset_specs:
                P, pubs, pub_index, res_rows, fresh = build_pubs_for_dataset(
                    X_ds,
                    angle_banks[regime],
                    pm,
                    backend,
                    n_qubits,
                    pipeline_cfg["num_layers_per_event"],
                    shots,
                    resilience_level,
                    iter_idx,
                    dataset_key,
                    args.checkpoint_dir,
                )
                pauli_matrices[dataset_key] = P
                all_pubs.extend(pubs)
                all_pub_index.extend(pub_index)
                fresh_by_dataset[dataset_key] = fresh
                iter_res_by_dataset[dataset_key] = res_rows
                all_resource_rows.extend(res_rows)

            usage_estimation = None
            if all_pubs:
                print(f"  Submitting {len(all_pubs)} pubs in a single Runtime job...")
                job = estimator.run(all_pubs, precision=None)
                results = job.result()
                usage_estimation = getattr(job, "usage_estimation", None)

                # Deinterleave results back into per-dataset Pauli matrices.
                # Each pub returns evs with shape (n_obs, n_bindings) which we transpose into (n_bindings, n_obs).
                for pub_idx, (dataset_key, event_idx, _n_bindings) in enumerate(
                    all_pub_index
                ):
                    r = results[pub_idx]
                    evs = np.asarray(r.data.evs)
                    if evs.ndim > 2:
                        evs = evs.reshape(n_obs, -1)
                    pauli_matrices[dataset_key][
                        :, event_idx * n_obs : (event_idx + 1) * n_obs
                    ] = evs.T

                # Persist per-event shards for any freshly-built (dataset, event) pairs.
                for dataset_key, fresh in fresh_by_dataset.items():
                    for event_idx in fresh:
                        block = pauli_matrices[dataset_key][
                            :, event_idx * n_obs : (event_idx + 1) * n_obs
                        ]
                        ckpt = (
                            args.checkpoint_dir
                            / f"iter{iter_idx}_{dataset_key}_event{event_idx}.npy"
                        )
                        np.save(ckpt, block)
            else:
                print("  All events restored from checkpoint; no submission needed.")

            if usage_estimation is not None:
                for res_rows in iter_res_by_dataset.values():
                    for row in res_rows:
                        row["runtime_usage_estimation"] = usage_estimation

            res_short = (
                iter_res_by_dataset.get("short_train", [])
                + iter_res_by_dataset.get("short_val", [])
                + iter_res_by_dataset.get("short_test", [])
            )
            res_long = (
                iter_res_by_dataset.get("long_train", [])
                + iter_res_by_dataset.get("long_val", [])
                + iter_res_by_dataset.get("long_test", [])
            )

            P_tr_short = pauli_matrices["short_train"]
            P_vl_short = pauli_matrices["short_val"]
            P_te_short = pauli_matrices["short_test"]
            P_tr_long = pauli_matrices["long_train"]
            P_vl_long = pauli_matrices["long_val"]
            P_te_long = pauli_matrices["long_test"]

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
            model_short.fit(
                H_tr_short,
                y_tr_short,
                eval_set=[(H_vl_short, y_vl_short)],
                verbose=False,
            )
            model_long.fit(
                H_tr_long, y_tr_long, eval_set=[(H_vl_long, y_vl_long)], verbose=False
            )

            test_pred = np.empty(len(X_test_q))
            test_pred[short_test_idx] = model_short.predict(H_te_short)
            test_pred[long_test_idx] = model_long.predict(H_te_long)

            val_pred = np.empty(len(X_val_q))
            short_val_positions = {
                idx: pos for pos, idx in enumerate(np.where(short_mask_val)[0])
            }
            long_val_positions = {
                idx: pos for pos, idx in enumerate(np.where(long_mask_val)[0])
            }
            for idx in short_val_idx:
                if idx in short_val_positions:
                    val_pred[idx] = model_short.predict(
                        H_vl_short[
                            short_val_positions[idx] : short_val_positions[idx] + 1
                        ]
                    )[0]
                elif idx in long_val_positions:
                    val_pred[idx] = model_short.predict(
                        H_vl_long[long_val_positions[idx] : long_val_positions[idx] + 1]
                    )[0]
            for idx in long_val_idx:
                if idx in long_val_positions:
                    val_pred[idx] = model_long.predict(
                        H_vl_long[long_val_positions[idx] : long_val_positions[idx] + 1]
                    )[0]
                elif idx in short_val_positions:
                    val_pred[idx] = model_long.predict(
                        H_vl_short[
                            short_val_positions[idx] : short_val_positions[idx] + 1
                        ]
                    )[0]

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
                    "runtime_usage_estimation": usage_estimation,
                }
            )
            print(
                f"Iteration {iter_idx} complete | test_mae={iter_results[-1]['test_mae']:.2f}"
            )

        # Built-in IBM Runtime usage metric for the batch when available.
        try:
            batch_usage = batch.usage()
        except Exception:
            pass

    if all_resource_rows:
        print("\n\tAggregate resource estimate across all submitted iterations")
        print_resource_report(all_resource_rows, args.backend)

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
