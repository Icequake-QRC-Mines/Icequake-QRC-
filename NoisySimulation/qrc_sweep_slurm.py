#!/usr/bin/env python3
"""

Noisy hardware studies: --noise-model fakefez --noise-sweep. Evolution-time
sweeps: --t-sweep (override --n-trotter-steps if large t needs finer Trotter).

Ridge architectures: --arch ridge-sequential reproduces (pick via --topology fc | nn) under
the shot-based noisy protocol. The notebooks' stateful reservoir is read out
by re-running PREFIX circuits (events 0..k) and measuring Z/X/Y with shots,
regime routing is loaded from a file (override via --regime-labels-csv, or pass
--regime-routing classifier to reuse the script's Shallow-Arch K-Means+XGBoost
routing instead of the CSV), and Optuna-tuned
Ridge sub-regressors replace XGBoost. Example SLURM
usage (array size = n_iterations * 2, same task layout as the default arch):
  sbatch --array=0-9 ... noisy_qrc_slurm.py --arch ridge-sequential \
      --topology fc --noise-model fakefez --task-id $SLURM_ARRAY_TASK_ID ...

Modes:
1) Full run (single process): compute quantum features + tune/train
2) Array task mode: compute one (iteration, regime, sweep-point) partial artifact
3) Aggregate mode: consume partial artifacts and run classical tuning/selection only
4) Estimate mode: print transpiled circuit resources and exit

Partial jobs persist Shallow-Arch regime routing to regime_routing.pkl so
aggregation reuses the exact train/val/test masks from the quantum run.
Use --partials-dir to aggregate from an existing run folder.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import RZZGate
from qiskit.quantum_info import Operator
from qiskit.transpiler import PassManager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    coherent_unitary_error,
    pauli_error,
)
from qiskit_ibm_runtime.fake_provider import FakeFez, FakeSherbrooke
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor


optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    # Built-in transpiler analysis bundle for depth/size/count_ops.
    from qiskit.transpiler.passes import ResourceEstimation
except Exception:
    ResourceEstimation = None


TOPOLOGY_DEFAULTS: Dict[str, Dict[str, float | int]] = {
    # Shallow-Arch.ipynb — Opt-NN-TFI
    "nn": {
        "num_layers": 1,
        "n_trotter_steps": 1,
        "h_field": 0.5,
        "t_evolution": 0.1,
    },
    # Hamiltonian-QRC(Best).ipynb — FC-TFI
    "fc": {
        "num_layers": 2,
        "n_trotter_steps": 3,
        "h_field": 1.0,
        "t_evolution": 0.5,
    },
}

N_CV_FOLDS = 3
XGB_N_ESTIMATORS = 2000
ROUTING_ARTIFACT = "regime_routing.pkl"


@dataclass
class QRCConfig:
    num_layers_per_event: int = 1
    shots: int = 4096
    n_iterations: int = 5
    top_k: int = 3
    random_seed: int = 42
    optuna_trials: int = 30
    short_threshold: int = 64_875
    n_previous_events: int = 20
    n_qubits: int = 6
    topology: str = "nn"  # "nn" | "fc"
    n_trotter_steps: int = 1
    h_field: float = 0.5
    t_evolution: float = 0.1
    arch: str = "xgb-per-event"  # "xgb-per-event" | "ridge-sequential"
    regime_labels_csv: Path | None = None  # only used by ridge-sequential
    regime_routing: str = "classifier"  # "classifier" | "csv" (csv: seq only)


@dataclass
class RegimeRouting:
    """Classifier-predicted regime masks (Shallow-Arch.ipynb routing).

    classifier is None when routing comes from a precomputed labels file
    (--regime-routing csv) instead of a trained model.
    """

    classifier: XGBClassifier | None
    short_mask_train: np.ndarray
    long_mask_train: np.ndarray
    short_mask_val: np.ndarray
    long_mask_val: np.ndarray
    short_test_idx: np.ndarray
    long_test_idx: np.ndarray
    short_val_idx: np.ndarray
    long_val_idx: np.ndarray


# ----------------------------------------------------------------------------
# Correlated-noise experiment harness.
#
# These knobs *compose on top* of NoiseModel.from_backend(FakeFez()); they
# do not replace it. With CorrelatedNoiseSpec() (all zeros) the simulator
# behaves identically to the original baseline.
#
# Inductive-bias rationale:
#   - p_zz_after_2q is the headline dial. Independent depolarizing noise
#     degrades <Z_i Z_j> smoothly. Correlated ZZ dephasing preserves single-
#     qubit Z populations but kills multi-qubit coherence, so it isolates
#     whether the QRC's predictive power leans on multi-qubit features.
#   - alpha_zz_coherent is the deterministic analog: a small RZZ(alpha)
#     after every 2q gate deforms the kernel without decohering it.
#   - readout_corr_strength tests sensitivity at the measurement layer only.
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrelatedNoiseSpec:
    """Correlated-noise knobs layered on top of a backend NoiseModel."""

    p_zz_after_2q: float = 0.0
    alpha_zz_coherent: float = 0.0
    readout_corr_strength: float = 0.0
    p_flip_readout: float = 0.00
    apply_to_pairs: str = "coupling"  # "coupling" | "nn-ring" | "all"

    @property
    def is_identity(self) -> bool:
        return (
            self.p_zz_after_2q <= 0.0
            and self.alpha_zz_coherent <= 0.0
            and self.readout_corr_strength <= 0.0
        )

    @property
    def label(self) -> str:
        if self.is_identity:
            return "baseline"
        return (
            f"pzz{self.p_zz_after_2q:g}"
            f"_azz{self.alpha_zz_coherent:g}"
            f"_rc{self.readout_corr_strength:g}"
            f"_{self.apply_to_pairs}"
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "p_zz_after_2q": self.p_zz_after_2q,
            "alpha_zz_coherent": self.alpha_zz_coherent,
            "readout_corr_strength": self.readout_corr_strength,
            "p_flip_readout": self.p_flip_readout,
            "apply_to_pairs": self.apply_to_pairs,
            "label": self.label,
        }


def _resolve_corr_pairs(backend, n_qubits: int, mode: str) -> List[Tuple[int, int]]:
    """Return (a, b) qubit pairs that correlated channels are attached to.

    "coupling" uses the backend coupling map restricted to the qubits the
    circuit actually touches (fake backends are much larger than n_qubits).
    Falls back to "nn-ring" if the restricted coupling map is empty.
    """
    if mode == "coupling":
        edges: List[Tuple[int, int]] = []
        seen = set()
        try:
            raw = list(backend.coupling_map.get_edges())
        except Exception:
            raw = []
        for a, b in raw:
            if a >= n_qubits or b >= n_qubits:
                continue
            key = tuple(sorted((int(a), int(b))))
            if key in seen:
                continue
            seen.add(key)
            edges.append((int(a), int(b)))
        if edges:
            return edges
        mode = "nn-ring"
    if mode == "nn-ring":
        return [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
    if mode == "all":
        return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    raise ValueError(f"Unknown apply_to_pairs mode: {mode}")


def _correlated_readout_error(p_flip: float, c: float) -> ReadoutError:
    """Two-qubit assignment matrix with correlation strength c in [0, 1].

    c=0 reproduces two independent per-qubit flips at rate p_flip; c=1
    sends all of the flip mass to the joint 00<->11 / 01<->10 channel.
    """
    pf = p_flip * (1.0 - c)
    pc = p_flip * c
    keep = 1.0 - 2.0 * pf - pc
    M = np.array(
        [
            [keep, pf, pf, pc],
            [pf, keep, pc, pf],
            [pf, pc, keep, pf],
            [pc, pf, pf, keep],
        ],
        dtype=float,
    )
    # Guard against tiny numerical drift before Aer normalization.
    M = np.clip(M, 0.0, 1.0)
    M = M / M.sum(axis=1, keepdims=True)
    return ReadoutError(M)


def augment_noise_model_with_correlations(
    noise_model: NoiseModel,
    backend,
    n_qubits: int,
    spec: CorrelatedNoiseSpec,
    two_qubit_gates: Tuple[str, ...] = ("ecr", "cx", "cz"),
) -> NoiseModel:
    """Compose correlated channels onto a backend-derived NoiseModel in-place.

    The original local errors from NoiseModel.from_backend are preserved;
    correlated channels are layered *in addition* via add_quantum_error,
    which composes after the targeted gate.
    """
    if spec.is_identity:
        return noise_model

    pairs = _resolve_corr_pairs(backend, n_qubits, spec.apply_to_pairs)

    if spec.p_zz_after_2q > 0.0:
        zz_err = pauli_error(
            [("ZZ", spec.p_zz_after_2q), ("II", 1.0 - spec.p_zz_after_2q)]
        )
        for a, b in pairs:
            for g in two_qubit_gates:
                noise_model.add_quantum_error(zz_err, [g], [a, b], warnings=False)

    if spec.alpha_zz_coherent > 0.0:
        coh_err = coherent_unitary_error(Operator(RZZGate(spec.alpha_zz_coherent)).data)
        for a, b in pairs:
            for g in two_qubit_gates:
                noise_model.add_quantum_error(coh_err, [g], [a, b], warnings=False)

    if spec.readout_corr_strength > 0.0:
        # NOTE: a multi-qubit readout error on [a, b] takes precedence over
        # per-qubit readout errors on those qubits at simulation time when
        # both are measured together. For clean inductive-bias experiments
        # prefer the gate-level p_zz_after_2q knob; use this only when you
        # explicitly want measurement-side correlations.
        ro_err = _correlated_readout_error(
            spec.p_flip_readout, spec.readout_corr_strength
        )
        for a, b in pairs:
            noise_model.add_readout_error(ro_err, [a, b])

    return noise_model


def build_noise_specs(args: argparse.Namespace) -> List[CorrelatedNoiseSpec]:
    """Resolve CLI args into one or more CorrelatedNoiseSpecs to run.

    --noise-sweep "v1,v2,..." expands p_zz_after_2q into that many specs
    while keeping the other knobs fixed at their scalar CLI values. Without
    it, a single spec is returned.
    """
    if args.noise_sweep:
        sweep_vals = [
            float(tok.strip()) for tok in args.noise_sweep.split(",") if tok.strip()
        ]
        if not sweep_vals:
            raise ValueError("--noise-sweep was provided but parsed empty.")
        return [
            CorrelatedNoiseSpec(
                p_zz_after_2q=v,
                alpha_zz_coherent=args.alpha_zz_coherent,
                readout_corr_strength=args.readout_corr,
                p_flip_readout=args.p_flip_readout,
                apply_to_pairs=args.apply_pairs,
            )
            for v in sweep_vals
        ]
    return [
        CorrelatedNoiseSpec(
            p_zz_after_2q=args.p_zz_corr,
            alpha_zz_coherent=args.alpha_zz_coherent,
            readout_corr_strength=args.readout_corr,
            p_flip_readout=args.p_flip_readout,
            apply_to_pairs=args.apply_pairs,
        )
    ]


@dataclass(frozen=True)
class SweepPoint:
    """One point in the experiment's sweep dimension.

    The sweep is either over Hamiltonian evolution time t (--t-sweep, the
    primary experiment) or over correlated-noise strength (--noise-sweep,
    legacy). Each point owns the partials/<label>/ subdirectory and the
    per-point aggregate artifacts.
    """

    t_evolution: float
    noise_spec: CorrelatedNoiseSpec
    label: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "t_evolution": self.t_evolution,
            "noise_spec": self.noise_spec.as_dict(),
            "label": self.label,
        }


def build_sweep_points(args: argparse.Namespace) -> List[SweepPoint]:
    """Resolve CLI args into the list of sweep points to run.

    --t-sweep "v1,v2,..." sweeps the evolution time t with noise knobs fixed
    (default: noiseless). --noise-sweep sweeps p_zz_after_2q at fixed t. The
    two are mutually exclusive. Without either, a single point is returned.
    """
    if args.t_sweep and args.noise_sweep:
        raise ValueError("--t-sweep and --noise-sweep are mutually exclusive.")
    if args.t_sweep:
        t_vals = [float(tok.strip()) for tok in args.t_sweep.split(",") if tok.strip()]
        if not t_vals:
            raise ValueError("--t-sweep was provided but parsed empty.")
        base_spec = build_noise_specs(args)[0]
        return [
            SweepPoint(t_evolution=t, noise_spec=base_spec, label=f"t{t:g}")
            for t in t_vals
        ]
    noise_specs = build_noise_specs(args)
    if args.noise_sweep:
        t_default = build_config_from_args(args).t_evolution
        return [
            SweepPoint(t_evolution=t_default, noise_spec=spec, label=spec.label)
            for spec in noise_specs
        ]
    spec = noise_specs[0]
    t_default = build_config_from_args(args).t_evolution
    label = f"t{t_default:g}" if spec.is_identity else spec.label
    return [SweepPoint(t_evolution=t_default, noise_spec=spec, label=label)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Noisy QRC pipeline with SLURM support."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("Classical/results/noisy_qrc_run")
    )
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--n-iterations", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--optuna-trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset-frac", type=float, default=1.0)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Max circuits per Aer run() batch"
    )
    parser.add_argument(
        "--max-memory-mb", type=int, default=None, help="Aer max_memory_mb override"
    )
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument(
        "--partials-dir",
        type=Path,
        default=None,
        help="Directory containing partials/ and regime_routing.pkl. "
        "Defaults to --output-dir. Use to aggregate an existing run folder "
        "while writing summaries elsewhere.",
    )
    parser.add_argument(
        "--write-routing-only",
        action="store_true",
        help="Build and save regime_routing.pkl (routing) then exit. "
        "Useful before launching SLURM partial array jobs.",
    )
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument(
        "--estimate-json",
        type=Path,
        default=None,
        help="Optional path to write structured estimate JSON.",
    )
    parser.add_argument(
        "--estimate-sweep-fracs",
        type=str,
        default=None,
        help="Comma-separated subset fractions for estimate sweep (e.g. 0.1,0.25,0.5,1.0).",
    )
    parser.add_argument(
        "--estimate-plot",
        type=Path,
        default=None,
        help="Optional output path for manuscript-ready estimate plot (png/pdf).",
    )
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    # --- Pipeline architecture ---
    parser.add_argument(
        "--arch",
        type=str,
        default="xgb-per-event",
        choices=("xgb-per-event", "ridge-sequential"),
        help="Pipeline architecture. 'xgb-per-event' (default) = existing "
        "pipeline: stateless per-event circuits, decay-weighted features, "
        "K-Means+XGBoost routing, XGBoost regressors. 'ridge-sequential' = "
        "stateful sequential reservoir read out "
        "via prefix circuits (events 0..k, measured with shots), regime "
        "routing per --regime-routing (default: --regime-labels-csv), "
        "Optuna-tuned Ridge regressors, no decay weighting. Combine with "
        "--topology fc (FC-TFI-Ridge) or nn (NN-TFI-Ridge) and "
        "--noise-model fakefez.",
    )
    parser.add_argument(
        "--regime-routing",
        type=str,
        default=None,
        choices=("csv", "classifier"),
        help="Regime routing source for --arch ridge-sequential. 'csv' "
        "(default) = precomputed labels from --regime-labels-csv, matching "
        "the noiseless Ridge notebook baselines. 'classifier' = the script's "
        "Shallow-Arch routing: K-Means labels on y_train + Optuna-tuned "
        "XGBoost classifier on the raw classical features, persisted to "
        "regime_routing.pkl and shared across SLURM tasks. --arch "
        "xgb-per-event always uses 'classifier'.",
    )
    parser.add_argument(
        "--regime-labels-csv",
        type=Path,
        default=None,
        help="Precomputed regime labels file for --arch ridge-sequential "
        "with --regime-routing csv. Defaults to "
        "ridge_noiseless_fc_classifier_events.csv next to this script "
        "(both Ridge notebooks use the FC file).",
    )
    parser.add_argument(
        "--topology",
        type=str,
        default="nn",
        choices=("nn", "fc"),
        help="Ising coupling topology. 'nn' = nearest-neighbour open chain "
        "'fc' = fully connected "
        "random couplings.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--n-trotter-steps",
        type=int,
        default=1,
        help="First-order Trotter steps per evolution call." 
    )
    parser.add_argument(
        "--h-field",
        type=float,
        default=None,
        help="Transverse-field strength h.",
    )
    parser.add_argument(
        "--t-evolution",
        type=float,
        default=None,
        help="Hamiltonian evolution time t per Trotter layer. Default: 0.1 "
        "(nn) or 0.5 (fc). Ignored when --t-sweep is set.",
    )
    parser.add_argument(
        "--t-sweep",
        type=str,
        default=None,
        help="Comma-separated evolution times to sweep (e.g. "
        "'0.05,0.1,0.25,0.5,1.0'). When set, the SLURM array grows by "
        "len(sweep) and partials are split across partials/t<value>/. "
        "Mutually exclusive with --noise-sweep.",
    )
    parser.add_argument(
        "--noise-model",
        type=str,
        default="none",
        choices=("none", "fakefez"),
        help="Base simulator noise. 'none' = ideal noiseless Aer (default for "
        "the evolution-time sweep); 'fakefez' = NoiseModel.from_backend("
        "FakeFez()) plus any correlated-noise knobs (original experiment).",
    )
    # --- Correlated-noise experiment knobs (compose on top of FakeFez) ---
    parser.add_argument(
        "--p-zz-corr",
        type=float,
        default=0.0,
        help="Stochastic joint-ZZ Pauli probability appended after every "
        "two-qubit gate on each pair from --apply-pairs. Main inductive-"
        "bias dial. Composes with the FakeFez NoiseModel.",
    )
    parser.add_argument(
        "--alpha-zz-coherent",
        type=float,
        default=0.0,
        help="Coherent RZZ(alpha) rotation appended after every two-qubit "
        "gate on each pair. Models coherent crosstalk.",
    )
    parser.add_argument(
        "--readout-corr",
        type=float,
        default=0.0,
        help="Correlation strength c in [0, 1] for joint 2q readout error. "
        "0 = independent per-qubit flips (matches baseline). Overrides "
        "per-qubit readout errors on the affected pairs when > 0.",
    )
    parser.add_argument(
        "--p-flip-readout",
        type=float,
        default=0.00,
        help="Base per-qubit flip probability for --readout-corr.",
    )
    parser.add_argument(
        "--apply-pairs",
        type=str,
        default="coupling",
        choices=("coupling", "nn-ring", "all"),
        help="Qubit pairs that correlated channels are attached to.",
    )
    parser.add_argument(
        "--noise-sweep",
        type=str,
        default=None,
        help="Comma-separated p_zz_after_2q sweep values (e.g. "
        "'0,0.005,0.01,0.02,0.05'). When set, the SLURM array grows by "
        "len(sweep) and partials are split across partials/<noise_label>/.",
    )
    # --- Multi-qubit packing experiments ---
    parser.add_argument(
        "--packing-mode",
        type=str,
        default="single",
        choices=("single", "joint-ising", "shot-parallel"),
        help="Circuit packing mode. 'single' = baseline 6q per event. "
        "'joint-ising' = pack pack_n consecutive events onto one "
        "(pack_n*n_qubits)-qubit circuit with a fully-coupled Ising "
        "disorder spanning all qubits (Exp 1). 'shot-parallel' = replicate "
        "the same 6q circuit pack_n times on (pack_n*n_qubits) qubits with "
        "block-diagonal Ising and divide shots by pack_n (Exp 2).",
    )
    parser.add_argument(
        "--pack-n",
        type=int,
        default=1,
        help="Number of events (joint-ising) or replicas (shot-parallel) "
        "packed per circuit. Total circuit qubits = n_qubits * pack_n. "
        "Required >= 2 when --packing-mode != single.",
    )
    parser.add_argument(
        "--max-parallel-experiments",
        type=int,
        default=1,
        help="Aer max_parallel_experiments. Increase on GPU to amortize "
        "kernel launch overhead across batched circuits. Set to 0 for "
        "Aer auto-tuning.",
    )
    parser.add_argument(
        "--sim-method",
        type=str,
        default="density_matrix",
        choices=("density_matrix", "automatic", "statevector", "matrix_product_state"),
        help="Aer simulation method. density_matrix is ~3-4x faster than "
        "automatic for noisy <=12 qubit circuits because it computes the "
        "exact noisy density matrix once per parameter binding and then "
        "samples shots cheaply. Memory is 2^(2n) complex = 256MB at 12q. "
        "Use 'automatic' to defer to Aer; 'matrix_product_state' for "
        "larger circuits with limited entanglement.",
    )
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> QRCConfig:
    """Resolve topology-aware Hamiltonian defaults"""
    topo = args.topology
    if topo not in TOPOLOGY_DEFAULTS:
        raise ValueError(f"Unknown topology: {topo}")
    defaults = TOPOLOGY_DEFAULTS[topo]
    t_evolution = (
        float(args.t_evolution)
        if args.t_evolution is not None
        else float(defaults["t_evolution"])
    )
    if args.num_layers is not None:
        num_layers = int(args.num_layers)
    elif args.arch == "ridge-sequential":
        # FC-TFI-Ridge.ipynb and NN-TFI-Ridge.ipynb both use 2 layers/event
        # (the nn topology default of 1 comes from Shallow-Arch.ipynb).
        num_layers = 2
    else:
        num_layers = int(defaults["num_layers"])
    if args.arch == "ridge-sequential" and args.packing_mode != "single":
        raise ValueError(
            "--arch ridge-sequential does not support --packing-mode "
            f"{args.packing_mode!r}; the sequential reservoir is 6-qubit only."
        )
    regime_labels_csv = (
        args.regime_labels_csv
        if args.regime_labels_csv is not None
        else Path(__file__).resolve().parent / "ridge_noiseless_fc_classifier_events.csv"
    )
    if args.regime_routing is None:
        regime_routing = "csv" if args.arch == "ridge-sequential" else "classifier"
    else:
        regime_routing = args.regime_routing
        if args.arch != "ridge-sequential" and regime_routing != "classifier":
            raise ValueError(
                "--regime-routing csv is only supported with "
                "--arch ridge-sequential; the xgb-per-event pipeline always "
                "uses the Shallow-Arch classifier routing."
            )
    return QRCConfig(
        num_layers_per_event=num_layers,
        shots=args.shots,
        n_iterations=args.n_iterations,
        top_k=args.top_k,
        random_seed=args.seed,
        optuna_trials=args.optuna_trials,
        topology=topo,
        n_trotter_steps=int(
            args.n_trotter_steps
            if args.n_trotter_steps is not None
            else defaults["n_trotter_steps"]
        ),
        h_field=float(
            args.h_field if args.h_field is not None else defaults["h_field"]
        ),
        t_evolution=t_evolution,
        arch=args.arch,
        regime_labels_csv=regime_labels_csv,
        regime_routing=regime_routing,
    )


# ----------------------------------------------------------------------------
# Multi-qubit packing harness.
#
# Two new experiments layer on top of the baseline 6q/event pipeline:
#
# Exp 1 - "joint-ising": pack pack_n consecutive history events onto one
#   (pack_n * n_qubits)-qubit circuit with a single fully-coupled Ising
#   disorder spanning ALL qubits. Each event's data is bound to its own
#   6-qubit block (event A -> qubits [0..n), event B -> qubits [n..2n)).
#   The per-event observable slicing keeps the baseline feature count
#   (4 * n_qubits per event); the boundary ZZ correlator z[n-1] (qubit
#   n-1 <-> qubit n, periodic) automatically encodes cross-event quantum
#   correlations. Shots stay at base shots; we use the same circuit for
#   the pair.
#
# Exp 2 - "shot-parallel": replicate the SAME 6q Ising circuit pack_n
#   times on (pack_n * n_qubits) qubits (block-diagonal J, shared thetas),
#   run with shots // pack_n shots, and sum per-block counts. Each block
#   is an i.i.d. sample of the same single-event distribution, so the
#   total effective shots match the baseline. On real hardware this
#   parallelizes across physical qubits and saves wall time; on simulator
#   it costs MORE (statevector dim grows 2^n) and is mainly useful as a
#   fidelity check that packing doesn't introduce inter-block leakage
#   under noise. The "fidelity-check" framing is the right one here.
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class PackingSpec:
    """How to pack circuits onto a wider qubit register."""

    mode: str = "single"  # "single" | "joint-ising" | "shot-parallel"
    pack_n: int = 1

    def __post_init__(self):
        if self.mode not in ("single", "joint-ising", "shot-parallel"):
            raise ValueError(f"Unknown packing mode: {self.mode}")
        if self.mode == "single" and self.pack_n != 1:
            raise ValueError("pack_n must be 1 for single mode")
        if self.mode != "single" and self.pack_n < 2:
            raise ValueError(
                f"pack_n must be >= 2 for mode {self.mode!r} (got {self.pack_n})"
            )

    def circuit_qubits(self, n_qubits_per_event: int) -> int:
        return n_qubits_per_event * self.pack_n

    def effective_shots(self, base_shots: int) -> int:
        if self.mode == "shot-parallel":
            return max(1, base_shots // self.pack_n)
        return base_shots

    @property
    def label(self) -> str:
        if self.mode == "single":
            return "single"
        return f"{self.mode}_N{self.pack_n}"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "pack_n": self.pack_n,
            "label": self.label,
        }


def packing_partials_root(out_dir: Path, packing: PackingSpec) -> Path:
    """partials/ for single (backward-compatible); partials/pack_<label>/ otherwise."""
    if packing.mode == "single":
        return out_dir / "partials"
    return out_dir / "partials" / f"pack_{packing.label}"


def _trotter_ising_layer_offset(
    qc: QuantumCircuit,
    n_qubits_block: int,
    offset: int,
    J: np.ndarray,
    h: float,
    t: float,
    n_trotter_steps: int = 3,
) -> None:
    """Apply a Trotter-Ising layer on qubits [offset, offset+n_qubits_block).

    Identical to trotter_ising_layer but shifted by `offset`. Used by the
    shot-parallel builder to place pack_n block-diagonal copies of the same
    Ising disorder on disjoint qubit ranges.
    """
    dt = t / n_trotter_steps
    for _ in range(n_trotter_steps):
        for i in range(n_qubits_block):
            for j in range(i + 1, n_qubits_block):
                if abs(J[i, j]) > 1e-10:
                    qc.cx(offset + i, offset + j)
                    qc.rz(2 * J[i, j] * dt, offset + j)
                    qc.cx(offset + i, offset + j)
        for i in range(n_qubits_block):
            qc.rx(2 * h * dt, offset + i)


def build_replicated_reservoir_circuit(
    ising_params,
    num_layers: int,
    n_qubits_per_event: int,
    pack_n: int,
    n_trotter_steps: int = 3,
) -> Tuple[QuantumCircuit, List[Parameter]]:
    """Build pack_n independent block-diagonal copies sharing parameters.

    Returns a (pack_n * n_qubits_per_event)-qubit circuit and the SAME
    n_qubits_per_event-length Parameter list used in every block. Bind one
    event's 6 features and every block ends up with the same per-row state.
    """
    J, h, t = ising_params
    total_qubits = n_qubits_per_event * pack_n
    thetas = [Parameter(f"theta_{i}") for i in range(n_qubits_per_event)]
    qc = QuantumCircuit(total_qubits)
    for b in range(pack_n):
        off = b * n_qubits_per_event
        for i in range(n_qubits_per_event):
            qc.h(off + i)
        for i in range(n_qubits_per_event):
            qc.ry(thetas[i], off + i)
    qc.barrier()
    for _ in range(num_layers):
        for b in range(pack_n):
            _trotter_ising_layer_offset(
                qc, n_qubits_per_event, b * n_qubits_per_event, J, h, t,
                n_trotter_steps,
            )
        qc.barrier()
        for b in range(pack_n):
            off = b * n_qubits_per_event
            for i in range(n_qubits_per_event):
                qc.ry(thetas[i], off + i)
        qc.barrier()
        for b in range(pack_n):
            _trotter_ising_layer_offset(
                qc, n_qubits_per_event, b * n_qubits_per_event, J, h, t,
                n_trotter_steps,
            )
        qc.barrier()
    return qc, thetas


def _split_packed_bitstring(
    bits: str, n_qubits_per_event: int, pack_n: int, block_idx: int
) -> str:
    """Slice a packed Qiskit bitstring into block_idx's 6-bit substring.

    Qiskit's get_counts returns big-endian strings (leftmost char = highest
    qubit). Block layout: block 0 = qubits [0..n), block 1 = qubits [n..2n).
    So block 0's substring is the RIGHTMOST n characters, block 1 the next
    n to the left, etc.
    """
    start = (pack_n - 1 - block_idx) * n_qubits_per_event
    end = (pack_n - block_idx) * n_qubits_per_event
    return bits[start:end]


def _per_block_marginal_counts(
    counts: Dict[str, int], n_qubits_per_event: int, pack_n: int
) -> List[Dict[str, int]]:
    """Marginalize joint counts onto each block independently.

    Used by joint-ising: each block measures a different event so we want
    each block's marginal distribution at the FULL shot count.
    """
    per_block: List[Dict[str, int]] = [{} for _ in range(pack_n)]
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        for b in range(pack_n):
            sub = _split_packed_bitstring(bits, n_qubits_per_event, pack_n, b)
            per_block[b][sub] = per_block[b].get(sub, 0) + count
    return per_block


def _aggregate_replica_counts(
    counts: Dict[str, int], n_qubits_per_event: int, pack_n: int
) -> Dict[str, int]:
    """Sum block sub-bitstring counts as i.i.d. samples of the same dist.

    Used by shot-parallel: blocks are independent copies of the SAME event,
    so each shot yields pack_n samples that can be summed. The returned
    dict has total counts == pack_n * shots == base_shots.
    """
    combined: Dict[str, int] = {}
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        for b in range(pack_n):
            sub = _split_packed_bitstring(bits, n_qubits_per_event, pack_n, b)
            combined[sub] = combined.get(sub, 0) + count
    return combined


def build_angle_bank(
    packing: PackingSpec,
    n_qubits_per_event: int,
    n_total_events: int,
    rng: np.random.Generator,
    topology: str = "nn",
    h: float = 0.5,
    t: float = 0.1,
) -> List[Tuple[np.ndarray, float, float]]:
    """Generate one Ising disorder tuple per circuit slot for the given packing.

    joint-ising: ceil(n_total_events / pack_n) entries. Full groups get a
        (pack_n * n_qubits_per_event)-qubit J; the last (partial) group
        gets a fall-back 6-qubit J. Pair-from-start, so the *current* event
        (the highest event_idx) is the one that ends up as a singleton
        when n_total_events % pack_n != 0 -- preserves baseline semantics
        for the most predictive event.
    shot-parallel / single: n_total_events entries, each a 6-qubit J.

    t does not consume rng draws, so a t-sweep with the same seed sees
    identical J disorder at every sweep point.
    """
    bank: List[Tuple[np.ndarray, float, float]] = []
    if packing.mode == "joint-ising":
        n_full = n_total_events // packing.pack_n
        for _ in range(n_full):
            bank.append(
                generate_ising_params(
                    packing.pack_n * n_qubits_per_event, rng, topology, h=h, t=t
                )
            )
        remainder = n_total_events - n_full * packing.pack_n
        for _ in range(remainder):
            bank.append(
                generate_ising_params(n_qubits_per_event, rng, topology, h=h, t=t)
            )
    else:
        for _ in range(n_total_events):
            bank.append(
                generate_ising_params(n_qubits_per_event, rng, topology, h=h, t=t)
            )
    return bank


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


def generate_ising_params(
    n_qubits: int,
    rng: np.random.Generator,
    topology: str = "nn",
    J_std: float = 0.5,
    h: float = 0.5,
    t: float = 0.1,
):
    """Sample a random Ising disorder tuple (J, h, t).

    topology="nn": nearest-neighbour open chain, J_{i,i+1} ~ U(-J_std, J_std)
        (Opt-NN-TFI from Shallow-Arch.ipynb; h/t defaults follow the sweet
        spot in Fig. 11 of Hamhoum et al. 2025).
    topology="fc": fully connected, J_ij ~ N(0, J_std) for all i < j. Pair
        with h=1.0, t=0.5 to reproduce the original FC-TFI runs exactly.
    """
    J = np.zeros((n_qubits, n_qubits))
    if topology == "nn":
        for i in range(n_qubits - 1):
            J[i, i + 1] = rng.uniform(-J_std, J_std)
    elif topology == "fc":
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                J[i, j] = rng.normal(0, J_std)
    else:
        raise ValueError(f"Unknown topology: {topology}")
    return J, h, t


def trotter_ising_layer(
    qc: QuantumCircuit,
    n_qubits: int,
    J: np.ndarray,
    h: float,
    t: float,
    n_trotter_steps: int = 1,
    topology: str = "nn",
) -> None:
    """First-order Trotterization of the TFI Hamiltonian.

    NN topology uses an explicit open-chain loop (Shallow-Arch.ipynb); FC uses
    all i<j pairs with non-zero J (Hamiltonian-QRC(Best).ipynb).
    """
    dt = t / n_trotter_steps
    for _ in range(n_trotter_steps):
        if topology == "nn":
            for i in range(n_qubits - 1):
                if abs(J[i, i + 1]) > 1e-10:
                    qc.cx(i, i + 1)
                    qc.rz(2 * J[i, i + 1] * dt, i + 1)
                    qc.cx(i, i + 1)
        else:
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    if abs(J[i, j]) > 1e-10:
                        qc.cx(i, j)
                        qc.rz(2 * J[i, j] * dt, j)
                        qc.cx(i, j)
        for i in range(n_qubits):
            qc.rx(2 * h * dt, i)


def build_parametric_reservoir_circuit(
    ising_params,
    num_layers: int,
    n_qubits: int,
    n_trotter_steps: int = 1,
    topology: str = "nn",
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
        trotter_ising_layer(qc, n_qubits, J, h, t, n_trotter_steps, topology)
        qc.barrier()
        for i in range(n_qubits):
            qc.ry(thetas[i], i)
        qc.barrier()
        trotter_ising_layer(qc, n_qubits, J, h, t, n_trotter_steps, topology)
        qc.barrier()
    return qc, thetas


def build_parametric_sequential_prefix(
    angle_bank,
    upto_event: int,
    num_layers: int,
    n_qubits: int,
    n_trotter_steps: int = 1,
    topology: str = "nn",
) -> Tuple[QuantumCircuit, List[Parameter]]:
    """Prefix circuit for the stateful reservoir (FC/NN-TFI-Ridge notebooks).

    Reproduces build_event_step_circuit composed over events 0..upto_event:
    H on every qubit once (event 0 only), then per event
    Ry-encode + num_layers x [Trotter-Ising, Ry-encode, Trotter-Ising],
    with each event using its own disorder tuple from angle_bank and its own
    six Parameters. The returned parameter list is event-major then
    qubit-minor, matching the column layout of the scaled feature matrix, so
    _run_batched can bind X_data[:, : (upto_event + 1) * n_qubits] directly.
    """
    qc = QuantumCircuit(n_qubits)
    params: List[Parameter] = []
    for i in range(n_qubits):
        qc.h(i)
    for event_idx in range(upto_event + 1):
        J, h, t = angle_bank[event_idx]
        thetas = [Parameter(f"theta_e{event_idx}_q{i}") for i in range(n_qubits)]
        params.extend(thetas)
        for i in range(n_qubits):
            qc.ry(thetas[i], i)
        qc.barrier()
        for _ in range(num_layers):
            trotter_ising_layer(qc, n_qubits, J, h, t, n_trotter_steps, topology)
            qc.barrier()
            for i in range(n_qubits):
                qc.ry(thetas[i], i)
            qc.barrier()
            trotter_ising_layer(qc, n_qubits, J, h, t, n_trotter_steps, topology)
            qc.barrier()
    return qc, params


def build_noisy_simulator(
    device: str,
    max_memory_mb: int | None,
    noise_spec: CorrelatedNoiseSpec | None = None,
    n_qubits: int = 6,
    max_parallel_experiments: int = 1,
    sim_method: str = "density_matrix",
    noise_backend: str = "fakefez",
):
    """Build (fake_backend, simulator, local_transpile).

    noise_backend="none" returns an ideal noiseless AerSimulator (the
    evolution-time-sweep experiment). Transpilation still targets the FakeFez
    basis gates either way, so depth/resource estimates stay comparable
    between noisy and ideal runs.
    """
    fake_backend = FakeFez()
    if noise_backend == "none":
        sim_kwargs: Dict[str, Any] = {
            "max_parallel_experiments": max_parallel_experiments,
        }
        if max_memory_mb is not None:
            sim_kwargs["max_memory_mb"] = max_memory_mb
        if device == "gpu":
            gpu_kwargs = dict(sim_kwargs)
            gpu_kwargs["device"] = "GPU"
            try:
                simulator = AerSimulator(**gpu_kwargs)
                probe = QuantumCircuit(1)
                probe.measure_all()
                simulator.run([probe], shots=1).result()
            except Exception as exc:
                msg = str(exc)
                if 'Simulation device "GPU" is not supported' not in msg:
                    raise
                print(
                    '[WARN] Requested --device gpu, but Aer GPU is unavailable on this '
                    "system. Falling back to CPU simulator."
                )
                simulator = AerSimulator(**sim_kwargs)
        else:
            simulator = AerSimulator(**sim_kwargs)

        # Shallow-Arch.ipynb: ideal Aer, native CX/RZ/RX — no hardware transpile.
        def local_transpile(circuit: QuantumCircuit) -> QuantumCircuit:
            return circuit

        return fake_backend, simulator, local_transpile

    noise_model = NoiseModel.from_backend(fake_backend)
    if noise_spec is not None and not noise_spec.is_identity:
        noise_model = augment_noise_model_with_correlations(
            noise_model, fake_backend, n_qubits, noise_spec
        )
    sim_kwargs: Dict[str, Any] = {
        # CPU laptops: keep 1 to avoid concurrent OOM. GPU: bump via CLI flag.
        # max_parallel_experiments=0 lets Aer auto-tune.
        "max_parallel_experiments": max_parallel_experiments,
        # density_matrix is what Aer's "automatic" method picks for the
        # FakeFez Kraus noise model anyway; making it explicit avoids
        # surprises if Aer's heuristics change. Memory is 2^(2n) complex
        # amplitudes: ~256MB at 12q, fine for any GPU; would be 4 TB at
        # 18q so use "matrix_product_state" or "tensor_network" beyond that.
        "method": sim_method,
    }
    if noise_backend != "none":
        sim_kwargs["noise_model"] = noise_model
    if max_memory_mb is not None:
        sim_kwargs["max_memory_mb"] = max_memory_mb
    if device == "gpu":
        gpu_kwargs = dict(sim_kwargs)
        gpu_kwargs["device"] = "GPU"
        try:
            simulator = AerSimulator(**gpu_kwargs)
            # Aer may accept device="GPU" at construction and fail only at
            # execution time. Probe once here so SLURM array tasks fail-safe.
            probe = QuantumCircuit(1)
            probe.measure_all()
            simulator.run([probe], shots=1).result()
        except Exception as exc:
            msg = str(exc)
            if 'Simulation device "GPU" is not supported' not in msg:
                raise
            print(
                '[WARN] Requested --device gpu, but Aer GPU is unavailable on this '
                "system. Falling back to CPU simulator."
            )
            simulator = AerSimulator(**sim_kwargs)
    else:
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


def _counts_to_exp_and_zz(counts, n_qubits: int, shots: int, open_chain: bool = False):
    """Z expectations + NN ZZ correlators.

    open_chain=True drops the wrap-around <Z_{n-1} Z_0> term (left at 0),
    matching the open-chain NN-topology observable set in Shallow-Arch.ipynb.
    The zz vector stays length n_qubits either way, so feature shapes are
    identical across topologies.
    """
    zexp = np.zeros(n_qubits)
    zz = np.zeros(n_qubits)
    n_zz = n_qubits - 1 if open_chain else n_qubits
    for bitstring, count in counts.items():
        for q in range(n_qubits):
            bi = _parse_bit(bitstring, n_qubits, q)
            zexp[q] += (1 - 2 * bi) * count / shots
            if q < n_zz:
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


def estimate_resources(
    isa_circuit: QuantumCircuit, backend, shots: int, n_bindings: int
) -> Dict[str, float]:
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
            total_duration = float(
                isa_circuit.estimate_duration(target=backend.target, unit="s")
            )
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


def _prepare_basis_templates(
    template: QuantumCircuit, local_transpile
) -> Dict[str, QuantumCircuit]:
    """Build the Z/X/Y basis-included measurement templates ONCE and
    transpile each to ISA gates. Doing this once per event (rather than
    per row) avoids repeated transpilation work and lets the basis-rotation
    gates pick up the FakeFez noise correctly.
    """
    return {
        b: local_transpile(add_measurement_basis(template, b)) for b in ("Z", "X", "Y")
    }


def _run_batched(
    isa_templates: Dict[str, QuantumCircuit],
    params: List[Parameter],
    X_event: np.ndarray,
    simulator: AerSimulator,
    shots: int,
    batch_size: int,
) -> Tuple[List[Dict[str, int]], List[Dict[str, int]], List[Dict[str, int]], float]:
    """Run Z/X/Y measurements via a SINGLE Aer call per row-batch.

    Two compounding wins over the previous per-basis loop:
    1. parameter_binds: Aer binds parameters internally over a single
       compiled circuit instead of receiving N pre-bound circuit copies.
       This eliminates a Python-side bind-and-copy step and lets Aer
       reuse the noise-model compilation across all bindings.
    2. Combined Z+X+Y submission: one Aer invocation receives all three
       basis variants at once, so max_parallel_experiments can fan them
       out in parallel (most useful on GPU) instead of serializing
       three separate runs.

    Measured ~2.2x speedup on 6-qubit FakeFez noisy circuits versus the
    original code path on a single CPU thread (33s -> 15s for 16 rows
    with 4096 shots). On GPU with max_parallel_experiments > 1 the gain
    is larger because the three basis circuits can run concurrently.
    Behavior is statistically equivalent -- same noise model, same
    shot count -- so features agree within shot noise.
    """
    n_rows = X_event.shape[0]
    n_params = len(params)
    z_counts: List[Dict[str, int]] = []
    x_counts: List[Dict[str, int]] = []
    y_counts: List[Dict[str, int]] = []
    aer_time_taken = 0.0
    circuits_zxy = [isa_templates["Z"], isa_templates["X"], isa_templates["Y"]]
    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        n_batch = end - start
        # Aer's parameter_binds: per-circuit dict mapping Parameter to a
        # list of values, one entry per binding. Same binds for all three
        # bases since Z/X/Y templates share Parameter identities (basis
        # rotation is appended in-place via .copy()).
        binds = {
            p: X_event[start:end, i].astype(float).tolist()
            for i, p in enumerate(params[:n_params])
        }
        result = simulator.run(
            circuits_zxy,
            shots=shots,
            parameter_binds=[binds, binds, binds],
        ).result()
        aer_time_taken += float(getattr(result, "time_taken", 0.0))
        # Aer lays out experiments as: [c0 bind 0..n-1, c1 bind 0..n-1, c2 bind 0..n-1].
        # So Z -> [0, n_batch), X -> [n_batch, 2*n_batch), Y -> [2*n_batch, 3*n_batch).
        for idx in range(n_batch):
            z_counts.append(result.get_counts(idx))
            x_counts.append(result.get_counts(n_batch + idx))
            y_counts.append(result.get_counts(2 * n_batch + idx))
    return z_counts, x_counts, y_counts, aer_time_taken


def _features_from_counts(
    counts_z: Dict[str, int],
    counts_x: Dict[str, int],
    counts_y: Dict[str, int],
    n_qubits: int,
    shots: int,
    open_chain: bool = False,
) -> np.ndarray:
    """Compute the per-event 4*n_qubits feature vector from Z/X/Y counts."""
    zexp, zz = _counts_to_exp_and_zz(counts_z, n_qubits, shots, open_chain)
    xexp = _counts_to_basis_exp(counts_x, n_qubits, shots)
    yexp = _counts_to_basis_exp(counts_y, n_qubits, shots)
    return np.concatenate([zexp, xexp, yexp, zz])


def _try_load_checkpoint(
    ckpt: Path | None, expected_shape: Tuple[int, ...], resume: bool
) -> np.ndarray | None:
    """Load a per-event checkpoint when --resume and shape still matches."""
    if ckpt is None or not resume or not ckpt.exists():
        return None
    block = np.load(ckpt)
    if tuple(block.shape) != expected_shape:
        print(
            f"[WARN] Stale checkpoint {ckpt.name}: shape {block.shape} "
            f"!= expected {expected_shape} (routing or data changed); recomputing."
        )
        return None
    return block


def _run_single_event_mode(
    X_data: np.ndarray,
    angle_bank,
    cfg: QRCConfig,
    simulator: AerSimulator,
    local_transpile,
    backend,
    checkpoint_prefix: Path | None,
    resume: bool,
    batch_size: int,
):
    """Baseline path: one 6q event per circuit, full shots."""
    m = X_data.shape[0]
    n_obs = 4 * cfg.n_qubits
    n_total_events = cfg.n_previous_events + 1
    pauli_matrix = np.zeros((m, n_total_events * n_obs))
    resources = []

    for event_idx in range(n_total_events):
        ckpt = (
            None
            if checkpoint_prefix is None
            else checkpoint_prefix.with_name(
                f"{checkpoint_prefix.name}_event{event_idx}.npy"
            )
        )
        if ckpt and resume and ckpt.exists():
            block = _try_load_checkpoint(ckpt, (m, n_obs), resume)
            if block is not None:
                pauli_matrix[:, event_idx * n_obs : (event_idx + 1) * n_obs] = block
                continue

        start_col = event_idx * cfg.n_qubits
        end_col = start_col + cfg.n_qubits
        X_event = X_data[:, start_col:end_col]
        template, params = build_parametric_reservoir_circuit(
            angle_bank[event_idx],
            cfg.num_layers_per_event,
            cfg.n_qubits,
            cfg.n_trotter_steps,
            cfg.topology,
        )
        isa_templates = _prepare_basis_templates(template, local_transpile)
        # Estimate from the Z-basis template (X/Y differ only by one extra
        # basis-rotation gate; the difference is negligible vs the reservoir
        # body). n_bindings here counts 1 Aer run per row covering all 3 bases.
        resources.append(
            estimate_resources(isa_templates["Z"], backend, cfg.shots, 3 * len(X_event))
        )

        z_counts, x_counts, y_counts, aer_time = _run_batched(
            isa_templates, params, X_event, simulator, cfg.shots, batch_size
        )

        event_block = np.zeros((m, n_obs))
        for sample_idx in range(m):
            event_block[sample_idx] = _features_from_counts(
                z_counts[sample_idx],
                x_counts[sample_idx],
                y_counts[sample_idx],
                cfg.n_qubits,
                cfg.shots,
                open_chain=(cfg.topology == "nn"),
            )

        pauli_matrix[:, event_idx * n_obs : (event_idx + 1) * n_obs] = event_block
        if ckpt:
            np.save(ckpt, event_block)
        resources[-1]["aer_time_taken_seconds"] = aer_time
        print(
            f"\tEvent {event_idx + 1}/{n_total_events} complete | aer_time={aer_time:.2f}s"
        )

    return pauli_matrix, resources


def _run_joint_ising_mode(
    X_data: np.ndarray,
    angle_bank,
    cfg: QRCConfig,
    packing: PackingSpec,
    simulator: AerSimulator,
    local_transpile,
    backend,
    checkpoint_prefix: Path | None,
    resume: bool,
    batch_size: int,
):
    """Exp 1: pack pack_n consecutive events onto one (pack_n * 6)-qubit circuit
    with a single fully-coupled Ising disorder. Pair-from-start so the last
    (possibly singleton) group contains the most recent events.

    Per-event features are sliced from the joint observable vector. The ZZ
    correlator at the inter-block boundary (qubit n-1 <-> qubit n, periodic)
    naturally encodes cross-event quantum correlation -- that's the whole
    point of this mode.
    """
    m = X_data.shape[0]
    n_obs = 4 * cfg.n_qubits
    n_total_events = cfg.n_previous_events + 1
    pauli_matrix = np.zeros((m, n_total_events * n_obs))
    resources = []

    bank_idx = 0
    event_idx = 0
    while event_idx < n_total_events:
        group_size = min(packing.pack_n, n_total_events - event_idx)
        group_events = list(range(event_idx, event_idx + group_size))
        ckpt = (
            None
            if checkpoint_prefix is None
            else checkpoint_prefix.with_name(
                f"{checkpoint_prefix.name}_group{bank_idx}.npy"
            )
        )
        if ckpt and resume and ckpt.exists():
            blocks = _try_load_checkpoint(ckpt, (m, group_size * n_obs), resume)
            if blocks is not None:
                for k, ev in enumerate(group_events):
                    pauli_matrix[:, ev * n_obs : (ev + 1) * n_obs] = blocks[
                        :, k * n_obs : (k + 1) * n_obs
                    ]
                event_idx += group_size
                bank_idx += 1
                continue

        # Collect this group's data: each row is (group_size * n_qubits) wide,
        # concatenated as [event_idx_data | event_idx+1_data | ...].
        X_group = np.concatenate(
            [
                X_data[:, ev * cfg.n_qubits : (ev + 1) * cfg.n_qubits]
                for ev in group_events
            ],
            axis=1,
        )

        # Full group: build the wide joint-Ising circuit.
        # Partial group (last remainder): fall back to single-event circuit;
        # those events stay on 6 qubits and behave like baseline.
        if group_size == packing.pack_n:
            circuit_qubits = packing.circuit_qubits(cfg.n_qubits)
            template, params = build_parametric_reservoir_circuit(
                angle_bank[bank_idx],
                cfg.num_layers_per_event,
                circuit_qubits,
                cfg.n_trotter_steps,
                cfg.topology,
            )
        else:
            circuit_qubits = cfg.n_qubits
            template, params = build_parametric_reservoir_circuit(
                angle_bank[bank_idx],
                cfg.num_layers_per_event,
                cfg.n_qubits,
                cfg.n_trotter_steps,
                cfg.topology,
            )

        isa_templates = _prepare_basis_templates(template, local_transpile)
        resources.append(
            estimate_resources(isa_templates["Z"], backend, cfg.shots, 3 * m)
        )

        z_counts, x_counts, y_counts, aer_time = _run_batched(
            isa_templates, params, X_group, simulator, cfg.shots, batch_size
        )

        if group_size == packing.pack_n:
            # Compute joint 4*circuit_qubits features once, then slice per event.
            for sample_idx in range(m):
                # open_chain only drops the outermost wrap-around term; the
                # inter-block boundary correlators are interior NN pairs and
                # survive, which is the point of joint-ising.
                joint = _features_from_counts(
                    z_counts[sample_idx],
                    x_counts[sample_idx],
                    y_counts[sample_idx],
                    circuit_qubits,
                    cfg.shots,
                    open_chain=(cfg.topology == "nn"),
                )
                # joint = [zexp(0..C-1), xexp(0..C-1), yexp(0..C-1), zz(0..C-1)]
                # Slice each event's 4*n_qubits features. The ZZ entries at
                # boundaries (zz[k*n_qubits + n_qubits - 1]) encode the
                # cross-event correlator -- this is by design.
                for k, ev in enumerate(group_events):
                    q_start = k * cfg.n_qubits
                    q_end = q_start + cfg.n_qubits
                    zexp_k = joint[q_start:q_end]
                    xexp_k = joint[circuit_qubits + q_start : circuit_qubits + q_end]
                    yexp_k = joint[
                        2 * circuit_qubits + q_start : 2 * circuit_qubits + q_end
                    ]
                    zz_k = joint[
                        3 * circuit_qubits + q_start : 3 * circuit_qubits + q_end
                    ]
                    feats = np.concatenate([zexp_k, xexp_k, yexp_k, zz_k])
                    pauli_matrix[:, ev * n_obs : (ev + 1) * n_obs][sample_idx] = feats
        else:
            # Singleton remainder: standard per-event extraction.
            ev = group_events[0]
            for sample_idx in range(m):
                pauli_matrix[:, ev * n_obs : (ev + 1) * n_obs][sample_idx] = (
                    _features_from_counts(
                        z_counts[sample_idx],
                        x_counts[sample_idx],
                        y_counts[sample_idx],
                        cfg.n_qubits,
                        cfg.shots,
                        open_chain=(cfg.topology == "nn"),
                    )
                )

        if ckpt:
            slab = np.concatenate(
                [pauli_matrix[:, ev * n_obs : (ev + 1) * n_obs] for ev in group_events],
                axis=1,
            )
            np.save(ckpt, slab)
        resources[-1]["aer_time_taken_seconds"] = aer_time
        print(
            f"\tGroup {bank_idx} events={group_events} "
            f"qubits={circuit_qubits} | aer_time={aer_time:.2f}s"
        )

        event_idx += group_size
        bank_idx += 1

    return pauli_matrix, resources


def _run_shot_parallel_mode(
    X_data: np.ndarray,
    angle_bank,
    cfg: QRCConfig,
    packing: PackingSpec,
    simulator: AerSimulator,
    local_transpile,
    backend,
    checkpoint_prefix: Path | None,
    resume: bool,
    batch_size: int,
):
    """Exp 2: replicate the same 6q event circuit pack_n times on
    (pack_n * 6) qubits (block-diagonal J, shared thetas), with shots
    divided by pack_n. Per-block counts are summed back to effective
    shots == cfg.shots.

    Purpose: noise-leakage check. If blocks were truly independent the
    aggregated single-event features should be statistically equivalent
    to the baseline single-mode features at the same total shots. Any
    deviation under the FakeFez noise model + correlated-noise channels
    is what this experiment quantifies.
    """
    m = X_data.shape[0]
    n_obs = 4 * cfg.n_qubits
    n_total_events = cfg.n_previous_events + 1
    pauli_matrix = np.zeros((m, n_total_events * n_obs))
    resources = []
    effective_shots_per_block = packing.effective_shots(cfg.shots)
    effective_total_shots = effective_shots_per_block * packing.pack_n

    for event_idx in range(n_total_events):
        ckpt = (
            None
            if checkpoint_prefix is None
            else checkpoint_prefix.with_name(
                f"{checkpoint_prefix.name}_event{event_idx}.npy"
            )
        )
        if ckpt and resume and ckpt.exists():
            block = _try_load_checkpoint(ckpt, (m, n_obs), resume)
            if block is not None:
                pauli_matrix[:, event_idx * n_obs : (event_idx + 1) * n_obs] = block
                continue

        start_col = event_idx * cfg.n_qubits
        end_col = start_col + cfg.n_qubits
        X_event = X_data[:, start_col:end_col]
        template, params = build_replicated_reservoir_circuit(
            angle_bank[event_idx],
            cfg.num_layers_per_event,
            cfg.n_qubits,
            packing.pack_n,
            cfg.n_trotter_steps,
        )
        isa_templates = _prepare_basis_templates(template, local_transpile)
        resources.append(
            estimate_resources(
                isa_templates["Z"], backend, effective_shots_per_block, 3 * len(X_event)
            )
        )

        z_counts, x_counts, y_counts, aer_time = _run_batched(
            isa_templates,
            params,
            X_event,
            simulator,
            effective_shots_per_block,
            batch_size,
        )

        event_block = np.zeros((m, n_obs))
        for sample_idx in range(m):
            # Sum block sub-bitstrings: pack_n i.i.d. samples per shot.
            agg_z = _aggregate_replica_counts(
                z_counts[sample_idx], cfg.n_qubits, packing.pack_n
            )
            agg_x = _aggregate_replica_counts(
                x_counts[sample_idx], cfg.n_qubits, packing.pack_n
            )
            agg_y = _aggregate_replica_counts(
                y_counts[sample_idx], cfg.n_qubits, packing.pack_n
            )
            event_block[sample_idx] = _features_from_counts(
                agg_z,
                agg_x,
                agg_y,
                cfg.n_qubits,
                effective_total_shots,
                open_chain=(cfg.topology == "nn"),
            )

        pauli_matrix[:, event_idx * n_obs : (event_idx + 1) * n_obs] = event_block
        if ckpt:
            np.save(ckpt, event_block)
        resources[-1]["aer_time_taken_seconds"] = aer_time
        resources[-1]["effective_shots_per_block"] = effective_shots_per_block
        resources[-1]["effective_total_shots"] = effective_total_shots
        print(
            f"\tEvent {event_idx + 1}/{n_total_events} (shot-parallel N={packing.pack_n}) "
            f"complete | aer_time={aer_time:.2f}s"
        )

    return pauli_matrix, resources


def _run_sequential_prefix_mode(
    X_data: np.ndarray,
    angle_bank,
    cfg: QRCConfig,
    simulator: AerSimulator,
    local_transpile,
    backend,
    checkpoint_prefix: Path | None,
    resume: bool,
    batch_size: int,
):
    """Stateful reservoir (FC/NN-TFI-Ridge notebooks) under the shot protocol.

    The notebooks evolve ONE state per sample through all events and read
    exact statevector expectations after each event without collapsing. With
    shots + a noise model that readout has no direct analog, so event k's
    features come from a fresh PREFIX circuit containing events 0..k,
    measured in Z/X/Y exactly like the per-event mode. Later events therefore
    run deeper circuits and accumulate more hardware noise -- the noisy
    analog of the notebooks' fading memory. Feature layout, checkpoints, and
    resume semantics are identical to _run_single_event_mode.
    """
    m = X_data.shape[0]
    n_obs = 4 * cfg.n_qubits
    n_total_events = cfg.n_previous_events + 1
    pauli_matrix = np.zeros((m, n_total_events * n_obs))
    resources = []

    for event_idx in range(n_total_events):
        ckpt = (
            None
            if checkpoint_prefix is None
            else checkpoint_prefix.with_name(
                f"{checkpoint_prefix.name}_event{event_idx}.npy"
            )
        )
        if ckpt and resume and ckpt.exists():
            block = _try_load_checkpoint(ckpt, (m, n_obs), resume)
            if block is not None:
                pauli_matrix[:, event_idx * n_obs : (event_idx + 1) * n_obs] = block
                continue

        X_prefix = X_data[:, : (event_idx + 1) * cfg.n_qubits]
        template, params = build_parametric_sequential_prefix(
            angle_bank,
            event_idx,
            cfg.num_layers_per_event,
            cfg.n_qubits,
            cfg.n_trotter_steps,
            cfg.topology,
        )
        isa_templates = _prepare_basis_templates(template, local_transpile)
        resources.append(
            estimate_resources(isa_templates["Z"], backend, cfg.shots, 3 * m)
        )

        z_counts, x_counts, y_counts, aer_time = _run_batched(
            isa_templates, params, X_prefix, simulator, cfg.shots, batch_size
        )

        event_block = np.zeros((m, n_obs))
        for sample_idx in range(m):
            event_block[sample_idx] = _features_from_counts(
                z_counts[sample_idx],
                x_counts[sample_idx],
                y_counts[sample_idx],
                cfg.n_qubits,
                cfg.shots,
                open_chain=(cfg.topology == "nn"),
            )

        pauli_matrix[:, event_idx * n_obs : (event_idx + 1) * n_obs] = event_block
        if ckpt:
            np.save(ckpt, event_block)
        resources[-1]["aer_time_taken_seconds"] = aer_time
        print(
            f"\tPrefix 0..{event_idx} ({event_idx + 1}/{n_total_events}) "
            f"complete | aer_time={aer_time:.2f}s"
        )

    return pauli_matrix, resources


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
    packing: PackingSpec | None = None,
):
    """Dispatch to the right packing-mode implementation.

    packing=None (default) or packing.mode=="single" reproduces the
    original behavior exactly. cfg.arch=="ridge-sequential" switches to the
    notebook-matching stateful prefix-circuit readout (single packing only).
    """
    if cfg.arch == "ridge-sequential":
        if packing is not None and packing.mode != "single":
            raise ValueError(
                "--arch ridge-sequential does not support packing modes."
            )
        return _run_sequential_prefix_mode(
            X_data,
            angle_bank,
            cfg,
            simulator,
            local_transpile,
            backend,
            checkpoint_prefix,
            resume,
            batch_size,
        )
    if packing is None or packing.mode == "single":
        return _run_single_event_mode(
            X_data,
            angle_bank,
            cfg,
            simulator,
            local_transpile,
            backend,
            checkpoint_prefix,
            resume,
            batch_size,
        )
    if packing.mode == "joint-ising":
        return _run_joint_ising_mode(
            X_data,
            angle_bank,
            cfg,
            packing,
            simulator,
            local_transpile,
            backend,
            checkpoint_prefix,
            resume,
            batch_size,
        )
    if packing.mode == "shot-parallel":
        return _run_shot_parallel_mode(
            X_data,
            angle_bank,
            cfg,
            packing,
            simulator,
            local_transpile,
            backend,
            checkpoint_prefix,
            resume,
            batch_size,
        )
    raise ValueError(f"Unknown packing mode: {packing.mode}")


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


def tune_and_train_regressor(
    X_train, y_train, seed: int, n_trials: int, n_cv_folds: int = N_CV_FOLDS
):
    """Optuna + 3-fold CV on train, then refit (Shallow-Arch.ipynb XGBoost path)."""

    def _cv_mae(params: Dict[str, Any], rng_seed: int) -> float:
        kf = KFold(n_splits=n_cv_folds, shuffle=True, random_state=rng_seed)
        maes: List[float] = []
        for fold_tr, fold_vl in kf.split(X_train):
            model = XGBRegressor(**params)
            model.fit(X_train[fold_tr], y_train[fold_tr], verbose=False)
            maes.append(
                float(
                    mean_absolute_error(
                        y_train[fold_vl], model.predict(X_train[fold_vl])
                    )
                )
            )
        return float(np.mean(maes))

    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "n_estimators": XGB_N_ESTIMATORS,
            "random_state": 42,
            "tree_method": "hist",
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.005, 0.15, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        return _cv_mae(params, seed)

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials)
    full_params = {
        "objective": "reg:squarederror",
        "n_estimators": XGB_N_ESTIMATORS,
        "random_state": 42,
        "tree_method": "hist",
        **study.best_trial.params,
    }
    model = XGBRegressor(**full_params)
    model.fit(X_train, y_train, verbose=False)
    return model, full_params


def tune_and_train_ridge(
    X_train, y_train, seed: int, n_trials: int, n_cv_folds: int = N_CV_FOLDS
):
    """Optuna + 3-fold CV Ridge, then refit (FC/NN-TFI-Ridge.ipynb Cell 10)."""

    def _cv_mae(alpha: float, rng_seed: int) -> float:
        kf = KFold(n_splits=n_cv_folds, shuffle=True, random_state=rng_seed)
        maes: List[float] = []
        for fold_tr, fold_vl in kf.split(X_train):
            model = Ridge(alpha=alpha)
            model.fit(X_train[fold_tr], y_train[fold_tr])
            maes.append(
                float(
                    mean_absolute_error(
                        y_train[fold_vl], model.predict(X_train[fold_vl])
                    )
                )
            )
        return float(np.mean(maes))

    def objective(trial):
        return _cv_mae(trial.suggest_float("alpha", 1e-2, 1e8, log=True), seed)

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials)
    full_params = {"alpha": float(study.best_trial.params["alpha"])}
    model = Ridge(**full_params)
    model.fit(X_train, y_train)
    return model, full_params


def kmeans_regime_labels(y_train: np.ndarray) -> Tuple[np.ndarray, float]:
    """K-Means on y_train only — Shallow-Arch.ipynb Cell 6."""
    km_y = KMeans(n_clusters=2, n_init=50, random_state=42)
    km_y.fit(y_train.reshape(-1, 1))
    order = np.argsort(km_y.cluster_centers_.ravel())
    remap = {old: new for new, old in enumerate(order)}
    centers = km_y.cluster_centers_.ravel()[order]
    raw = km_y.predict(y_train.reshape(-1, 1))
    y_clf_train = np.array([remap[r] for r in raw])
    boundary = float(centers.mean())
    return y_clf_train, boundary


def train_regime_classifier(
    X_train_raw: np.ndarray,
    y_clf_train: np.ndarray,
    n_trials: int,
) -> XGBClassifier:
    """Optuna-tuned classifier on raw features (Shallow-Arch.ipynb Cell 6)."""
    sample_weights = compute_sample_weight("balanced", y_clf_train)

    def clf_objective(trial):
        clf = XGBClassifier(
            objective="binary:logistic",
            n_estimators=500,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 6),
            subsample=trial.suggest_float("subsample", 0.6, 0.9),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 0.9),
            scale_pos_weight=trial.suggest_float("scale_pos_weight", 0.5, 3.0),
            random_state=42,
            eval_metric="logloss",
        )
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        fold_accs: List[float] = []
        for train_idx, val_idx in skf.split(X_train_raw, y_clf_train):
            clf.fit(
                X_train_raw[train_idx],
                y_clf_train[train_idx],
                sample_weight=sample_weights[train_idx],
                verbose=False,
            )
            fold_accs.append(
                float(accuracy_score(y_clf_train[val_idx], clf.predict(X_train_raw[val_idx])))
            )
        return 1.0 - float(np.mean(fold_accs))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(clf_objective, n_trials=n_trials)
    classifier = XGBClassifier(
        objective="binary:logistic",
        n_estimators=500,
        **study.best_params,
        random_state=42,
        eval_metric="logloss",
    )
    classifier.fit(
        X_train_raw,
        y_clf_train,
        sample_weight=sample_weights,
        verbose=False,
    )
    return classifier


def build_regime_routing(
    X_train_raw: np.ndarray,
    X_val_raw: np.ndarray,
    X_test_raw: np.ndarray,
    y_train: np.ndarray,
    optuna_trials: int,
) -> RegimeRouting:
    """Classifier-predicted routing on all splits (Shallow-Arch.ipynb)."""
    y_clf_train, boundary = kmeans_regime_labels(y_train)
    classifier = train_regime_classifier(X_train_raw, y_clf_train, optuna_trials)
    clf_train_labels = classifier.predict(X_train_raw)
    clf_val_labels = classifier.predict(X_val_raw)
    clf_test_labels = classifier.predict(X_test_raw)
    print(
        f"K-Means boundary (midpoint): {boundary:.0f} s | "
        f"classifier train acc: {accuracy_score(y_clf_train, clf_train_labels):.4f}"
    )
    return RegimeRouting(
        classifier=classifier,
        short_mask_train=clf_train_labels == 0,
        long_mask_train=clf_train_labels == 1,
        short_mask_val=clf_val_labels == 0,
        long_mask_val=clf_val_labels == 1,
        short_test_idx=np.where(clf_test_labels == 0)[0],
        long_test_idx=np.where(clf_test_labels == 1)[0],
        short_val_idx=np.where(clf_val_labels == 0)[0],
        long_val_idx=np.where(clf_val_labels == 1)[0],
    )


def load_csv_regime_routing(
    csv_path: Path,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    subset_indices: Dict[str, np.ndarray | None] | None = None,
) -> Tuple[RegimeRouting, np.ndarray, float]:
    """Regime routing from a precomputed labels file (FC/NN-TFI-Ridge Cell 6).

    The file's first line is a comment header carrying the K-Means centers
    and decision boundary; the rest is split,event_index,y_true_ttns_s,
    regime_label,regime rows for every event in every split. Labels are
    verified against the loaded y arrays so misalignment (wrong file, wrong
    preprocessing, wrong subset) fails fast instead of wasting a SLURM run.
    Returns (routing, centers, boundary).
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Regime labels file not found: {csv_path}. "
            "--arch ridge-sequential needs it (see --regime-labels-csv)."
        )
    with csv_path.open("r") as f:
        header_line = f.readline()
    match = re.search(
        r"short_center_s=([\d.eE+-]+)\s+long_center_s=([\d.eE+-]+)\s+boundary_s=([\d.eE+-]+)",
        header_line,
    )
    if match is None:
        raise ValueError(
            f"Could not parse cluster centers from header line: {header_line!r}"
        )
    short_center_s, long_center_s, boundary = (float(g) for g in match.groups())
    centers = np.array([short_center_s, long_center_s])

    regime_df = pd.read_csv(csv_path, skiprows=1, sep=None, engine="python")
    regime_df.columns = [str(c).strip() for c in regime_df.columns]
    regime_df = regime_df.loc[:, ~regime_df.columns.str.match(r"^Unnamed")]
    regime_df = regime_df.dropna(axis=1, how="all")
    required_cols = {"split", "event_index", "y_true_ttns_s", "regime_label"}
    missing = required_cols - set(regime_df.columns)
    if missing:
        raise ValueError(
            f"Regime labels file {csv_path} is missing columns {missing}; "
            f"found {list(regime_df.columns)}"
        )
    regime_df["split"] = regime_df["split"].astype(str).str.strip()
    regime_df["event_index"] = pd.to_numeric(regime_df["event_index"])
    regime_df["y_true_ttns_s"] = pd.to_numeric(regime_df["y_true_ttns_s"])
    regime_df["regime_label"] = pd.to_numeric(regime_df["regime_label"]).astype(int)

    def _labels_for_split(split_name: str, y_split: np.ndarray) -> np.ndarray:
        sub = (
            regime_df[regime_df["split"] == split_name]
            .sort_values("event_index")
            .reset_index(drop=True)
        )
        idx = None if subset_indices is None else subset_indices.get(split_name)
        if idx is not None:
            if len(sub) and int(np.max(idx)) >= len(sub):
                raise ValueError(
                    f"'{split_name}' subset index {int(np.max(idx))} out of range "
                    f"for {len(sub)} label rows -- labels file does not match "
                    "the full pre-subset split."
                )
            sub = sub.iloc[idx].reset_index(drop=True)
        if len(sub) != len(y_split):
            raise ValueError(
                f"'{split_name}' split length mismatch: labels file has "
                f"{len(sub)} rows but loaded split has {len(y_split)}. Check "
                "that the file was generated from the same preprocessing/"
                "subset settings as this run."
            )
        file_y = sub["y_true_ttns_s"].to_numpy()
        if not np.allclose(file_y, y_split, atol=1e-6):
            n_bad = int(np.sum(~np.isclose(file_y, y_split, atol=1e-6)))
            raise ValueError(
                f"{n_bad}/{len(file_y)} '{split_name}' y-values differ between "
                f"{csv_path.name} and the loaded data -- the file and this "
                "run are not row-aligned."
            )
        return sub["regime_label"].to_numpy().astype(int)

    train_labels = _labels_for_split("train", y_train)
    val_labels = _labels_for_split("val", y_val)
    test_labels = _labels_for_split("test", y_test)
    print(
        f"Loaded regime labels from {csv_path.name} | short center "
        f"{centers[0]:.0f}s, long center {centers[1]:.0f}s, "
        f"boundary {boundary:.0f}s | train short/long="
        f"{int((train_labels == 0).sum())}/{int((train_labels == 1).sum())}"
    )
    routing = RegimeRouting(
        classifier=None,
        short_mask_train=train_labels == 0,
        long_mask_train=train_labels == 1,
        short_mask_val=val_labels == 0,
        long_mask_val=val_labels == 1,
        short_test_idx=np.where(test_labels == 0)[0],
        long_test_idx=np.where(test_labels == 1)[0],
        short_val_idx=np.where(val_labels == 0)[0],
        long_val_idx=np.where(val_labels == 1)[0],
    )
    return routing, centers, boundary


def resolve_partials_root(args: argparse.Namespace) -> Path:
    """Root directory for partials/ and regime_routing.pkl."""
    return args.partials_dir if args.partials_dir is not None else args.output_dir


def routing_artifact_path(partials_root: Path) -> Path:
    return partials_root / ROUTING_ARTIFACT


def routing_snapshot(routing: RegimeRouting) -> Dict[str, Any]:
    """Serializable routing masks/indices saved with partial artifacts."""
    return {
        "short_mask_train": routing.short_mask_train,
        "long_mask_train": routing.long_mask_train,
        "short_mask_val": routing.short_mask_val,
        "long_mask_val": routing.long_mask_val,
        "short_test_idx": routing.short_test_idx,
        "long_test_idx": routing.long_test_idx,
        "short_val_idx": routing.short_val_idx,
        "long_val_idx": routing.long_val_idx,
    }


def routing_from_snapshot(snapshot: Dict[str, Any], classifier: XGBClassifier) -> RegimeRouting:
    return RegimeRouting(
        classifier=classifier,
        short_mask_train=np.asarray(snapshot["short_mask_train"], dtype=bool),
        long_mask_train=np.asarray(snapshot["long_mask_train"], dtype=bool),
        short_mask_val=np.asarray(snapshot["short_mask_val"], dtype=bool),
        long_mask_val=np.asarray(snapshot["long_mask_val"], dtype=bool),
        short_test_idx=np.asarray(snapshot["short_test_idx"], dtype=int),
        long_test_idx=np.asarray(snapshot["long_test_idx"], dtype=int),
        short_val_idx=np.asarray(snapshot["short_val_idx"], dtype=int),
        long_val_idx=np.asarray(snapshot["long_val_idx"], dtype=int),
    )


def save_regime_routing(
    artifact_path: Path,
    routing: RegimeRouting,
    metadata: Dict[str, Any],
) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "routing": routing_snapshot(routing),
        "classifier": routing.classifier,
    }
    tmp_path = artifact_path.with_suffix(".pkl.tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(payload, f)
    tmp_path.replace(artifact_path)


def load_regime_routing(artifact_path: Path) -> RegimeRouting:
    with artifact_path.open("rb") as f:
        payload = pickle.load(f)
    return routing_from_snapshot(payload["routing"], payload["classifier"])


def _routing_metadata(args: argparse.Namespace, cfg: QRCConfig) -> Dict[str, Any]:
    return {
        "subset_frac": float(args.subset_frac),
        "random_seed": int(cfg.random_seed),
        "optuna_trials": int(cfg.optuna_trials),
        "n_previous_events": int(cfg.n_previous_events),
    }


def _assert_routing_metadata(
    saved: Dict[str, Any], args: argparse.Namespace, cfg: QRCConfig
) -> None:
    expected = _routing_metadata(args, cfg)
    mismatches = [
        key for key, value in expected.items() if saved.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "Saved regime routing metadata does not match current CLI/config: "
            f"{', '.join(mismatches)}. "
            f"Saved={ {k: saved.get(k) for k in mismatches} }, "
            f"current={ {k: expected[k] for k in mismatches} }. "
            "Re-run with matching --subset-frac, --seed, --optuna-trials, "
            "and --n-previous-events, or regenerate regime_routing.pkl."
        )


def ensure_regime_routing(
    partials_root: Path,
    X_train_raw: np.ndarray,
    X_val_raw: np.ndarray,
    X_test_raw: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
    cfg: QRCConfig,
) -> RegimeRouting:
    """Load saved Shallow-Arch routing or build once and persist for all SLURM tasks."""
    artifact_path = routing_artifact_path(partials_root)
    if artifact_path.exists():
        with artifact_path.open("rb") as f:
            payload = pickle.load(f)
        _assert_routing_metadata(payload.get("metadata", {}), args, cfg)
        routing = routing_from_snapshot(payload["routing"], payload["classifier"])
        print(f"Loaded saved regime routing from {artifact_path}")
        return routing

    lock_path = artifact_path.with_suffix(".lock")
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        for _ in range(600):
            if artifact_path.exists():
                return load_regime_routing(artifact_path)
            time.sleep(0.5)
        raise TimeoutError(
            f"Timed out waiting for another task to write {artifact_path}"
        )

    try:
        if artifact_path.exists():
            with artifact_path.open("rb") as f:
                payload = pickle.load(f)
            _assert_routing_metadata(payload.get("metadata", {}), args, cfg)
            return routing_from_snapshot(payload["routing"], payload["classifier"])

        routing = build_regime_routing(
            X_train_raw, X_val_raw, X_test_raw, y_train, cfg.optuna_trials
        )
        save_regime_routing(
            artifact_path, routing, _routing_metadata(args, cfg)
        )
        print(f"Saved regime routing -> {artifact_path}")
        return routing
    finally:
        lock_path.unlink(missing_ok=True)


def load_saved_regime_routing(
    partials_root: Path,
    args: argparse.Namespace,
    cfg: QRCConfig,
) -> RegimeRouting:
    """Load routing for aggregation; never re-trains the classifier."""
    artifact_path = routing_artifact_path(partials_root)
    if artifact_path.exists():
        with artifact_path.open("rb") as f:
            payload = pickle.load(f)
        _assert_routing_metadata(payload.get("metadata", {}), args, cfg)
        routing = routing_from_snapshot(payload["routing"], payload["classifier"])
        print(f"Loaded saved regime routing from {artifact_path}")
        return routing

    partials_root_glob = partials_root / "partials"
    if partials_root_glob.exists():
        for partial_path in sorted(partials_root_glob.rglob("partial_iter*_short.pkl")):
            with partial_path.open("rb") as f:
                payload = pickle.load(f)
            if "routing" not in payload or "classifier" not in payload:
                continue
            routing = routing_from_snapshot(payload["routing"], payload["classifier"])
            print(
                f"Loaded regime routing embedded in partial artifact {partial_path}"
            )
            return routing

    raise FileNotFoundError(
        f"Missing {artifact_path}. Partial jobs save this file on first launch. "
        f"Run once with --write-routing-only --output-dir {partials_root} "
        "before aggregating, using the same data/subset settings as the partial run."
    )


def _assert_partial_routing_shapes(
    regime: str,
    iteration: int,
    point_label: str,
    P_train: np.ndarray,
    P_val: np.ndarray,
    P_test: np.ndarray,
    mask_train: np.ndarray,
    mask_val: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    expected_train = int(np.sum(mask_train))
    expected_val = int(np.sum(mask_val))
    expected_test = int(len(test_idx))
    actual_train = int(P_train.shape[0])
    actual_val = int(P_val.shape[0])
    actual_test = int(P_test.shape[0])
    if (
        actual_train != expected_train
        or actual_val != expected_val
        or actual_test != expected_test
    ):
        raise ValueError(
            f"{regime} routing mismatch for {point_label} iter {iteration}: "
            f"saved routing expects train/val/test="
            f"{expected_train}/{expected_val}/{expected_test}, "
            f"but partial P arrays are "
            f"{actual_train}/{actual_val}/{actual_test}. "
            "Use the regime_routing.pkl from the same run folder as the partials."
        )


def maybe_subset_split(
    X_df: pd.DataFrame, y: pd.Series | np.ndarray, frac: float, rng: np.random.Generator
):
    """Randomly subsample one split while keeping X/y aligned (Shallow-Arch.ipynb).

    Also returns the sorted row indices kept (None when no subsetting) so
    external per-row artifacts like the regime-labels CSV can be aligned.
    """
    if frac >= 1.0 or len(X_df) == 0:
        return X_df, y, None
    keep = max(1, int(len(X_df) * frac))
    idx = np.sort(rng.choice(len(X_df), size=keep, replace=False))
    y_sub = y.iloc[idx] if hasattr(y, "iloc") else y[idx]
    if hasattr(y_sub, "reset_index"):
        y_sub = y_sub.reset_index(drop=True)
    return X_df.iloc[idx].reset_index(drop=True), y_sub, idx


def load_data(cfg: QRCConfig, subset_frac: float):
    from Preprocess import preprocess_data_window

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
        filtered_time, data_orig, cfg.n_previous_events
    )
    subset_idx: Dict[str, np.ndarray | None] = {
        "train": None,
        "val": None,
        "test": None,
    }
    if subset_frac < 1.0:
        rng = np.random.default_rng(cfg.random_seed)
        X_train, y_train, subset_idx["train"] = maybe_subset_split(
            X_train, y_train, subset_frac, rng
        )
        X_val, y_val, subset_idx["val"] = maybe_subset_split(
            X_val, y_val, subset_frac, rng
        )
        X_test, y_test, subset_idx["test"] = maybe_subset_split(
            X_test, y_test, subset_frac, rng
        )
    X_train_raw = X_train.to_numpy()
    X_val_raw = X_val.to_numpy()
    X_test_raw = X_test.to_numpy()
    X_train_q, X_val_q, X_test_q, train_min, train_max = scale_to_pi_range(
        X_train_raw, X_val_raw, X_test_raw
    )
    return (
        X_train_q,
        X_val_q,
        X_test_q,
        X_train_raw,
        X_val_raw,
        X_test_raw,
        y_train.to_numpy(),
        y_val.to_numpy(),
        y_test.to_numpy(),
        train_min,
        train_max,
        subset_idx,
    )


def task_to_iter_regime_corr(
    task_id: int, n_iterations: int, n_corr_levels: int
) -> Tuple[int, str, int]:
    """Map a SLURM task_id to (iter_idx, regime, corr_idx).

    Layout: tasks 0..(2*n_iter - 1) belong to corr_idx=0, the next block to
    corr_idx=1, and so on. Within a corr block, even task_ids are 'short',
    odd are 'long'. This keeps the original 2*n_iter layout intact when
    n_corr_levels == 1.
    """
    per_corr = n_iterations * 2
    total = per_corr * n_corr_levels
    if task_id < 0 or task_id >= total:
        raise ValueError(f"task_id must be in [0, {total - 1}]")
    corr_idx = task_id // per_corr
    inner = task_id % per_corr
    iter_idx = inner // 2
    regime = "short" if (inner % 2 == 0) else "long"
    return iter_idx, regime, corr_idx


def task_to_iter_regime(task_id: int, n_iterations: int):
    """Backward-compat wrapper: drops corr_idx for non-sweep callers."""
    iter_idx, regime, _ = task_to_iter_regime_corr(task_id, n_iterations, 1)
    return iter_idx, regime


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_partial_task(args: argparse.Namespace, cfg: QRCConfig):
    sweep_points = build_sweep_points(args)
    iter_idx, regime, sweep_idx = task_to_iter_regime_corr(
        args.task_id, cfg.n_iterations, len(sweep_points)
    )
    point = sweep_points[sweep_idx]
    noise_spec = point.noise_spec
    packing = PackingSpec(mode=args.packing_mode, pack_n=args.pack_n)
    partials_root = resolve_partials_root(args)
    partials_root.mkdir(parents=True, exist_ok=True)
    partial_dir = packing_partials_root(partials_root, packing) / point.label
    partial_dir.mkdir(parents=True, exist_ok=True)

    (
        X_train_q,
        X_val_q,
        X_test_q,
        X_train_raw,
        X_val_raw,
        X_test_raw,
        y_train,
        y_val,
        y_test,
        _train_min,
        _train_max,
        subset_idx,
    ) = load_data(cfg, args.subset_frac)
    if cfg.arch == "ridge-sequential" and args.noise_model == "none":
        print(
            "[NOTE] --arch ridge-sequential with --noise-model none runs "
            "the shot-based protocol on an ideal simulator. Pass "
            "--noise-model fakefez for the FakeFez noisy study."
        )
    if cfg.arch == "ridge-sequential" and cfg.regime_routing == "csv":
        routing, _, _ = load_csv_regime_routing(
            Path(cfg.regime_labels_csv), y_train, y_val, y_test, subset_idx
        )
    else:
        routing = ensure_regime_routing(
            partials_root,
            X_train_raw,
            X_val_raw,
            X_test_raw,
            y_train,
            args,
            cfg,
        )

    short_mask_train = routing.short_mask_train
    long_mask_train = routing.long_mask_train
    short_mask_val = routing.short_mask_val
    long_mask_val = routing.long_mask_val
    short_test_idx = routing.short_test_idx
    long_test_idx = routing.long_test_idx

    # NOTE: sweep_idx is deliberately *not* mixed into this seed, and t does
    # not consume rng draws, so every sweep point (t value or noise level)
    # sees identical Ising disorder for the same (iter, regime). Performance
    # variation across the sweep then attributes to the swept knob only.
    # Packing mode IS reflected in the angle-bank shape via build_angle_bank.
    rng = np.random.default_rng(
        cfg.random_seed + iter_idx + (0 if regime == "short" else 10_000)
    )
    angle_bank = build_angle_bank(
        packing,
        cfg.n_qubits,
        cfg.n_previous_events + 1,
        rng,
        topology=cfg.topology,
        h=cfg.h_field,
        t=point.t_evolution,
    )

    # Build simulator sized for the packed circuit. For packed modes the
    # correlated-noise pair resolution needs to consider the full circuit
    # qubit count, so pass that in.
    sim_n_qubits = packing.circuit_qubits(cfg.n_qubits)
    _, sim, local_transpile = build_noisy_simulator(
        args.device,
        args.max_memory_mb,
        noise_spec,
        n_qubits=sim_n_qubits,
        max_parallel_experiments=args.max_parallel_experiments,
        sim_method=args.sim_method,
        noise_backend=args.noise_model,
    )
    fake_backend = FakeSherbrooke()

    if regime == "short":
        Xtr, Xvl, Xte = (
            X_train_q[short_mask_train],
            X_val_q[short_mask_val],
            X_test_q[short_test_idx],
        )
    else:
        Xtr, Xvl, Xte = (
            X_train_q[long_mask_train],
            X_val_q[long_mask_val],
            X_test_q[long_test_idx],
        )

    print(
        f"Running partial: iteration={iter_idx} regime={regime} "
        f"sweep={point.label} sweep_idx={sweep_idx} "
        f"arch={cfg.arch} topology={cfg.topology} t={point.t_evolution:g} "
        f"noise_model={args.noise_model} noise={noise_spec.label} "
        f"packing={packing.label} circuit_qubits={sim_n_qubits} "
        f"train={len(Xtr)} val={len(Xvl)} test={len(Xte)}"
    )
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
        packing=packing,
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
        packing=packing,
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
        packing=packing,
    )
    elapsed = time.time() - t0

    payload = {
        "iteration": iter_idx,
        "regime": regime,
        "sweep_idx": sweep_idx,
        "sweep_label": point.label,
        "t_evolution": point.t_evolution,
        "arch": cfg.arch,
        "regime_routing": cfg.regime_routing,
        "topology": cfg.topology,
        "noise_model": args.noise_model,
        "noise_spec": noise_spec.as_dict(),
        "packing": packing.as_dict(),
        "angle_bank": angle_bank,
        "P_train": P_tr,
        "P_val": P_vl,
        "P_test": P_te,
        "routing": routing_snapshot(routing),
        "classifier": routing.classifier,
        "resources": resources,
        "elapsed_seconds": elapsed,
    }
    partial_path = partial_dir / f"partial_iter{iter_idx}_{regime}.pkl"
    with partial_path.open("wb") as f:
        pickle.dump(payload, f)
    print(f"Saved partial artifact: {partial_path}")


def _aggregate_one_spec(
    args: argparse.Namespace,
    cfg: QRCConfig,
    point: SweepPoint,
    packing: PackingSpec,
    partials_root: Path,
    out_dir: Path,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    short_mask_train: np.ndarray,
    long_mask_train: np.ndarray,
    short_mask_val: np.ndarray,
    long_mask_val: np.ndarray,
    short_val_idx: np.ndarray,
    long_val_idx: np.ndarray,
    short_test_idx: np.ndarray,
    long_test_idx: np.ndarray,
    n_test: int,
    n_val: int,
    clf,
    train_min,
    train_max,
) -> Dict[str, Any]:
    partial_dir = packing_partials_root(partials_root, packing) / point.label
    if not partial_dir.exists():
        raise FileNotFoundError(
            f"No partials directory for packing={packing.label} "
            f"sweep={point.label!r}: expected {partial_dir}"
        )

    all_results = []
    n_total_events = cfg.n_previous_events + 1
    n_obs = 4 * cfg.n_qubits
    for i in range(cfg.n_iterations):
        with (partial_dir / f"partial_iter{i}_short.pkl").open("rb") as f:
            ps = pickle.load(f)
        with (partial_dir / f"partial_iter{i}_long.pkl").open("rb") as f:
            pl = pickle.load(f)

        for payload in (ps, pl):
            saved_arch = payload.get("arch", "xgb-per-event")
            if saved_arch != cfg.arch:
                raise ValueError(
                    f"Partial iter {payload['iteration']} ({payload['regime']}) "
                    f"was computed with arch={saved_arch!r} but aggregation "
                    f"requested arch={cfg.arch!r}. Re-run partials or pass "
                    "the matching --arch."
                )
            # Routing decides which events each partial simulated, so a
            # mismatch silently misaligns features with labels. Older
            # partials lack the key; those fall back to the shape checks.
            saved_routing = payload.get("regime_routing")
            if saved_routing is not None and saved_routing != cfg.regime_routing:
                raise ValueError(
                    f"Partial iter {payload['iteration']} ({payload['regime']}) "
                    f"was computed with regime_routing={saved_routing!r} but "
                    f"aggregation requested {cfg.regime_routing!r}. Re-run "
                    "partials or pass the matching --regime-routing."
                )

        _assert_partial_routing_shapes(
            "short",
            i,
            point.label,
            ps["P_train"],
            ps["P_val"],
            ps["P_test"],
            short_mask_train,
            short_mask_val,
            short_test_idx,
        )
        _assert_partial_routing_shapes(
            "long",
            i,
            point.label,
            pl["P_train"],
            pl["P_val"],
            pl["P_test"],
            long_mask_train,
            long_mask_val,
            long_test_idx,
        )

        if cfg.arch == "ridge-sequential":
            # FC/NN-TFI-Ridge notebooks feed raw quantum features to Ridge —
            # no exponential decay weighting.
            H_tr_short, H_vl_short, H_te_short = (
                ps["P_train"], ps["P_val"], ps["P_test"],
            )
            H_tr_long, H_vl_long, H_te_long = (
                pl["P_train"], pl["P_val"], pl["P_test"],
            )
            tuner = tune_and_train_ridge
        else:
            H_tr_short = make_hybrid_features_decay(ps["P_train"], n_total_events, n_obs)
            H_vl_short = make_hybrid_features_decay(ps["P_val"], n_total_events, n_obs)
            H_te_short = make_hybrid_features_decay(ps["P_test"], n_total_events, n_obs)
            H_tr_long = make_hybrid_features_decay(pl["P_train"], n_total_events, n_obs)
            H_vl_long = make_hybrid_features_decay(pl["P_val"], n_total_events, n_obs)
            H_te_long = make_hybrid_features_decay(pl["P_test"], n_total_events, n_obs)
            tuner = tune_and_train_regressor

        y_tr_short = y_train[short_mask_train]
        y_tr_long = y_train[long_mask_train]

        model_short, short_params = tuner(
            H_tr_short,
            y_tr_short,
            cfg.random_seed + i,
            cfg.optuna_trials,
        )
        model_long, long_params = tuner(
            H_tr_long,
            y_tr_long,
            cfg.random_seed + i + 1,
            cfg.optuna_trials,
        )

        test_pred = np.empty(n_test)
        test_pred[short_test_idx] = model_short.predict(H_te_short)
        test_pred[long_test_idx] = model_long.predict(H_te_long)

        val_pred = np.empty(n_val)
        short_val_positions = {
            idx: pos for pos, idx in enumerate(np.where(short_mask_val)[0])
        }
        long_val_positions = {
            idx: pos for pos, idx in enumerate(np.where(long_mask_val)[0])
        }
        for idx in short_val_idx:
            if idx in short_val_positions:
                val_pred[idx] = model_short.predict(
                    H_vl_short[short_val_positions[idx] : short_val_positions[idx] + 1]
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
                    H_vl_short[short_val_positions[idx] : short_val_positions[idx] + 1]
                )[0]

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
        r = all_results[-1]
        print(
            f"[{point.label}] Aggregated iteration {i + 1}/{cfg.n_iterations} | "
            f"Val R2: {r['val_r2']:.4f} | Val RMSE: {r['val_rmse']:.2f} | "
            f"Val MAE: {r['val_mae']:.2f} | Test RMSE: {r['test_rmse']:.2f} | "
            f"Test MAE: {r['test_mae']:.2f}"
        )

    top_results = sorted(all_results, key=lambda r: r["val_mae"])[: cfg.top_k]
    top_indices = [r["iteration"] for r in top_results]
    all_test_preds = np.array([r["test_pred"] for r in all_results])
    mean_all_pred = all_test_preds.mean(axis=0)
    top_k_pred = np.mean([r["test_pred"] for r in top_results], axis=0)

    ensemble_test_mae = float(mean_absolute_error(y_test, mean_all_pred))
    ensemble_test_rmse = float(root_mean_squared_error(y_test, mean_all_pred))
    ensemble_test_r2 = float(r2_score(y_test, mean_all_pred))
    top_k_test_mae = float(mean_absolute_error(y_test, top_k_pred))
    top_k_test_rmse = float(root_mean_squared_error(y_test, top_k_pred))
    top_k_test_r2 = float(r2_score(y_test, top_k_pred))

    print(f"\n[{point.label}] Ensemble (all {len(all_results)} iterations — primary headline):")
    print(f"  Test RMSE: {ensemble_test_rmse:.2f}")
    print(f"  Test MAE:  {ensemble_test_mae:.2f}")
    print(f"  Test R2:   {ensemble_test_r2:.4f}")
    print(
        f"\n[{point.label}] Top-{cfg.top_k} ensemble by val MAE "
        "(optimistic — val used for selection):"
    )
    print(f"  Iterations: {[idx + 1 for idx in top_indices]}")
    print(f"  Test RMSE: {top_k_test_rmse:.2f}")
    print(f"  Test MAE:  {top_k_test_mae:.2f}")
    print(f"  Test R2:   {top_k_test_r2:.4f}")

    ensemble_short_test_mae = None
    ensemble_long_test_mae = None
    print(f"\n[{point.label}] === Ensemble Per-Regime Test MAE (routing-predicted) ===")
    if len(short_test_idx) > 0:
        ensemble_short_test_mae = float(
            mean_absolute_error(y_test[short_test_idx], mean_all_pred[short_test_idx])
        )
        print(
            f"Short regime (n={len(short_test_idx)}): "
            f"MAE = {ensemble_short_test_mae:.2f}s"
        )
    if len(long_test_idx) > 0:
        ensemble_long_test_mae = float(
            mean_absolute_error(y_test[long_test_idx], mean_all_pred[long_test_idx])
        )
        print(
            f"Long  regime (n={len(long_test_idx)}):  "
            f"MAE = {ensemble_long_test_mae:.2f}s"
        )

    summary = {
        "sweep_label": point.label,
        "t_evolution": point.t_evolution,
        "arch": cfg.arch,
        "regime_routing": cfg.regime_routing,
        "topology": cfg.topology,
        "noise_spec": point.noise_spec.as_dict(),
        "packing": packing.as_dict(),
        "top_indices": top_indices,
        "ensemble_test_mae": ensemble_test_mae,
        "ensemble_test_rmse": ensemble_test_rmse,
        "ensemble_test_r2": ensemble_test_r2,
        "top_k_ensemble_test_mae": top_k_test_mae,
        "top_k_ensemble_test_rmse": top_k_test_rmse,
        "top_k_ensemble_test_r2": top_k_test_r2,
        "ensemble_short_test_mae": ensemble_short_test_mae,
        "ensemble_long_test_mae": ensemble_long_test_mae,
        "n_short_test": int(len(short_test_idx)),
        "n_long_test": int(len(long_test_idx)),
        "per_iteration_val_mae": [r["val_mae"] for r in all_results],
        "per_iteration_val_rmse": [r["val_rmse"] for r in all_results],
        "per_iteration_val_r2": [r["val_r2"] for r in all_results],
        "per_iteration_test_mae": [r["test_mae"] for r in all_results],
        "per_iteration_test_rmse": [r["test_rmse"] for r in all_results],
    }
    file_tag = (
        point.label if packing.mode == "single" else f"{packing.label}_{point.label}"
    )
    summary_path = out_dir / f"aggregate_summary_{file_tag}.json"
    save_json(summary_path, summary)

    hardware_config = {
        "sweep_label": point.label,
        "t_evolution": point.t_evolution,
        "arch": cfg.arch,
        "regime_routing": cfg.regime_routing,
        "topology": cfg.topology,
        "regime_labels_csv": (
            str(cfg.regime_labels_csv)
            if cfg.arch == "ridge-sequential" and cfg.regime_routing == "csv"
            else None
        ),
        "noise_spec": point.noise_spec.as_dict(),
        "packing": packing.as_dict(),
        "top_k_indices": top_indices,
        "top_k_seeds": [cfg.random_seed + i for i in top_indices],
        "ising_params_per_iteration": {
            r["iteration"]: {
                "short": r["angle_bank_short"],
                "long": r["angle_bank_long"],
            }
            for r in top_results
        },
        "xgb_params_per_iteration": {
            r["iteration"]: {"short": r["short_params"], "long": r["long_params"]}
            for r in top_results
        },
        "regime_classifier": clf,
        "pipeline_config": cfg.__dict__,
        "scaling_params": {"train_min": train_min, "train_max": train_max},
        "short_threshold": cfg.short_threshold,
    }
    hw_path = out_dir / f"hardware_config_{file_tag}.pkl"
    with hw_path.open("wb") as f:
        pickle.dump(hardware_config, f)
    print(f"Wrote hardware config -> {hw_path}")
    return summary


def aggregate_partials(args: argparse.Namespace, cfg: QRCConfig):
    sweep_points = build_sweep_points(args)
    packing = PackingSpec(mode=args.packing_mode, pack_n=args.pack_n)
    partials_root = resolve_partials_root(args)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (
        _X_train_q,
        _X_val_q,
        _X_test_q,
        X_train_raw,
        X_val_raw,
        X_test_raw,
        y_train,
        y_val,
        y_test,
        train_min,
        train_max,
        subset_idx,
    ) = load_data(cfg, args.subset_frac)
    if cfg.arch == "ridge-sequential" and cfg.regime_routing == "csv":
        routing, _, _ = load_csv_regime_routing(
            Path(cfg.regime_labels_csv), y_train, y_val, y_test, subset_idx
        )
    else:
        routing = load_saved_regime_routing(partials_root, args, cfg)
    clf = routing.classifier
    short_mask_train = routing.short_mask_train
    long_mask_train = routing.long_mask_train
    short_mask_val = routing.short_mask_val
    long_mask_val = routing.long_mask_val
    short_test_idx = routing.short_test_idx
    long_test_idx = routing.long_test_idx
    short_val_idx = routing.short_val_idx
    long_val_idx = routing.long_val_idx

    sweep_summaries = []
    for point in sweep_points:
        try:
            spec_summary = _aggregate_one_spec(
                args,
                cfg,
                point,
                packing,
                partials_root,
                out_dir,
                y_train,
                y_val,
                y_test,
                short_mask_train,
                long_mask_train,
                short_mask_val,
                long_mask_val,
                short_val_idx,
                long_val_idx,
                short_test_idx,
                long_test_idx,
                len(_X_test_q),
                len(_X_val_q),
                clf,
                train_min,
                train_max,
            )
        except FileNotFoundError as exc:
            print(f"Skipping {point.label}: {exc}")
            continue
        sweep_summaries.append(spec_summary)

    if len(sweep_summaries) > 1:
        base = "t_sweep_summary" if args.t_sweep else "noise_sweep_summary"
        sweep_name = (
            f"{base}.json"
            if packing.mode == "single"
            else f"{base}_{packing.label}.json"
        )
        save_json(out_dir / sweep_name, sweep_summaries)
        print(
            f"Wrote sweep summary across "
            f"{len(sweep_summaries)} points -> "
            f"{out_dir / sweep_name}"
        )


def parse_fraction_sweep_arg(raw: str | None, default_frac: float) -> List[float]:
    if raw is None or not raw.strip():
        fracs = [default_frac]
    else:
        fracs = []
        for token in raw.split(","):
            tok = token.strip()
            if not tok:
                continue
            fracs.append(float(tok))
        if not fracs:
            fracs = [default_frac]
    clean = sorted(set(fracs))
    for frac in clean:
        if not (0.0 < frac <= 1.0):
            raise ValueError(
                f"Invalid fraction {frac}. Fractions must satisfy 0 < frac <= 1."
            )
    return clean


def _compute_estimate(
    cfg: QRCConfig,
    shots: int,
    subset_frac: float,
    backend,
    local_transpile,
    packing: PackingSpec | None = None,
    t_evolution: float = 0.1,
) -> Dict[str, Any]:
    if packing is None:
        packing = PackingSpec()
    (
        X_train_q,
        X_val_q,
        X_test_q,
        X_train_raw,
        X_val_raw,
        X_test_raw,
        y_train,
        y_val,
        y_test,
        _train_min,
        _train_max,
        subset_idx,
    ) = load_data(cfg, subset_frac)
    if cfg.arch == "ridge-sequential" and cfg.regime_routing == "csv":
        routing, _, _ = load_csv_regime_routing(
            Path(cfg.regime_labels_csv), y_train, y_val, y_test, subset_idx
        )
    else:
        routing = build_regime_routing(
            X_train_raw, X_val_raw, X_test_raw, y_train, cfg.optuna_trials
        )
    short_mask_train = routing.short_mask_train
    long_mask_train = routing.long_mask_train
    short_mask_val = routing.short_mask_val
    long_mask_val = routing.long_mask_val
    short_test_idx = routing.short_test_idx
    long_test_idx = routing.long_test_idx

    regime_sizes = {
        "short": {
            "train": int(np.sum(short_mask_train)),
            "val": int(np.sum(short_mask_val)),
            "test": int(len(short_test_idx)),
        },
        "long": {
            "train": int(np.sum(long_mask_train)),
            "val": int(np.sum(long_mask_val)),
            "test": int(len(long_test_idx)),
        },
    }

    # Noisy path measures Z, X, and Y bases separately (ZZ comes from Z counts),
    # so each binding incurs 3 circuit executions.
    basis_multiplier = 3
    effective_shots = packing.effective_shots(shots)

    n_events = cfg.n_previous_events + 1
    grand_total = 0.0
    total_circuit_executions = 0
    per_iteration = []

    for iter_idx in range(cfg.n_iterations):
        iter_total = 0.0
        iter_execs = 0
        iter_regime_breakdown = {}
        for regime in ("short", "long"):
            rng = np.random.default_rng(
                cfg.random_seed + iter_idx + (0 if regime == "short" else 10_000)
            )
            angle_bank = build_angle_bank(
                packing,
                cfg.n_qubits,
                n_events,
                rng,
                topology=cfg.topology,
                h=cfg.h_field,
                t=t_evolution,
            )
            n_rows = (
                regime_sizes[regime]["train"]
                + regime_sizes[regime]["val"]
                + regime_sizes[regime]["test"]
            )
            n_bindings_per_circuit = n_rows * basis_multiplier
            regime_total = 0.0
            if cfg.arch == "ridge-sequential":
                # One prefix circuit per event (events 0..k), depth grows
                # with the prefix length.
                for event_idx in range(n_events):
                    template, _ = build_parametric_sequential_prefix(
                        angle_bank,
                        event_idx,
                        cfg.num_layers_per_event,
                        cfg.n_qubits,
                        cfg.n_trotter_steps,
                        cfg.topology,
                    )
                    isa = local_transpile(template)
                    row = estimate_resources(
                        isa, backend, effective_shots, n_bindings_per_circuit
                    )
                    regime_total += row["est_qpu_seconds"]
                    iter_execs += n_bindings_per_circuit
                iter_regime_breakdown[regime] = regime_total
                iter_total += regime_total
                continue
            # Each entry in angle_bank corresponds to ONE circuit on the
            # device (joint-ising: one circuit per group of pack_n events;
            # shot-parallel: one wide circuit per event; single: one per event).
            for bank_idx, ising_params in enumerate(angle_bank):
                J = ising_params[0]
                n_q_circuit = J.shape[0]
                if packing.mode == "shot-parallel":
                    template, _ = build_replicated_reservoir_circuit(
                        ising_params,
                        cfg.num_layers_per_event,
                        cfg.n_qubits,
                        packing.pack_n,
                        cfg.n_trotter_steps,
                    )
                else:
                    template, _ = build_parametric_reservoir_circuit(
                        ising_params,
                        cfg.num_layers_per_event,
                        n_q_circuit,
                        cfg.n_trotter_steps,
                        cfg.topology,
                    )
                isa = local_transpile(template)
                row = estimate_resources(
                    isa, backend, effective_shots, n_bindings_per_circuit
                )
                regime_total += row["est_qpu_seconds"]
                iter_execs += n_bindings_per_circuit
            iter_regime_breakdown[regime] = regime_total
            iter_total += regime_total
        per_iteration.append(
            {
                "iteration": iter_idx,
                "short_est_qpu_equiv_seconds": float(iter_regime_breakdown["short"]),
                "long_est_qpu_equiv_seconds": float(iter_regime_breakdown["long"]),
                "iteration_est_qpu_equiv_seconds": float(iter_total),
                "iteration_circuit_executions": int(iter_execs),
            }
        )
        total_circuit_executions += iter_execs
        grand_total += iter_total

    return {
        "subset_frac": float(subset_frac),
        "shots": int(shots),
        "iterations": int(cfg.n_iterations),
        "events_per_regime": int(n_events),
        "basis_multiplier": int(basis_multiplier),
        "regime_sizes": regime_sizes,
        "per_iteration": per_iteration,
        "total_circuit_executions": int(total_circuit_executions),
        "grand_total_est_qpu_equiv_seconds": float(grand_total),
    }


def save_estimate_plot(estimates: List[Dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for --estimate-plot output."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    xs = [e["subset_frac"] for e in estimates]
    ys_minutes = [
        max(e["grand_total_est_qpu_equiv_seconds"] / 60.0, 1e-12) for e in estimates
    ]
    ys_circuits = [max(float(e["total_circuit_executions"]), 1.0) for e in estimates]

    fig, ax1 = plt.subplots(figsize=(9.2, 5.4))
    line1 = ax1.plot(
        xs,
        ys_minutes,
        marker="o",
        markersize=7,
        markeredgecolor="white",
        markeredgewidth=0.8,
        linewidth=2.2,
        color="#1f77b4",
        label="Est. QPU-equivalent minutes",
    )
    ax1.set_yscale("log")
    ax1.set_xlabel("Fraction of events/circuits used")
    ax1.set_ylabel("Estimated QPU-equivalent runtime (log minutes)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    line2 = ax2.plot(
        xs,
        ys_circuits,
        marker="s",
        markersize=6.5,
        markeredgecolor="white",
        markeredgewidth=0.8,
        linewidth=2.0,
        color="#d62728",
        label="Circuit executions",
    )
    ax2.set_yscale("log")
    ax2.set_ylabel("Total circuit executions (log scale)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    # Annotate each marker with exact values for manuscript readability.
    for x, y in zip(xs, ys_minutes):
        ax1.annotate(
            f"{y:.1f}m",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            color="#1f77b4",
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.15",
                "fc": "white",
                "ec": "none",
                "alpha": 0.75,
            },
        )
    for x, y in zip(xs, ys_circuits):
        ax2.annotate(
            f"{int(y):,}",
            (x, y),
            textcoords="offset points",
            xytext=(0, -12),
            ha="center",
            color="#d62728",
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.15",
                "fc": "white",
                "ec": "none",
                "alpha": 0.75,
            },
        )

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=True)
    ax1.set_title("Estimated Hardware Workload vs Subset Fraction")
    fig.tight_layout()
    fig.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def print_estimate(
    cfg: QRCConfig,
    shots: int,
    subset_frac: float,
    device: str,
    estimate_json: Path | None = None,
    estimate_sweep_fracs: str | None = None,
    estimate_plot: Path | None = None,
    packing: PackingSpec | None = None,
    t_evolution: float = 0.1,
):
    if packing is None:
        packing = PackingSpec()
    fractions = parse_fraction_sweep_arg(estimate_sweep_fracs, subset_frac)
    backend, _, local_transpile = build_noisy_simulator(
        device, None, n_qubits=packing.circuit_qubits(cfg.n_qubits)
    )
    estimates = []

    print("=== QRC Resource Estimate ===")
    print(
        f"topology={cfg.topology} | layers={cfg.num_layers_per_event} | "
        f"trotter={cfg.n_trotter_steps} | h={cfg.h_field} | t={t_evolution} | "
        f"iterations={cfg.n_iterations} | events_per_regime={cfg.n_previous_events + 1} "
        f"| shots={shots} (effective={packing.effective_shots(shots)}) "
        f"| basis_multiplier=3 | packing={packing.label}"
    )
    for frac in fractions:
        estimate = _compute_estimate(
            cfg,
            shots,
            frac,
            backend,
            local_transpile,
            packing=packing,
            t_evolution=t_evolution,
        )
        estimates.append(estimate)
        regime_sizes = estimate["regime_sizes"]
        print(
            f"\nsubset_frac={frac:.4f} | "
            f"short(train={regime_sizes['short']['train']}, val={regime_sizes['short']['val']}, test={regime_sizes['short']['test']}), "
            f"long(train={regime_sizes['long']['train']}, val={regime_sizes['long']['val']}, test={regime_sizes['long']['test']})"
        )
        for row in estimate["per_iteration"]:
            print(
                f"Iteration {row['iteration']:02d} | short={row['short_est_qpu_equiv_seconds']:.2f}s "
                f"long={row['long_est_qpu_equiv_seconds']:.2f}s "
                f"total={row['iteration_est_qpu_equiv_seconds']:.2f}s"
            )
        print(
            f"subset_frac={frac:.4f} totals | est_qpu_equiv_s={estimate['grand_total_est_qpu_equiv_seconds']:.2f} "
            f"| circuit_executions={estimate['total_circuit_executions']}"
        )

    if estimate_json is not None:
        payload = {
            "config": {
                "shots": shots,
                "device": device,
                "n_iterations": cfg.n_iterations,
                "n_qubits": cfg.n_qubits,
                "n_previous_events": cfg.n_previous_events,
                "num_layers_per_event": cfg.num_layers_per_event,
                "packing": packing.as_dict(),
            },
            "fractions": fractions,
            "estimates": estimates,
        }
        save_json(estimate_json, payload)
        print(f"Wrote estimate JSON -> {estimate_json}")

    if estimate_plot is not None:
        save_estimate_plot(estimates, estimate_plot)
        print(f"Wrote estimate plot -> {estimate_plot}")


def write_routing_artifact(args: argparse.Namespace, cfg: QRCConfig) -> None:
    partials_root = resolve_partials_root(args)
    partials_root.mkdir(parents=True, exist_ok=True)
    (
        _X_train_q,
        _X_val_q,
        _X_test_q,
        X_train_raw,
        X_val_raw,
        X_test_raw,
        y_train,
        _y_val,
        _y_test,
        *_,
    ) = load_data(cfg, args.subset_frac)
    ensure_regime_routing(
        partials_root,
        X_train_raw,
        X_val_raw,
        X_test_raw,
        y_train,
        args,
        cfg,
    )


def main():
    args = parse_args()
    cfg = build_config_from_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.estimate_only:
        print_estimate(
            cfg,
            args.shots,
            args.subset_frac,
            args.device,
            estimate_json=args.estimate_json,
            estimate_sweep_fracs=args.estimate_sweep_fracs,
            estimate_plot=args.estimate_plot,
            packing=PackingSpec(mode=args.packing_mode, pack_n=args.pack_n),
            t_evolution=cfg.t_evolution,
        )
        return

    if args.write_routing_only:
        if cfg.arch == "ridge-sequential" and cfg.regime_routing == "csv":
            print(
                "--arch ridge-sequential with --regime-routing csv loads "
                f"routing from {cfg.regime_labels_csv}; no regime_routing.pkl "
                "is needed."
            )
            return
        write_routing_artifact(args, cfg)
        return

    if args.task_id is not None:
        run_partial_task(args, cfg)
        return

    if args.aggregate:
        aggregate_partials(args, cfg)
        return

    # Single-node full execution: compute partials locally then aggregate.
    sweep_points = build_sweep_points(args)
    total_tasks = cfg.n_iterations * 2 * len(sweep_points)
    for task_id in range(total_tasks):
        args_task = argparse.Namespace(**vars(args))
        args_task.task_id = task_id
        run_partial_task(args_task, cfg)
    aggregate_partials(args, cfg)


if __name__ == "__main__":
    main()
