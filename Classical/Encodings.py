"""
ry               : Ry(x_i) per qubit. Product state. (current baseline)
dense_angle      : Rz(x_{2i+1}) Ry(x_{2i}) per qubit. 2 features/qubit.
iqp              : H^n D(x) H^n D(x) with Z + ZZ interactions.
amplitude        : Features as state amplitudes. ceil(log2(n)) qubits.
data_reuploading : Re-encode between each reservoir layer.
"""

from typing import Callable
import numpy as np
from qiskit.circuit import QuantumCircuit


# Scaling functions
# All fit on X_train to avoid data leakage.
def _minmax_scale(X_train, X_val, X_test, lo, hi):
    """Min-max scale all sets to [lo, hi] using X_train min/max."""
    train_min = X_train.min(axis=0)
    train_max = X_train.max(axis=0)
    denom = train_max - train_min
    denom[denom == 0] = 1.0

    def transform(X):
        scaled = np.clip((X - train_min) / denom, 0.0, 1.0)
        return scaled * (hi - lo) + lo

    return transform(X_train), transform(X_val), transform(X_test)


def scale_to_pi(X_train, X_val, X_test):
    """Scale to [0, pi]. For angle-based encodings (ry, dense_angle, data_reuploading)."""
    return _minmax_scale(X_train, X_val, X_test, 0.0, np.pi)


def scale_to_2pi(X_train: np., X_val, X_test):
    """Scale to [0, 2pi]. For phase-based encodings (iqp)."""
    return _minmax_scale(X_train, X_val, X_test, 0.0, 2 * np.pi)


def scale_to_unit_norm(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min-max to [0,1] then L2-normalize each row. For amplitude encoding."""
    train_min = X_train.min(axis=0)
    train_max = X_train.max(axis=0)
    denom = train_max - train_min
    denom[denom == 0] = 1.0

    def transform(X):
        scaled = np.clip((X - train_min) / denom, 0.0, 1.0)
        norms = np.linalg.norm(scaled, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return scaled / norms

    return transform(X_train), transform(X_val), transform(X_test)


# Encoding functions
#
# All have signature: encode(qc, data_sample, n_qubits) -> None
# They modify qc in-place, adding only the encoding gates.
def encode_ry(qc: QuantumCircuit, data_sample: np.ndarray, n_qubits: int) -> None:
    """
    Ry rotation encoding: Ry(x_i) on qubit i.

    Product state, zero entanglement from encoding alone.
    Requires: n_qubits = n_features, data in [0, pi].
    """
    for i in range(n_qubits):
        qc.ry(float(data_sample[i]), i)


def encode_dense_angle(
    qc: QuantumCircuit, data_sample: np.ndarray, n_qubits: int
) -> None:
    """
    Dense angle encoding: Rz(x_{2i+1}) Ry(x_{2i}) on qubit i.

    Packs 2 features per qubit (LaRose & Coyle, 2020).
    Still a product state but halves the qubit count.
    Requires: n_qubits = ceil(n_features / 2), data in [0, pi].
    """
    n_features = len(data_sample)
    for i in range(n_qubits):
        idx_y = 2 * i
        idx_z = 2 * i + 1
        if idx_y < n_features:
            qc.ry(float(data_sample[idx_y]), i)
        if idx_z < n_features:
            qc.rz(float(data_sample[idx_z]), i)


def encode_iqp(qc: QuantumCircuit, data_sample: np.ndarray, n_qubits: int) -> None:
    """
       IQP-style encoding
    = H^n * D(x) repeated twice.

       D(x) applies:
         - Rz(x_i) on each qubit (single-feature phases)
         - ZZ(x_i * x_j) on nearest-neighbor pairs (feature correlations)

       Requires: n_qubits = n_features, data in [0, 2pi].
    """
    n_features = min(len(data_sample), n_qubits)

    for _ in range(2):
        # Hadamard layer
        for i in range(n_qubits):
            qc.h(i)

        # Single qubit Z rotations
        for i in range(n_features):
            qc.rz(float(data_sample[i]), i)

        # ZZ interactions on nearest-neighbor pairs
        # Rzz(theta) = CX * (I (x) Rz(theta)) * CX
        for i in range(n_features - 1):
            angle = float(data_sample[i] * data_sample[i + 1])
            qc.cx(i, i + 1)
            qc.rz(angle, i + 1)
            qc.cx(i, i + 1)


def encode_amplitude(
    qc: QuantumCircuit, data_sample: np.ndarray, n_qubits: int
) -> None:
    """
    Amplitude encoding: features become amplitudes of the quantum state.

    Uses only ceil(log2(n_features)) qubits. Input must be L2-normalized.
    """
    n_states = 2**n_qubits
    padded = np.zeros(n_states)
    padded[: len(data_sample)] = data_sample
    norm = np.linalg.norm(padded)
    if norm > 0:
        padded = padded / norm
    qc.initialize(padded, qc.qubits)  # type: ignore


# Circuit builders
# TODO: consider different entanglement topologies
def _reservoir_layers(
    qc: QuantumCircuit, random_angles: np.ndarray, layer: int, n_qubits: int
) -> None:
    """Apply one reservoir layer: Rx/Rz/Ry rotations + CNOT ring"""
    # TODO: decide if we want to detach entanglement scheme from reservoir itself
    for i in range(n_qubits):
        qc.rx(float(random_angles[layer, i, 0]), i)
        qc.rz(float(random_angles[layer, i, 1]), i)
        qc.ry(float(random_angles[layer, i, 2]), i)
    # CNOT ring (cyclic)
    qc.cx(n_qubits - 1, 0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)


def build_reservoir_circuit(
    data_sample: np.ndarray,
    random_angles: np.ndarray,
    num_layers: int,
    n_qubits: int,
    encode_fn: Callable[[QuantumCircuit, np.ndarray, int], None],
) -> QuantumCircuit:
    """
    Generic reservoir circuit with options to use different encodings.

    Structure: encode -> [random rotations & CNOT ring] x num_layers -> measure.

    Parameters
    ---
    data_sample : array-like
        Scaled feature vector.
    random_angles : ndarray (num_layers, n_qubits, 3)
        Rx, Rz, Ry angles per layer per qubit.
    num_layers : int
        Number of reservoir layers.
    n_qubits : int
        Number of qubits.
    encode_fn : callable
        Encoding function: (qc, data_sample, n_qubits) -> None which modifies the quantum circuit in-place to add the encoding gates.

    Returns
    ---
    QuantumCircuit
    """
    qc = QuantumCircuit(n_qubits)

    encode_fn(qc, data_sample, n_qubits)
    qc.barrier()

    for layer in range(num_layers):
        _reservoir_layers(qc, random_angles, layer, n_qubits)
        qc.barrier()

    qc.measure_all()
    return qc


def build_reuploading_circuit(
    data_sample: np.ndarray, random_angles: np.ndarray, num_layers: int, n_qubits: int
) -> QuantumCircuit:
    """
    Data re-uploading circuit which re-encodes the data before every reservoir layer.

    Re-encodes data (Ry) before every reservoir layer, adding +1 Fourier
    frequency per layer. With L layers this is a universal approximator.

    Structure: [Ry(x) -> random rotations -> CNOT ring] x num_layers -> measure.

    Parameters
    ---
    data_sample : array-like
        Feature vector in [0, pi].
    random_angles : ndarray (num_layers, n_qubits, 3)
        Rx, Rz, Ry angles per layer per qubit.
    num_layers, n_qubits : int
        Number of reservoir layers and number of qubits.

    Returns
    ---
    QuantumCircuit
    """
    qc = QuantumCircuit(n_qubits)

    for layer in range(num_layers):
        # Re-encode data before each reservoir layer
        n_features = min(len(data_sample), n_qubits)
        for i in range(n_features):
            qc.ry(float(data_sample[i]), i)
        qc.barrier()

        _reservoir_layers(qc, random_angles, layer, n_qubits)
        qc.barrier()

    qc.measure_all()
    return qc
