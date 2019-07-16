import qiskit as qk
import numpy as np


class MeasurementCircuit(object):
    def __init__(self, circuit, stabilizer_matrix, N):
        self.circuit = circuit
        self.stabilizer_matrix = stabilizer_matrix
        self.N = N


    def __str__(self):
        return str(self.circuit) + '\n' + str(self.stabilizer_matrix) + '\n\n\n'


def _get_measurement_circuit(stabilizer_matrix, N):
    """Return MeasurementCircuit for simultaneous measurement of N operators in stabilizer_matrix.

    Each column of stabilizer_matrix represents a Pauli string that we seek to measure.
    Thus, stabilizer_matrix should have dimensions of 2 * N rows by N columns. The first N rows
    indicate the presence of a Z in each index of the Pauli String. The last N rows indicate X's.

    For instance, simultaneous measurement of YYI, XXY, IYZ would be represented by
    [[1, 0, 0],  ========
     [1, 0, 1],  Z matrix
     [0, 1, 1],  ========
     [1, 1, 0],  ========
     [1, 1, 1],  X matrix
     [0, 1, 0]   ========

    As annotated above, the submatrix of the first (last) N rows is referred to as the Z (X) matrix.

    All operators must commute and be independent (i.e. can't express any column as a base-2
    product of the other columns) for this code to work.
    """
    _validate_stabilizer_matrix(stabilizer_matrix, N)
    measurement_circuit = MeasurementCircuit(qk.QuantumCircuit(N, N), stabilizer_matrix, N)
    _prepare_X_matrix(measurement_circuit)
    _row_reduce_X_matrix(measurement_circuit)
    _patch_Z_matrix(measurement_circuit)
    _change_X_to_Z_basis(measurement_circuit)
    _terminate_with_measurements(measurement_circuit)

    return measurement_circuit


def _validate_stabilizer_matrix(stabilizer_matrix, N):
    assert stabilizer_matrix.shape == (2 * N, N), '%s qubits, but matrix shape: %s' % (N, stabilizer_matrix.shape)
    # i, j will always denote row, column index
    for i in range(2 * N):
        for j in range(N):
            value = stabilizer_matrix[i, j]
            assert value in [0, 1], '[%s, %s] index is %s' % (i, j, value)


def _prepare_X_matrix(measurement_circuit):
    # apply H's to ensure that diagonal of X matrix has 1's
    N = measurement_circuit.N
    for j in range(N):
        i = j + N
        if measurement_circuit.stabilizer_matrix[i, j] == 0:
            _apply_H(measurement_circuit, j)

def _row_reduce_X_matrix(measurement_circuit):
    """Use Gaussian elimination to reduce the Z matrix to the Identity matrix."""
    _transform_X_matrix_to_row_echelon_form(measurement_circuit)
    _transform_X_matrix_to_reduced_row_echelon_form(measurement_circuit)


def _transform_X_matrix_to_row_echelon_form(measurement_circuit):
    N = measurement_circuit.N
    for j in range(N):
        if measurement_circuit.stabilizer_matrix[j + N, j] == 0:
            i = j + 1
            while measurement_circuit.stabilizer_matrix[i + N, j] == 0:
                i += 1
            _apply_SWAP(measurement_circuit, i, j)

        for i in range(N + j + 1, 2 * N):
            if measurement_circuit.stabilizer_matrix[i, j] == 1:
                _apply_CNOT(measurement_circuit, j, i - N)


def _transform_X_matrix_to_reduced_row_echelon_form(measurement_circuit):
    N = measurement_circuit.N
    for j in range(N - 1, 0, -1):
        for i in range(N, N + j):
            if measurement_circuit.stabilizer_matrix[i, j] == 1:
                _apply_CNOT(measurement_circuit, j, i - N)


def _patch_Z_matrix(measurement_circuit):
    stabilizer_matrix, N = measurement_circuit.stabilizer_matrix, measurement_circuit.N
    assert np.allclose(stabilizer_matrix[:N],
            stabilizer_matrix[:N].T), 'Z-matrix,\n%s is not symmetric' % stabilizer_matrix
    
    for i in range(N):
        for j in range(0, i):
            if stabilizer_matrix[i, j] == 1:
                _apply_CZ(measurement_circuit, i, j)

        j = i
        if stabilizer_matrix[i, j] ==  1:
            _apply_S(measurement_circuit, i)


def _change_X_to_Z_basis(measurement_circuit):
    # change each qubit from X basis to Z basis via H
    N = measurement_circuit.N
    for j in range(N):
        _apply_H(measurement_circuit, j)


def _terminate_with_measurements(measurement_circuit):
    for j in range(measurement_circuit.N):
        measurement_circuit.circuit.measure(j, j)


def _apply_H(measurement_circuit, i):
    N = measurement_circuit.N
    measurement_circuit.stabilizer_matrix[[i, i + N]] = measurement_circuit.stabilizer_matrix[[i + N, i]]
    measurement_circuit.circuit.h(i)


def _apply_S(measurement_circuit, i):
    measurement_circuit.stabilizer_matrix[i, i] = 0
    measurement_circuit.circuit.s(i)


def _apply_CZ(measurement_circuit, i, j):
    measurement_circuit.stabilizer_matrix[i, j] = 0
    measurement_circuit.stabilizer_matrix[j, i] = 0
    measurement_circuit.circuit.cz(i, j)


def _apply_CNOT(measurement_circuit, control_index, target_index):
    N = measurement_circuit.N
    measurement_circuit.stabilizer_matrix[control_index] = (measurement_circuit.stabilizer_matrix[
        control_index] + measurement_circuit.stabilizer_matrix[target_index]) % 2
    measurement_circuit.stabilizer_matrix[target_index + N] = (measurement_circuit.stabilizer_matrix[
        control_index + N] + measurement_circuit.stabilizer_matrix[target_index + N]) % 2
    measurement_circuit.circuit.cx(control_index, target_index)


def _apply_SWAP(measurement_circuit, i, j):
    N = measurement_circuit.N
    measurement_circuit.stabilizer_matrix[[i, j]] = measurement_circuit.stabilizer_matrix[[j, i]]
    measurement_circuit.stabilizer_matrix[[i + N, j + N]] = measurement_circuit.stabilizer_matrix[[j + N, i + N]]
    measurement_circuit.circuit.swap(i, j)
