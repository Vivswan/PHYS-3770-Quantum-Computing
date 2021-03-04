import numpy as np

from QC.gates import IDENTITY_2
from QC.helper_func import kron


def density_matrix_of(density_matrix, qubit):
    num_qubits = int(np.log2(np.shape(density_matrix)[0]))
    if not (0 <= qubit < num_qubits):
        raise Exception(f"Invalid qubit: 0 <= qubit: {qubit} <= num of qubit in matrix = {num_qubits - 1}")

    qubit_position_list = [IDENTITY_2] * num_qubits
    qubit_position_list[qubit] = np.array([[1, 2], [3, 4]])
    positions = kron(*qubit_position_list)

    single_density_matrix = np.zeros((2, 2), dtype=complex)

    for i in range(0, 4):
        single_density_matrix[int(i / 2)][i % 2] = np.sum(density_matrix[positions == i + 1])

    return single_density_matrix


def trace_or_throw(density_matrix, remove_qubit):
    num_qubits = int(np.log2(np.shape(density_matrix)[0]))

    if not (0 <= remove_qubit < num_qubits):
        raise Exception(f"Invalid remove qubit: "
                        f"0 <= remove_qubit: {remove_qubit} <= num of qubit in matrix = {num_qubits - 1}")

    qubit_position_list = [np.ones((2, 2))] * num_qubits
    qubit_position_list[remove_qubit] = np.array([[1, 0], [0, -1]])
    indexes = kron(*qubit_position_list)

    new_matrix = (density_matrix[indexes == 1] + density_matrix[indexes == -1])
    new_matrix = new_matrix.reshape((np.power(2, num_qubits - 1), np.power(2, num_qubits - 1)))

    return new_matrix


def trace_of_all(density_matrix, remove_qubits: list):
    r = list.copy(remove_qubits)
    r.sort()
    r.reverse()
    for i in r:
        density_matrix = trace_or_throw(density_matrix, i)

    return density_matrix
