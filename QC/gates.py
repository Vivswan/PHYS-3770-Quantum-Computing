import numpy as np

from QC.checks import is_unitary
from QC.helper_func import kron

IDENTITY_2 = np.identity(2)

ZERO = np.array([
    [1],
    [0]
])
ONE = np.array([
    [0],
    [1]
])

# Measurements
Measurement_ZERO = np.matmul(ZERO, ZERO.conjugate().transpose())
Measurement_ONE = np.matmul(ONE, ONE.conjugate().transpose())

# Gates
X_GATE = np.array([
    [0, 1],
    [1, 0]
])
Y_GATE = np.array([
    [0, -complex(0, 1)],
    [complex(0, 1), 0]
])
Z_GATE = np.array([
    [1, 0],
    [0, -1]
])
H_GATE = (1 / np.sqrt(2)) * np.array([
    [1, 1],
    [1, -1]
])
CNOT_GATE = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
])


def full_unitary_at_time_step(number_qubits, gates):
    full_gate = None
    for index in range(0, number_qubits):
        if index in gates:
            if not gates[index].shape == (2, 2):
                raise Exception(f"Gates on {index} is not a single qubit gate.")
            gate = gates[index]
        else:
            gate = IDENTITY_2

        if index == 0:
            full_gate = gate
        else:
            full_gate = np.kron(full_gate, gate)

    return full_gate


def create_control_gate(gate, targets, one_controls, total_number_of_qubits):  # , zero_controls=[]
    if not gate.shape == (2, 2):
        raise Exception(f"Gates is not a single qubit gate.")
    if not is_unitary(gate):
        raise Exception("Full Gate not unitary.")

    zero_term = []
    one_term = []

    if one_controls is not list:
        one_controls = [one_controls]

    # if zero_controls is not list:
    #     targets = [targets]

    if targets is not list:
        targets = [targets]

    for i in range(0, total_number_of_qubits):
        if i in one_controls:
            zero_term.append(Measurement_ZERO)
            one_term.append(Measurement_ONE)
            continue

        # if i in zero_controls:
        #     zero_term.append(Measurement_ONE)
        #     one_term.append(Measurement_ZERO)
        #     continue

        if i in targets:
            zero_term.append(IDENTITY_2)
            one_term.append(gate)
            continue

        zero_term.append(IDENTITY_2)
        one_term.append(IDENTITY_2)

    return np.array(kron(*zero_term) + kron(*one_term))
