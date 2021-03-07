import numpy as np

from QC.checks import is_unitary
from QC.helper_func import kron, nullspace

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

    if not isinstance(one_controls, list):
        one_controls = [one_controls]

    # if not isinstance(zero_controls, list):
    #     targets = [targets]

    if not isinstance(targets, list):
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


def create_unitary(initial, final):
    if np.shape(initial) != np.shape(final):
        raise Exception("shape of initial and final must be same.")

    if np.isclose(initial, final).all():
        return np.identity(initial.shape[0])

    check = np.zeros(np.shape(initial))
    check[0][0] = 1

    check_initial = np.all(np.isclose(initial, check))
    check_final = np.all(np.isclose(final, check))

    def get_q(matrix):
        measurement = np.matmul(matrix, check.transpose().conjugate())
        null_spaced = measurement + np.pad(nullspace(measurement), ((0, 0), (1, 0)))
        q, _ = np.linalg.qr(null_spaced)
        return -q

    if check_initial and not check_final:
        return get_q(final)

    if check_final and not check_initial:
        return np.linalg.inv(get_q(initial))

    return np.matmul(get_q(final), np.linalg.inv(get_q(initial)))
