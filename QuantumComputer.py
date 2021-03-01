# Source Code: https://github.com/Vivswan/PHYS-3770-Quantum-Information

import numpy as np
from numpy import random, ndarray

I = complex(0, 1)
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
    [0, -I],
    [I, 0]
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


def is_unitary(m):
    m = np.array(m)
    return np.all(np.isclose(m.conjugate().transpose().dot(m), np.identity(m.shape[0])))


def is_hermitian(m):
    m = np.array(m)
    return np.all(np.isclose(m.conjugate().transpose(), m))


def apply_to_list(list, func):
    c = None
    for i in list:
        if c is None:
            c = i
        else:
            c = func(c, i)
    return c


def kron(*args):
    return apply_to_list(args, np.kron)


def matmul(*args):
    return apply_to_list(args, np.matmul)


def full_unitary_at_time_step(number_qubits, gates):
    full_gate = None
    for index in range(0, number_qubits):
        if index in gates:
            if not gates[index].shape == (2, 2):
                raise Exception(f"Gates on {index} is not a single qubit gate.")
            # if not is_unitary(gates[index]):
            #     raise Exception(f"Gates on {index} is not unitary.")
            gate = gates[index]
        else:
            gate = IDENTITY_2

        if index == 0:
            full_gate = gate
        else:
            full_gate = np.kron(full_gate, gate)

    return full_gate


def density_matrix_to_state(density_matrix: ndarray):
    state = np.zeros((density_matrix.shape[0], 1), dtype=complex)
    for i in range(0, density_matrix.shape[0]):
        state[i] = np.sqrt(density_matrix[i][i])

    return state


# noinspection PyPep8Naming
class QuantumComputer:
    state: ndarray
    density_matrix: ndarray
    unitary: ndarray

    def __init__(self, number_of_qubits):
        self.state, self.density_matrix = self.reset_state(number_of_qubits)
        self.unitary = np.identity(np.power(2, number_of_qubits), dtype=complex)

    def reset_state(self, number_of_qubits):
        dim = np.power(2, number_of_qubits)
        self.state = np.zeros((dim, 1), dtype=complex)
        self.state[0] = 1
        self.density_matrix = np.matmul(self.state, self.state.conjugate().transpose())
        return self.state, self.density_matrix

    def set_state(self, state):
        self.state = np.copy(state)
        self.density_matrix = np.matmul(self.state, self.state.conjugate().transpose())

    def set_density_matrix(self, density_matrix):
        self.density_matrix = np.copy(density_matrix)
        self.state = density_matrix_to_state(self.density_matrix)

    def num_qubit(self):
        return int(np.log2(self.state.shape[0]))

    def gates(self, gates, qubit=None):
        if type(gates) != 'dict':
            gates = {qubit: gates}

        self.full_gate(full_unitary_at_time_step(self.num_qubit(), gates))

    def full_gate(self, gate):
        if not gate.shape == self.unitary.shape:
            raise Exception("Gate is not a Full Gate.")
        if not is_unitary(gate):
            raise Exception("Full Gate not unitary.")

        self.unitary = np.matmul(gate, self.unitary)
        self.density_matrix = matmul(gate.conj().T, self.density_matrix, gate)
        self.state = np.matmul(gate, self.state)

    def controlled_gate(self, gate, targets, controls):
        if not gate.shape == (2, 2):
            raise Exception(f"Gates is not a single qubit gate.")
        if not is_unitary(gate):
            raise Exception("Full Gate not unitary.")

        zero_term = []
        one_term = []

        if controls is not list:
            controls = [controls]

        if targets is not list:
            targets = [targets]

        for i in range(0, self.num_qubit()):
            if i in controls:
                zero_term.append(Measurement_ZERO)
                one_term.append(Measurement_ONE)
            elif i in targets:
                zero_term.append(IDENTITY_2)
                one_term.append(gate)
            else:
                zero_term.append(IDENTITY_2)
                one_term.append(IDENTITY_2)

        return self.full_gate(np.array(kron(*zero_term) + kron(*one_term)))

    def X(self, target):
        return self.gates(X_GATE, target)

    def Y(self, target):
        return self.gates(Y_GATE, target)

    def Z(self, target):
        return self.gates(Z_GATE, target)

    def H(self, target):
        return self.gates(H_GATE, target)

    def CNOT(self, control, target):
        return self.controlled_gate(X_GATE, target, control)

    def physicality(self):
        return np.trace(self.density_matrix)

    def purity(self):
        return np.trace(np.matmul(self.density_matrix, self.density_matrix))

    def is_physical(self):
        return np.isclose(self.physicality(), 1)

    def is_pure(self):
        return np.isclose(self.purity(), 1)

    def bra_notation(self, probabilites=False):
        max_state = int(np.log2(len(self.state)))
        notation = ""

        for index, state in enumerate(self.state):
            state = state[0]
            if not np.isclose(state, 0):
                if len(notation) > 0:
                    notation += " + "

                if probabilites:
                    notation += f"{np.power(np.abs(state), 2):.3f}"
                else:
                    is_state_real = not np.isclose(complex(state).real, 0)
                    is_state_imag = not np.isclose(complex(state).imag, 0)
                    if is_state_real:
                        notation += f"{complex(state).real:.3f}"
                    if is_state_real and is_state_imag:
                        notation += " + "
                    if is_state_imag:
                        notation += f"{complex(state).imag:.3f}i"

                n_state = "{0:b}".format(index).zfill(max_state)
                notation += f" |{n_state}⟩"

        return notation

    def bra_ket_notation(self):
        num_qubits = int(np.log2(np.shape(self.density_matrix)[0]))
        bra_ket_notation = ""

        notation_array = []
        for i in range(0, 2 ** num_qubits):
            notation_array.append("{0:b}".format(i).zfill(num_qubits))

        for row_index, row in enumerate(notation_array):
            for col_index, col in enumerate(notation_array):
                if not np.isclose(self.density_matrix[row_index][col_index], 0):
                    if len(bra_ket_notation) > 0:
                        bra_ket_notation += " + "
                    if np.isclose(np.imag(self.density_matrix[row_index][col_index]), 0):
                        value = np.real(self.density_matrix[row_index][col_index])
                    else:
                        value = np.imag(self.density_matrix[row_index][col_index])

                    bra_ket_notation += f"{str(np.round(value, 3)).replace('j', 'i')} |{row}⟩⟨{col}|"

        return bra_ket_notation

    def get_probabilities(self):
        return np.power(np.abs(self.state), 2)

    def get_probability_of(self, qubit: int, outcome: [0, 1], get_states=False):
        if 0 > qubit >= self.num_qubit():
            raise Exception(f"invalid qubit number ({qubit})")

        if outcome == 0:
            gate = Measurement_ZERO
        else:
            gate = Measurement_ONE

        measurement = full_unitary_at_time_step(self.num_qubit(), {qubit: gate})
        d = matmul(measurement, self.density_matrix, measurement)
        p = np.trace(np.abs(d))

        if get_states:
            if p != 0:
                d = d / p
                s = density_matrix_to_state(d)
            else:
                d = None
                s = None

            return p, s, d
        else:
            return p

    def get_probabilities_of(self, qubit: int, get_states=False):
        p = [0, 0]
        d = [None, None]
        s = [None, None]

        p[0], s[0], d[0] = self.get_probability_of(qubit, 0, get_states=True)
        p[1], s[1], d[1] = self.get_probability_of(qubit, 1, get_states=True)

        if get_states:
            return p, s, d
        else:
            return p

    def simulate_measurement(self, qubit, get_states=False):
        random_p = random.rand()
        p, s, d = self.get_probabilities_of(qubit, get_states=True)

        observation = 0 if random_p <= p[0] else 1
        if get_states:
            return observation, s[observation], d[observation]
        else:
            return observation

    def measurement(self, qubit):
        observation, s, d = self.simulate_measurement(qubit, get_states=True)
        self.state = s
        self.density_matrix = d
        return observation

    def force_measurement(self, qubit, outcome):
        if outcome not in [0, 1]:
            raise Exception(f"invalid outcome: {outcome}")

        _, s, d = self.get_probability_of(qubit, outcome, get_states=True)
        self.state = s
        self.density_matrix = d

    def measurement_all(self):
        observations = []

        for i in range(0, self.num_qubit()):
            observations.append(self.measurement(i))

        return observations
