# Source Code: https://github.com/Vivswan/PHYS-3770-Quantum-Information

import numpy as np
from numpy import random, ndarray

I = complex(0, 1)

ZERO = np.array([
    [1],
    [0]
])
ONE = np.array([
    [0],
    [1]
])

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


def kron(args):
    c = None
    for i in args:
        if c is None:
            c = i
        else:
            c = np.kron(c, i)
    return c


def full_unitary_at_time_step(number_qubits, gates):
    full_gate = None
    for index in range(0, number_qubits):
        if index in gates:
            if not gates[index].shape == (2, 2):
                raise Exception(f"Gates on {index} is not a single qubit gate.")
            if not is_unitary(gates[index]):
                raise Exception(f"Gates on {index} is not unitary.")
            gate = gates[index]
        else:
            gate = np.identity(2)

        if index == 0:
            full_gate = gate
        else:
            full_gate = np.kron(full_gate, gate)

    return full_gate


# noinspection PyPep8Naming
class QuantumComputer:
    state: ndarray
    density_matrix: ndarray
    unitary: ndarray

    def __init__(self, number_of_qubits):
        self.state, self.density_matrix = self.reset_state(number_of_qubits)
        self.unitary = np.identity(np.power(2, number_of_qubits))

    def reset_state(self, number_of_qubits):
        dim = np.power(2, number_of_qubits)
        self.state = np.zeros((dim, 1))
        self.state[0] = 1
        self.density_matrix = np.matmul(self.state, self.state.conjugate().transpose())
        return self.state, self.density_matrix

    def set_state(self, state):
        self.state = np.copy(state)
        self.density_matrix = np.matmul(self.state, self.state.conjugate().transpose())

    def set_density_matrix(self, density_matrix):
        self.density_matrix = np.copy(density_matrix)
        self.state = np.zeros((self.density_matrix.shape[0], 1))
        for i in range(0, self.density_matrix.shape[0]):
            self.state[i] = np.sqrt(self.density_matrix[i][i])

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
        self.density_matrix = np.matmul(np.matmul(gate.conj().T, self.density_matrix), gate)
        self.state = np.matmul(gate, self.state)

    def controlled_gate(self, gate, controls, targets):
        if not gate.shape == (2, 2):
            raise Exception(f"Gates is not a single qubit gate.")
        if not is_unitary(gate):
            raise Exception("Full Gate not unitary.")

        zero_term = []
        one_term = []
        two_by_two_i = np.identity(2)

        if controls is not list:
            controls = [controls]

        if targets is not list:
            targets = [targets]

        for i in range(0, self.num_qubit()):
            if i in controls:
                zero_term.append(np.matmul(ZERO, ZERO.conjugate().T))
                one_term.append(np.matmul(ONE, ONE.conjugate().T))
            elif i in targets:
                zero_term.append(two_by_two_i)
                one_term.append(gate)
            else:
                zero_term.append(two_by_two_i)
                one_term.append(two_by_two_i)

        return self.full_gate(np.array(kron(zero_term) + kron(one_term)))

    def X(self, target):
        return self.gates(X_GATE, target)

    def Y(self, target):
        return self.gates(Y_GATE, target)

    def Z(self, target):
        return self.gates(Z_GATE, target)

    def H(self, target):
        return self.gates(H_GATE, target)

    def CNOT(self, control, target):
        return self.controlled_gate(X_GATE, control, target)

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

    def get_probabilities_of(self, qubit: int):
        p = [0, 0]

        for i in range(0, len(self.state)):
            n_state = int("{0:b}".format(i).zfill(self.num_qubit())[qubit])
            p[n_state] += np.power(np.abs(self.state[i]), 2)

        return np.array(p)

    def normalise(self):
        self.state = self.state / np.linalg.norm(self.state)
        return self.state

    def sim_measurement(self, qubit):
        random_p = random.rand()
        p0 = self.get_probabilities_of(qubit)[0]

        observation = 0 if random_p <= p0 else 1
        return observation

    def measurement(self, qubit):
        observation = self.sim_measurement(qubit)
        self.force_measurement(qubit, observation)
        return observation

    def force_measurement(self, qubit, outcome):
        if qubit >= self.num_qubit() or qubit < 0:
            raise Exception(f"invalid qubit number ({qubit})")

        if outcome not in [0, 1]:
            raise Exception(f"invalid outcome: {outcome}")

        for i in range(0, len(self.state)):
            n_state = bin(i).replace('0b', '')
            n_state = ("0" * (self.num_qubit() - len(n_state))) + n_state
            self.state[i] *= 1 if outcome == int(n_state[qubit]) else 0

        self.normalise()
