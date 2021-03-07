# Source Code: https://github.com/Vivswan/PHYS-3770-Quantum-Information
from multipledispatch import dispatch

from numpy import random, ndarray

from QC.checks import *
from QC.gates import *
from QC.helper_func import *
from QC.throw_or_trace import density_matrix_of
from QC.to_notation import *


def density_matrix_to_state(density_matrix: ndarray):
    return np.sqrt(density_matrix.diagonal().reshape((density_matrix.shape[0], 1)))


# noinspection PyPep8Naming
class QuantumComputer:
    state: ndarray
    density_matrix: ndarray
    unitary: ndarray

    @dispatch(int)
    def __init__(self, number_of_qubits):
        self.reset_state(number_of_qubits)

        self._save_unitary = self.unitary
        self._save_state = self.state
        self._save_density_matrix = self.density_matrix

    @dispatch(ndarray)
    def __init__(self, state):
        self.set_state(state)

        self._save_unitary = self.unitary
        self._save_state = self.state
        self._save_density_matrix = self.density_matrix

    def reset_state(self, number_of_qubits):
        dim = np.power(2, number_of_qubits)
        self.state = np.zeros((dim, 1), dtype=complex)
        self.state[0] = 1
        self.density_matrix = np.matmul(self.state, self.state.conjugate().transpose())
        self.unitary = np.identity(np.power(2, self.num_qubit()), dtype=complex)

    def set_state(self, state):
        if not np.power(2, int(np.log2(np.shape(state)[0]))) == np.shape(state)[0]:
            raise Exception("invalid state.")

        if not np.isclose(np.sum(np.abs(np.power(state, 2))), 1):
            raise Exception("State is not normalised.")

        self.state = np.copy(state)
        self.density_matrix = np.matmul(self.state, self.state.conjugate().transpose())
        self.unitary = create_unitary(zero_state_matrix(self.num_qubit()), self.state)

    def set_density_matrix(self, density_matrix):
        if not np.power(2, int(np.log2(np.shape(density_matrix)[0]))):
            raise Exception("invalid density_matrix.")

        if not np.shape(density_matrix)[0] == np.shape(density_matrix)[1]:
            raise Exception("density_matrix is not square.")

        self.density_matrix = np.copy(density_matrix)
        self.state = density_matrix_to_state(self.density_matrix)

    def get_density_matrix_of(self, qubit):
        return density_matrix_of(self.density_matrix, qubit)

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
        return self.full_gate(create_control_gate(X_GATE, targets, controls, self.num_qubit()))

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
        return physicality(self.density_matrix)

    def purity(self):
        return purity(self.density_matrix)

    def is_physical(self):
        return is_physical(self.density_matrix)

    def is_pure(self):
        return is_pure(self.density_matrix)

    def ket_notation(self, probabilities=False):
        return to_ket_notation(self.state, probabilities=probabilities)

    def bra_ket_notation(self):
        return to_bra_ket_notation(self.density_matrix)

    def get_probabilities(self):
        return np.power(np.abs(self.state), 2)

    def get_probability_of(self, qubit: int, outcome: [0, 1], get_states=False):
        if not (0 <= qubit < self.num_qubit()):
            raise Exception(f"invalid qubit number ({qubit})")

        gate = Measurement_ZERO if outcome == 0 else Measurement_ONE
        measurement = full_unitary_at_time_step(self.num_qubit(), {qubit: gate})

        new_density_matrix = matmul(measurement, self.density_matrix, measurement)
        probability = np.trace(np.abs(new_density_matrix))
        probability_check = np.abs(np.trace(matmul(self.state.conjugate().transpose(), measurement, self.state)))

        if not np.isclose(probability, probability_check):
            raise Exception(f"State and Density Matrix are not connected. QC is faulty.")

        if get_states:
            if probability != 0:
                new_density_matrix = new_density_matrix / probability
                new_state = density_matrix_to_state(new_density_matrix)
            else:
                new_density_matrix = None
                new_state = None

            return probability, new_state, new_density_matrix
        else:
            return probability

    def get_probabilities_of(self, qubit: int, get_states=False):
        probabilities = [0, 0]
        states = [None, None]
        density_matrices = [None, None]

        probabilities[0], states[0], density_matrices[0] = self.get_probability_of(qubit, 0, get_states=True)
        probabilities[1], states[1], density_matrices[1] = self.get_probability_of(qubit, 1, get_states=True)

        if get_states:
            return probabilities, states, density_matrices
        else:
            return probabilities

    def simulate_measurement(self, qubit, get_states=False):
        random_p = random.rand()
        probabilities, states, density_matrices = self.get_probabilities_of(qubit, get_states=True)

        observation = 0 if random_p <= probabilities[0] else 1
        if get_states:
            return observation, states[observation], density_matrices[observation]
        else:
            return observation

    def measurement(self, qubit):
        observation, state, density_matrix = self.simulate_measurement(qubit, get_states=True)
        self.state = state
        self.density_matrix = density_matrix
        return observation

    def force_measurement(self, qubit, outcome):
        if outcome not in [0, 1]:
            raise Exception(f"invalid outcome: {outcome}")

        _, state, density_matrix = self.get_probability_of(qubit, outcome, get_states=True)
        self.state = state
        self.density_matrix = density_matrix

    def measurement_all(self):
        observations = []

        for i in range(0, self.num_qubit()):
            observations.append(self.measurement(i))

        return observations

    def save(self):
        self._save_unitary = self.unitary
        self._save_state = self.state
        self._save_density_matrix = self.density_matrix

    def load(self):
        self.unitary = self._save_unitary
        self.state = self._save_state
        self.density_matrix = self._save_density_matrix
