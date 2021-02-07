import numpy as np

I = complex(0, 1)

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


def make_full_gate_matrix(gate, qubit, qubits):
    np_gate = np.array(gate)
    gate_size = np_gate.shape[0] / 2
    all_gates = []

    for i in range(0, qubits):
        if qubit == i:
            all_gates.append(np_gate)
        elif qubit < i < qubit + gate_size:
            all_gates.append(None)
        else:
            all_gates.append(np.identity(2))

    full_gate = all_gates[0]

    for index, value in enumerate(all_gates):
        if index != 0 and value is not None:
            full_gate = np.kron(full_gate, value)

    return full_gate


# noinspection PyPep8Naming
class QuantumComputer:
    def __init__(self, number_of_qubits):
        self.state = self.reset_state(number_of_qubits)

    def reset_state(self, number_of_qubits):
        self.state = np.zeros(np.power(2, number_of_qubits))
        self.state[0] = 1
        return self.state

    def set_state(self, state):
        state = np.copy(state)
        if int(np.log2(len(state))) != len(state):
            raise Exception("invalid state")
        self.state = np.copy(state)

    def get_state(self):
        return np.copy(self.state)

    def num_qubit(self):
        return int(np.log2(len(self.state)))

    def gate(self, gate, qubit: int):
        gate = np.array(gate)

        if not is_unitary(gate):
            raise Exception("gate is not unitary")

        full_gate = make_full_gate_matrix(gate, qubit, self.num_qubit())
        self.state = np.matmul(full_gate, self.state)

    def X(self, *args):
        return self.gate(X_GATE, *args)

    def Y(self, *args):
        return self.gate(Y_GATE, *args)

    def Z(self, *args):
        return self.gate(Z_GATE, *args)

    def H(self, *args):
        return self.gate(H_GATE, *args)

    def CNOT(self, *args):
        return self.gate(CNOT_GATE, *args)

    def print_wavefunction(self, with_zeros=False, probabilites=False):
        max_state = int(np.log2(len(self.state)))
        first = False
        print("Wavefunction = ", end='')

        for index, state in enumerate(self.state):
            if np.isclose(state, 0) and not with_zeros:
                continue

            if index > 0 and first:
                print(" + ", end='')

            if probabilites:
                print(f"{np.power(np.abs(state), 2):.3f} ", end='')
            else:
                is_state_real = not np.isclose(complex(state).real, 0)
                is_state_imag = not np.isclose(complex(state).imag, 0)
                if is_state_real:
                    print(f"{complex(state).real:.3f} ", end='')
                if is_state_real and is_state_imag:
                    print("+ ", end='')
                if is_state_imag:
                    print(f"{complex(state).imag:.3f}i ", end='')

            n_state = bin(index).replace('0b', '')
            n_state = ("0" * (max_state - len(n_state))) + n_state
            print(f"|{n_state}>", end='')

            first = True

        print()

    def get_probabilities(self):
        return np.power(np.abs(self.state), 2)

    def get_probabilities_of(self, qubit: int):
        p = [0, 0]

        for i in range(0, len(self.state)):
            n_state = bin(i).replace('0b', '')
            n_state = ("0" * (self.num_qubit() - len(n_state))) + n_state
            p[int(n_state[qubit])] += np.power(np.abs(self.state[i]), 2)

        return np.array(p)

    def print_probabilities(self):
        self.print_wavefunction(probabilites=True)

    def normalise(self):
        self.state = self.state / np.linalg.norm(self.state)
        return self.state

    def measurement(self, qubit):
        pass

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
