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


class QuantumComputer:
    def __init__(self, number_of_qubits):
        self.number_of_qubits = number_of_qubits
