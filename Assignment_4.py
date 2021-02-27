# Source Code: https://github.com/Vivswan/PHYS-3770-Quantum-Information
#
# Output:

import numpy as np

from QuantumComputer import X_GATE, Y_GATE, QuantumComputer


def question1():
    s1 = np.matmul(X_GATE, Y_GATE)
    s2 = np.matmul(s1, X_GATE)
    print(s2, np.all(s2 == - Y_GATE))


def question4():
    qc = QuantumComputer(2)
    qc.H(0)
    qc.H(1)
    qc.CNOT(0, 1)
    qc.H(0)
    qc.H(1)
    print(np.round(qc.unitary, 3))


def question5():
    T_GATE = np.array([
        [1, 0],
        [0, np.power(np.e, complex(0, np.pi / 4))]
    ])

    TOFFOLI_GATE = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0]
    ])

    qc = QuantumComputer(3)
    qc.H(2)
    qc.CNOT(1, 2)
    qc.gates(T_GATE.conjugate().transpose(), 2)
    qc.CNOT(0, 2)
    qc.gates(T_GATE, 2)
    qc.CNOT(1, 2)
    qc.gates(T_GATE.conjugate().transpose(), 2)
    qc.CNOT(0, 2)
    qc.gates(T_GATE, 1)
    qc.gates(T_GATE, 2)
    qc.CNOT(0, 1)
    qc.H(2)
    qc.gates(T_GATE, 0)
    qc.gates(T_GATE.conjugate().transpose(), 1)
    qc.CNOT(0, 1)

    print("Question 5:", np.isclose(qc.unitary, TOFFOLI_GATE).all())  # True


def main():
    # question1()
    question4()
    question5()


if __name__ == '__main__':
    main()
