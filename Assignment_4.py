# Source Code: https://github.com/Vivswan/PHYS-3770-Quantum-Information
#
# Output:
# Question 1 (XYX = -Y): True
# Question 3: True
# Question 4: True
# Question 5: True

import numpy as np

from QuantumComputer import Y_GATE, QuantumComputer, Z_GATE, CNOT_GATE


def question1():
    qc = QuantumComputer(1)
    qc.H(0)
    qc.Y(0)
    qc.H(0)
    print("Question 1 (XYX = -Y):", np.isclose(qc.unitary, -Y_GATE).all())  # True


def question3():
    qc = QuantumComputer(2)
    qc.H(1)
    qc.controlled_gate(Z_GATE, 0, 1)
    qc.H(1)
    print("Question 3:", np.isclose(qc.unitary, CNOT_GATE).all())  # True


def question4():
    qc = QuantumComputer(2)
    qc.H(0)
    qc.H(1)
    qc.CNOT(0, 1)
    qc.H(0)
    qc.H(1)

    qc_dash = QuantumComputer(2)
    qc_dash.CNOT(1, 0)

    print("Question 4:", np.isclose(qc.unitary, qc_dash.unitary).all())  # True


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
    question1()
    question3()
    question4()
    question5()


if __name__ == '__main__':
    main()
