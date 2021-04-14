# Source Code: https://github.com/Vivswan/PHYS-3770-Quantum-Information
#
# Output:
#
# Question 2:
# Initial state: 0.577 |0⟩ + 0.816i |1⟩
# Initial state: 0.333 |0⟩⟨0| + -0.471i |0⟩⟨1| + 0.471i |1⟩⟨0| + 0.667 |1⟩⟨1|
# New state: 0.383 |0⟩⟨0| + -0.330i |0⟩⟨1| + 0.330i |1⟩⟨0| + 0.617 |1⟩⟨1|
# Fidelity: 0.9219544457292888
#
# Question 4:
# Logical qubit with no bit flip:    0.707 |000⟩ + 0.707 |111⟩
# Z1Z2 measurement: 1.0 [+1] + 0.0 [-1]
# Z2Z3 measurement: 1.0 [+1] + 0.0 [-1]
# No correction needed
#
# Logical qubit with bit flip on 1:  0.707 |011⟩ + 0.707 |100⟩
# Z1Z2 measurement: 0.0 [+1] + 1.0 [-1]
# Z2Z3 measurement: 1.0 [+1] + 0.0 [-1]
# For correction: X Gate on qubit 1
#
# Logical qubit with bit flip on 2:  0.707 |010⟩ + 0.707 |101⟩
# Z1Z2 measurement: 0.0 [+1] + 1.0 [-1]
# Z2Z3 measurement: 0.0 [+1] + 1.0 [-1]
# For correction: X Gate on qubit 2
#
# Logical qubit with bit flip on 3:  0.707 |001⟩ + 0.707 |110⟩
# Z1Z2 measurement: 1.0 [+1] + 0.0 [-1]
# Z2Z3 measurement: 0.0 [+1] + 1.0 [-1]
# For correction: X Gate on qubit 3

import numpy as np

from QC.gates import Z_GATE, ZERO, ONE, IDENTITY_2
from QC.helper_func import matmul, kron
from QC.to_notation import to_ket_notation, to_ket_bra_notation, complex_to_str


def parity_measurement(state, measurement_gate):
    plus_one_m = (measurement_gate == 1).astype(int)
    minus_one_m = (measurement_gate == -1).astype(int)

    probability_plus_one = matmul(state.conj().T, plus_one_m.conj().T, plus_one_m, state)[0][0]
    probability_minus_one = matmul(state.conj().T, minus_one_m.conj().T, minus_one_m, state)[0][0]

    if probability_plus_one != 0:
        new_state_plus_one = matmul(plus_one_m, state) / np.sqrt(probability_plus_one)
    else:
        new_state_plus_one = None

    if probability_minus_one != 0:
        new_state_minus_one = matmul(minus_one_m, state) / np.sqrt(probability_minus_one)
    else:
        new_state_minus_one = None

    probabilities = (probability_plus_one, probability_minus_one)
    new_states = (new_state_plus_one, new_state_minus_one)

    return probabilities, new_states


def question2():
    p = 0.3

    initial_state = 1 / np.sqrt(3) * ZERO + complex(0, 1) * np.sqrt(2 / 3) * ONE
    initial_density_matrix = np.kron(initial_state, initial_state.T.conj())

    new_state = p / 2 * IDENTITY_2 + (1 - p) * initial_density_matrix

    fidelity = np.trace(np.sqrt(matmul(initial_state.T.conj(), new_state, initial_state))).real

    print()
    print("Question 2: ")
    print(f"Initial state: {to_ket_notation(initial_state)}")
    print(f"Initial state: {to_ket_bra_notation(initial_density_matrix)}")
    print(f"New state: {to_ket_bra_notation(new_state)}")
    print(f"Fidelity: {fidelity}")


def question4():
    Z1_Z2 = kron(Z_GATE, Z_GATE, IDENTITY_2)
    Z2_Z3 = kron(IDENTITY_2, Z_GATE, Z_GATE)

    def measurement_and_correction(logical_qubit):
        pm_Z1_Z2, _ = parity_measurement(logical_qubit, Z1_Z2)
        pm_Z2_Z3, _ = parity_measurement(logical_qubit, Z2_Z3)
        print(
            f"Z1Z2 measurement: "
            f"{complex_to_str(pm_Z1_Z2[0], 1, True)} [+1] + "
            f"{complex_to_str(pm_Z1_Z2[1], 1, True)} [-1]"
        )
        print(
            f"Z2Z3 measurement: "
            f"{complex_to_str(pm_Z2_Z3[0], 1, True)} [+1] + "
            f"{complex_to_str(pm_Z2_Z3[1], 1, True)} [-1]"
        )

        pm_Z1_Z2 = pm_Z1_Z2[0] - pm_Z1_Z2[1]
        pm_Z2_Z3 = pm_Z2_Z3[0] - pm_Z2_Z3[1]

        if np.isclose(pm_Z1_Z2, 1) and np.isclose(pm_Z2_Z3, 1):
            print("No correction needed")

        if np.isclose(pm_Z1_Z2, -1) and np.isclose(pm_Z2_Z3, 1):
            print("For correction: X Gate on qubit 1")

        if np.isclose(pm_Z1_Z2, -1) and np.isclose(pm_Z2_Z3, -1):
            print("For correction: X Gate on qubit 2")

        if np.isclose(pm_Z1_Z2, 1) and np.isclose(pm_Z2_Z3, -1):
            print("For correction: X Gate on qubit 3")

    qubit_with_no_bit_flip = np.sqrt(1 / 2) * (kron(ZERO, ZERO, ZERO) + kron(ONE, ONE, ONE))
    qubit_with_bit_flip_1 = np.sqrt(1 / 2) * (kron(ONE, ZERO, ZERO) + kron(ZERO, ONE, ONE))
    qubit_with_bit_flip_2 = np.sqrt(1 / 2) * (kron(ZERO, ONE, ZERO) + kron(ONE, ZERO, ONE))
    qubit_with_bit_flip_3 = np.sqrt(1 / 2) * (kron(ZERO, ZERO, ONE) + kron(ONE, ONE, ZERO))

    print()
    print("Question 4: ")
    print("Logical qubit with no bit flip:   ", to_ket_notation(qubit_with_no_bit_flip))
    measurement_and_correction(qubit_with_no_bit_flip)
    print()
    print("Logical qubit with bit flip on 1: ", to_ket_notation(qubit_with_bit_flip_1))
    measurement_and_correction(qubit_with_bit_flip_1)
    print()
    print("Logical qubit with bit flip on 2: ", to_ket_notation(qubit_with_bit_flip_2))
    measurement_and_correction(qubit_with_bit_flip_2)
    print()
    print("Logical qubit with bit flip on 3: ", to_ket_notation(qubit_with_bit_flip_3))
    measurement_and_correction(qubit_with_bit_flip_3)


if __name__ == '__main__':
    question2()
    question4()
