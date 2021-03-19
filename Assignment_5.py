# Source Code: https://github.com/Vivswan/PHYS-3770-Quantum-Information
#
# Output:
#
# Question 1:
# [[ 1  0  0  0]
#  [ 0 -1  0  0]
#  [ 0  0 -1  0]
#  [ 0  0  0  1]]
# Is this the ZZ parity gate:  True
#
# Question 2:
# Probabilities for A:  0.00 [+1] + 1.00 [-1]  is entangled:  False
# Probabilities for B:  1.00 [+1] + 0.00 [-1]  is entangled:  True
# Probabilities for C:  0.50 [+1] + 0.50 [-1]  is entangled:  False
#
# Question 3:
# After H gate on each qubit.
# 0.707 |00⟩ + 0.707 |11⟩ with prob = 0.50+0.00j
# 0.707 |01⟩ + 0.707 |10⟩ with prob = 0.50+0.00j
#
# Question 4:
# IX(0.707 |00⟩ + -0.707 |11⟩) = 0.707 |01⟩ + -0.707 |10⟩ = |O-⟩
# IY(0.707 |00⟩ + -0.707 |11⟩) = 0.707i |01⟩ + 0.707i |10⟩ = i|O+⟩
# XY(0.707 |00⟩ + -0.707 |11⟩) = 0.707i |00⟩ + 0.707i |11⟩ = i|E+⟩

import numpy as np

from QC.QuantumComputer import QuantumComputer
from QC.checks import is_pure
from QC.gates import Z_GATE, ZERO, ONE
from QC.helper_func import state_to_density_matrix, matmul
from QC.throw_or_trace import density_matrix_of
from QC.to_notation import to_ket_notation

ZZ_GATE: np.ndarray = np.kron(Z_GATE, Z_GATE)

measurement_plus_one = (ZZ_GATE == 1).astype(int)
measurement_minus_one = (ZZ_GATE == -1).astype(int)


def zz_parity_measurement(state):
    probability_plus_one = matmul(state.conj().T, measurement_plus_one.conj().T, measurement_plus_one, state)[0][0]
    probability_minus_one = matmul(state.conj().T, measurement_minus_one.conj().T, measurement_minus_one, state)[0][0]

    new_state_plus_one = matmul(measurement_plus_one, state) / np.sqrt(probability_plus_one) if probability_plus_one != 0 else None
    new_state_minus_one = matmul(measurement_minus_one, state) / np.sqrt(probability_minus_one) if probability_minus_one != 0 else None

    probabilities = (probability_plus_one, probability_minus_one)
    new_states = (new_state_plus_one, new_state_minus_one)

    return probabilities, new_states


def question1():
    to_mat = lambda vector: np.matmul(vector, vector.conj().T)

    zz = np.kron(ZERO, ZERO)
    zo = np.kron(ZERO, ONE)
    oz = np.kron(ONE, ZERO)
    oo = np.kron(ONE, ONE)

    gate = to_mat(zz) + to_mat(oo) - to_mat(zo) - to_mat(oz)

    print()
    print("Question 1: ")
    print(ZZ_GATE)
    print("Is this the ZZ parity gate: ", np.isclose(gate, ZZ_GATE).all())  # True


def question2():
    a = np.kron(ZERO, ONE)
    b = 1 / np.sqrt(2) * (np.kron(ZERO, ZERO) + np.kron(ONE, ONE))
    c = 1 / 2 * (np.kron(ZERO, ZERO) - np.kron(ZERO, ONE) + np.kron(ONE, ZERO) - np.kron(ONE, ONE))

    pa, _ = zz_parity_measurement(a)
    pb, _ = zz_parity_measurement(b)
    pc, _ = zz_parity_measurement(c)

    is_entangled_a = not is_pure(density_matrix_of(state_to_density_matrix(a), 1))
    is_entangled_b = not is_pure(density_matrix_of(state_to_density_matrix(b), 1))
    is_entangled_c = not is_pure(density_matrix_of(state_to_density_matrix(c), 1))

    print()
    print("Question 2: ")
    print("Probabilities for A: ", f"{pa[0]:.2f} [+1] + {pa[1]:.2f} [-1]", " is entangled: ", is_entangled_a)  # False
    print("Probabilities for B: ", f"{pb[0]:.2f} [+1] + {pb[1]:.2f} [-1]", " is entangled: ", is_entangled_b)  # True
    print("Probabilities for C: ", f"{pc[0]:.2f} [+1] + {pc[1]:.2f} [-1]", " is entangled: ", is_entangled_c)  # False


def question3():
    qc = QuantumComputer(2)
    qc.H(0)
    qc.H(1)
    probabilities, new_states = zz_parity_measurement(qc.state)
    print()
    print("Question 3: ")
    print("After H gate on each qubit.")
    print(f"{to_ket_notation(new_states[0])} with prob = {probabilities[0]:.2f}")
    print(f"{to_ket_notation(new_states[1])} with prob = {probabilities[1]:.2f}")


def question4():
    state = 1 / np.sqrt(2) * (np.kron(ZERO, ZERO) - np.kron(ONE, ONE))
    ix = QuantumComputer(state).I(0).X(1).ket_notation()
    iy = QuantumComputer(state).I(0).Y(1).ket_notation()
    xy = QuantumComputer(state).X(0).Y(1).ket_notation()

    print()
    print("Question 4: ")
    print(f"IX({to_ket_notation(state)}) = {ix} = |O-⟩")
    print(f"IY({to_ket_notation(state)}) = {iy} = i|O+⟩")
    print(f"XY({to_ket_notation(state)}) = {xy} = i|E+⟩")


if __name__ == '__main__':
    question1()
    question2()
    question3()
    question4()
