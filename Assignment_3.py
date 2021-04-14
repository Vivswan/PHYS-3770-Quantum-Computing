# Source Code: https://github.com/Vivswan/PHYS-3770-Quantum-Information
#
# Output:
# Question 1:
# Trace of 0 on E_plus_density => phi[1] = 0.500 |0⟩⟨0| + 0.500 |1⟩⟨1| (Physical: 1.0, Purity: 0.5)
# Trace of 1 on E_plus_density => phi[0] = 0.500 |0⟩⟨0| + 0.500 |1⟩⟨1| (Physical: 1.0, Purity: 0.5)
# Trace of 0 on E_minus_density => phi[1] = 0.500 |0⟩⟨0| + 0.500 |1⟩⟨1| (Physical: 1.0, Purity: 0.5)
# Trace of 1 on E_minus_density => phi[0] = 0.500 |0⟩⟨0| + 0.500 |1⟩⟨1| (Physical: 1.0, Purity: 0.5)
# Trace of 0 on O_plus_density => phi[1] = 0.500 |0⟩⟨0| + 0.500 |1⟩⟨1| (Physical: 1.0, Purity: 0.5)
# Trace of 1 on O_plus_density => phi[0] = 0.500 |0⟩⟨0| + 0.500 |1⟩⟨1| (Physical: 1.0, Purity: 0.5)
# Trace of 0 on O_minus_density => phi[1] = 0.500 |0⟩⟨0| + 0.500 |1⟩⟨1| (Physical: 1.0, Purity: 0.5)
# Trace of 1 on O_minus_density => phi[0] = 0.500 |0⟩⟨0| + 0.500 |1⟩⟨1| (Physical: 1.0, Purity: 0.5)
#
# Question 2:
#
# Trace of 0 => phi[1, 2] = 0.167 |00⟩⟨00| + -0.236 |00⟩⟨10| + 0.167 |01⟩⟨01| + -0.236 |01⟩⟨11| + -0.236 |10⟩⟨00| + 0.333 |10⟩⟨10| + -0.236 |11⟩⟨01| + 0.333 |11⟩⟨11|
# Physical: 1.0, Purity: 0.5
# Qubit[1, 2] are not fully entangled
#
# Trace of 1 => phi[0, 2] = 0.500 |01⟩⟨01| + 0.500i |01⟩⟨10| + -0.500i |10⟩⟨01| + 0.500 |10⟩⟨10|
# Physical: 1.0, Purity: 1.0
# Qubit[0, 2] are fully entangled
#
# Trace of 2 => phi[0, 1] = 0.167 |00⟩⟨00| + -0.236 |00⟩⟨01| + -0.236 |01⟩⟨00| + 0.333 |01⟩⟨01| + 0.167 |10⟩⟨10| + -0.236 |10⟩⟨11| + -0.236 |11⟩⟨10| + 0.333 |11⟩⟨11|
# Physical: 1.0, Purity: 0.5
# Qubit[0, 1] are not fully entangled
#
# Question 3:
# H: [[0.7071067811865475, 0.7071067811865475], [0.7071067811865475, -0.7071067811865475]]
# Eigenvector for 1.0: [[0.9238795325112867], [0.3826834323650897]]
# Eigenvector for -1.0: [[-0.3826834323650897], [0.9238795325112867]]
# Completeness Relation:  True
# Eigenvector test for 0:  True
# Eigenvector test for 1:  True
# M_0: [[0.85355339 0.35355339]
#  [0.35355339 0.14644661]]
# M_1: [[ 0.14644661 -0.35355339]
#  [-0.35355339  0.85355339]]

import numpy as np

from QC.QuantumComputer import ZERO, ONE, H_GATE
from QC.throw_or_trace import trace_or_throw
from QC.to_notation import to_ket_bra_notation


def physicality(pho):
    return np.trace(pho)


def purity(pho):
    return np.trace(np.matmul(pho, pho))


def kron(*args):
    c = None
    for i in args:
        if c is None:
            c = i
        else:
            c = np.kron(c, i)
    return c


def question1():
    print()
    print('Question 1:')

    zz = kron(ZERO, ZERO)
    zo = kron(ZERO, ONE)
    oz = kron(ONE, ZERO)
    oo = kron(ONE, ONE)

    e_plus = np.array(1 / np.sqrt(2) * (zz + oo))
    e_minus = np.array(1 / np.sqrt(2) * (zz - oo))
    o_plus = np.array(1 / np.sqrt(2) * (zo + oz))
    o_minus = np.array(1 / np.sqrt(2) * (zo - oz))

    density_matrices = {
        'E_plus_density': np.matmul(e_plus, e_plus.conjugate().transpose()),
        'E_minus_density': np.matmul(e_minus, e_minus.conjugate().transpose()),
        'O_plus_density': np.matmul(o_plus, o_plus.conjugate().transpose()),
        'O_minus_density': np.matmul(o_minus, o_minus.conjugate().transpose())
    }

    for i in density_matrices:
        for j in [0, 1]:
            new_matrix = trace_or_throw(density_matrices[i], j)
            print(f'Trace of {j} on {i} => phi[{1 if j == 0 else 0}] = {to_ket_bra_notation(new_matrix)}'
                  f' (Physical: {abs(np.round(physicality(new_matrix), 3))},'
                  f' Purity: {abs(np.round(purity(new_matrix), 3))})')


def question2():
    print()
    print('Question 2:')

    zzo = kron(ZERO, ZERO, ONE)
    ozz = kron(ONE, ZERO, ZERO)
    zoo = kron(ZERO, ONE, ONE)
    ooz = kron(ONE, ONE, ZERO)

    phi = np.array(
        (np.sqrt(1 / 6) * zzo - 1j * np.sqrt(1 / 6) * ozz - np.sqrt(2 / 6) * zoo + 1j * np.sqrt(2 / 6) * ooz))

    phi_density_matrix = np.matmul(phi, phi.conjugate().transpose())

    for i in [0, 1, 2]:
        after_trace = trace_or_throw(phi_density_matrix, i)
        xx = [0, 1, 2]
        xx.remove(i)
        print()
        print(f"Trace of {i} => phi{xx} = {to_ket_bra_notation(after_trace)}")
        print(f"Physical: {abs(np.round(physicality(after_trace), 3))}, "
              f"Purity: {abs(np.round(purity(after_trace), 3))}")
        if abs(purity(after_trace)) == 1:
            print(f"Qubit{xx} are fully entangled")
        else:
            print(f"Qubit{xx} are not fully entangled")


def question3():
    print()
    print("Question 3:")
    eigenvalues, eigenvectors = np.linalg.eig(H_GATE)
    print("H:", H_GATE.tolist())
    eigenvectors_list = [eigenvectors[:, 0].reshape(2, 1), eigenvectors[:, 1].reshape(2, 1)]
    m_h0 = np.matmul(eigenvectors_list[0], eigenvectors_list[0].conjugate().transpose())
    m_h1 = np.matmul(eigenvectors_list[1], eigenvectors_list[1].conjugate().transpose())
    print(f"Eigenvector for {eigenvalues[0]}:", eigenvectors_list[0].tolist())
    print(f"Eigenvector for {np.round(eigenvalues[1], 1)}:", eigenvectors_list[1].tolist())
    print("Completeness Relation: ", np.all(
        np.matmul(m_h0.conjugate().transpose(), m_h0) + np.matmul(m_h1.conjugate().transpose(), m_h1) == np.identity(
            2)))
    print("Eigenvector test for 0: ", np.all(np.matmul(m_h0, eigenvectors_list[0]) == eigenvectors_list[0]))
    print("Eigenvector test for 1: ", np.all(np.matmul(m_h1, eigenvectors_list[1]) == eigenvectors_list[1]))
    print("M_0:", m_h0)
    print("M_1:", m_h1)


def main():
    question1()
    question2()
    question3()


if __name__ == '__main__':
    main()
