# Source Code: https://github.com/Vivswan/PHYS-3770-Quantum-Information
#
# Output:
# Question 1
# Trace of 0 on E_plus_density => phi[1] = 0.5 |0⟩⟨0| + 0.5 |1⟩⟨1| (Physical: 1.0, Purity: 0.5)
# Trace of 1 on E_plus_density => phi[0] = 0.5 |0⟩⟨0| + 0.5 |1⟩⟨1| (Physical: 1.0, Purity: 0.5)
# Trace of 0 on E_minus_density => phi[1] = 0.5 |0⟩⟨0| + 0.5 |1⟩⟨1| (Physical: 1.0, Purity: 0.5)
# Trace of 1 on E_minus_density => phi[0] = 0.5 |0⟩⟨0| + 0.5 |1⟩⟨1| (Physical: 1.0, Purity: 0.5)
# Trace of 0 on O_plus_density => phi[1] = 0.5 |0⟩⟨0| + 0.5 |1⟩⟨1| (Physical: 1.0, Purity: 0.5)
# Trace of 1 on O_plus_density => phi[0] = 0.5 |0⟩⟨0| + 0.5 |1⟩⟨1| (Physical: 1.0, Purity: 0.5)
# Trace of 0 on O_minus_density => phi[1] = 0.5 |0⟩⟨0| + 0.5 |1⟩⟨1| (Physical: 1.0, Purity: 0.5)
# Trace of 1 on O_minus_density => phi[0] = 0.5 |0⟩⟨0| + 0.5 |1⟩⟨1| (Physical: 1.0, Purity: 0.5)
#
# Question 2
#
# Trace of 0 => phi[1, 2] = 0.167 |00⟩⟨00| + -0.236 |00⟩⟨10| + 0.167 |01⟩⟨01| + -0.236 |01⟩⟨11| + -0.236 |10⟩⟨00| + 0.333 |10⟩⟨10| + -0.236 |11⟩⟨01| + 0.333 |11⟩⟨11|
# Physical: 1.0, Purity: 0.5
# Qubit[1, 2] are not fully entangled
#
# Trace of 1 => phi[0, 2] = 0.5 |01⟩⟨01| + 0.5 |01⟩⟨10| + -0.5 |10⟩⟨01| + 0.5 |10⟩⟨10|
# Physical: 1.0, Purity: 1.0
# Qubit[0, 2] are fully entangled
#
# Trace of 2 => phi[0, 1] = 0.167 |00⟩⟨00| + -0.236 |00⟩⟨01| + -0.236 |01⟩⟨00| + 0.333 |01⟩⟨01| + 0.167 |10⟩⟨10| + -0.236 |10⟩⟨11| + -0.236 |11⟩⟨10| + 0.333 |11⟩⟨11|
# Physical: 1.0, Purity: 0.5
# Qubit[0, 1] are not fully entangled


import numpy as np

from QuantumComputer import ZERO, ONE


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


def density_matrix_to_ket_bra(density_matrix):
    num_qubits = int(np.log2(np.shape(density_matrix)[0]))
    bra_ket_notation = ""

    notation_array = []
    for i in range(0, 2 ** num_qubits):
        notation_array.append("{0:b}".format(i).zfill(num_qubits))

    for row_index, row in enumerate(notation_array):
        for col_index, col in enumerate(notation_array):
            if not np.isclose(density_matrix[row_index][col_index], 0):
                if len(bra_ket_notation) > 0:
                    bra_ket_notation += " + "
                if np.isclose(np.imag(density_matrix[row_index][col_index]), 0):
                    value = np.real(density_matrix[row_index][col_index])
                else:
                    value = np.imag(density_matrix[row_index][col_index])

                bra_ket_notation += f"{str(np.round(value, 3)).replace('j', 'i')} |{row}⟩⟨{col}|"

    return bra_ket_notation


def trace_of(density_matrix, remove_qubit):
    num_qubits = int(np.log2(np.shape(density_matrix)[0]))

    if not (0 <= remove_qubit < num_qubits):
        raise Exception(f"Invalid remove qubit: "
                        f"0 <= remove_qubit: {remove_qubit} <= num of qubit in matrix = {num_qubits - 1}")

    new_matrix_shape = (2 ** (num_qubits - 1), 2 ** (num_qubits - 1))
    new_matrix = np.zeros(new_matrix_shape, dtype=complex)
    test_matrix = np.zeros(new_matrix_shape)

    notation_array = []
    for i in range(0, 2 ** num_qubits):
        notation_array.append("{0:b}".format(i).zfill(num_qubits))

    for row_index, row in enumerate(notation_array):
        for col_index, col in enumerate(notation_array):
            if row[remove_qubit] == col[remove_qubit]:
                new_row = int(row[0:remove_qubit] + row[remove_qubit + 1:], 2)
                new_col = int(col[0:remove_qubit] + col[remove_qubit + 1:], 2)
                new_matrix[new_row][new_col] += density_matrix[row_index][col_index]
                test_matrix[new_row][new_col] += 1

    if not np.all(test_matrix == 2):
        raise Exception("Not able to Trace properly")

    return new_matrix


def question1():
    print()
    print('Question 1')

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
            new_matrix = trace_of(density_matrices[i], j)
            print(f'Trace of {j} on {i} => phi[{1 if j == 0 else 0}] = {density_matrix_to_ket_bra(new_matrix)}'
                  f' (Physical: {abs(np.round(physicality(new_matrix), 3))},'
                  f' Purity: {abs(np.round(purity(new_matrix), 3))})')


def question2():
    print()
    print('Question 2')

    zzo = kron(ZERO, ZERO, ONE)
    ozz = kron(ONE, ZERO, ZERO)
    zoo = kron(ZERO, ONE, ONE)
    ooz = kron(ONE, ONE, ZERO)

    phi = np.array(
        (np.sqrt(1 / 6) * zzo - 1j * np.sqrt(1 / 6) * ozz - np.sqrt(2 / 6) * zoo + 1j * np.sqrt(2 / 6) * ooz))

    phi_density_matrix = np.matmul(phi, phi.conjugate().transpose())

    for i in [0, 1, 2]:
        after_trace = trace_of(phi_density_matrix, i)
        xx = [0, 1, 2]
        xx.remove(i)
        print()
        print(f"Trace of {i} => phi{xx} = {density_matrix_to_ket_bra(after_trace)}")
        print(f"Physical: {abs(np.round(physicality(after_trace), 3))}, "
              f"Purity: {abs(np.round(purity(after_trace), 3))}")
        if abs(purity(after_trace)) == 1:
            print(f"Qubit{xx} are fully entangled")
        else:
            print(f"Qubit{xx} are not fully entangled")


def main():
    question1()
    question2()


if __name__ == '__main__':
    main()
