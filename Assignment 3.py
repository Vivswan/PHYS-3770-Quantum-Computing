import math

import numpy as np


def is_physical(pho):
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


def trace_of(density_matrix, remove_qubit):
    num_qubits = int(math.log2(np.shape(density_matrix)[0]))
    new_matrix = np.zeros((2 ** (num_qubits - 1), 2 ** (num_qubits - 1))).tolist()
    notation_array = []
    for i in range(0, 2 ** num_qubits):
        notation_array.append("{0:b}".format(i).zfill(num_qubits))
    for row_index, row in enumerate(notation_array):
        for col_index, col in enumerate(notation_array):
            if row[remove_qubit] == col[remove_qubit]:
                new_row = row[0:remove_qubit] + row[remove_qubit + 1:]
                new_col = col[0:remove_qubit] + col[remove_qubit + 1:]
                new_row_num = int(new_row, 2)
                new_col_num = int(new_col, 2)
                # print(f"{row},{col} -> {new_row},{new_col} -> {density_matrix[row_index][col_index]}")
                new_matrix[new_row_num][new_col_num] += density_matrix[row_index][col_index]

    return np.array(new_matrix)


def main():
    zero = np.array([
        [1],
        [0]
    ])
    one = np.array([
        [0],
        [1]
    ])

    # zz = kron(zero, zero)
    # zo = kron(zero, one)
    # oz = kron(one, zero)
    # oo = kron(one, one)
    #
    # E_plus = np.array(1 / np.sqrt(2) * (zz + oo))
    # E_minus = np.array(1 / np.sqrt(2) * (zz - oo))
    # O_plus = np.array(1 / np.sqrt(2) * (zo + oz))
    # O_minus = np.array(1 / np.sqrt(2) * (zo - oz))
    #
    # E_plus_density = np.matmul(E_plus, E_plus.conjugate().transpose())
    # E_minus_density = np.matmul(E_minus, E_minus.conjugate().transpose())
    # O_plus_density = np.matmul(O_plus, O_plus.conjugate().transpose())
    # O_minus_density = np.matmul(O_minus, O_minus.conjugate().transpose())
    #
    # for i in [E_plus_density, E_minus_density, O_plus_density, O_minus_density]:
    #     for j in [0, 1]:
    #         new_matrix = trace_of(i, j)
    #         print(is_physical(new_matrix), purity(new_matrix))

    zzo = kron(zero, zero, one)
    ozz = kron(one, zero, zero)
    zoo = kron(zero, one, one)
    ooz = kron(one, one, zero)

    phi = np.array((zzo - 1j * ozz - zoo + 1j * ooz) / math.sqrt(4))
    # print(phi)
    # print()
    phi_density_matrix = np.matmul(phi, phi.conjugate().transpose())
    # print(phi_density_matrix)
    # print()
    for i in [0, 1, 2]:
        after_trace = trace_of(phi_density_matrix, i)
        xx = [0, 1, 2]
        xx.remove(i)
        print()
        print(xx, abs(is_physical(after_trace)), abs(purity(after_trace)))
        print(after_trace)

        # for j in [0, 1]:
        #     after_trace = trace_of(trace_of(phi_density_matrix, i), j)
        #     xx = [0, 1, 2]
        #     xx.remove(i)
        #     xx.remove(xx[j])
        #     # print()
        #     print(xx, abs(is_physical(after_trace)), abs(purity(after_trace)))
        #     # print(after_trace)


if __name__ == '__main__':
    main()
