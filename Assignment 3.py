import numpy as np


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


def main():
    zero = np.array([
        [1],
        [0]
    ])
    one = np.array([
        [0],
        [1]
    ])

    print('Question 1')
    zz = kron(zero, zero)
    zo = kron(zero, one)
    oz = kron(one, zero)
    oo = kron(one, one)

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
            print()
            print(f"Trace of {j} on {i} => "
                  f"Physical: {abs(np.round(physicality(new_matrix), 1))}, "
                  f"Purity: {abs(np.round(purity(new_matrix), 1))}")
            print(new_matrix.round(2))

    print()
    print('Question 2')
    zzo = kron(zero, zero, one)
    ozz = kron(one, zero, zero)
    zoo = kron(zero, one, one)
    ooz = kron(one, one, zero)

    phi = np.array((zzo - 1j * ozz - zoo + 1j * ooz) / np.sqrt(4))
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
        print(f"Trace of {i} => phi{xx} => "
              f"Physical: {abs(physicality(after_trace))}, "
              f"Purity: {abs(purity(after_trace))}, "
              f"Fully Entangled: {abs(purity(after_trace)) == 1}")
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
