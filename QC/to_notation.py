import numpy as np


def to_ket_notation(state, probabilities=False):
    state = np.atleast_2d(state)
    max_state = int(np.log2(np.shape(state)[0]))
    notation = ""

    for index, state in enumerate(state):
        state = state[0]
        if not np.isclose(state, 0):
            if len(notation) > 0:
                notation += " + "

            if probabilities:
                notation += f"{np.power(np.abs(state), 2):.3f}"
            else:
                is_state_real = not np.isclose(complex(state).real, 0)
                is_state_imag = not np.isclose(complex(state).imag, 0)
                if is_state_real:
                    notation += f"{complex(state).real:.3f}"
                if is_state_real and is_state_imag:
                    notation += " + "
                if is_state_imag:
                    notation += f"{complex(state).imag:.3f}i"

            n_state = "{0:b}".format(index).zfill(max_state)
            notation += f" |{n_state}⟩"

    return notation


def to_ket_bra_notation(density_matrix):
    density_matrix = np.atleast_2d(density_matrix)
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
