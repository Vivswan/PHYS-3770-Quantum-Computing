import numpy as np


def complex_to_str(value, floating_point=3, with_zero=False):
    notation = ""

    is_state_real = not np.isclose(complex(value).real, 0)
    is_state_imag = not np.isclose(complex(value).imag, 0)
    if is_state_real:
        notation += f"{complex(value).real:.{floating_point}f}"
    if is_state_real and is_state_imag:
        notation += " + "
    if is_state_imag:
        notation += f"{complex(value).imag:.{floating_point}f}i"
    if with_zero and not is_state_real and not is_state_imag:
        notation += f"{0:.{floating_point}f}"

    return notation


def to_ket_notation(state, probabilities=False):
    state = np.atleast_2d(state)
    max_state = int(np.log2(np.shape(state)[0]))
    notation = ""

    for index, state in enumerate(state):
        state = state[0]
        if not np.isclose(state, 0):
            if len(notation) > 0:
                notation += " + "

            n_state = "{0:b}".format(index).zfill(max_state)
            if probabilities:
                notation += f"{np.power(np.abs(state), 2):.3f} |{n_state}⟩"
            else:
                notation += f"{complex_to_str(state)} |{n_state}⟩"

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

                bra_ket_notation += f"{complex_to_str(density_matrix[row_index][col_index])} |{row}⟩⟨{col}|"

    return bra_ket_notation
