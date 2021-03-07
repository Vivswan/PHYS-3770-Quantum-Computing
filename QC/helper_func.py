import numpy as np


def apply_to_list(list, func):
    c = None
    for i in list:
        if c is None:
            c = i
        else:
            c = func(c, i)
    return c


def kron(*args):
    return apply_to_list(args, np.kron)


def matmul(*args):
    return apply_to_list(args, np.matmul)


def nullspace(matrix, atol=1e-13, rtol=0):
    matrix = np.atleast_2d(matrix)
    u, s, vh = np.linalg.svd(matrix)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


def zero_state_matrix(number_of_qubits, dim=1):
    shape = [np.power(number_of_qubits, 2)] * dim
    if dim > 1:
        shape = tuple(shape)
    else:
        shape = (shape[0], 1)

    state = np.zeros(shape)
    state[0][0] = 1
    return state
