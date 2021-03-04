import numpy as np


def is_unitary(m):
    m = np.array(m)
    return np.all(np.isclose(m.conjugate().transpose().dot(m), np.identity(m.shape[0])))


def is_hermitian(m):
    m = np.array(m)
    return np.all(np.isclose(m.conjugate().transpose(), m))


def physicality(density_matrix):
    return np.trace(density_matrix)


def purity(density_matrix):
    return np.trace(np.matmul(density_matrix, density_matrix))


def is_physical(density_matrix):
    return np.isclose(physicality(density_matrix), 1)


def is_pure(density_matrix):
    return np.isclose(purity(density_matrix), 1)
