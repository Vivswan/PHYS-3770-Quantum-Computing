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
