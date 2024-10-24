import torch
import numpy as np


def DCG_at_K(sorted_list, teacher_t_items, user, K, lam=10.0):
    """Calculation of DCG@K.
    DCG@K(pi) = sum_k [(2^y - 1)/(log(k+1))]

    Args:
        sorted_list (_type_): _description_
        teacher_t_items (_type_): _description_
        user (_type_): _description_
        K (int): top K
        lam (float, optional): lambda in calculation of y_i. \
            The hyperparameter that controls the sharpness of the distribution. \
            Defaults to 10.
    """
    # y_i = exp(-rank(pi, i) / lambda)
    y = np.asarray([np.exp(-t / lam) for t in range(1, K + 1)])
    unit = np.asarray([((2**y - 1) / (np.log(k + 1))) for k in range(1, K + 1)])
    return np.sum(unit)
