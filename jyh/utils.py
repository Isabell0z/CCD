import torch
import numpy as np


def DCG_at_K(pi_t, pi, K, lam=10.0):
    """Calculation of DCG@K.
    DCG@K(pi) = sum_k [(2^y - 1)/(log(k+1))]

    Args:
        pi_t (ndarray): permutation of teachers
        pi (ndarray): permutation
        K (int): top K
        lam (float, optional): lambda in calculation of y_i. \
            The hyperparameter that controls the sharpness of the distribution. \
            Defaults to 10.
    """
    # y_i = exp(-rank(pi, i) / lambda)
    s = 0
    for i in range(K):
        yi = 0 if pi[i] not in pi_t else np.exp(-np.where(pi_t == pi[i]) / lam)
        s += (2**yi - 1) / np.log(i + 2)
    return s


def NDCG_at_K(pi_t, pi, K):
    dcg = DCG_at_K(pi_t, pi, K)
    dcg_t = DCG_at_K(pi_t, pi_t, K)
    return dcg / dcg_t


def D_at_K(pi_t, pi, K):
    return 1 - NDCG_at_K(pi_t, pi, K)


def gamma(dx, pi, pi_t):
    """_summary_

    Args:
        x (_type_): teacher's index
    """
    return dx / D_at_K(pi, pi_t)


def generate_pi_d():
    pass


def DKC(pi_t_mat, pi, v, alpha=1.05):
    """_summary_

    Args:
        pi_t_mat (_type_): teachers predictions from early stages to convergenced stage\
                            (M * E * K), M is the number of teachers, E is the max training stage
        pi (_type_): permutation of the student model
        v (_type_): the training state of teachers
        alpha (float): a threshold
    """
    M, E, K = pi_t_mat.shape
    for x in range(M):
        pi_t_next = pi_t_mat[x, v[x] + 1]
        dx = D_at_K(pi_t_next, pi)
        if (v[x] < E) and (gamma(dx, pi, pi_t_next)):
            v[x] = v[x] + 1
    pi_d = generate_pi_d(pi_t_mat, v)
    return pi_d
