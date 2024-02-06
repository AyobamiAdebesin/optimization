#!/usr/bin/python3
import matplotlib.pyplot as plt
import os
import numpy as np
import math
from typing import Tuple, Sequence, List
from numpy.linalg import eig, inv, norm, LinAlgError


def steepest_descent(Q: np.ndarray, b: np.ndarray, c: float) -> np.ndarray:
    """
    Compute the minimizer using stepeest gradient descent
    Assume that Q is Hermitian and positive definite.

    The function f that represents Q, b and c is 2-D
    f : R^2 -> R
    """
    if Q.ndim != 2 or Q.shape != (2, 2):
        raise ValueError("Q must be a 2x2 matrix")
    if b.ndim != 2 or b.shape != (2, 1):
        raise ValueError("b must be a 2x1 matrix")
    x_min = np.zeros((2, 1))

    print(f":::::Steepest Gradient Descent Algorithm::::::::")

    for i in range(1, 10):
        x_0 = x_min
        g_k = np.dot(Q, x_0) - b
        try:
            alpha_k = np.dot(g_k.T, g_k) / np.dot(np.dot(g_k.T, Q), g_k)
        except Exception:
            print(f"Unable to compute step size alpha for step {i}")
            return
        alpha_k = float(np.squeeze(alpha_k))
        if not isinstance(alpha_k, float) or math.isnan(alpha_k) == True:
            raise TypeError(
                "step size is not of type float or valid number. Check if correct values are passed")
        print(f"Learning rate at step {i}: {alpha_k}")
        x_min = x_0 - alpha_k*g_k
        if norm((x_min - x_0), 2) < 1e-4:
            break
    print(f"Approximate minimizer: {x_min}")
    try:
        x = np.dot(inv(Q), b)
    except LinAlgError as err:
        print(f"Unable to find inverse of Q:::::: Error message: {err}")
        return
    print(f"Exact minimizer: {x}")
    return x_min
