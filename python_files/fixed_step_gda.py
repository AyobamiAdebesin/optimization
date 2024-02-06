#!/usr/bin/python3
""" Fixed step Gradient Descent Algorithm """
import os
from typing import List, Union
import numpy as np
from numpy.linalg import inv, norm
from numpy.linalg import LinAlgError


def gradient_descent(Q: np.ndarray, b: np.ndarray, c: float, alpha=None) -> Union[np.ndarray, List]:
    """
    Compute the minimizer of a quadratic function using fixed step gradient descent
    Assume that Q is Hermitian and positive definite.

    The function f that represents Q, b and c is 2-D
    f : R^2 -> R
    """
    if Q.ndim != 2 or Q.shape != (2, 2):
        raise ValueError("Q must be a 2x2 matrix")
    if b.ndim != 2 or b.shape != (2, 1):
        raise ValueError("b must be a 2x1 matrix")
    x_0 = np.zeros((2, 1))
    x_min = np.zeros((2, 1))
    values = []
    if not alpha:
        alpha = 1.6
    print(f"::::::Fixed step Gradient Descent Algorithm::::::::")
    print(f"Here is my learning rate: {alpha}")

    for i in range(20):
        x_0 = x_min
        x_min = x_0 - alpha*(np.dot(Q, x_0) - b)
        # compute the value of the objective function
        value = np.squeeze(
            0.5 * np.dot(x_min.T, np.dot(Q, x_min)) - np.dot(b.T, x_min) + c)
        values.append(value)
        if norm((x_min - x_0), 2) < 1e-4:
            break
        # print(f"Error difference at step {i}: {norm((x_1 - x_0), 2)}")
    print(f"Approximate minimizer: {x_min}")
    try:
        x = np.dot(inv(Q), b)
    except LinAlgError as err:
        print(f"Unable to find inverse of Q:::::: Error message: {err}")
        return
    print(f"Exact minimizer: {x}")
    return x_min, values
