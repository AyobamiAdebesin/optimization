#!/usr/bin/python3
""" Compute the exact minimizer for a quadratic function """
import numpy as np
from numpy.linalg import inv
from numpy.linalg import LinAlgError


def exact_minimizer(Q: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Returns the exact minimizer of a linear equation
    Qx = b
    """
    if Q.ndim != 2 or Q.shape != (2, 2):
        raise ValueError("Q must be a 2x2 matrix")
    if b.ndim != 2 or b.shape != (2, 1):
        raise ValueError("b must be a 2x1 matrix")
    try:
        x = np.dot(inv(Q), b)
    except LinAlgError as err:
        print(f"Unable to find inverse of Q:::::: Error message: {err}")
        return
    return x
