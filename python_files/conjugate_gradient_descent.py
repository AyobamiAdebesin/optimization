#!/usr/bin/python3
""" Conjugate Gradient Descent """
import numpy as np
import os
import shutil


def conjugate_gradient(Q: np.ndarray, b: np.ndarray, c: float) -> None:
    """
    Compute the minimizer of a quadratic
    function using conjugate gradient descent algorithm.

    Q must be symmetric and positive definite
    """
    if Q.ndim != 2 or Q.shape != (2, 2):
        raise ValueError("Q must be a 2x2 matrix")
    if b.ndim != 2 or b.shape != (2, 1):
        raise ValueError("b must be a 2x1 matrix")

    x_0 = np.zeros((2, 1))
    g_0 = np.dot(Q, x_0) - b
    try:
        alpha_0 = np.dot(g_0.T, g_0) / np.dot(g_0.T, np.dot(Q, g_0))
    except ZeroDivisionError:
        print("Division by zero occured:::: Check that")
    d_0 = -g_0
    x_1 = x_0 + (alpha_0 * d_0)

    for _ in range(len(Q)):
        cur_pos = x_1
        cur_grad = np.dot(Q, cur_pos) - b
        prev_dir = d_0
        prev_beta = np.dot(cur_grad.T, np.dot(Q, prev_dir)) / \
            np.dot(prev_dir.T, np.dot(Q, prev_dir))
        d_0 = -cur_grad + (prev_beta*prev_dir)
        alpha_k = -1 * np.dot(cur_grad.T, d_0) / \
            np.dot(d_0.T, np.dot(Q, d_0))
        x_1 = cur_pos + (alpha_k * d_0)
    return x_1
