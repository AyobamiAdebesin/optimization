#!/usr/bin/python3
import matplotlib.pyplot as plt
import os
import numpy as np
import math
from typing import Tuple, Sequence, List, Union
from numpy.linalg import eig, inv, norm
from numpy.linalg import LinAlgError


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
    x_0 = np.zeros((2, 1))
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


def exponential_least_squares(data_points: Sequence[Tuple]) -> Tuple:
    """ Approximate a set of data points with an exponential function """
    np.random.seed(0)
    nrows = len(data_points)
    ncols = len(data_points[0])
    A = np.ones((nrows, ncols))
    B = np.zeros((nrows, 1))
    for i in range(len(A)):
        A[i, 0] = data_points[i][0]
        B[i, 0] = np.log(data_points[i][1])
    Q = 2*np.dot(A.T, A)
    b = 2*np.dot(A.T, B)
    c = np.dot(B.T, B)
    try:
        x_min = steepest_descent(Q, b, c,)
    except Exception as err:
        print(
            f"Error while calculating gradient descent::::Error message: {err}")
    vec = np.squeeze(x_min)
    alpha = vec[0]
    C = np.exp(vec[1])
    return (alpha, C)


def generate_synthetic_data():
    """ Generate synthetic data """
    np.random.seed(0)
    # Parameters for the exponential function
    A = 3       # Amplitude
    alpha = 0.1  # Growth rate

    # Number of points
    n = 100

    # Generate t-values (x-coordinates)
    t = np.linspace(0, 50, n)

    # Compute y-values based on the exponential function
    y = A * np.exp(alpha * t)

    # Add random noise to y-values
    noise = np.random.normal(0, 0.3 * y, n)
    y_noisy = y + noise

    return [(x_val, y_val) for x_val, y_val in zip(t, y_noisy)]


def plot_exp_points(data_points, alpha, C):
    """ Plot the points """
    # Generate x-coordinates
    x = np.zeros((len(data_points),))
    y = np.zeros((len(data_points),))
    for i in range(len(data_points)):
        x[i] = (data_points[i][0])
        y[i] = (data_points[i][1])

    # Plotting the random points
    plt.scatter(x, y, label='Random Points')
    # Plotting the least squares curve
    plt.plot(x, C * np.exp(alpha*x), color='red', label='Exponential Curve')

    # Adding title and labels
    plt.title('Random Points and Exponential Curve')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')

    # Adding a legend
    plt.legend()

    # Showing the plot
    plt.show()
    return


if __name__ == "__main__":
    data_points = generate_synthetic_data()
    alpha, C = exponential_least_squares(data_points)
    print(f"C: {C}, alpha: {alpha}")
    plot_exp_points(data_points, alpha, C)
