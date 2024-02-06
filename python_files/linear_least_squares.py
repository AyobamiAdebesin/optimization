#!/usr/bin/python3
""" Linear Least Squares using Fixed Step Gradient Descent """
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy.linalg import eig, LinAlgError
from typing import Tuple, Sequence, List, Union
from fixed_step_gda import gradient_descent


def linear_least_squares(data_points: Sequence[Tuple]) -> Union[np.ndarray, List]:
    """ Approximate a set of data points with a linear function """
    np.random.seed(0)
    nrows = len(data_points)
    ncols = len(data_points[0])
    A = np.ones((nrows, ncols))
    B = np.zeros((nrows, 1))
    for i in range(len(A)):
        A[i, 0] = data_points[i][0]
        B[i, 0] = data_points[i][1]
    Q, b, c = 2*np.dot(A.T, A), 2*np.dot(A.T, B), np.dot(B.T, B)
    try:
        eig_val, _ = eig(Q)
    except LinAlgError as e:
        print(
            f"Error calculating eigenvectors and eigenvalues of A::::: Error message: {err}")
    alpha = np.random.uniform(0, 2/max(eig_val))
    try:
        x_min, values = gradient_descent(Q, b, c, alpha)
    except Exception as err:
        print(
            f"Error while calculating gradient descent::::Error message: {err}")
    return np.squeeze(x_min), values


def plot_points(data_points, m, c, values):
    # Number of points
    n = 100
    x = np.zeros((len(data_points),))
    y = np.zeros((len(data_points),))
    for i in range(len(data_points)):
        x[i] = (data_points[i][0])
        y[i] = (data_points[i][1])

    # Plotting the random points
    plt.scatter(x, y, label='Random Points')

    # Plotting the original straight line
    plt.plot(x, m * x + c, color='red', label='Straight Line')

    # Adding title and labels
    plt.title('Random Points and Straight Line')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')

    # Adding a legend
    plt.legend()

    # Showing the plot
    plt.show()

    # plot function against iterations
    plt.plot(values)
    plt.xlabel('Iteration')
    plt.ylabel('Objective function value')
    plt.show()
    return
