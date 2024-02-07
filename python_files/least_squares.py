#!/usr/bin/python3
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy.linalg import eig, LinAlgError
from typing import Tuple, Sequence, List, Union
from conjugate_gradient_descent import conjugate_gradient
from steepest_descent import steepest_descent


def least_squares(data_points: Sequence[Tuple], method="steepest_descent") -> Tuple:
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
    if method == "steepest_descent":
        try:
            x_min = steepest_descent(Q, b, c,)
        except Exception as err:
            print(
                f"Error while calculating gradient descent::::Error message: {err}")
    elif method == "conjugate_gradient":
        try:
            x_min = conjugate_gradient(Q, b, c,)
        except Exception as err:
            print(
                f"Error while calculating gradient descent::::Error message: {err}")
    elif method == None:
        raise ValueError("Provide an optimization method!")
    vec = np.squeeze(x_min)
    alpha = vec[0]
    C = np.exp(vec[1])
    return (alpha, C)

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
    # Plotting the original exponential curve
    plt.plot(x, C* np.exp(alpha*x), color='red', label='Exponential Curve')

    # Adding title and labels
    plt.title('Random Points and Exponential Curve')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')

    # Adding a legend
    plt.legend()

    # Showing the plot
    plt.show()
    return

