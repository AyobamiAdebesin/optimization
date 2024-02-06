#!/usr/bin/python3
import numpy as np


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
