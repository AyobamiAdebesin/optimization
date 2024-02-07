#!/usr/bin/python3
import os
import sys
from linear_least_squares import linear_least_squares, plot_points
from least_squares import least_squares, plot_exp_points
from generate_data import generate_synthetic_data
from conjugate_gradient_descent import conjugate_gradient


data_points = generate_synthetic_data()
alpha, C = least_squares(data_points)
print(f"C: {C}, alpha: {alpha}")
plot_exp_points(data_points, alpha, C)
