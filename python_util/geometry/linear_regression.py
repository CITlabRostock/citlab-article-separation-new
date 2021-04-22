# -*- coding: utf-8 -*-

import numpy as np


def calc_line(x_points, y_points):
    assert type(x_points) == list and type(y_points) == list, "x_points and y_points have to be lists"
    assert all([type(x) == int for x in x_points]), "elements of x_points have to be ints"
    assert all([type(y) == int for y in y_points]), "elements of y_points have to be ints"
    assert len(x_points) == len(y_points), "same x_points- and y_points-list length required"
    n_points = len(x_points)

    min_x = 100000
    max_x = 0
    sum_x = 0.0

    a = np.zeros([n_points, 2])
    y = np.zeros([n_points])

    for i in range(n_points):
        y[i] = y_points[i]

        px = x_points[i]
        a[i, 0] = 1.0
        a[i, 1] = px

        min_x = min(px, min_x)
        max_x = max(px, max_x)
        sum_x += px
    if max_x - min_x < 2:
        return sum_x / len(x_points), float("inf")

    return solve_lin(a, y)


def solve_lin(a, y):
    assert isinstance(a, np.ndarray), "a has to np.ndarray"
    assert isinstance(y, np.ndarray), "y has to np.ndarray"

    a_t = np.transpose(a)
    ls = np.matmul(a_t, a)
    rs = np.matmul(a_t, y)
    assert ls.shape == (2, 2)

    det = ls[0, 0] * ls[1, 1] - ls[0, 1] * ls[1, 0]
    if det < 1e-9:
        print("LinearRegression Error: Numerically unstable.")
        return a[0, 1], float("inf")
    else:
        d = 1.0 / det
        inv = np.empty_like(ls)
        inv[0, 0] = d * ls[1, 1]
        inv[1, 1] = d * ls[0, 0]
        inv[1, 0] = -d * ls[1, 0]
        inv[0, 1] = -d * ls[0, 1]

    return np.matmul(inv, rs)

# if __name__ == '__main__':
#     from scipy.stats import linregress
#
#     # (1,30) (10,25) (25, 22) (60,36)
#     xs = [1, 10, 25, 60]
#     ys = [30, 25, 22, 36]
#
#     # # (1,10) (2, 20) (3,15)
#     # x_points = [1, 2, 3]
#     # y_points = [10, 20, 15]
#
#     res = calc_line(xs, ys)
#     res2 = linregress(xs, ys)
#
#     print("res = \n", res)
#     print("res2 = \n", [res[0], res[1]])
