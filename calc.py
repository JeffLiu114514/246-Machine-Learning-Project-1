import math

import matplotlib.pyplot as plt
import numpy as np
from util import *


def design_matrix(x, m):
    # matrix = np.power(x, 0)
    # for i in range(1, m):
    #     np.column_stack((matrix, np.power(x, i)))
    return np.column_stack([np.power(x, i) for i in range(m)])


def calculate_w(m, gamma, d_matrix, t):
    d_matrix_trans = np.transpose(d_matrix)
    return np.linalg.inv(gamma * np.identity(m) + d_matrix_trans.dot(d_matrix)).dot(d_matrix_trans).dot(t)


def phi(xi, m):
    return [math.pow(xi, i) for i in range(m)]


def predict_y(w, x, m, n):
    y = np.empty([n], dtype = float)
    for i in range(n):
        phi_xi = phi(x[i], m)
        dot = np.dot(np.transpose(w), phi_xi)
        y[i] = dot
    return y  # np.array(np.dot(np.transpose(w), phi(x[i], m)) for i in range(n))


def get_Erms(y, t, n):
    sum = 0
    for i in range(n):
        sum += math.pow((y[i] - t[i]), 2)
    return math.sqrt(sum / n)


def plot_result(x, t, y):
    plt.scatter(x, t)
    plt.plot(x, y, c = "red")
    plt.show()


def auto_fit(x, t, gamma):
    result_list = []
    error_list = []
    n = x.size
    for a in range(20):
        m = a + 1
        d_matrix = design_matrix(x, m)
        w = calculate_w(m, gamma, d_matrix, t)
        y = predict_y(w, x, m, n)
        Erms = get_Erms(y, t, n)

        error_list.append(Erms)
        result_list.append([Erms, a, w, y])
    best_m = error_list.index(min(error_list))
    print("Best polynomial is of degree ", best_m,
          "\nRoot mean square error is ", result_list[best_m][0],
          "\nParameters are ", result_list[best_m][2], "\n")
    plot_result(x, t, result_list[best_m][3])


def m_fit(x, t, a, gamma):
    m = a + 1
    n = x.size
    d_matrix = design_matrix(x, m)
    w = calculate_w(m, gamma, d_matrix, t)
    y = predict_y(w, x, m, n)
    Erms = get_Erms(y, t, n)

    print("Fit ", a, " degree polynomial:",
          "\nRoot mean square error is ", Erms,
          "\nParameters are ", w, "\n")
    plot_result(x, t, y)


def main():
    x, t = read_input("levelOne/A")
    gamma = 0.0000001
    #auto_fit(x, t, gamma)
    m_fit(x, t, 1, gamma)


if __name__ == "__main__":
    main()
