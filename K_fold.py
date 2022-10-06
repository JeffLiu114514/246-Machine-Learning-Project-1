from calc import *
import math
import matplotlib.pyplot as plt
import numpy as np


def fit(train_x, train_t, test_x, test_t, gamma, m):
    n = test_x.size
    # train
    d_matrix = design_matrix(train_x, m)
    w = calculate_w(m, gamma, d_matrix, train_t)
    # test
    y = predict_y(w, test_x, m, n)
    Erms = get_Erms(y, test_t, n)
    return Erms


def k_fold(x, t, gamma, k):
    # twoD_array = np.vstack((x, t)).T
    # np.random.shuffle(twoD_array)
    # shuffled_array = np.transpose(twoD_array)
    # shuffled_x, shuffled_t = shuffled_array[0], shuffled_array[1]

    shuffled_x, shuffled_t = x, t
    size = math.ceil(x.size / k)
    klist_x = np.array([shuffled_x[i:i + size] for i in range(0, len(x), size)])
    klist_t = np.array([shuffled_t[i:i + size] for i in range(0, len(t), size)])

    result_list_m = []
    for m in range(20):  # loop m
        result_list_k = []  # list of errors in k iterations with same m
        for i in range(k):
            test_x, test_t = np.array(klist_x[i]), np.array(klist_t[i])  # testing set
            train_x, train_t = np.empty(0), np.empty(0)
            for j in range(1, k):  # training set
                train_x = np.concatenate((train_x, klist_x[j]), axis = 0)
                train_t = np.concatenate((train_t, klist_t[j]), axis = 0)
            result = fit(train_x, train_t, test_x, test_t, gamma, m + 1)
            result_list_k.append(result)
        mean_error_k = sum(result_list_k) / k
        result_list_m.append(mean_error_k)
        print("Now is at degree ", m,
              "\nAverage root mean square error is ", mean_error_k, "\n")
    best_m = result_list_m.index(min(result_list_m))
    print("The best m is ", best_m, " with ERMS ", result_list_m[best_m], "\n")
