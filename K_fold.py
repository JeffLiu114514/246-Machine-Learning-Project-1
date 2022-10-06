# Name: Junfei Liu
# Email: jliu137@u.rochester.edu

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
    return [Erms, w]  # np.array([[Erms], [test_x], [test_t], [y]])


def shuffle_partition(x, t, k):
    twoD_array = np.vstack((x, t)).T
    np.random.shuffle(twoD_array)
    shuffled_array = np.transpose(twoD_array)
    shuffled_x, shuffled_t = shuffled_array[0], shuffled_array[1]

    # shuffled_x, shuffled_t = x, t
    size = math.ceil(x.size / k)
    klist_x = np.array([shuffled_x[i:i + size] for i in range(0, len(x), size)])
    klist_t = np.array([shuffled_t[i:i + size] for i in range(0, len(t), size)])
    return klist_x, klist_t


def write_output(erms, m, w, gamma, output_path):
    if output_path != "nopath":
        np.savetxt(output_path, w, header="m = %d\ngamma = %f\nERMS = %f"%(m, gamma, erms))


def k_fold_givenm(x, t, gamma, k, m, output_path):
    klist_x, klist_t = shuffle_partition(x, t, k)
    erms_list_k = []
    w_list_k = []
    for i in range(k):
        test_x, test_t = np.array(klist_x[i]), np.array(klist_t[i])  # testing set
        train_x, train_t = np.empty(0), np.empty(0)
        train_list = [a for a in range(k) if a != i]
        for j in train_list:  # training set
            train_x = np.concatenate((train_x, klist_x[j]), axis = 0)
            train_t = np.concatenate((train_t, klist_t[j]), axis = 0)
        result = fit(train_x, train_t, test_x, test_t, gamma, m + 1)
        erms_list_k.append(result[0])
        w_list_k.append(result[1])
    mean_w, mean_erms = sum(w_list_k) / k, sum(erms_list_k) / k
    print("The result of given m ", m, "is with ERMS ", mean_erms, " with weights", mean_w, "\n")
    write_output(mean_erms, m, mean_w, gamma, output_path)

def k_fold_autofit(x, t, gamma, k, max_m, output_path):
    klist_x, klist_t = shuffle_partition(x, t, k)

    erms_list_m = []
    w_list_m = []
    for m in range(max_m + 1):  # loop m
        erms_list_k = []  # list of errors in k iterations with same m
        w_list_k = []
        for i in range(k):
            test_x, test_t = np.array(klist_x[i]), np.array(klist_t[i])  # testing set
            train_x, train_t = np.empty(0), np.empty(0)
            train_list = [a for a in range(k) if a != i]
            for j in train_list:  # training set
                train_x = np.concatenate((train_x, klist_x[j]), axis = 0)
                train_t = np.concatenate((train_t, klist_t[j]), axis = 0)
            result = fit(train_x, train_t, test_x, test_t, gamma, m + 1)
            erms_list_k.append(result[0])
            w_list_k.append(result[1])
            # result_list_k.append(result)
        mean_w, mean_erms = sum(w_list_k) / k, sum(erms_list_k) / k
        w_list_m.append(mean_w)
        erms_list_m.append(mean_erms)
        # result_this_m = result_list_k[0]
        # for i in result_list_k[1:]:
        #     result_this_m = np.hstack((result_this_m, i))
        # result_list_m.append(result_this_m)
        print("Now is at degree ", m,
              "\nAverage root mean square error is ", mean_erms,
              "\nAverage weight is ", mean_w, "\n")
    best_m = erms_list_m.index(min(erms_list_m))
    print("The best m is ", best_m, " with ERMS ", erms_list_m[best_m], " with weights", w_list_m[best_m], "\n")
    # plot_result(result_list_m[best_m][1], result_list_m[best_m][2], result_list_m[best_m][3])
    write_output(erms_list_m[best_m], best_m, w_list_m[best_m], gamma, output_path)
