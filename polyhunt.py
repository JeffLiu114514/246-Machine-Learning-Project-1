# Name: Junfei Liu
# Email: jliu137@u.rochester.edu

import argparse
import csv
import numpy as np
from calc import *
from K_fold import *
from distutils.util import strtobool


def read_input(dir):
    x, y = np.empty(0), np.empty(0)
    with open(dir, newline = '\n') as csvfile:
        csvreader = csv.reader(csvfile, delimiter = ',')
        for row in csvreader:
            x = np.append(x, float(row[0]))
            y = np.append(y, float(row[1]))
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type = int, required = True)
    parser.add_argument("--gamma", type = float)
    parser.add_argument("--trainPath", type = str, required = True)
    parser.add_argument("--modelOutput", type = str)
    parser.add_argument("--autofit", type=lambda x: bool(strtobool(str(x))))
    parser.add_argument("--info", action='store_true')
    parser.add_argument("--numFolds", type = int)

    args = parser.parse_args()
    x, t = read_input(args.trainPath)
    m = int(args.m)
    if args.gamma is None:
        gamma = 0
    else:
        gamma = float(args.gamma)
    if args.numFolds is None:
        k = 5
    else:
        k = int(args.numFolds)
    if args.modelOutput is None:
        output_path = "nopath"
    else:
        output_path = args.modelOutput
    if args.info is True:
        print("Name: Junfei Liu\n",
              "Email: jliu137@u.rochester.edu\n")

    if args.autofit is None or args.autofit is False:
        k_fold_autofit(x, t, gamma, k, m, output_path)
    elif args.autofit is True:
        k_fold_givenm(x, t, gamma, k, m, output_path)
    else:
        print("Please enter correct param for --autofit")
        exit(0)


if __name__ == "__main__":
    main()
