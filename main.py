import argparse
import csv
import numpy as np
from calc import *


def read_input(dir):
    x, y = np.empty(0), np.empty(0)
    with open(dir, newline='\n') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            x = np.append(x, float(row[0]))
            y = np.append(y, float(row[1]))
    return x, y


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--m", type = int, required = True)
    # parser.add_argument("--gamma", type = float)
    # parser.add_argument("--trainPath", type = str, required = True)
    # parser.add_argument("--modelOutput", type = str)
    # parser.add_argument("--autofit", type = bool)
    # parser.add_argument("--info", type = bool)
    x, t = read_input("levelOne/B")
    gamma = 0.0000001
    #auto_fit(x, t, gamma)
    m_fit(x, t, 2, gamma)


if __name__ == "__main__":
    main()



