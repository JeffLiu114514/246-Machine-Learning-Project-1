import numpy as np
import csv

def read_input(dir):
    x, y = np.empty(0), np.empty(0)
    with open(dir, newline='\n') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            x = np.append(x, float(row[0]))
            y = np.append(y, float(row[1]))
    return x, y






