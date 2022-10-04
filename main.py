import argparse
from util import *
from calc import *

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--m", type = int, required = True)
    # parser.add_argument("--gamma", type = float)
    # parser.add_argument("--trainPath", type = str, required = True)
    # parser.add_argument("--modelOutput", type = str)
    # parser.add_argument("--autofit", type = bool)
    # parser.add_argument("--info", type = bool)
    x, t = read_input("levelOne/A")
    matrix = design_matrix(x, 5)
    print(matrix)



