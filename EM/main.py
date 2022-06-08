import csv
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from mixed_gaussian_EM_algorithm import MixedGaussianModel


def main():
    dataset_file = open('Test set.csv')
    csvReader = csv.reader(dataset_file)
    sample_data = []
    for row in csvReader:
        try:
            sample_data.append([float(row[0]) / 100, float(row[1]) / 100])
        except ValueError as v:
            print(v)
    sample_data = np.array(sample_data)
    model = MixedGaussianModel(2)
    model.import_test_data(sample_data)
    u_list = [[1.4, 0.65], [1.70, 0.8]]
    model.init_mean(np.array(u_list))
    s_list = [[0.1, 0.01, 0.1], [0.1, 0.01, 0.1]]
    model.init_sigma(s_list)
    p_list = [0.5, 0.5]
    model.init_pi(np.array(p_list))
    model.iteration_calculate()


if __name__ == '__main__':
    main()
