import csv
import numpy as np
from mixed_gaussian_EM_algorithm import MixedGaussianModel


def main():
    dataset_file = open('Test set.csv')
    csvReader = csv.reader(dataset_file)
    sample_data = []
    male_data = [[], []]
    female_data = [[], []]
    for row in csvReader:
        try:
            if row[2] == 'Female':
                female_data[0].append(float(row[0]) / 100)
                female_data[1].append(float(row[1]) / 100)
                sample_data.append([float(row[0]) / 100, float(row[1]) / 100, 0])
            else:
                male_data[0].append(float(row[0]) / 100)
                male_data[1].append(float(row[1]) / 100)
                sample_data.append([float(row[0]) / 100, float(row[1]) / 100, 1])
        except ValueError as v:
            print(v)
    male_array = np.array(male_data)
    female_array = np.array(female_data)
    print(np.mean(male_array, axis=1))
    print(np.mean(female_array, axis=1))
    print(np.cov(male_array))
    print(np.cov(female_array))
    sample_data = np.array(sample_data)
    model = MixedGaussianModel(2)
    model.import_test_data(sample_data)
    u_list = [[1.5, 0.5], [1.75, 0.75]]
    model.init_mean(np.array(u_list))
    s_list = [[1, 0.5, 1], [1, 0.5, 1]]
    model.init_sigma(s_list)
    p_list = [0.5, 0.5]
    model.init_pi(np.array(p_list))
    model.plot_raw_data()
    model.iteration_calculate()
    # model.classification()
    model.demarcation()


if __name__ == '__main__':
    main()
