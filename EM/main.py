import csv
import os 
import re
import numpy as np
import matplotlib.pyplot as plt


def checkData():
    male_height_sample = []
    female_height_sample = []
    male_weight_sample = []
    female_weight_sample = []
    dataset_file = open('Test set.csv')
    csvreader = csv.reader(dataset_file)
    for row in csvreader:
        try:
            if row[-1] == 'Male':
                male_height_sample.append(float(row[0]) / 100)
                male_weight_sample.append(float(row[1]))
            else:
                female_height_sample.append(float(row[0]) / 100)
                female_weight_sample.append(float(row[1]))
        except ValueError as v:
            print(v)

    male_count = len(male_height_sample)
    female_count = len(female_height_sample)
    print(str(male_count) + '*' + str(female_count))
    plt.scatter(np.array(male_height_sample), np.array(male_weight_sample))
    plt.scatter(np.array(female_height_sample), np.array(female_weight_sample))
    plt.show()
        

def main():
    checkData()


if __name__ == '__main__':
    main()
