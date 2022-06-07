"""
This algorithm is designed for two dimension mixed gaussian model
github url: https://github.com/OminousBlackCat
"""
import numpy as np
import matplotlib.pyplot as plt
import math

"""
standard two dimension gaussian function
"""


def gaussian_function(u_x, u_y, s_x, c, s_y, x, y):
    coe = 1 / (2 * math.pi * s_x * s_y * math.sqrt(1 - c * c))
    pow_coe = ((x - u_x) / s_x) ** 2 - 2 * c * ((x - u_x) / s_x) * ((y - u_y) / s_y) + ((y - u_y) / s_y) ** 2
    pow_coe = pow_coe * (-1 / 2 * (1 - c ** 2))
    return coe * math.exp(pow_coe)


class MixedGaussianModel:
    def __init__(self, cluster_count: int):
        self.cluster_count = cluster_count
        self.u_list = np.zeros((cluster_count, 2))  # with each cluster's (u_x, u_y)
        self.sigma_list = np.zeros(cluster_count, 3)  # with each cluster's (sigma_x, cor_parm, sigma_y)
        self.pi_list = np.zeros(cluster_count, )  # with each item represent the probability of which cluster
        self.sample_list = None  # save the test sample

    def init_mean(self, init_list):
        self.u_list = np.array(init_list)

    def init_sigma(self, init_list):
        self.sigma_list = np.array(init_list)

    def init_pi(self, init_list):
        self.pi_list = init_list

    def import_test_data(self, data):
        self.sample_list = np.array(data)

    def plot_raw_data(self):
        plt.scatter(self.sample_list[:, 0], self.sample_list[:, 1])
        plt.show()

    def iteration_calculate(self):
        print('Using EM algorithm fits mixed gaussian model')
        print('Current cluster count:' + str(self.cluster_count))
        print('Initial gaussian parameter:')
        print('U List:' + str(self.u_list))
        print('Sigma List' + str(self.sigma_list))
        print('Pi List' + str(self.pi_list))

        # Using initial parameter to start iterating
        # E step: according to current model parameter, calculate the influence facter of each cluster
        influence_facter = np.zeros((self.sample_list, self.cluster_count))
        for j in range(self.sample_list.shape[0]):
            # j is the iterator of sample
            sumOfExpectation = 0
            for k in range(self.cluster_count):
                # k is the iterator of cluster
                influence_facter[j][k] = self.pi_list[k] * gaussian_function(self.u_list[k][0], self.u_list[k][1],
                                                                             self.sigma_list[k][0],
                                                                             self.sigma_list[k][1],
                                                                             self.sigma_list[k][2],
                                                                             self.sample_list[j][0],
                                                                             self.sample_list[j][1])
                sumOfExpectation += influence_facter[j][k]
            for k in range(self.cluster_count):
                influence_facter[j][k] /= sumOfExpectation
        # M step: calculate the new parameter of mixed gaussian function which will be used in the next iteration
        sum_of_Q_table = np.sum(influence_facter, axis=0)
        for k in range(self.cluster_count):
            self.u_list[k][0] = np.sum(influence_facter[:, k] * self.sample_list[:, 0])/sum_of_Q_table
            self.u_list[k][1] = np.sum(influence_facter[:, k] * self.sample_list[:, 1])/sum_of_Q_table
