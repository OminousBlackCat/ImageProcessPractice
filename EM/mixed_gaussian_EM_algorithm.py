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
    sigma_matrix = np.array([
        [s_x, c],
        [c, s_y]
    ])
    sigma_matrix = np.matrix(sigma_matrix)
    coe = 1 / (2 * math.pi * math.sqrt(np.linalg.det(sigma_matrix)))
    vector = np.array([x - u_x, y - u_y])
    pow_coe = -0.5 * (np.dot(np.dot(vector, sigma_matrix.I), vector.T))
    return coe * math.exp(pow_coe)


def gaussian_function_with_input(u_x, u_y, s_x, c, s_y, x: np.array, y: np.array):
    z = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i][j] = gaussian_function(u_x, u_y, s_x, c, s_y, x[i][j], y[i][j])
    return z


class MixedGaussianModel:
    def __init__(self, cluster_count: int):
        self.cluster_count = cluster_count
        self.u_list = np.zeros((cluster_count, 2))  # with each cluster's (u_x, u_y)
        self.sigma_list = np.zeros((cluster_count, 3))  # with each cluster's (sigma_x, cor_parm, sigma_y)
        self.pi_list = np.zeros((cluster_count,))  # with each item represent the probability of which cluster
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

    def draw_gaussian_with_current_parm(self, cluster_index):
        x = np.arange(1.2, 2.0, 0.01)
        y = np.arange(0.3, 1.3, 0.01)
        xx, yy = np.meshgrid(x, y)
        z1 = gaussian_function_with_input(self.u_list[cluster_index][0], self.u_list[cluster_index][1],
                                          self.sigma_list[cluster_index][0],
                                          self.sigma_list[cluster_index][1], self.sigma_list[cluster_index][2], xx, yy)
        fig, ax = plt.subplots()
        c = ax.pcolormesh(x, y, z1, cmap='RdBu', vmin=np.min(z1), vmax=np.max(z1))
        fig.colorbar(c, ax=ax)
        self.plot_raw_data()
        plt.show()

    def iteration_calculate(self):
        print('Using EM algorithm fits mixed gaussian model')
        print('Current cluster count:' + str(self.cluster_count))
        print('Initial gaussian parameter:')
        print('U List:' + str(self.u_list))
        print('Sigma List' + str(self.sigma_list))
        print('Pi List' + str(self.pi_list))

        iteration_count = 0
        self.draw_gaussian_with_current_parm(0)
        self.draw_gaussian_with_current_parm(1)
        while iteration_count < 100000:
            # Using initial parameter to start iterating
            # E step: according to current model parameter, calculate the influence facter of each cluster
            influence_facter = np.zeros((self.sample_list.shape[0], self.cluster_count))
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
                    if influence_facter[j][k] == 0:
                        continue
                    influence_facter[j][k] /= sumOfExpectation
            # M step: calculate the new parameter of mixed gaussian function which will be used in the next iteration
            sum_of_Q_table = np.sum(influence_facter, axis=0)
            # record original u parameter which will using at this iteration
            upper_u_list = np.array(self.u_list)
            # Update u parameter
            for k in range(self.cluster_count):
                self.u_list[k][0] = np.sum(influence_facter[:, k] * self.sample_list[:, 0]) / sum_of_Q_table[k]
                self.u_list[k][1] = np.sum(influence_facter[:, k] * self.sample_list[:, 1]) / sum_of_Q_table[k]
            # Update sigma parameter
            for k in range(self.cluster_count):
                a = (self.sample_list[:, 1] - upper_u_list[k][1]) * (self.sample_list[:, 1] - upper_u_list[k][1])
                a = a * influence_facter[:, k]
                sum_temp = np.sum(a)
                self.sigma_list[k][0] = np.sum(influence_facter[:, k] *
                                               ((self.sample_list[:, 0] - self.u_list[k][0]) * (
                                                       self.sample_list[:, 0] - self.u_list[k][0]))) / \
                                        sum_of_Q_table[k]
                self.sigma_list[k][1] = np.sum(influence_facter[:, k] * (
                        (self.sample_list[:, 0] - self.u_list[k][0]) * (
                        self.sample_list[:, 1] - self.u_list[k][1]))) / sum_of_Q_table[k]
                self.sigma_list[k][2] = np.sum(influence_facter[:, k] * (
                        (self.sample_list[:, 1] - self.u_list[k][1]) * (
                        self.sample_list[:, 1] - self.u_list[k][1]))) / sum_of_Q_table[k]
            # Update pi parameter
            for k in range(self.cluster_count):
                self.pi_list[k] = sum_of_Q_table[k] / self.sample_list.shape[0]
            print('The' + str(iteration_count) + 'th time update')
            print('U List:' + str(self.u_list))
            print('Sigma List' + str(self.sigma_list))
            print('Pi List' + str(self.pi_list))
            if (upper_u_list[0][0] - self.u_list[0][0]) ** 2 < 0.0001 and (
                    upper_u_list[1][1] - self.u_list[1][1]) ** 2 < 0.0001:
                break
            else:
                iteration_count += 1
            self.draw_gaussian_with_current_parm(cluster_index=0)
            self.draw_gaussian_with_current_parm(cluster_index=1)
