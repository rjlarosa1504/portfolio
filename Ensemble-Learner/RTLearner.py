"""
Author: Renzo La Rosa
Date: 03/25/2022

NOTE: This script is to be used for the author's personal use only. It is stored in the author's
GitHub portfolio and is for portfolio use only.

This is a random tree learner. It finds the factor to split randomly. The split
value is found using the median of the factor column.
An ndarray approach (as opposed to a node-branch OOP approach) will be used to create learner.
"""
import numpy as np
import random
from scipy import stats

class RTLearner(object):

    def __init__(self, leaf_size = 1, verbose = False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        # self.name = "RTLearner"

    def __author__(self):
        return "Renzo_LaRosa"

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """

        data = np.concatenate((data_x, data_y.reshape(len(data_y), 1)), axis = 1)

        if data.shape[0] == 1: #number of rows = 1 means leaf
            return [-1, data[0,-1], -1, -1]

        if len(np.unique(data[:, -1])) == 1: #same y across all remaining rows
            return [-1, data[0, -1], -1, -1]
        else:
            if data.shape[0] <= self.leaf_size:
                rowMode = stats.mode(data[:,-1])

                self.model = [-1, rowMode[0][-1], -1, -1]
                return self.model

            index = np.random.randint(data_x.shape[1])
            # factor is at index
            splitVal = np.median(data[:, index])

            tmp = data[data[:, index] <= splitVal]

            # if median equals the max or min of the data table, then return median y
            if (data[:, index].max() == splitVal) or (data[:, index].min() == splitVal):
                rowMode = stats.mode(data[:,-1])
                # rowMedian = np.median(data, axis=0)  # median of all rows
                self.model = [-1, rowMode[0][-1], -1, -1]
                return self.model

            lefttree = np.asarray(self.add_evidence(tmp[:,:-1], tmp[:,-1]))

            tmp = data[data[:, index] > splitVal]
            righttree = np.asarray(self.add_evidence(tmp[:,:-1], tmp[:,-1]))

            if lefttree.ndim == 1:
                lefttree = np.reshape(lefttree, (1, 4))
            if righttree.ndim == 1:
                righttree = np.reshape(righttree, (1, 4))
            root = np.asarray([[index, splitVal, 1, lefttree.shape[0] + 1]])

            self.model = np.concatenate([root, lefttree, righttree], axis=0)
            return self.model


    def query(self, data_x):

        y_pred = np.ones(data_x.shape[0])
        for i in range(data_x.shape[0]):
            working_model = self.model
            working_data_x_row = data_x[i,:]
            factor = 0
            while factor != -1:
                factor = int(working_model[0, 0])
                if (factor == -1):
                    # We have hit a leaf row
                    working_y = working_model[0, 1]
                    break
                start = 0
                stop = 0
                if (working_data_x_row[factor] <= working_model[0, 1]):
                    start += int(working_model[0, 2])
                    stop += int(working_model[0, 3])
                    working_model = working_model[start:stop, :]
                else:
                    start += int(working_model[0, 3])
                    working_model = working_model[start:, :]
            y_pred[i] = working_y

        # return ndarray y_pred
        return y_pred