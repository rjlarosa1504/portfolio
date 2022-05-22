"""
Author: Renzo La Rosa
Date: 03/25/2022

NOTE: This script is to be used for the author's personal use only. It is stored in the author's
GitHub portfolio and is for portfolio use only.

This is a bootstrap aggregation learner. It creates the input number of bags using the input learner type.
An ndarray approach (as opposed to a node-branch OOP approach) will be used to create learner.
"""

import numpy as np

class BagLearner(object):

    def __init__(self, learner, kwargs, bags = 2, boost = False, verbose = False):
        self.learner = learner(**kwargs)
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        # self.name = "BagLearner"

    def __author__(self):
        return "Renzo_LaRosa"

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner
        Use a randomizer with replacements to make self.bags number of bags of the type
        learner and take mean as final model.

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        # pass the arguments to the learner
        # self.learner = self.learner(**self.kwargs)
        self.model = []

        # call learner's add_evidence "bags" number of times
        for i in range(self.bags):
            # Create randomized data_x and data_y
            row_indices = np.random.randint(data_x.shape[0], size = data_x.shape[0])
            new_data_x = data_x[row_indices, :]
            new_data_y = data_y[row_indices]

            returnLearner = self.learner.add_evidence(new_data_x, new_data_y)

            if returnLearner is None:
                # If learner returns None then return array of 0's
                returnLearner = np.zeros(3)

            self.model.append(returnLearner)
        return self.model

    def query(self, data_x):
        y_pred_arr = []
        for model in self.model:

            if model.ndim == 1:
                y_pred = (model[:-1] * data_x).sum(axis=1) + model[-1]
            else:
                y_pred = np.ones(data_x.shape[0])
                for i in range(data_x.shape[0]):
                    working_model = model
                    working_data_x_row = data_x[i, :]
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
            y_pred_arr.append(y_pred)

        y_pred_arr = np.asarray(y_pred_arr)
        y_pred_arr = np.median(y_pred_arr, axis=0)  # median of all rows
        return y_pred_arr