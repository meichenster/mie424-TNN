import random, time, math
import numpy as np
from docplex.cp.model import *

"""
This is a CP model that aims to find the TNNs with the maximal sum of the neuron's margins that fits the training set.
"""

class MultiLayerPerceptron:
    def __init__(self, layers, data, labels):
        """
        "layers" is a list of numbers indicating how many neurons each layer has
          e.g. [50, 16, 3] -> 50 input neurons, then 16 neurons on a hidden layer, and 3 output neurons
        "data" is a numpy array of shape (n_samples, 50, 1) from sentiment140
        "labels" is a numpy array of shape (n_samples, 3) with one-hot encoding
        """
        self.layers = layers.copy()
        self.m = CpoModel()

        # Preprocessing: Identify non-constant input features across samples
        self.id2input = []
        # data shape: (n_samples, 50, 1), squeeze to (n_samples, 50)
        data_squeezed = np.squeeze(data, axis=2) # Shape: (1, 50)
        in_equal = np.all(data_squeezed == data_squeezed[0, :], axis=0) # Shape: (50,)
        self.id2input = [n_in for n_in in range(self.layers[0]) if not in_equal[n_in]]
        print("id2input:", self.id2input)
        self.original_input = self.layers[0]  # Should be 50
        print(self.original_input)
        self.layers[0] = len(self.id2input)


        # Weights
        self.weights = {}
        for layer_id in range(1, len(self.layers)):
            for neuron_out in range(self.layers[layer_id]):
                # Incoming weights (last weight is the bias)
                weight_id = (layer_id, neuron_out)
                w = self.m.integer_var_list(self.layers[layer_id-1] + 1, domain=[-1, 0, 1], name="w_%d-%d" % (layer_id, neuron_out))
                self.weights[weight_id] = w

        # Margins per neuron
        self.margins = {}
        for layer_id in range(1, len(self.layers)):
            for neuron_out in range(self.layers[layer_id]):
                self.margins[(layer_id, neuron_out)] = []

        # Auxiliary attributes
        self.msol = None
        self.eg_id = 0

    def add_example(self, data, label, show=False):
        """
        "data" is a single sample of shape (50, 1)
        "label" is a one-hot vector of shape (3,) e.g., [0, 0, 1]
        """
        # Output number (index of 1 in one-hot label)
        out_number = np.where(label == 1)[0][0]

        print("Data, label, weights----------")
        # print(data)
        # print(label)
        print(self.weights)

        # Adding the layers
        activations = None
        n_layers = len(self.layers)
        for layer_id in range(1, n_layers):
            h_size = self.layers[layer_id]
            if layer_id < n_layers - 1:
                activations = [None for _ in range(h_size)]

            for n_out in range(h_size):
                # Computing preactivation
                if layer_id == 1:
                    # Squeeze data from (50, 1) to (50,) and select non-constant inputs
                    x_input = np.squeeze(data)[self.id2input]
                else:
                    x_input = activations_prev  # Hidden layers

                print("x_input, weights[(layer_id, n_out)], pre_activation")
                print(x_input)
                print(self.weights[(layer_id, n_out)])

                # Computing preactivations including an extra 1.0 input for the bias
                pre_activation = self.m.scal_prod(x_input + [1], self.weights[(layer_id, n_out)])
                # Saving the margin for this neuron
                self.margins[(layer_id, n_out)].append(self.m.abs(pre_activation))

                # Computing activation
                if layer_id == n_layers - 1:
                    # Output unit
                    if label[n_out] > 0:
                        self.m.add(pre_activation >= 0)
                    else:
                        self.m.add(pre_activation <= -1)
                else:
                    # Hidden unit (ternary activation: -1 or 1)
                    activations[n_out] = (2 * (pre_activation >= 0) - 1)

            activations_prev = activations

        self.eg_id += 1

    def train(self, n_threads, time_out):
        """
        Returns True if a feasible solution exists
        """
        # Objective: Maximize the minimum margin
        sum_margins = sum([self.m.min(m) for m in self.margins.values()])
        self.m.maximize(sum_margins)

        # Optimize
        self.msol = self.m.solve(Workers=n_threads, TimeLimit=time_out * 60, LogVerbosity="Quiet")
        return bool(self.msol)

    def get_info(self):
        info_all = {}
        info = self.msol.get_solver_infos()
        info_all["bound"] = self.msol.get_objective_bounds()[0]
        info_all["is_optimal"] = self.msol.is_solution_optimal()
        info_all["num_branches"] = info['NumberOfBranches']
        info_all["num_vars"] = info['NumberOfIntegerVariables']

        if bool(self.msol):
            info_all["objective"] = self.msol.get_objective_values()[0]
            info_all["gap"] = self.msol.get_objective_gaps()[0]
            print("objective: %0.2f" % info_all["objective"])
            print("bound: %0.2f" % info_all["bound"])
            print("gap: %0.2f" % info_all["gap"])

        return info_all

    def get_weights(self):
        """
        Returns the best weights found so far
        """
        w_ret, b_ret = [], []
        for layer_id in range(1, len(self.layers)):
            n_in = self.layers[layer_id - 1]
            n_out = self.layers[layer_id]
            if layer_id == 1:
                weights = np.zeros((self.original_input, n_out))  # Full 50 x n_out matrix
            else:
                weights = np.zeros((n_in, n_out))
            biases = np.zeros((n_out,))
            for j in range(n_out):
                weight_id = (layer_id, j)
                for i in range(n_in):
                    if layer_id == 1:
                        weights[self.id2input[i], j] = self.msol[self.weights[weight_id][i]]
                    else:
                        weights[i, j] = self.msol[self.weights[weight_id][i]]
                biases[j] = self.msol[self.weights[weight_id][n_in]]
            w_ret.append(weights)
            b_ret.append(biases)

        return w_ret, b_ret