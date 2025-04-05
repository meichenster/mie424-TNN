import random, time, math
import numpy as np
from docplex.cp.model import *

"""
This is a CP model that aims to find the BNNs with fewer non-zero weights that fits the training set.
"""

class MultiLayerPerceptron:
    def __init__(self, layers, data, labels):
        """
        "layers" is a list of numbers indicating how many neurons each layer has
          e.g. [2,2,1] -> 2 input neurons, then 2 neurons on a hidden layer, and one output neuron
        """
        self.layers = layers.copy()
        self.m = CpoModel()        

        # Preprocessing (we only learn parameters for weights that has non-zero input for some image)
        self.id2input = []
        in_equal = np.all(data == data[0,:], axis = 0)
        self.id2input = [n_in for n_in in range(self.layers[0]) if not in_equal[n_in]]
        self.original_input = self.layers[0]
        self.layers[0] = len(self.id2input)

        # Weights
        self.weights = {}
        for layer_id in range(1,len(self.layers)):
            for neuron_out in range(self.layers[layer_id]):
                # Adding incomming weights (last weight is the bias)
                weight_id = (layer_id, neuron_out)
                w = self.m.integer_var_list(self.layers[layer_id-1] + 1, domain=[-1,0,1], name="w_%d-%d"%(layer_id, neuron_out))
                self.weights[weight_id] = w

        # computing the objective function
        sum_w = sum([abs(w) for layer_id in range(1,len(self.layers)) for neuron_out in range(self.layers[layer_id]) for w in self.weights[(layer_id, neuron_out)]])
        self.m.minimize(sum_w)

        # Auxiliary attributes
        self.msol = None
        self.eg_id = 0

    def add_example(self, data, label, show = False):

        # Output number
        out_number = np.where(label==1)[0][0]

        # Adding the layers
        activations = None
        n_layers = len(self.layers)
        for layer_id in range(1, n_layers):
            h_size = self.layers[layer_id]
            if layer_id < n_layers-1:
                activations = [None for _ in range(h_size)]
            
            for n_out in range(h_size):
                # computing preactivation
                if layer_id == 1: 
                    x_input = [data[i] for i in self.id2input] # Image
                else: 
                    x_input = activations_prev # Hidden layers
                
                # Computing preactivations including an extra 1.0 input for the bias
                pre_activation = scal_prod(x_input + [1], self.weights[(layer_id,n_out)])
                
                # NOTE: EDITED HERE FOR TNN
                # computing activation
                if layer_id == n_layers-1:
                    # This is an output unit - allow ternary {-1, 0, +1}
                    if label[n_out] > 0:
                        self.m.add(pre_activation >= 1)  # Forces +1 for positive examples
                    elif label[n_out] < 0:
                        self.m.add(pre_activation <= -1)  # Forces -1 for negative examples
                    # If label is 0, no constraint added, allowing any ternary value
                # Hidden layers: ternary {-1, 0, +1}   
                else:
                    # This is a hidden unit
                    activations[n_out] = (2*(pre_activation >= 0)-1)

            activations_prev = activations

        # Keeping track of the example id is important to name the new auxiliary variables
        self.eg_id += 1

    def train(self, n_threads, time_out):
        """
        Returns True if no feasible solution exists
        """

        # Optimize
        self.msol = self.m.solve(Workers=n_threads, TimeLimit=time_out*60, LogVerbosity="Quiet")

        # Is feasible?
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
            print("objective: %0.2f"%info_all["objective"])
            print("bound: %0.2f"%info_all["bound"])
            print("gap: %0.2f"%info_all["gap"])

        return info_all


    def get_weights(self):
        """
        Returns the best weights found so far
        """
        w_ret, b_ret = [], []
        for layer_id in range(1,len(self.layers)):
            n_in = self.layers[layer_id-1]
            n_out = self.layers[layer_id]
            if layer_id == 1: weights = np.zeros((self.original_input, n_out))
            else: weights = np.zeros((n_in, n_out))
            biases  = np.zeros((n_out,))
            for j in range(n_out):
                weight_id = (layer_id, j)
                for i in range(n_in):
                    if layer_id == 1:
                        weights[self.id2input[i],j] = self.msol[self.weights[weight_id][i]]
                    else:
                        weights[i,j] = self.msol[self.weights[weight_id][i]]
                biases[j] = self.msol[self.weights[weight_id][n_in]]
            w_ret.append(weights)
            b_ret.append(biases)
        
        return w_ret, b_ret
