"""
This is a MIP model that aims to find the BNNs with fewer non-zero weights that fits the training set.
"""
import random, time, math
from gurobipy import *
import numpy as np

class MultiLayerPerceptron:
    def __init__(self, neurons_per_layer, data, labels):
        """
        "neurons_per_layer" is a list of numbers indicating how many neurons each layer has
          e.g. [2,2,1] -> 2 input neurons, then 2 neurons on a hidden layer, and one output neuron
        """
        self.neurons_per_layer = neurons_per_layer
        self.m = Model("MLP")

        # Removing dead inputs
        dead_inputs = np.all(data == data[0,:], axis = 0)

        # Weights and biases
        self.weights = {}
        self.biases  = {}
        for layer_id in range(1,len(self.neurons_per_layer)):
            for neuron_out in range(self.neurons_per_layer[layer_id]):
                # Adding weights
                for neuron_in in range(self.neurons_per_layer[layer_id-1]):
                    # NOTE: layer_id correspond to the layer in which the output neuron is
                    w_id = (neuron_in, layer_id, neuron_out)
                    if layer_id == 1 and dead_inputs[neuron_in]:
                        w = 0
                    else:            
                        w = self.m.addVar(vtype=GRB.INTEGER, name="w%d_%d-%d"%w_id, lb=-1, ub=1)
                    self.weights[w_id] = w
                # Adding biases
                b_id = (layer_id, neuron_out)
                b = self.m.addVar(vtype=GRB.INTEGER, name="b_%d-%d"%b_id, lb=-1, ub=1)
                self.biases[b_id] = b

        # Min weights loss
        w_abs = []
        for w_id in self.weights:
            if type(self.weights[w_id]) is int:
                continue
            w = self.m.addVar(vtype=GRB.BINARY, name="aw%d_%d-%d"%w_id)
            self.m.addConstr(self.weights[w_id] <=  w)
            self.m.addConstr(self.weights[w_id] >= -w)
            w_abs.append(w)
        for b_id in self.biases:
            b = self.m.addVar(vtype=GRB.BINARY, name="ab_%d-%d"%b_id)
            self.m.addConstr(self.biases[b_id] <=  b)
            self.m.addConstr(self.biases[b_id] >= -b)
            w_abs.append(w)
        self.loss = sum(w_abs)

        # No loss function
        self.eg_id = 0
        self.activations = {}

    def _add_neuron_weight_binding(self, n, w, n_in, layer_id, n_out):
        I = self.m.addVar(vtype=GRB.CONTINUOUS, name="I_%d-%d-%d_%d"%(n_in, layer_id, n_out, self.eg_id), lb=-1, ub=1)
        self.m.addConstr(I - w + 2*n <=  2)
        self.m.addConstr(I + w - 2*n <=  0)
        self.m.addConstr(I - w - 2*n >= -2)
        self.m.addConstr(I + w + 2*n >=  0)
        return I

    def add_example(self, data, label, show = False):

        """
        NOTE:
            - the neurons are binary variables (0,1)
            - however, the '0' value has to be mapped to '-1' when adding the constraints (i.e. replace 'n' by '2*n-1')
        """

        # Adding the layers
        neurons = {}
        for layer_id in range(1, len(self.neurons_per_layer)):
            for n_out in range(self.neurons_per_layer[layer_id]):
                # computing preactivation
                if layer_id == 1:
                    # First layer neuron
                    pre_activation = sum([data[i] * self.weights[(i,1,n_out)] for i in range(len(data))])
                else:
                    inputs = []
                    for n_in in range(self.neurons_per_layer[layer_id-1]):
                        n = neurons[(layer_id-1, n_in)]
                        w = self.weights[(n_in, layer_id, n_out)]
                        I = self._add_neuron_weight_binding(n, w, n_in, layer_id, n_out)
                        inputs.append(I)
                    pre_activation = sum(inputs)
                # adding the bias
                pre_activation += self.biases[(layer_id,n_out)]
                # computing activation
                if layer_id == len(self.neurons_per_layer)-1:
                    # This is an output unit
                    if label[n_out] > 0:
                        self.m.addConstr(pre_activation >=  0)
                    else:
                        self.m.addConstr(pre_activation <= -1)
                else:
                    # This is a hidden unit
                    n = self.m.addVar(vtype=GRB.BINARY, name="n%d-%d_%d"%(layer_id, n_out, self.eg_id))
                    # Indicator constraint version
                    self.m.addConstr((n == 1) >> (pre_activation >=  0))
                    self.m.addConstr((n == 0) >> (pre_activation <= -1))
                    neurons[(layer_id, n_out)] = n
                    self.activations[(layer_id, n_out, self.eg_id)] = n

        # Keeping track of the example id is important to name the new auxiliary variables
        self.eg_id += 1

    def add_warmup(self, weights, biases, activations):
        # setting the weights and biases
        for layer_id in range(1,len(self.neurons_per_layer)):
            for neuron_out in range(self.neurons_per_layer[layer_id]):
                # Adding weights
                for neuron_in in range(self.neurons_per_layer[layer_id-1]):
                    w = self.weights[(neuron_in, layer_id, neuron_out)]
                    if not(type(w) is int):
                        w.start = weights[layer_id-1][neuron_in,neuron_out]
                # Adding biases
                b = self.biases[(layer_id, neuron_out)]
                b.start = biases[layer_id-1][neuron_out]
        # setting the activations
        for layer_id in range(1, len(self.neurons_per_layer)-1):
            for eg_id in range(self.eg_id):
                for n_out in range(self.neurons_per_layer[layer_id]):
                    n = self.activations[(layer_id, n_out, eg_id)]
                    n.start = activations[layer_id][eg_id,n_out]

    def train(self, n_threads, time_out):
        """
        Returns True if no feasible solution exists
        """

        # Params
        self.m.Params.OutputFlag = 0
        self.m.Params.Threads = n_threads
        self.m.Params.TimeLimit = time_out*60

        # Optimize
        self.m.setObjective(self.loss, GRB.MINIMIZE)
        self.m.update()
        self.m.optimize()

        # Is feasible?
        return self.m.SolCount > 0

    def get_info(self):
        info_all = {}
        info_all["objective"] = self.m.ObjVal
        info_all["bound"] = self.m.ObjBound
        info_all["gap"] = self.m.MIPGap
        info_all["is_optimal"] = (self.m.status == GRB.OPTIMAL)
        info_all["num_nodes"] = self.m.NodeCount
        info_all["num_vars"] = self.m.NumIntVars + self.m.NumBinVars

        if self.m.SolCount > 0:
            print("objective: %0.2f"%info_all["objective"])
            print("bound: %0.2f"%info_all["bound"])
            print("gap: %0.2f"%info_all["gap"])

        return info_all

    def get_weights(self):
        """
        Returns the best weights found so far
        """
        w_ret, b_ret = [], []
        for layer_id in range(1,len(self.neurons_per_layer)):
            n_in = self.neurons_per_layer[layer_id-1]
            n_out = self.neurons_per_layer[layer_id]
            weights = np.zeros((n_in, n_out))
            biases  = np.zeros((n_out,))
            for j in range(n_out):
                for i in range(n_in):
                    w_id = (i, layer_id, j)
                    w = self.weights[w_id]
                    if type(w) is int: weights[i,j] = 0
                    else:              weights[i,j] = w.X
                biases[j] = self.biases[(layer_id, j)].X
            w_ret.append(weights)
            b_ret.append(biases)
        
        return w_ret, b_ret
