"""
This is a MIP model that aims to find the BNNs with the maximal sum of the neuron's margins that fits the training set.
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

        # Input shape is (n_samples, 50)
        dead_inputs = np.all(data == data[0,:], axis = 0)

        # Weights and biases
        self.weights = {}
        self.biases  = {}
        self.margins = {}
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
                bias_id = (layer_id, neuron_out)
                b = self.m.addVar(vtype=GRB.INTEGER, name="b_%d-%d"%bias_id, lb=-1, ub=1)
                self.biases[bias_id] = b
                # Adding margins per neuron
                n_id = (layer_id, neuron_out)
                self.margins[n_id] = self.m.addVar(vtype=GRB.CONTINUOUS, name="m%d-%d"%n_id, lb=0)

        # Max margin loss
        self.loss = sum(list(self.margins.values()))
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
                margin = self.margins[(layer_id, n_out)]
                # computing activation
                if layer_id == len(self.neurons_per_layer)-1:
                    # This is an output unit
                    # NOTE: Changed from oriignal code to be TNN
                    if label[0] == 1: # Negative (class=-1)
                        self.m.addConstr(pre_activation <= -margin - 0.001)
                    elif label[1] == 1:
                        self.m.addConstr(-margin <= pre_activation)
                        self.m.addConstr(pre_activation <= margin)
                    elif label[2] == 1:
                        self.m.addConstr(pre_activation >= margin)
                        
                else:
                    # This is a hidden unit
                    n = self.m.addVar(vtype=GRB.INTEGER, lb=-1, ub=1, name="n%d-%d_%d"%(layer_id, n_out, self.eg_id))
                    # NOTE: Added two new binary variables p and q where n = p-q such that n can take on values -1, 0, 1
                    p = self.m.addVar(vtype=GRB.BINARY, name="p%d-%d_%d"%(layer_id, n_out, self.eg_id) )
                    q = self.m.addVar(vtype=GRB.BINARY, name="q%d-%d_%d"%(layer_id, n_out, self.eg_id) )
                    # NOTE: Added constraint for p and q
                    self.m.addConstr(n == p - q) # p=1, q=0 means n=1 (positive) and p=0,q=0 means n=1 (neutral), etc.
                    self.m.addConstr(p + q <= 1) # p and q cannot both == 1 at the same time
                    # Indicator constraint version
                    # NOTE: Changed from original code to be TNN and added M to scale p and q values
                    M = 100000 # M * (1-p)
                    self.m.addConstr(pre_activation <= -margin - 0.001 + M*(1-q)) # applies for n=-1
                    self.m.addConstr(pre_activation >=  -margin - M*(p+q)) # applies for n=0
                    self.m.addConstr(pre_activation <= margin + M*(p+q)) # applies for n=0
                    self.m.addConstr(pre_activation >=  margin - M*(1-p)) # applies for n=1
  
                    neurons[(layer_id, n_out)] = n
                    self.activations[(layer_id, n_out, self.eg_id)] = n

        # Keeping track of the example id is important to name the new auxiliary variables
        self.eg_id += 1

    def train(self, n_threads, time_out):
        """
        Returns True if no feasible solution exists
        """

        # Params
        self.m.Params.OutputFlag = 0
        self.m.Params.Threads = n_threads
        self.m.Params.TimeLimit = time_out*60

        # Optimize
        self.m.setObjective(self.loss, GRB.MAXIMIZE)
        self.m.update()
        self.m.optimize()

        # Is feasible?
        return self.m.SolCount > 0

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
