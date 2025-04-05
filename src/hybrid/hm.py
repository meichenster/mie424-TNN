"""
This is a CP/MIP hybrid approach that:
    1) It uses CP without an objective function to find a feasible solution and, then,
    2) It uses MIP to optimize for min-w or max-margin using the CP solution as warm start or to fix activations
"""
import random, time, math
import numpy as np
from tnn import TernaryNeuralNetwork
from docplex.cp.model import *
from gurobipy import *
import mip.mip_w, mip.mip_m

class HybridMethod:
    def __init__(self, solver, layers, data, labels):
        """
        "layers" is a list of numbers indicating how many neurons each layer has
          e.g. [2,2,1] -> 2 input neurons, then 2 neurons on a hidden layer, and one output neuron
        """
        self.solver = solver
        self.layers = layers
        self.data = data
        self.labels = labels
        self.cp_feasibility = FeasCP(layers, data)
        self.n_train = 0

    def add_example(self, data, label, show = False):
        self.cp_feasibility.add_example(data, label, show)
        self.n_train += 1


    def train(self, n_threads, time_out):
        """
        Returns True if no feasible solution exists
        """
        # Step 1: Find a feasible solution using CP
        cp_start = time.time()
        self.is_cp_sat = self.cp_feasibility.train(n_threads, time_out)
        self.cp_total_time = (time.time() - cp_start)/60.0
        if self.is_cp_sat:
            # Step 2: Optimize the solution using MIP
            weights, biases = self.cp_feasibility.get_weights()
            self.cp_weights, self.cp_biases = weights, biases
            # NOTE: Changed this from BNN to TNN
            tnn = TernaryNeuralNetwork(self.layers)
            for i in range(len(weights)):
                tnn.update_layer(i, weights[i], biases[i])
            activations = tnn.get_activations(self.data)

            # Loading the mip model
            if self.solver == "hw_w": self.mip_optimality = mip.mip_w.MultiLayerPerceptron(self.layers, self.data, self.labels)
            if self.solver == "hw_m": self.mip_optimality = mip.mip_m.MultiLayerPerceptron(self.layers, self.data, self.labels)
            if self.solver == "ha_w": self.mip_optimality = OptWeightMIP(self.layers, self.data, weights, biases)
            if self.solver == "ha_m": self.mip_optimality = OptMarginMIP(self.layers, self.data, weights, biases)

            # Adding the examples for warmup approaches
            if self.solver in ["hw_w", "hw_m"]:
                for i in range(self.n_train):
                    self.mip_optimality.add_example(self.data[i], self.labels[i], show=False)
                self.mip_optimality.add_warmup(weights, biases, activations)

            # Adding the examples for fixed-activation approaches
            if self.solver in ["ha_w", "ha_m"]:
                for i in range(self.n_train):
                    activations_i = [acts[i,:] for acts in activations]
                    self.mip_optimality.add_example(activations_i)

            # Training for the remaining of the time ("time_out-self.cp_total_time")
            if time_out-self.cp_total_time > 0:
                self.is_mip_sat = self.mip_optimality.train(n_threads, time_out-self.cp_total_time)
            else:
                self.is_mip_sat = False

        return self.is_cp_sat

    def get_info(self):
        info = {}
        # Adding cp info
        info["cp_total_time"] = self.cp_total_time
        info["cp_is_sat"] = self.is_cp_sat
        cp_info = self.cp_feasibility.get_info()
        for k in cp_info:
            info["cp_" + k] = cp_info[k]
        # Adding MIP info
        if self.is_cp_sat and self.is_mip_sat:
            info["mip_is_sat"] = self.is_mip_sat
            mip_info = self.mip_optimality.get_info()
            for k in mip_info:
                info["mip_" + k] = mip_info[k]
            
            # Adding the cp weights, just in case
            info["cp_weights"] = [w.tolist() for w in self.cp_weights]
            info["cp_biases"]  = [b.tolist() for b in self.cp_biases]
        else:
            info["mip_is_sat"] = False

        return info

    def get_weights(self):
        """
        Returns the best weights found so far
        """
        if self.is_mip_sat:
            return self.mip_optimality.get_weights()
        return self.cp_feasibility.get_weights()


class OptWeightMIP:
    """
    This MIP optimize the weights given fixed activations
    """

    def __init__(self, layers, data, w_weights, w_biases):
        """
        "layers" is a list of numbers indicating how many neurons each layer has
          e.g. [2,2,1] -> 2 input neurons, then 2 neurons on a hidden layer, and one output neuron
        """
        self.layers = layers
        self.m = Model("MLP")

        # Removing dead inputs
        dead_inputs = np.all(data == data[0,:], axis = 0)

        # Weights and biases (w/warmup solutions)
        self.weights = {}
        self.biases  = {}
        for layer_id in range(1,len(self.layers)):
            for neuron_out in range(self.layers[layer_id]):
                # Adding weights
                for neuron_in in range(self.layers[layer_id-1]):
                    # NOTE: layer_id correspond to the layer in which the output neuron is
                    w_id = (neuron_in, layer_id, neuron_out)
                    if layer_id == 1 and dead_inputs[neuron_in]:
                        w = 0
                    else:            
                        w = self.m.addVar(vtype=GRB.INTEGER, name="w%d_%d-%d"%w_id, lb=-1, ub=1)
                        w.start = w_weights[layer_id-1][neuron_in,neuron_out]
                    self.weights[w_id] = w
                # Adding biases
                b_id = (layer_id, neuron_out)
                b = self.m.addVar(vtype=GRB.INTEGER, name="b_%d-%d"%b_id, lb=-1, ub=1)
                b.start = w_biases[layer_id-1][neuron_out]
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
            w_abs.append(b)
        self.loss = sum(w_abs)


    def add_example(self, activations):
        # Adding the constraints given the activations
        neurons = {}
        for layer_id in range(1, len(self.layers)):
            for n_out in range(self.layers[layer_id]):
                # computing preactivation
                pre_activation = sum([activations[layer_id-1][n_in] * self.weights[(n_in,layer_id,n_out)] for n_in in range(self.layers[layer_id-1])])
                # adding the bias
                pre_activation += self.biases[(layer_id,n_out)]
                # NOTE: Changed activation for TNN
                if activations[layer_id][n_out] == 1:
                    self.m.addConstr(pre_activation >= 0)
                elif activations[layer_id][n_out] == -1:
                    self.m.addConstr(pre_activation <= -1)
                else:  # 0
                    self.m.addConstr(pre_activation >= -1)
                    self.m.addConstr(pre_activation <= 0)

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
        for layer_id in range(1,len(self.layers)):
            n_in = self.layers[layer_id-1]
            n_out = self.layers[layer_id]
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


class OptMarginMIP:
    """
    This MIP optimize the margin given fixed activations
    """

    def __init__(self, layers, data, w_weights, w_biases):
        """
        "layers" is a list of numbers indicating how many neurons each layer has
          e.g. [2,2,1] -> 2 input neurons, then 2 neurons on a hidden layer, and one output neuron
        """
        self.layers = layers
        self.m = Model("MLP")

        # Removing dead inputs
        dead_inputs = np.all(data == data[0,:], axis = 0)

        # Weights and biases (w/warmup solutions)
        self.weights = {}
        self.biases  = {}
        self.margins = {}
        for layer_id in range(1,len(self.layers)):
            for neuron_out in range(self.layers[layer_id]):
                # Adding weights
                for neuron_in in range(self.layers[layer_id-1]):
                    # NOTE: layer_id correspond to the layer in which the output neuron is
                    w_id = (neuron_in, layer_id, neuron_out)
                    if layer_id == 1 and dead_inputs[neuron_in]:
                        w = 0
                    else:            
                        w = self.m.addVar(vtype=GRB.INTEGER, name="w%d_%d-%d"%w_id, lb=-1, ub=1)
                        w.start = w_weights[layer_id-1][neuron_in,neuron_out]
                    self.weights[w_id] = w
                # Adding biases
                b_id = (layer_id, neuron_out)
                b = self.m.addVar(vtype=GRB.INTEGER, name="b_%d-%d"%b_id, lb=-1, ub=1)
                b.start = w_biases[layer_id-1][neuron_out]
                self.biases[b_id] = b
                # Adding margins per neuron
                n_id = (layer_id, neuron_out)
                self.margins[n_id] = self.m.addVar(vtype=GRB.CONTINUOUS, name="m%d-%d"%n_id, lb=0)

        # Max margin loss
        self.loss = sum(list(self.margins.values()))


    def add_example(self, activations):
        # Adding the constraints given the activations
        neurons = {}
        for layer_id in range(1, len(self.layers)):
            for n_out in range(self.layers[layer_id]):
                # computing preactivation
                pre_activation = sum([activations[layer_id-1][n_in] * self.weights[(n_in,layer_id,n_out)] for n_in in range(self.layers[layer_id-1])])
                # adding the bias
                pre_activation += self.biases[(layer_id,n_out)]
                # computing activation
                margin = self.margins[(layer_id, n_out)]
                # NOTE: Change activation for TNN
                if activations[layer_id][n_out] == 1:
                        self.m.addConstr(pre_activation >= margin)
                elif activations[layer_id][n_out] == -1:
                    self.m.addConstr(pre_activation <= -margin - 1)
                else:  # 0
                    self.m.addConstr(pre_activation >= -margin)
                    self.m.addConstr(pre_activation <= margin)
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
        for layer_id in range(1,len(self.layers)):
            n_in = self.layers[layer_id-1]
            n_out = self.layers[layer_id]
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

class FeasCP:
    """
    CP model that looks for a feasible CP solution
    """
    def __init__(self, layers, data):
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
        #self.msol = self.m.solve(Workers=n_threads, TimeLimit=time_out*60)
        #self.msol = self.m.solve()

        # Is feasible?
        return bool(self.msol)

    def get_info(self):
        info_all = {}
        info = self.msol.get_solver_infos()
        info_all["num_branches"] = info['NumberOfBranches']
        info_all["num_vars"] = info['NumberOfIntegerVariables']

        if bool(self.msol):
            print("Solution found!")

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
