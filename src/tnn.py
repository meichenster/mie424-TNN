"""
The following code was adapted from Toro Icarte et. al's paper on BNNs and modified for
a ternary neural network (TNN).

This code is used for evaluating the test performance of the TNNs found by each approach.
The "TernaryNeuralNetwork" class creates the tnn, imports the weights given by a model, 
and evaluate the network performance over a dataset.

This class is also used by the HA methods to compute the layer activations.
"""
import tensorflow as tf
import numpy as np

class TernaryNeuralNetwork:

    def __init__(self, neurons_per_layer):
        """
        "neurons_per_layer" is a list of numbers indicating how many neurons each layer has
          e.g. [2,2,1] -> 2 input neurons, then 2 neurons on a hidden layer, and one output neuron
        """
        self.neurons_per_layer = neurons_per_layer
        self._create_network(neurons_per_layer)

    def _create_network(self, neurons_per_layer):
        self.sess   = tf.Session()
        n_inputs  = neurons_per_layer[0]
        n_outputs = neurons_per_layer[-1]
        self.seqs   = tf.placeholder(tf.float32, shape=[None, n_inputs])
        self.labels = tf.placeholder(tf.float32, shape=[None, n_outputs])
        self.weight_values = tf.placeholder(tf.float32)
        self.bias_values   = tf.placeholder(tf.float32)

        # list of operators to update weights on a given layer
        self.update_weights = [] 
        self.patterns = [self.seqs]
        self.weights = []
        self.biases = []

        # Adding hidden layers
        x = self.seqs
        for layer_id in range(1, len(neurons_per_layer)):
            x_prev = x

            # Computing hidden patterns
            fc_shape = [neurons_per_layer[layer_id-1], neurons_per_layer[layer_id]]
            
            n_outputs = fc_shape[1]
            W = _weight_variable(fc_shape)
            b = _bias_variable([n_outputs])
            a = tf.matmul(x_prev, W) + b

            # NOTE: Edited to turn into TNN
            # x = tf.sign(tf.sign(a) + 0.1) 
            x = tf.where(a > 0.001, tf.ones_like(a), tf.where(a < -0.001, tf.ones_like(a) * -1.0, tf.zeros_like(a)))

            # Methods to set weights to values that were found using MIP
            w_copy = tf.assign(W, tf.reshape(self.weight_values, fc_shape))
            b_copy = tf.assign(b, tf.reshape(self.bias_values, [n_outputs]))
            self.update_weights.append([w_copy, b_copy])

            # Methods to get the weights and biases from this layer
            # (this allows for computing the costs of each pattern)
            self.weights.append(W)
            self.biases.append(b)

            # Methods to get the activations in the current layer
            self.patterns.append(x)

        # Computing performance
        threshold = 2
        score = tf.multiply(self.labels, x)
        score = tf.reduce_sum(score, 1) >= n_outputs - threshold
        self.performance = tf.reduce_mean(tf.cast(score, tf.float32)) # Should return a scalar!

        # initialize variables
        self.sess.run(tf.global_variables_initializer())

    def get_activations(self, seqs):
        return self.sess.run(self.patterns, {self.seqs: seqs})

    def get_number_of_weights(self):
        total = 0
        for i in range(1,len(self.neurons_per_layer)):
            weights, biases = self.sess.run([self.weights[i-1], self.biases[i-1]], {})
            total += np.sum(np.abs(weights)) + np.sum(np.abs(biases))
        return total        

    def test_network(self, seqs, labels):
        """
        The "all-good" evaluation metric:
            A multiclass instance is considered to be correctly classified iff
            its one-hot embedding is perfectly outputed by the network.
            Hence, an always yes classifier would have 0.0 performance for a 10-classes problem
        """

        return self.sess.run(self.performance, {self.seqs: seqs, self.labels:labels})

    def update_layer(self, layer_id, weights, biases):
        self.sess.run(self.update_weights[layer_id], {self.weight_values: weights, self.bias_values:biases})

    def close(self):
        self.sess.close()

        
def _weight_variable(shape):
    initial = tf.random_uniform(shape, minval=-1, maxval=2, dtype=tf.int32)
    initial = tf.cast(initial, tf.float32)
    return tf.Variable(initial)

def _bias_variable(shape):
    initial = tf.random_uniform(shape, minval=-1, maxval=2, dtype=tf.int32)
    initial = tf.cast(initial, tf.float32)
    return tf.Variable(initial)
