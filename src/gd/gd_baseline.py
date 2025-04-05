"""
This code is based on the Uranus Yu's implementation of BNNs in Tensorflow. 
https://github.com/uranusx86/BinaryNet-on-tensorflow
"""

import time, random, math, os, argparse
import numpy as np
import tensorflow as tf
from sst3 import get_sst3_train_per_class, get_sst3_test_numpy
from bnn import BinarizedNetwork

class StandardNeuralNet:
    def __init__(self, solver, n_input_units, n_hidden_units, n_hidden_layers, n_output_units, lr, tf_seed):
        
        tf.set_random_seed(tf_seed)

        self.show = False
        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = 1
        config.inter_op_parallelism_threads= 1
        self.sess = tf.Session(config=config)
        
        # Parameters
        self.n_input_units   = n_input_units
        self.n_hidden_units  = n_hidden_units
        self.n_hidden_layers = n_hidden_layers
        self.n_output_units  = n_output_units
        
        # Input layer
        self.x  = tf.placeholder(tf.float32, [None, self.n_input_units], name='x')
        self.y_ = tf.placeholder(tf.float32, [None, self.n_output_units],  name='y_')
        self.lr = lr
        self.keep_prob  = tf.placeholder(tf.float32)
        self.accuracy   = None
        self.train_step = None

        if solver == "gd_b":
            self._mlp_architecture_binary()
        if solver == "gd_t":
            self._mlp_architecture_ternary()
            

    def _mlp_architecture_binary(self):
        n_input_units   = self.n_input_units
        n_hidden_units  = self.n_hidden_units
        n_hidden_layers = self.n_hidden_layers
        n_output_units  = self.n_output_units
        if n_hidden_layers == 0: n_hidden_units  = n_input_units
        self.weights = []
        self.biases  = []

        x = self.x
        for i in range(n_hidden_layers):
            # Fully connected layer
            n_in = n_input_units if i == 0 else n_hidden_units

            W_fc = binary_tanh_unit(weight_variable([n_in, n_hidden_units]))
            b_fc = binary_tanh_unit(bias_variable([n_hidden_units]))
            # NOTE: the 0.001 ensures that a zero preactivation is mapped to +1 activation
            x = binary_tanh_unit(tf.matmul(x, W_fc) + b_fc + 0.001) 
            self.weights.append(W_fc)
            self.biases.append(b_fc)

        # Fully connected layer 2 (Output layer)
        W_o = binary_tanh_unit(weight_variable([n_hidden_units, n_output_units]))
        b_o = binary_tanh_unit(bias_variable([n_output_units]))
        y = tf.matmul(x, W_o) + b_o

        self.weights.append(W_o)
        self.biases.append(b_o)

        # Evaluation functions (square hinge loss)
        self.hinge = tf.square(tf.losses.hinge_loss(self.y_, y))

        score = tf.multiply(2*self.y_-1, tf.sign(tf.sign(y)+0.1))
        score = tf.reduce_sum(score, 1) >= n_output_units
        self.accuracy = tf.reduce_mean(tf.cast(score, tf.float32), name='accuracy')

        # Training algorithm
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.hinge)

    def _mlp_architecture_ternary(self):
        n_input_units   = self.n_input_units
        n_hidden_units  = self.n_hidden_units
        n_hidden_layers = self.n_hidden_layers
        n_output_units  = self.n_output_units
        if n_hidden_layers == 0: n_hidden_units  = n_input_units
        self.weights = []
        self.biases  = []
        threshold = 0.5 # OPTIONAL parameter to pas sinto ternary tanh unit

        x = self.x
        for i in range(n_hidden_layers):
            # Fully connected layer
            n_in = n_input_units if i == 0 else n_hidden_units

            W1_fc = ternary_tanh_unit(weight_variable([n_in, n_hidden_units]))
            b1_fc = ternary_tanh_unit(bias_variable([n_hidden_units]))
            W2_fc = ternary_tanh_unit(weight_variable([n_in, n_hidden_units]))
            b2_fc = ternary_tanh_unit(bias_variable([n_hidden_units]))
            W_fc = (W1_fc + W2_fc)/2
            b_fc = (b1_fc + b2_fc)/2
            # NOTE: Changed the activation
            x = ternary_tanh_unit(tf.matmul(x, W_fc) + b_fc + 0.001)
            self.weights.append(W_fc)
            self.biases.append(b_fc)

        # Fully connected layer 2 (Output layer)
        W1_o = ternary_tanh_unit(weight_variable([n_hidden_units, n_output_units]))
        b1_o = ternary_tanh_unit(bias_variable([n_output_units]))
        W2_o = ternary_tanh_unit(weight_variable([n_hidden_units, n_output_units]))
        b2_o = ternary_tanh_unit(bias_variable([n_output_units]))
        W_o = (W1_o + W2_o)/2
        b_o = (b1_o + b2_o)/2
        y = tf.matmul(x, W_o) + b_o

        self.weights.append(W_o)
        self.biases.append(b_o)

        # Evaluation functions (square hinge loss)
        self.hinge = tf.square(tf.losses.hinge_loss(self.y_, y))

        # NOTE: ORIGINAL SCORE
        # score = tf.multiply(2*self.y_-1, tf.sign(tf.sign(y)+0.1))
        # score = tf.reduce_sum(score, 1) >= n_output_units
        # self.accuracy = tf.reduce_mean(tf.cast(score, tf.float32), name='accuracy')

        # NOTE: NEW SCORE TO REWARD PARTIAL CORRECTNESS
        correct_prediction = tf.equal(y, self.y_)
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(correct_prediction)

        # Training algorithm
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.hinge)

    def get_accuracy(self, train_data, train_labels):
        return self.sess.run(self.accuracy, feed_dict={self.x: train_data, self.y_: train_labels, self.keep_prob: 1.0})

    def train(self, train_data, train_labels, test_data, test_labels, time_out):
        start = time.time()
        # Training steps
        self.sess.run(tf.global_variables_initializer())
        step = 0
        while (time.time()-start) <= time_out * 60:
            batch_xs, batch_ys = train_data, train_labels
            if (step % 100) == 0:
                train_performance, loss = self.sess.run([self.accuracy,self.hinge], feed_dict={self.x: train_data, self.y_: train_labels, self.keep_prob: 1.0})
                if self.show: print(step, train_performance, "%0.4f"%loss)
                if loss == 0.0:
                    # A zero loss will not generate gradients
                    break
            self.sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys, self.keep_prob: 1.0})
            step += 1

    def get_weights(self):
        """
        Returns the best weights found so far
        """
        w_ret = self.sess.run(self.weights)
        b_ret = self.sess.run(self.biases)
        return w_ret, b_ret

    def close(self):
        self.sess.close()

def hard_sigmoid(x):
    return tf.clip_by_value((x + 1.)/2., 0, 1)

def round_through(x):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    a op that behave as f(x) in forward mode,
    but as g(x) in the backward mode.
    '''
    rounded = tf.round(x) # tf round sends 0.5 to 0
    return x + tf.stop_gradient(rounded-x)

# The neurons' activations binarization function
# It behaves like the sign function during forward propagation
# And like:
#   hard_tanh(x) = 2*hard_sigmoid(x)-1
# during back propagation
def binary_tanh_unit(x):
    return 2.*round_through(hard_sigmoid(x))-1.

# NOTE: New ternary activation function using a threshold to output {-1, 0, 1}
def ternary_tanh_unit(x, threshold=0.5):
    # Get input shape
    x_shape = tf.shape(x)

    # Tensor with x's shape filled with constants {-1, 0, 1}
    ones = tf.ones(x_shape, dtype=tf.float32)
    zeros = tf.zeros(x_shape, dtype=tf.float32)

    # Activation
    ternary = tf.where(x > threshold, ones, 
                       tf.where(x < -threshold, -ones, zeros))
    
    # Same as round_through function
    return x + tf.stop_gradient(ternary - x)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
