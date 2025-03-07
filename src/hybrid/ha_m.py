"""
This is a CP/MIP hybrid that uses fixed-activations and aims to find the BNNs 
with the maximal sum of the neuron's margins that fits the training set.
"""
from hybrid.hm import HybridMethod

class MultiLayerPerceptron:
    def __init__(self, layers, data, labels):
        self.hm = HybridMethod("ha_m", layers, data, labels)

    def add_example(self, data, label, show = False):
        self.hm.add_example(data, label, show)

    def train(self, n_threads, time_out):
        """
        Returns True if no feasible solution exists
        """
        return self.hm.train(n_threads, time_out)

    def get_info(self):
        return self.hm.get_info()

    def get_weights(self):
        """
        Returns the best weights found so far
        """
        return self.hm.get_weights()
