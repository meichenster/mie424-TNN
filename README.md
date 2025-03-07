# Training Binarized Neural Networks using MIP and CP

Binarized Neural Networks (BNNs) are an important class of neural network characterized by weights and activations restricted to the set {-1,+1}. BNNs provide simple compact descriptions and as such have a wide range of applications in low-power devices. This project investigates a model-based approach to training BNNs using constraint programming (CP), mixed-integer programming (MIP), and CP/MIP hybrids. We formulate the training problem as finding a set of weights that correctly classify the training set instances while optimizing objective functions that have been proposed in the literature as proxies for generalizability. Our experimental results on the MNIST digit recognition dataset suggest that---when training data is limited---the BNNs found by our hybrid approach generalize better than those obtained from gradient descent. A detailed description of all our models and main results can be found in the following paper ([link](http://www.cs.toronto.edu/~rntoro/docs/cp19_bnns.pdf)):

    @inproceedings{tor-etal-cp19,
        author = {Toro Icarte, Rodrigo and Illanes, León and Castro, Margarita P. and Cire, Andre and McIlraith, Sheila A. and Beck, J. Christopher},
        title     = {Training Binarized Neural Networks using MIP and CP},
        booktitle = {Proceedings of the 25th International Conference on Principles and Practice of Constraint Programming (CP)},
        year      = {2019}
    }

This code is meant to be a clean and usable version of our approach. If you find any bugs or have questions about it, please let us know. We'll be happy to help you!

## Installation instructions

You can clone this repository by running:

    git clone https://bitbucket.org/RToroIcarte/bnn.git

All our models require [Python 3.6](https://www.python.org/) with [numpy](http://www.numpy.org/) and [tensorflow 1.9.0](https://www.tensorflow.org/). Additionally, the MIP models are solved using [Gurobi 8.1](https://www.gurobi.com/) via its python interface [gurobipy](https://pypi.org/project/gurobipy/). The CP models are solved using [IBM ILOG CP Optimizer 12.8](https://www.ibm.com/analytics/cplex-cp-optimizer) via its python interface [docplex](https://pypi.org/project/docplex/). Gurobi and CP Optimizer are commercial solvers, but you can get free academic licenses if you are a faculty member, student, or staff of a recognized degree-granting academic institution.

## Running examples

To run one of our model-based approaches, move to the *src* directory and execute *run.py*. This script receives 6 parameters: The model to use (which can be "mip", "cp", "hw", or "ha"), the generalization criteria (which can be "min-w" for min-weight or "max-m" for max-margin), the number of hidden layers (by default, every hidden layer consists of 16 binary neurons), the number of training examples per class, the training set id (which is a non-negative integer), and the timeout in minutes. We use the training set id to systematically generate different training sets of the same size. For instance, the following command uses the MIP model with a min-weight objective function to train a BNN with no hidden layers using a training set with one example per class and a 5-minute time limit:

```
python3 run.py --model="mip" --obj="min-w" --hls=0 --exs=1 --ins=0 --to=5
```

To run one of our GD baselines, also execute *run.py* but using the following parameters: the model has to be either "gd_b" or "gd_t", select a learning rate, random seed, number of hidden layers, number of training examples per class, the training set id, and the timeout in minutes. The following example trains a BNN with one hidden layer using gd_b, a learning rate of 1e-3, a training set with one example per class, and a 5-minute time limit:

```
python3 run.py --model="gd_b" --lr=1e-3 --seed=0 --hls=1 --exs=1 --ins=0 --to=5
```

Once the time limit is reached, learning is stopped and the solution is tested over the full MNIST test set. The train and test performance will be printed on the console and many statistics from the learning process are saved on the './results' folder.

## Additional scripts

For reference only, we included the following three scripts that would allow you to replicate all the experiments that were included in our paper:

  - './scripts/run_gd.sh': runs all the GD experiments from the paper sequentially.
  - './scripts/run_mb.sh': runs all the model-based approaches from the paper sequentially.
  - './scripts/show.sh': reads the log files from the experiments that have been run and prints a summary table with the main statistics

However, running all those experiments sequentially would take around 1,200 days (but only 2 hours if you run them in parallel using 14,400 cores :thinking:). To test that the code is properly working, we included three lightweight versions of the previous scripts: './scripts/test_gd.sh', './scripts/test_mb.sh', and './scripts/test_show.sh'. They run experiments with up to 3 examples per class using a 5 minutes time limit. The 'test_gd.sh' takes around 7.5 hours and 'test_mb.sh' takes around 6.0 hours. Then, you can run 'test_show.sh' to observe a summary of the test performance obtained by each approach. For instance, ha_m (the HA model using the max-margin objective) should reach test performances between 20% and 30% (or timeout) whereas gd_t should reach test performances between 5% and 13%.
