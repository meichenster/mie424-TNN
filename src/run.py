from mnist import get_mnist_train_per_class, get_mnist_test_numpy
from bnn import BinarizedNetwork
import time, os, argparse, math
import numpy as np


def _get_one_hot_encoding(a, n_classes=10):
    b = np.zeros((a.size, n_classes))
    b[np.arange(a.size), a] = 1
    return b
        
def test_weights(net, weights, biases, images, labels):
    bnn = BinarizedNetwork(net)
    for i in range(len(weights)):
        bnn.update_layer(i, weights[i], biases[i])
    train_performance = bnn.test_network(images, labels)
    print("Train performance = %0.2f"%train_performance)
    images, labels = get_mnist_test_numpy()
    labels =  2.0*_get_one_hot_encoding(labels) - 1.0 # mapping labels to -1/1 vectors
    test_performance = bnn.test_network(images, labels)
    print("Test performance = %0.2f"%test_performance)

    # clossing the network sessions
    bnn.close()

    return train_performance, test_performance

def _save_mb_results(solver, n_hidden_layers, examples_per_class, examples_skip, time_out, info):
    folder_out = "../results/%s/%d/"%(solver,examples_per_class)
    file_out   = folder_out + "%d_%d_%d.txt"%(time_out, n_hidden_layers, examples_skip)
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    f = open(file_out,"w")
    f.write("solver=%s\n"%solver)
    f.write("n_hidden_layers=%d\n"%n_hidden_layers)
    f.write("examples_per_class=%d\n"%examples_per_class)
    f.write("examples_skip=%d\n"%examples_skip)
    f.write("time_out=%d\n"%time_out)
    for k in info:
        f.write(k+"="+str(info[k])+"\n")
    f.close()

def run_mb_experiment(solver, n_hidden_layers, examples_per_class, examples_skip, time_out):
    """
    This code runs a experiment using a model-based approach.
    @params
        - solver: 
            - this indicates which model-based approach to used:
            - The valid values are 
                "mip_w": mip model using a min-weight objective
                "mip_m": mip model using a max-margin objective
                "cp_w":  cp model using a min-weight objective
                "cp_m":  cp model using a max-margin objective
                "hw_w":  cp/mip hybrid model via warm-starts using a min-weight objective
                "hw_m":  cp/mip hybrid model via warm-starts using a max-margin objective
                "ha_w":  cp/mip hybrid model via fixed-activations using a min-weight objective
                "ha_m":  cp/mip hybrid model via fixed-activations using a max-margin objective
        - n_hidden_layers: 
            - this is the number of hidden layers to use 
            - Note that the number of hidden units is fixed to 16
        - examples_per_class: 
            - This indicates how many examples per class will be used to construct the training set in this experiment
        - examples_skip:
            - To construct different instance problems of the same size (without overlap), we skip examples_skip*examples_per_class examples when creating the training set
        - time_out:
            - This indicates the time out in minutes
    """

    print("solver:",solver)
    print("n_hidden_layers:",n_hidden_layers)
    print("examples_per_class:",examples_per_class)
    print("examples_skip:",examples_skip)
    print("time_out:",time_out)
    
    # Configuration
    n_hidden_units  = 16     # number of neurons per hidden layer
    n_threads       = 1      # number of threads

    # Code to run the experiment
    assert solver in ["mip_w", "mip_m", "cp_w", "cp_m", "hw_w", "hw_m", "ha_w", "ha_m"], "Selected solver is not supported yet!"

    if solver == "mip_w": from mip.mip_w import MultiLayerPerceptron
    if solver == "mip_m": from mip.mip_m import MultiLayerPerceptron
    if solver == "cp_w":  from cp.cp_w  import MultiLayerPerceptron
    if solver == "cp_m":  from cp.cp_m  import MultiLayerPerceptron
    if solver == "hw_w":  from hybrid.hw_w  import MultiLayerPerceptron
    if solver == "hw_m":  from hybrid.hw_m  import MultiLayerPerceptron
    if solver == "ha_w":  from hybrid.ha_w  import MultiLayerPerceptron
    if solver == "ha_m":  from hybrid.ha_m  import MultiLayerPerceptron

    # Net architecture
    net = [28*28] + [n_hidden_units for _ in range(n_hidden_layers)] + [10]
    n_train = 10 * examples_per_class

    # loading the training set
    images, labels = get_mnist_train_per_class(examples_per_class, examples_skip)
    labels =  2.0*_get_one_hot_encoding(labels) - 1.0 # mapping labels to -1/1 vectors

    # Training the network
    start = time.time()
    mlp = MultiLayerPerceptron(net, images, labels)
    for i in range(n_train):
        mlp.add_example(images[i], labels[i], show=False)
    is_sat = mlp.train(n_threads, time_out)
    total_time = ((time.time()-start)/60.0)
    print("-----------------------------")
    print("Total time = %0.2f[m]"%total_time)
    info = mlp.get_info()
    info["total_time"] = total_time
    info["is_sat"] = is_sat

    # Testing the solution
    if is_sat:
        weights, biases = mlp.get_weights()
        # Show standard performance
        train_performance, test_performance = test_weights(net, weights, biases, images, labels)
        info["train_performance"] = train_performance
        info["test_performance"] = test_performance
        info["weights"] = [w.tolist() for w in weights]
        info["biases"] = [b.tolist() for b in biases]
    else:
        print("TIMEOUT or INFEASIBLE! :O")

    # Saving the results
    _save_mb_results(solver, n_hidden_layers, examples_per_class, examples_skip, time_out, info)

    print("-----------------------------\n")

def _save_gd_results(solver, n_hidden_layers, examples_per_class, examples_skip, time_out, tf_seed, lr, info):
    folder_out = "../results/%s/%d/"%(solver,examples_per_class)
    log_rl = -round(math.log(lr,10))
    file_out   = folder_out + "%d_%d_%d_%d_%d.txt"%(time_out, n_hidden_layers, examples_skip, log_rl, tf_seed)
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    f = open(file_out,"w")
    f.write("solver=%s\n"%solver)
    f.write("n_hidden_layers=%d\n"%n_hidden_layers)
    f.write("examples_per_class=%d\n"%examples_per_class)
    f.write("examples_skip=%d\n"%examples_skip)
    f.write("time_out=%d\n"%time_out)
    for k in info:
        f.write(k+"="+str(info[k])+"\n")
    f.close()


def run_gd_experiments(solver, lr, tf_seed, n_hidden_layers, examples_per_class, examples_skip, time_out):
    """
    This code runs a experiment using a gradient-based approach.
    @params
        - solver: 
            - this indicates which gradient-based approach to used:
            - The valid values are 
                "gd_b": gradient descend using binary weights  (-1,1)
                "gd_t": gradient descend using ternary weights (-1,0,1)
        - lr: This is the learning rate
        - tf_seed: This is the random seed used to initialize the weights of the network
        - n_hidden_layers: 
            - this is the number of hidden layers to use 
            - Note that the number of hidden units is fixed to 16
        - examples_per_class: 
            - This indicates how many examples per class will be used to construct the training set in this experiment
        - examples_skip:
            - To construct different instance problems of the same size (without overlap), we skip examples_skip*examples_per_class examples when creating the training set
        - time_out:
            - This indicates the time out in minutes
    """

    print("solver:",solver)
    print("n_hidden_layers:",n_hidden_layers)
    print("examples_per_class:",examples_per_class)
    print("examples_skip:",examples_skip)
    print("time_out:",time_out)
    print("lr:", lr)
    print("tf_seed:", tf_seed)

    # Setting the network's architecture
    n_hidden_units  = 16
    n_input_units   = 28*28
    n_output_units  = 10
    net = [28*28] + [n_hidden_units for _ in range(n_hidden_layers)] + [10]

    # loading the training set
    train_data, train_labels = get_mnist_train_per_class(examples_per_class, examples_skip)
    train_labels = _get_one_hot_encoding(train_labels)

    # Training and testing the net
    from gd.gd_baseline import StandardNeuralNet
    nn = StandardNeuralNet(solver, n_input_units, n_hidden_units, n_hidden_layers, n_output_units, lr, tf_seed)
    start = time.time()
    is_sat = nn.train(train_data, train_labels, train_data, train_labels, time_out)
    total_time = ((time.time()-start)/60.0)
    
    # Testing the solution (weights of dead neurons are set to zero here)
    weights, biases = nn.get_weights()
    dead_inputs = np.all(train_data == train_data[0,:], axis = 0)
    for neuron_in in range(n_input_units):
        if dead_inputs[neuron_in]: weights[0][neuron_in,:] = np.zeros(net[1])
    train_performance, test_performance = test_weights(net, weights, biases, train_data, 2*train_labels-1)
    print("-----------------------------")
    print("Total time = %0.2f[m]"%total_time)
    print("Train performance = %0.2f"%train_performance)
    print("Test performance = %0.2f"%test_performance)

    # saving results
    info = {}
    info["lr"] = lr
    info["tf_seed"] = tf_seed
    info["train_performance"] = train_performance
    info["test_performance"] = test_performance
    info["total_time"] = total_time
    info["is_sat"] = (train_performance == 1.0)
    info["weights"] = [w.tolist() for w in weights]
    info["biases"] = [b.tolist() for b in biases]
    _save_gd_results(solver, n_hidden_layers, examples_per_class, examples_skip, time_out, tf_seed, lr, info)

    # clossing the network sessions
    nn.close()

    print("-----------------------------\n")


if __name__ == "__main__":

    # EXAMPLES: 
    #   python3 run.py --model="mip" --obj="min-w" --hls=0 --exs=1 --ins=0 --to=5
    #   python3 run.py --model="mip" --obj="max-m" --hls=0 --exs=1 --ins=0 --to=5
    #   python3 run.py --model="cp" --obj="min-w" --hls=0 --exs=1 --ins=0 --to=5
    #   python3 run.py --model="cp" --obj="max-m" --hls=0 --exs=1 --ins=0 --to=5
    #   python3 run.py --model="hw" --obj="min-w" --hls=2 --exs=1 --ins=0 --to=5
    #   python3 run.py --model="hw" --obj="max-m" --hls=2 --exs=1 --ins=0 --to=5
    #   python3 run.py --model="ha" --obj="min-w" --hls=2 --exs=1 --ins=0 --to=5
    #   python3 run.py --model="ha" --obj="max-m" --hls=2 --exs=1 --ins=0 --to=5
    #   python3 run.py --model="gd_b" --lr=1e-3 --seed=0 --hls=1 --exs=1 --ins=0 --to=5
    #   python3 run.py --model="gd_t" --lr=1e-3 --seed=0 --hls=1 --exs=1 --ins=0 --to=5

    # Getting params
    parser = argparse.ArgumentParser(prog="run", description='TODO')
    models    = ["mip", "cp", "hw", "ha", "gd_b", "gd_t"]
    objective = ["min-w","max-m"] # min-weight or max-margin
    parser.add_argument('--model', default='mip', type=str, help='Model to use. The options are: ' + str(models))
    parser.add_argument('--hls', default=1, type=int, help='number of hidden layers')
    parser.add_argument('--exs', default=1, type=int, help='number of training examples per class')
    parser.add_argument('--ins', default=0, type=int, help='TODO')
    parser.add_argument('--to', default=120, type=int, help='timeout (in minutes)')
    # [MIP/CP/HW/HA]-only params
    parser.add_argument('--obj', default='min-w', type=str, help='objective function for model-based approaches. The options are: ' + str(objective))
    # [GD_B,GD_T]-only params
    parser.add_argument('--seed', default=0, type=int, help='random seed for gd methods')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate for gd methods')

    # Checking that the value of the parameters are valid
    args = parser.parse_args()
    assert args.model in models,  f"invalid model {args.model}"
    assert args.obj in objective, f"invalid objective {args.obj}"
    assert args.to > 0,    "invalid timeout"
    assert args.exs > 0,   "invalid number of examples per class"
    assert args.ins >= 0,  "invalid instance id"
    assert args.hls >= 0,  "invalid number of hidden layers"
    assert args.seed >= 0, "invalid seed"
    assert args.lr > 0,    "invalid learning rate"
    
    # Running the experiment
    if args.model in ["mip", "cp", "hw", "ha"]:
        # Running model-based approach
        solver = f"{args.model}_{args.obj[-1]}"
        run_mb_experiment(solver, args.hls, args.exs, args.ins, args.to)
    else:
        # Running GD baseline
        run_gd_experiments(args.model, args.lr, args.seed, args.hls, args.exs, args.ins, args.to)
    
