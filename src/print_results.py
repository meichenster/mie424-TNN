"""
This code computes average results using the experiment's log files.
"""
import time, os, argparse, math
import numpy as np

def read_model_based_results(file):
    # reads the results from a model based approach
    f = open(file)
    for l in f:
        k,v = l.rstrip().split("=")
        if k == "is_sat": is_sat = eval(v)
        if k == "train_performance": train_performance = eval(v)
        if k == "test_performance": test_performance = eval(v)
        if k == "n_hidden_layers": n_hidden_layers = eval(v)        
        if k == "examples_per_class": examples_per_class = eval(v)
        if k == "examples_skip": examples_skip = eval(v)
        if k in ["objective", "mip_objective"]: objective = float(f"{eval(v):0.2f}")
    f.close()
    if is_sat: return n_hidden_layers, examples_per_class, examples_skip, train_performance, test_performance, objective
    return n_hidden_layers, examples_per_class, examples_skip, '-', '-', '-'

def read_gradient_based_results(file):
    # reads the results from a model based approach
    f = open(file)
    for l in f:
        k,v = l.rstrip().split("=")
        if k == "train_performance": train_performance = eval(v)
        if k == "test_performance": test_performance = eval(v)
        if k == "n_hidden_layers": n_hidden_layers = eval(v)        
        if k == "examples_per_class": examples_per_class = eval(v)
        if k == "examples_skip": examples_skip = eval(v)
        if k == "lr": lr = eval(v)
        
    f.close()
    return n_hidden_layers, examples_per_class, examples_skip, lr, train_performance, test_performance

def get_average(l):
    return float(f"{sum(l)/len(l):0.4f}")

def compute_average_over_random_seeds(raw_results):
    # computing the average across different seeds
    results_lr = []
    current = None, None, None, None
    for hls, exs, ins, lr, train, test in raw_results:
        if (hls, exs, ins, lr) != current:
            if current[0] is not None:
                results_lr.append(current+(get_average(train_values),get_average(test_values)))
            train_values = []
            test_values  = []
            current      = hls, exs, ins, lr
        train_values.append(train)
        test_values.append(test)
    results_lr.append(current+(get_average(train_values),get_average(test_values)))
    return results_lr

def compute_maximum_over_lrs(results_lr):
    results = []
    current = None, None, None
    for hls, exs, ins, lr, train, test in results_lr:
        if (hls, exs, ins) != current:
            if current[0] is not None:
                results.append(current+(get_average(train_values),get_average(test_values)))
            train_values = []
            test_values  = []
            current      = hls, exs, ins
        train_values.append(train)
        test_values.append(test)
    results.append(current+(max(train_values),max(test_values)))
    return results

def show_results(solver, time_out):
    print("-----------------------------------------------------")
    print(f"Results {solver} using a {time_out} min timeout...")

    # Collecting data from all the available results
    folder = f"../results/{solver}/"
    subfolders = [folder+subfolder+'/' for subfolder in os.listdir(folder) if os.path.isdir(folder+subfolder) and subfolder.isdigit()]
    experients = [subfolder + exp for subfolder in subfolders for exp in os.listdir(subfolder) if exp.endswith('.txt') and exp.startswith(f'{time_out}_')]
    if solver.startswith('gd'):
        # gradient descent results
        raw_results = [read_gradient_based_results(exp) for exp in experients]
        raw_results.sort()
        # reporting the performance of the best learning rate only
        results_lr = compute_average_over_random_seeds(raw_results)
        results = compute_maximum_over_lrs(results_lr)
    else:
        # model based results
        results = [read_model_based_results(exp) for exp in experients]
    results.sort()

    # printing the results
    header = "hls\texs\tins\ttrain\ttest"
    if solver.endswith("_w"): header += "\tmin-w"
    if solver.endswith("_m"): header += "\tmax-m"
    print(header)
    for result in results:
        print('\t'.join([str(r) for r in result]))


if __name__ == "__main__":

    # EXAMPLES: 
    #   python3 print_results.py --model="mip" --obj="min-w" --to=5
    #   python3 print_results.py --model="mip" --obj="max-m" --to=5
    #   python3 print_results.py --model="cp" --obj="min-w" --to=5
    #   python3 print_results.py --model="cp" --obj="max-m" --to=5
    #   python3 print_results.py --model="hw" --obj="min-w" --to=5
    #   python3 print_results.py --model="hw" --obj="max-m" --to=5
    #   python3 print_results.py --model="ha" --obj="min-w" --to=5
    #   python3 print_results.py --model="ha" --obj="max-m" --to=5
    #   python3 print_results.py --model="gd_b" --to=5
    #   python3 print_results.py --model="gd_t" --to=5

    # Getting params
    parser = argparse.ArgumentParser(prog="run", description='TODO')
    models    = ["mip", "cp", "hw", "ha", "gd_b", "gd_t"]
    objective = ["min-w","max-m"] # min-weight or max-margin
    parser.add_argument('--model', default='mip', type=str, help='Model to use. The options are: ' + str(models))
    parser.add_argument('--to', default=120, type=int, help='timeout (in minutes)')
    parser.add_argument('--obj', default='min-w', type=str, help='objective function for model-based approaches. The options are: ' + str(objective))

    # Checking that the value of the parameters are valid
    args = parser.parse_args()
    assert args.model in models,  f"invalid model {args.model}"
    assert args.obj in objective, f"invalid objective {args.obj}"
    assert args.to > 0,    "invalid timeout"
    
    # Running the experiment
    if args.model in ["mip", "cp", "hw", "ha"]:
        solver = f"{args.model}_{args.obj[-1]}"
    else:
        solver = args.model
    show_results(solver, args.to)