from task2 import main
from time import strftime
from itertools import product
import argparse
import os

def grid_search(args):
    timestamp = strftime("%Y%m%d-%H%M%S")

    # TODO: Could be made more efficient by precomputing the templates here are culling them from the list when necessary in the search loop.
    rotations = [0, 4, 8, 16]
    min_t_sizes = [8, 16, 32]
    max_t_sizes = [512, 256, 128]

    template_sizes = list(product(max_t_sizes, min_t_sizes))

    arg_config = {
        "train_dir": args.train_dir,
        "test_dir": args.test_dir,
        "anno_dir": args.anno_dir,
        "rotations": None,
        "max_temp_size": None,
        "min_temp_size": None,
        "threshold": 0.8,
        "silent": args.silent
    }
    
    search_len = len(rotations) * len(template_sizes)

    all_combinations = list(product(rotations, template_sizes))

    for i in range(args.i, len(all_combinations)):
        rot, (t_max, t_min) = all_combinations[i]
        arg_config["rotations"] = rot
        arg_config["max_temp_size"] = t_max
        arg_config["min_temp_size"] = t_min

        print(f"[{i+1}/{search_len}] Searching for config [r={rot}|t_max={t_max}|t_min={t_min}]")
        stats = main(**arg_config)
        total_t, average_t, accuracy, fn, tp, fp = stats

        print(f"Done. {stats}")

        with open(os.path.join(args.out, f"{timestamp}_task2_search_results.txt"), 'a') as f:
            f.write(f"{i}, {rot}, {t_max}, {t_min}, {total_t}, {average_t}, {accuracy}, {fn}, {tp}, {fp}\n")

def threshold_search(args):
    timestamp = strftime("%Y%m%d-%H%M%S")

    thresholds = [0.50,0.60,0.70,0.80,0.90]

    arg_config = {
        "train_dir": args.train_dir,
        "test_dir": args.test_dir,
        "anno_dir": args.anno_dir,
        "rotations": 16,
        "max_temp_size": 128,
        "min_temp_size": 32,
        "threshold": None,
        "silent": args.silent
    }
    
    search_len = len(thresholds)

    for i in range(args.i, len(thresholds)):
        arg_config["threshold"] = thresholds[i]

        print(f"[{i+1}/{search_len}] Searching for config [t={thresholds[i]}]")
        stats = main(**arg_config)
        total_t, average_t, accuracy, fn, tp, fp = stats

        print(f"Done. {stats}")

        with open(os.path.join(args.out, f"{timestamp}_task2_search_results.txt"), 'a') as f:
            f.write(f"{i}, {thresholds[i]}, {total_t}, {average_t}, {accuracy}, {fn}, {tp}, {fp}\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="Task 2 Grid Search")
    parser.add_argument('--train-dir', type=str, required=True, help="Training data directory containing *png* image templates.")
    parser.add_argument('--test-dir', type=str, required=True, help="Test data directory containing *png* test images.")
    parser.add_argument('--anno-dir', type=str, required=True, help="Test annotation directory containing *txt* files with comma separated values: '{template name}, {min_bound}, {max_bound}'.")

    parser.add_argument('-o', '--out', type=str, required=True, help="Results output directory.")

    parser.add_argument('-s', '--silent', action='store_true', help="Whether to run the tmeplate matching in silent mode or not.")
    parser.add_argument('-i', type=int, default=0, help="Parameter combination index to start at.")

    parser.add_argument('--search-threshold', action='store_true', help="Perform a threshold search instead for set params.")

    args = parser.parse_args()

    if args.search_threshold:
        threshold_search(args)
    else:
        grid_search(args)
