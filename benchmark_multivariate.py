import os
import pickle 

import numpy as np

from basic_models import predict_naive, predict_theta, predict_arima, predict_shift, predict_nbeats, predict_gru, predict_dlinear, predict_timesnet, predict_fedformer, predict_nonstationary_transformer
from data_wrangling import read_dataset_mts, flatten, scale, inverse_scale_predictions
from dataset_config import Config
from evaluation import evaluate_errors

results_path = "results_multivariate/"

H_SHORT = [24,36,48,60]
BH_SHORT = [36]
H_LONG = [96,192,336,720]
BH_LONG = [96]

# Error metrics used for model evaluation and comparison
error_metrics = ["mae", "mse", "smape", "mase", "owa"]

# Benchmark algorithms
algorithms = {
    "Naive": predict_naive,
    "Theta": predict_theta,
    "SHIFT": predict_shift,
    "N-BEATS": predict_nbeats,
    "GRU": predict_gru,
    "DLinear": predict_dlinear,
    "NonStationaryT": predict_nonstationary_transformer,
    "FEDformer": predict_fedformer,
}

# Datasets for the experiment
short_dataset_names = Config.multivariate_dataset_names_short
long_dataset_names = Config.multivariate_dataset_names_long

# Random states for non-deterministic models
random_states = np.arange(1)

def benchmark(horizon, backhorizon, algorithms, datasets, error_metrics, random_states):
    all_errors = {}

    for dataset_name in datasets:
        dataset = read_dataset_mts(dataset_name, Config.train_sizes[dataset_name], Config.val_sizes[dataset_name], Config.stride_lengths[dataset_name], backhorizon, horizon)

        dataset_flat, _ = flatten(dataset)
        dataset_scaled, scalers = scale(dataset)
        dataset_scaled_flat, original_shapes = flatten(dataset_scaled)

        print(f"Starting dataset {dataset_name}\n")
        
        if len(dataset["y_train"]) == 0:
            print(f"All time series in dataset {dataset_name} are too short to generate train/test instances, skipping dataset")
            continue

        algo_errors = {}
        for algo_name, algorithm in algorithms.items():
            print(f"Starting {algo_name}")
            if algo_name == "SHIFT":
                y_pred, _ = algorithm(dataset_scaled)
            elif algo_name in ["NonStationaryT", "FEDformer", "TimesNet"]:
                y_pred, _ = algorithm(dataset_scaled, Config.d_ff[dataset_name], Config.d_model[dataset_name], random_states)
            elif algo_name in ["DLinear", "N-BEATS", "GRU"]:
                y_pred, _ = algorithm(dataset_scaled, random_states)
            else:
                y_pred, _ = algorithm(dataset_scaled)

            errors = evaluate_errors(dataset_scaled["X_test"], dataset_scaled["y_test"], y_pred, error_metrics, dataset=dataset_scaled_flat)
            algo_errors[algo_name] = errors
            print(errors)
            print(f"DONE {algo_name}")
        all_errors[dataset_name] = algo_errors
        print(f"DONE with dataset {dataset_name}\n")
    print(f"DONE with configuration - H: {horizon}; BH: {backhorizon}")
    return all_errors

def update_results_file(results, filename):
    if os.path.exists(filename):
        with open(filename, "rb") as file:
            e = pickle.load(file)
            for dataset in results.keys():
                if dataset not in e:
                    e[dataset] = results[dataset]
                else:
                    for algo in results[dataset].keys():
                        e[dataset][algo] = results[dataset][algo]
        with open(filename, "wb") as file:
            pickle.dump(e, file)                
    else:
        with open(filename, "wb") as file:
            pickle.dump(results, file)

if __name__ == "__main__":
    # Short horizon datasets
    for i,h in enumerate(H_SHORT):
        for j,bh in enumerate(BH_SHORT):
            errors = benchmark(h, bh, algorithms, short_dataset_names, error_metrics, random_states)

            error_filename = f"{results_path}errors_h_{h}_bh_{bh}.pkl"
            update_results_file(errors, error_filename)

    # Long horizon datasets
    for i,h in enumerate(H_LONG):
        for j,bh in enumerate(BH_LONG):
            errors = benchmark(h, bh, algorithms, long_dataset_names, error_metrics, random_states)

            error_filename = f"{results_path}errors_h_{h}_bh_{bh}.pkl"
            update_results_file(errors, error_filename)