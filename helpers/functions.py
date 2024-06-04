import pandas as pd
from datetime import timedelta
import json

from helpers.paths import *

from sklearn.preprocessing import MinMaxScaler


def extract_target_consumption(df, start_time, end_time, household_id):
    return df.loc[(df.index >= start_time) & (df.index < end_time), household_id]


def extract_historical_series(df, end_time, nb_historical_days, household_id):
    start_time = end_time - timedelta(days=nb_historical_days)
    return extract_target_consumption(df, start_time, end_time, household_id)


def normalize_data(raw_data, method):
    if method == "min_max":
        scaler = MinMaxScaler()
        normalized_data_tmp = scaler.fit_transform(raw_data.transpose())
        return normalized_data_tmp.transpose()
    else:
        print("Not implemented yet")


def add_lags(df, col, lag):
    for i in range(lag, 0, -1):
        df[f'{col}_lag_{i}'] = df[col].shift(i)
    return df


def convert_series_to_table_format(series, lag, operations):
    df = pd.DataFrame(series.rename("consumption"))
    df = add_lags(df, "consumption", lag=lag)

    for operation in operations:
        if operation == "add_time_variables":
            print("Implement add_time_variables")
        else:
            print("Not implemented yet")

    df = df.dropna()

    return df


def get_best_params(model, dataset):
    hyperparameter_dictionary = {}
    optimal_score = 99999999

    if model == "historical_sampling":

        performance_dictionary_hs_path = os.path.join(model_dir, f'performance_dictionary_{dataset}_historical_sampling.json')

        with open(performance_dictionary_hs_path, 'r') as fp:
            performance_dictionary = json.load(fp)

        optimal_historical_sampling_days = None

        for key in performance_dictionary:
            value = performance_dictionary[key]
            if value["performance"] < optimal_score:
                optimal_score = value["performance"]
                optimal_historical_sampling_days = value["historical_sampling_days"]

        hyperparameter_dictionary["historical_sampling_days"] = optimal_historical_sampling_days

        print(f"Best parameters for historical sampling method (score = {optimal_score}): "
              f"historical_sampling_days = {optimal_historical_sampling_days}")

    elif model == "global_method":

        performance_dictionary_gm_path = os.path.join(model_dir, f'performance_dictionary_{dataset}_global_method.json')

        with open(performance_dictionary_gm_path, 'r') as fp:
            performance_dictionary = json.load(fp)

        optimal_historical_days = None
        optimal_nb_neighbors = None
        optimal_global_prediction_method = None

        for key in performance_dictionary:
            value = performance_dictionary[key]
            if value["performance"] < optimal_score:
                optimal_score = value["performance"]
                optimal_historical_days = value["historical_days"]
                optimal_nb_neighbors = value["nb_neighbors"]
                optimal_global_prediction_method = value["global_prediction_method"]

        hyperparameter_dictionary["historical_days"] = optimal_historical_days
        hyperparameter_dictionary["nb_neighbors"] = optimal_nb_neighbors
        hyperparameter_dictionary["global_prediction_method"] = optimal_global_prediction_method

        print(f"Best parameters for global method (score = {optimal_score}): historical_days = "
              f"{optimal_historical_days}, nb_neighbors = {optimal_nb_neighbors}, global_prediction_method = "
              f"{optimal_global_prediction_method}")

    elif model == "random_forest":

        performance_dictionary_rf_path = os.path.join(model_dir, f'performance_dictionary_{dataset}_random_forest.json')

        with open(performance_dictionary_rf_path, 'r') as fp:
            performance_dictionary = json.load(fp)

        optimal_lag = None
        optimal_nb_trees = None
        optimal_max_depth = None

        for key in performance_dictionary:
            value = performance_dictionary[key]
            if value["performance"] < optimal_score:
                optimal_score = value["performance"]
                optimal_lag = value["lag"]
                optimal_nb_trees = value["nb_trees"]
                optimal_max_depth = value["max_depth"]

        hyperparameter_dictionary["lag"] = optimal_lag
        hyperparameter_dictionary["nb_trees"] = optimal_nb_trees
        hyperparameter_dictionary["max_depth"] = optimal_max_depth

        print(f"Best parameters for random forest method (score = {optimal_score}): lag = {optimal_lag}, nb_trees = "
              f"{optimal_nb_trees}, max_depth = {optimal_max_depth}")

    elif model == "quantile_regression":

        performance_dictionary_qr_path = os.path.join(model_dir, f'performance_dictionary_{dataset}_quantile_regression.json')

        with open(performance_dictionary_qr_path, 'r') as fp:
            performance_dictionary = json.load(fp)

        optimal_lag = None
        optimal_l2_penalty = None
        optimal_solver = None

        for key in performance_dictionary:
            value = performance_dictionary[key]
            if value["performance"] < optimal_score:
                optimal_score = value["performance"]
                optimal_lag = value["lag"]
                optimal_l2_penalty = value["l2_penalty"]
                optimal_solver = value["solver"]

        hyperparameter_dictionary["lag"] = optimal_lag
        hyperparameter_dictionary["l2_penalty"] = optimal_l2_penalty
        hyperparameter_dictionary["solver"] = optimal_solver

        print(f"Best parameters for quantile regression method (score = {optimal_score}): lag = {optimal_lag}, l2_penalty = "
              f"{optimal_l2_penalty}, solver = {optimal_solver}")

    elif model == "neural_network":

        performance_dictionary_nn_path = os.path.join(model_dir, f'performance_dictionary_{dataset}_neural_network.json')

        with open(performance_dictionary_nn_path, 'r') as fp:
            performance_dictionary = json.load(fp)

        optimal_lag = None
        optimal_neurons_layer_1 = None
        optimal_neurons_layer_2 = None
        optimal_epochs = None

        for key in performance_dictionary:
            value = performance_dictionary[key]
            if value["performance"] < optimal_score:
                optimal_score = value["performance"]
                optimal_lag = value["lag"]
                optimal_neurons_layer_1 = value["neurons_layer_1"]
                optimal_neurons_layer_2 = value["neurons_layer_2"]
                optimal_epochs = value["epochs"]

        hyperparameter_dictionary["lag"] = optimal_lag
        hyperparameter_dictionary["neurons_layer_1"] = optimal_neurons_layer_1
        hyperparameter_dictionary["neurons_layer_2"] = optimal_neurons_layer_2
        hyperparameter_dictionary["epochs"] = optimal_epochs

        print(f"Best parameters for neural network method (score = {optimal_score}): lag = {optimal_lag}, "
              f"neurons_layer_1 = {optimal_neurons_layer_1}, neurons_layer_2 = {optimal_neurons_layer_2}, "
              f"epochs = {optimal_epochs}")

    return hyperparameter_dictionary


def get_periods(list_with_dates):
    periods = []
    for i in range(len(list_with_dates) - 1):
        period = (list_with_dates[i], list_with_dates[i + 1])
        periods.append(period)
    return periods
