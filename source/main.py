import warnings

import pandas as pd
import numpy as np
import yaml
import json
from datetime import datetime


from helpers.paths import *
from helpers.classes import *
from helpers.functions import get_best_params, get_periods
from source.Solver import Solver

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    with open(config_file_path, "r") as config_file:
        config = yaml.load(config_file, yaml.FullLoader)

    dataset = config["dataset"]

    if dataset == "london":
        df = pd.read_csv(london_data_path, index_col=0).transpose()
        missing_count = df.isnull().sum()
        columns_to_exclude = missing_count > 100
        filtered_df = df.loc[:, ~columns_to_exclude]
        df = filtered_df.fillna(method='ffill')
        with open(households_split_path, 'r') as households_split:
            loaded_lists = json.load(households_split)
            households = loaded_lists['list_global_local']
    elif dataset == "irish":
        df = pd.read_csv(irish_data_path, index_col=0).transpose()
        households = df.columns.tolist()


        ("bla")
    else:
        df = pd.DataFrame()
        raise Exception("Incorrect dataframe specified")
    df.index = pd.to_datetime(df.index)


    households = households[:2]

    validation_periods = get_periods(config["validation_periods"])
    prediction_periods = get_periods(config["prediction_periods"])

    if config["hyperparameter_scan"]:

        if "historical_sampling" in config["model_hyperparameter_scan"]:
            performance_dictionary = {}
            counter_hs = 0
            for historical_sampling_day in config["historical_sampling_days"]:
                print(f"Hyperparameter scan started for historical sampling with following parameters: "
                      f"historical_sampling_days = {historical_sampling_day}")

                parameters = Parameters(
                    historical_sampling_days=historical_sampling_day
                )

                performance_all_validation_periods = []
                elapsed_time_all_validation_periods = []

                for validation_period in validation_periods:

                    # Define validation period
                    date_information_validation = DateInformation(
                        training_start=config["training_start"],
                        prediction_start=validation_period[0],
                        prediction_end=validation_period[1]
                    )

                    solver = Solver(
                        df=df,
                        dataset=dataset,
                        households=households,
                        config=config,
                        date_information=date_information_validation,
                        params=parameters,
                        hyperparameter_scan=config["hyperparameter_scan"],
                        model_hyperparameter_scan="historical_sampling"
                    )

                    performance, elapsed_time = solver.run()

                    performance_all_validation_periods.append(performance)
                    elapsed_time_all_validation_periods.extend(elapsed_time)

                performance_dictionary[f"historical_sampling_{counter_hs}"] = {
                    "historical_sampling_days": historical_sampling_day,
                    "performance": np.mean(performance_all_validation_periods),
                    "elapsed_time": elapsed_time_all_validation_periods
                }

                counter_hs += 1

                performance_dictionary_hs_path = os.path.join(model_dir, f'performance_dictionary_{dataset}_historical_sampling.json')

                with open(performance_dictionary_hs_path, 'w') as fp:
                    json.dump(performance_dictionary, fp)

            hyperparameters = get_best_params(model="historical_sampling", dataset=dataset)

            parameters = Parameters(
                historical_sampling_days=hyperparameters["historical_sampling_days"]
            )

            for prediction_period in prediction_periods:

                # Define prediction period
                date_information_prediction = DateInformation(
                    training_start=config["training_start"],
                    prediction_start=prediction_period[0],
                    prediction_end=prediction_period[1]
                )

                solver = Solver(
                    df=df,
                    dataset=dataset,
                    households=households,
                    config=config,
                    date_information=date_information_prediction,
                    params=parameters,
                    hyperparameter_scan=config["hyperparameter_scan"],
                    model_hyperparameter_scan="historical_sampling"
                )

                _, _ = solver.run()

        if "global_method" in config["model_hyperparameter_scan"]:
            performance_dictionary = {}
            counter_gb = 0
            for historical_days in config["historical_days"]:
                for nb_neighbors in config["nb_neighbors"]:
                    for global_prediction_method in config["global_prediction_method"]:

                        print(f"Hyperparameter scan started for global method with following parameters: "
                              f"historical_days = {historical_days}, nb_neighbors = {nb_neighbors}, "
                              f"global_prediction_method = {global_prediction_method}")

                        parameters = Parameters(
                            historical_days=historical_days,
                            nb_neighbors=nb_neighbors,
                            global_prediction_method=global_prediction_method
                        )

                        performance_all_validation_periods = []
                        elapsed_time_all_validation_periods = []

                        for validation_period in validation_periods:

                            # Define validation period
                            date_information_validation = DateInformation(
                                training_start=config["training_start"],
                                prediction_start=validation_period[0],
                                prediction_end=validation_period[1]
                            )

                            solver = Solver(
                                df=df,
                                dataset=dataset,
                                households=households,
                                config=config,
                                date_information=date_information_validation,
                                params=parameters,
                                hyperparameter_scan=config["hyperparameter_scan"],
                                model_hyperparameter_scan="global_method"
                            )

                            performance, elapsed_time = solver.run()

                            performance_all_validation_periods.append(performance)
                            elapsed_time_all_validation_periods.extend(elapsed_time)

                        performance_dictionary[f"global_method_{counter_gb}"] = {
                            "historical_days": historical_days,
                            "nb_neighbors": nb_neighbors,
                            "global_prediction_method": global_prediction_method,
                            "performance": np.mean(performance_all_validation_periods),
                            "elapsed_time": elapsed_time_all_validation_periods
                        }

                        counter_gb += 1

                        performance_dictionary_gm_path = os.path.join(model_dir, f'performance_dictionary_{dataset}_global_method.json')

                        with open(performance_dictionary_gm_path, 'w') as fp:
                            json.dump(performance_dictionary, fp)

            hyperparameters = get_best_params(model="global_method", dataset=dataset)

            parameters = Parameters(
                historical_days=hyperparameters["historical_days"],
                nb_neighbors=hyperparameters["nb_neighbors"],
                global_prediction_method=hyperparameters["global_prediction_method"]
            )

            for prediction_period in prediction_periods:

                # Define prediction period
                date_information_prediction = DateInformation(
                    training_start=config["training_start"],
                    prediction_start=prediction_period[0],
                    prediction_end=prediction_period[1]
                )

                solver = Solver(
                    df=df,
                    dataset=dataset,
                    households=households,
                    config=config,
                    date_information=date_information_prediction,
                    params=parameters,
                    hyperparameter_scan=config["hyperparameter_scan"],
                    model_hyperparameter_scan="global_method"
                )

                _, _ = solver.run()

        if "random_forest" in config["model_hyperparameter_scan"]:
            performance_dictionary = {}
            counter_rf = 0
            for lag in config["lag"]:
                for nb_trees in config["nb_trees"]:
                    for max_depth in config["max_depth"]:
                        print(f"Hyperparameter scan started for random forest with following parameters: "
                              f"lag = {lag}, nb_trees = {nb_trees}, max_depth = {max_depth}")

                        parameters = Parameters(
                            lag=lag,
                            nb_trees=nb_trees,
                            max_depth=max_depth
                        )

                        performance_all_validation_periods = []
                        elapsed_time_all_validation_periods = []

                        for validation_period in validation_periods:

                            # Define validation period
                            date_information_validation = DateInformation(
                                training_start=config["training_start"],
                                prediction_start=validation_period[0],
                                prediction_end=validation_period[1]
                            )

                            solver = Solver(
                                df=df,
                                dataset=dataset,
                                households=households,
                                config=config,
                                date_information=date_information_validation,
                                params=parameters,
                                hyperparameter_scan=config["hyperparameter_scan"],
                                model_hyperparameter_scan="random_forest"
                            )

                            performance, elapsed_time = solver.run()

                            performance_all_validation_periods.append(performance)
                            elapsed_time_all_validation_periods.extend(elapsed_time)

                        performance_dictionary[f"random_forest_{counter_rf}"] = {
                            "lag": lag,
                            "nb_trees": nb_trees,
                            "max_depth": max_depth,
                            "performance": np.mean(performance_all_validation_periods),
                            "elapsed_time": elapsed_time_all_validation_periods
                        }

                        counter_rf += 1

                        performance_dictionary_rf_path = os.path.join(model_dir, f'performance_dictionary_{dataset}_random_forest.json')

                        with open(performance_dictionary_rf_path, 'w') as fp:
                            json.dump(performance_dictionary, fp)

            hyperparameters = get_best_params(model="random_forest", dataset=dataset)

            parameters = Parameters(
                lag=hyperparameters["lag"],
                nb_trees=hyperparameters["nb_trees"],
                max_depth=hyperparameters["max_depth"]
            )

            for prediction_period in prediction_periods:

                # Define prediction period
                date_information_prediction = DateInformation(
                    training_start=config["training_start"],
                    prediction_start=prediction_period[0],
                    prediction_end=prediction_period[1]
                )

                solver = Solver(
                    df=df,
                    dataset=dataset,
                    households=households,
                    config=config,
                    date_information=date_information_prediction,
                    params=parameters,
                    hyperparameter_scan=config["hyperparameter_scan"],
                    model_hyperparameter_scan="random_forest"
                )

                _, _ = solver.run()

        if "quantile_regression" in config["model_hyperparameter_scan"]:
            performance_dictionary = {}
            counter_qr = 0
            for lag in config["lag"]:
                for l2_penalty in config["l2_penalty"]:
                    for solver_method in config["solver"]:
                        print(f"Hyperparameter scan started for quantile regression with following parameters: "
                              f"lag = {lag}, l2_penalty = {l2_penalty}, solver = {solver_method}")

                        parameters = Parameters(
                            lag=lag,
                            l2_penalty=l2_penalty,
                            solver=solver_method
                        )

                        performance_all_validation_periods = []
                        elapsed_time_all_validation_periods = []

                        for validation_period in validation_periods:

                            # Define validation period
                            date_information_validation = DateInformation(
                                training_start=config["training_start"],
                                prediction_start=validation_period[0],
                                prediction_end=validation_period[1]
                            )

                            solver = Solver(
                                df=df,
                                dataset=dataset,
                                households=households,
                                config=config,
                                date_information=date_information_validation,
                                params=parameters,
                                hyperparameter_scan=config["hyperparameter_scan"],
                                model_hyperparameter_scan="quantile_regression"
                            )

                            performance, elapsed_time = solver.run()

                            performance_all_validation_periods.append(performance)
                            elapsed_time_all_validation_periods.extend(elapsed_time)

                        performance_dictionary[f"quantile_regression_{counter_qr}"] = {
                            "lag": lag,
                            "l2_penalty": l2_penalty,
                            "solver": solver_method,
                            "performance": np.mean(performance_all_validation_periods),
                            "elapsed_time": elapsed_time_all_validation_periods
                        }

                        counter_qr += 1

                        performance_dictionary_qr_path = os.path.join(model_dir, f'performance_dictionary_{dataset}_quantile_regression.json')

                        with open(performance_dictionary_qr_path, 'w') as fp:
                            json.dump(performance_dictionary, fp)

            hyperparameters = get_best_params(model="quantile_regression", dataset=dataset)

            parameters = Parameters(
                lag=hyperparameters["lag"],
                l2_penalty=hyperparameters["l2_penalty"],
                solver=hyperparameters["solver"]
            )

            for prediction_period in prediction_periods:

                # Define prediction period
                date_information_prediction = DateInformation(
                    training_start=config["training_start"],
                    prediction_start=prediction_period[0],
                    prediction_end=prediction_period[1]
                )

                solver = Solver(
                    df=df,
                    dataset=dataset,
                    households=households,
                    config=config,
                    date_information=date_information_prediction,
                    params=parameters,
                    hyperparameter_scan=config["hyperparameter_scan"],
                    model_hyperparameter_scan="quantile_regression"
                )

                _, _ = solver.run()

        if "neural_network" in config["model_hyperparameter_scan"]:
            performance_dictionary = {}
            counter_nn = 0
            for lag in config["lag"]:
                for neurons_layer_1 in config["neurons_layer_1"]:
                    for neurons_layer_2 in config["neurons_layer_2"]:
                        for epochs in config["epochs"]:
                            print(f"Hyperparameter scan started for quantile regression with following parameters: "
                                  f"lag = {lag}, neurons_layer_1 = {neurons_layer_1}, neurons_layer_2 = "
                                  f"{neurons_layer_2}, epochs = {epochs}")

                            parameters = Parameters(
                                lag=lag,
                                neurons_layer_1=neurons_layer_1,
                                neurons_layer_2=neurons_layer_2,
                                epochs=epochs
                            )

                            performance_all_validation_periods = []
                            elapsed_time_all_validation_periods = []

                            for validation_period in validation_periods:

                                # Define validation period
                                date_information_validation = DateInformation(
                                    training_start=config["training_start"],
                                    prediction_start=validation_period[0],
                                    prediction_end=validation_period[1]
                                )

                                solver = Solver(
                                    df=df,
                                    dataset=dataset,
                                    households=households,
                                    config=config,
                                    date_information=date_information_validation,
                                    params=parameters,
                                    hyperparameter_scan=config["hyperparameter_scan"],
                                    model_hyperparameter_scan="neural_network"
                                )

                                performance, elapsed_time = solver.run()

                                performance_all_validation_periods.append(performance)
                                elapsed_time_all_validation_periods.extend(elapsed_time)

                            performance_dictionary[f"neural_network_{counter_nn}"] = {
                                "lag": lag,
                                "neurons_layer_1": neurons_layer_1,
                                "neurons_layer_2": neurons_layer_2,
                                "epochs": epochs,
                                "performance": np.mean(performance_all_validation_periods),
                                "elapsed_time": elapsed_time_all_validation_periods
                            }

                            counter_nn += 1

                            performance_dictionary_nn_path = os.path.join(model_dir, f'performance_dictionary_{dataset}_neural_network.json')

                            with open(performance_dictionary_nn_path, 'w') as fp:
                                json.dump(performance_dictionary, fp)

            hyperparameters = get_best_params(model="neural_network", dataset=dataset)

            parameters = Parameters(
                lag=hyperparameters["lag"],
                neurons_layer_1=hyperparameters["neurons_layer_1"],
                neurons_layer_2=hyperparameters["neurons_layer_2"],
                epochs=hyperparameters["epochs"]
            )

            for prediction_period in prediction_periods:

                # Define prediction period
                date_information_prediction = DateInformation(
                    training_start=config["training_start"],
                    prediction_start=prediction_period[0],
                    prediction_end=prediction_period[1]
                )

                solver = Solver(
                    df=df,
                    dataset=dataset,
                    households=households,
                    config=config,
                    date_information=date_information_prediction,
                    params=parameters,
                    hyperparameter_scan=config["hyperparameter_scan"],
                    model_hyperparameter_scan="neural_network"
                )

                _, _ = solver.run()

    else:

        print(f"Simple run started")

        parameters = Parameters(
            config["historical_days"][0],
            config["nb_neighbors"][0],
            config["global_prediction_method"][0],
            config["lag"][0],
            config["nb_trees"][0],
            config["max_depth"][0],
            config["historical_sampling_days"][0]
        )

        for prediction_period in prediction_periods:

            # Define prediction period
            date_information_prediction = DateInformation(
                training_start=config["training_start"],
                prediction_start=prediction_period[0],
                prediction_end=prediction_period[1]
            )

            solver = Solver(
                df=df,
                dataset=dataset,
                households=households,
                config=config,
                date_information=date_information_prediction,
                params=parameters,
                hyperparameter_scan=config["hyperparameter_scan"]
            )

            _, _ = solver.run()


# TODO: Normalization behavior?
# TODO: Outlier removal?
# TODO: Fix data leakage (using "next day" data in database) - only if method is "next day"
# TODO: Implement different normalization/evaluation/prediction methods for experimentation purposes
