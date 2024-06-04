import pandas as pd
import numpy as np
from datetime import timedelta
import json

from collections import defaultdict

from helpers.functions import extract_target_consumption, extract_historical_series, normalize_data
from helpers.constants import *
from helpers.paths import *
from source.Database import Database
from source.PerformanceEvaluator import PerformanceEvaluator
from source.PredictionRunner import PredictionRunner
from source.PredictionDispatcher import PredictionDispatcher


class Solver:

    def __init__(self,
                 df,
                 dataset,
                 households,
                 config,
                 date_information,
                 params,
                 hyperparameter_scan,
                 model_hyperparameter_scan=False,
                 ):
        """
        :param config: dictionary with configuration parameters
        """
        self.df = df
        self.dataset = dataset
        self.households = households
        self.performance = None

        # Period definitions
        self.training_start = date_information.training_start
        self.prediction_start = date_information.prediction_start
        self.prediction_end = date_information.prediction_end
        self.prediction_dates = pd.date_range(self.prediction_start, self.prediction_end).strftime('%Y-%m-%d').tolist()

        # Standard parameters (all methods)
        self.quantile_start = config["quantile_start"]
        self.quantile_end = config["quantile_end"]
        self.quantile_step = config["quantile_step"]

        self.normalization_method = config["normalization_method"]
        self.evaluation_method = config["evaluation_method"]

        # GeneralParameters global method
        self.algorithm_sample_selector = config["algorithm_sample_selector"]

        # HyperParameters global method
        self.historical_days = params.historical_days
        self.nb_neighbors = params.nb_neighbors
        self.global_prediction_method = params.global_prediction_method

        # HyperParameters historical sampling
        self.historical_sampling_days = params.historical_sampling_days

        # HyperParameters local methods
        self.lag = params.lag

        # HyperParameters quantile regression
        self.l2_penalty = params.l2_penalty
        self.solver = params.solver

        # HyperParameters random forest
        self.nb_trees = params.nb_trees
        self.max_depth = params.max_depth

        # HyperParameters neural network
        self.neurons_layer_1 = params.neurons_layer_1
        self.neurons_layer_2 = params.neurons_layer_2
        self.epochs = params.epochs

        # HyperParameter model
        self.hyperparameter_scan = hyperparameter_scan
        self.model_hyperparameter_scan = model_hyperparameter_scan

        self.concept_drift_detection_method = config["concept_drift_detection_method"]

    def run(self):

        results_dictionary = defaultdict(lambda: defaultdict(dict))

        database = Database(
            df=self.df,
            dataset=self.dataset,
            lag=self.lag,
            historical_days=self.historical_days,
            start_training_date=self.training_start,
            end_training_date=self.prediction_dates[0],
            method=self.global_prediction_method
        )

        database.initiate_training_data_global()
        database.initiate_target_data_global()
        database.initiate_data_local()

        normalized_training_data_global = normalize_data(
            raw_data=database.get_training_data_global(),
            method=self.normalization_method
        )
        database.add_training_data_global_normalized(normalized_training_data_global)

        all_scores = []
        all_training_times = []

        for counter, prediction_date in enumerate(self.prediction_dates):

            print(f"Making prediction for {prediction_date}")

            start_prediction_time = pd.Timestamp(prediction_date)
            end_prediction_time = start_prediction_time + timedelta(days=1)

            rows_to_add_to_training_data = []
            rows_to_add_to_target_data = []
            rows_to_add_to_training_data_normalized = []

            for household in self.households:
                target_consumption = extract_target_consumption(
                    df=database.df,
                    start_time=start_prediction_time,
                    end_time=end_prediction_time,
                    household_id=household
                )
                historical_series_global = extract_historical_series(
                    df=database.df,
                    end_time=start_prediction_time,
                    nb_historical_days=database.historical_days,
                    household_id=household
                )
                historical_series_global_normalized = pd.Series(normalize_data(
                    raw_data=np.array(historical_series_global).reshape(1, -1),
                    method=self.normalization_method)[0]
                )
                historical_series_local = extract_historical_series(
                    df=database.df,
                    end_time=start_prediction_time,
                    nb_historical_days=database.lag,
                    household_id=household
                )

                # Select best model based on observed concept drift
                prediction_dispatcher = PredictionDispatcher(
                    database=database,
                    household=household,
                    hyperparameter_scan=self.hyperparameter_scan,
                    model_hyperparameter_scan=self.model_hyperparameter_scan,
                    prediction_date=prediction_date,
                    concept_drift_detection_method=self.concept_drift_detection_method
                )

                selected_algorithm = prediction_dispatcher.define_optimal_model()

                # Trigger prediction with selected model for specific household and date
                prediction_runner = PredictionRunner(
                    start_prediction_time=start_prediction_time,
                    end_prediction_time=end_prediction_time,
                    start_prediction_batch=self.prediction_start,
                    end_prediction_batch=self.prediction_end,
                    dataset=self.dataset,
                    household=household,
                    database=database,
                    quantiles=np.arange(self.quantile_start, self.quantile_end, self.quantile_step),
                    algorithm=selected_algorithm,
                    algorithm_sample_selector=self.algorithm_sample_selector,
                    nb_neighbors=self.nb_neighbors,
                    lag=self.lag,
                    nb_trees=self.nb_trees,
                    max_depth=self.max_depth,
                    historical_sampling_days=self.historical_sampling_days,
                    l2_penalty=self.l2_penalty,
                    solver=self.solver,
                    neurons_layer_1=self.neurons_layer_1,
                    neurons_layer_2=self.neurons_layer_2,
                    epochs=self.epochs
                )

                probabilistic_predictions, elapsed_time = prediction_runner.predict()
                all_training_times.append(elapsed_time)

                # Evaluate performance for specific household and date (24 hour evaluation)
                evaluator = PerformanceEvaluator(
                    probabilistic_predictions=probabilistic_predictions,
                    target_consumption=target_consumption,
                    method=self.evaluation_method
                )

                score = evaluator.compute_metric()
                all_scores.append(score)

                percentiles = evaluator.extract_percentiles([0.10, 0.25, 0.50, 0.75, 0.90])

                results_dictionary[prediction_date][household] = (score, selected_algorithm, list(percentiles.iloc[0, :]), list(percentiles.iloc[1, :]), list(percentiles.iloc[2, :]), list(percentiles.iloc[3, :]), list(percentiles.iloc[4, :]), list(target_consumption))

                # print(f"Timestep: {prediction_date} - Household: {household} - Score: {score}")

                target_consumption_global_processed = database.process_series_to_database_row(
                    row=target_consumption,
                    cols=list(database.target_data_global.columns)
                )
                target_consumption_local_processed = database.process_series_to_database_row(
                    row=target_consumption,
                    cols=list(database.target_data_local[household].columns)
                )
                historical_series_global_processed = database.process_series_to_database_row(
                    row=historical_series_global,
                    cols=list(database.training_data_global.columns)
                )
                historical_series_global_normalized_processed = database.process_series_to_database_row(
                    row=historical_series_global_normalized,
                    cols=list(database.training_data_global_normalized.columns)
                )

                historical_series_local_processed = database.process_series_to_database_row(
                    row=pd.concat([historical_series_local, pd.Series([start_prediction_time.dayofweek, int(start_prediction_time.dayofweek == 5 or start_prediction_time.dayofweek == 6)])]).reset_index(drop=True),
                    cols=list(database.training_data_local[household].columns)
                )

                rows_to_add_to_target_data.append(target_consumption_global_processed)
                rows_to_add_to_training_data.append(historical_series_global_processed)
                rows_to_add_to_training_data_normalized.append(historical_series_global_normalized_processed)

                database.add_row_to_training_data_local(historical_series_local_processed, household)
                database.add_row_to_target_data_local(target_consumption_local_processed, household)

            # Add new rows to database for future training/prediction
            database.add_rows_to_training_data_global(rows=rows_to_add_to_training_data)
            database.add_rows_to_target_data_global(rows=rows_to_add_to_target_data)
            database.add_rows_to_training_data_global_normalized(rows=rows_to_add_to_training_data_normalized)

        start_prediction_date = self.prediction_dates[0]
        end_prediction_date = self.prediction_dates[-1]

        if self.hyperparameter_scan:
            model_indicator = self.model_hyperparameter_scan
        else:
            model_indicator = "combined_model"

        if model_indicator == "global_method":
            file_name = f"results_{self.dataset}_from_{start_prediction_date}_to_{end_prediction_date}_model_" \
                        f"{model_indicator}_historical_days_{self.historical_days}_nb_neighbors_{self.nb_neighbors}_" \
                        f"global_prediction_method_{self.global_prediction_method}.csv"

        elif model_indicator == "random_forest":
            file_name = f"results_{self.dataset}_from_{start_prediction_date}_to_{end_prediction_date}_model_" \
                        f"{model_indicator}_lag_{self.lag}_nb_trees_{self.nb_trees}_max_depth_{self.max_depth}.csv"

        elif model_indicator == "historical_sampling":
            file_name = f"results_{self.dataset}_from_{start_prediction_date}_to_{end_prediction_date}_model_" \
                        f"{model_indicator}_historical_sampling_days_{self.historical_sampling_days}.csv"

        elif model_indicator == "quantile_regression":
            file_name = f"results_{self.dataset}_from_{start_prediction_date}_to_{end_prediction_date}_model_" \
                        f"{model_indicator}_lag_{self.lag}_l2_penalty_{self.l2_penalty}_solver_{self.solver}.csv"

        elif model_indicator == "neural_network":
            file_name = f"results_{self.dataset}_from_{start_prediction_date}_to_{end_prediction_date}_model_" \
                        f"{model_indicator}_lag_{self.lag}_neurons_layer_1_{self.neurons_layer_1}_neurons_layer_2_" \
                        f"{self.neurons_layer_2}_epochs_{self.epochs}.csv"

        else:
            file_name = f"results_{self.dataset}_from_{start_prediction_date}_to_{end_prediction_date}_model_" \
                        f"{model_indicator}_historical_days_{self.historical_days}_nb_neighbors_{self.nb_neighbors}_" \
                        f"global_prediction_method_{self.global_prediction_method}_lag_{self.lag}_nb_trees_" \
                        f"{self.nb_trees}_max_depth_{self.max_depth}.csv"

        path_results = os.path.join(results_dir, file_name)

        print("***** End of run *****")
        print(f"Final score for run: {np.mean(all_scores)}")
        print(f"Results will be stored in: {file_name}")
        print("***** End of run *****")

        with open(path_results, 'w') as json_file:
            json.dump(results_dictionary, json_file)

        all_training_times = np.array([np.nan if x is None else x for x in all_training_times]).tolist()
        return np.mean(all_scores), all_training_times
