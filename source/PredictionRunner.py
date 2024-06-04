import pandas as pd
import numpy as np
import joblib
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import QuantileRegressor

from keras.models import Sequential, load_model
from keras.layers import Dense
import keras.backend as K

from helpers.functions import extract_historical_series, normalize_data
from helpers.constants import *
from helpers.paths import *
from source.SampleSelector import SamplesSelector
from source.GlobalPredictor import GlobalPredictor


class PredictionRunner:

    def __init__(
            self,
            start_prediction_time,
            end_prediction_time,
            start_prediction_batch,
            end_prediction_batch,
            dataset,
            household,
            database,
            quantiles,
            algorithm,
            algorithm_sample_selector,
            **hyperparameters
    ):
        """
        :param start_prediction_time:
        :param end_prediction_time:
        :param household:
        :param algorithm:
        """
        self.start_prediction_time = start_prediction_time
        self.end_prediction_time = end_prediction_time
        self.start_prediction_batch = start_prediction_batch
        self.end_prediction_batch = end_prediction_batch
        self.dataset = dataset
        self.household = household
        self.database = database
        self.quantiles = quantiles
        self.algorithm = algorithm
        self.algorithm_sample_selector = algorithm_sample_selector
        self.hyperparameters = hyperparameters

    def predict(self):

        if self.algorithm == "global_method":
            predictions, elapsed_time = self.predict_with_global_method()
            return predictions, elapsed_time
        elif self.algorithm == "random_forest":
            predictions, elapsed_time = self.predict_with_random_forest()
            return predictions, elapsed_time
        elif self.algorithm == "historical_sampling":
            predictions, elapsed_time = self.predict_with_historical_sampling()
            return predictions, elapsed_time
        elif self.algorithm == "quantile_regression":
            predictions, elapsed_time = self.predict_with_quantile_regression()
            return predictions, elapsed_time
        elif self.algorithm == "neural_network":
            predictions, elapsed_time = self.predict_with_neural_network()
            return predictions, elapsed_time
        else:
            print("Not implemented yet")

    def predict_with_global_method(self):
        start_time = time.perf_counter()

        historical_series = extract_historical_series(
            self.database.df,
            self.start_prediction_time,
            self.database.historical_days,
            self.household
        )
        historical_series_normalized = pd.Series(normalize_data(
            raw_data=np.array(historical_series).reshape(1, -1),
            method="min_max")[0]
        )

        selector = SamplesSelector(
            historical_series=historical_series_normalized,
            database=self.database,
            algorithm_sample_selector=self.algorithm_sample_selector,
            aggregation_method="mean",
            hyperparameters=self.hyperparameters
        )

        selector.fit_and_apply_sample_selector()
        prediction_samples = selector.get_similar_prediction_samples()

        predictor = GlobalPredictor(
            prediction_samples=prediction_samples,
            method="quantiles",
            quantiles=self.quantiles
        )

        probabilistic_predictions = predictor.get_probabilistic_predictions()
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        # print(f"Prediction made with global method for household: {self.household}")

        return probabilistic_predictions, elapsed_time

    def get_random_forest_model(self, start_prediction_batch, end_prediction_batch, household, training_data, target_data):

        lag = self.hyperparameters["lag"]
        nb_trees = self.hyperparameters["nb_trees"]
        max_depth = self.hyperparameters["max_depth"]

        model_name = f"model_random_forest_{self.dataset}_{start_prediction_batch}_{end_prediction_batch}_{household}_lag_{lag}_nb_trees_{nb_trees}_max_depth_{max_depth}.joblib"
        path = os.path.join(rf_dir, model_name)

        if os.path.exists(path):
            # print(f"Model was found - Loading random forest model - {model_name}")

            model = joblib.load(path)
            return model, np.nan

        else:
            # print(f"Model was not found - Training random forest model - {model_name}")
            start_time = time.perf_counter()

            model = RandomForestRegressor(
                n_estimators=self.hyperparameters["nb_trees"],
                max_depth=self.hyperparameters["max_depth"]
            )
            model.fit(training_data, target_data)

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            # print(f"The model training took {elapsed_time} seconds to run.")

            joblib.dump(model, path)

            return model, elapsed_time

    def get_quantile_regression_models(self, start_prediction_batch, end_prediction_batch, household, training_data, target_data):

        lag = self.hyperparameters["lag"]
        l2_penalty = self.hyperparameters["l2_penalty"]
        solver = self.hyperparameters["solver"]

        models = []

        start_time = time.perf_counter()
        invalid_time = False

        for half_hour in range(TIME_POINTS_PER_DAY):
            for quantile in [0.10, 0.25, 0.50, 0.75, 0.90]:

                quantile = np.round(quantile, 2)

                model_name = f"model_quantile_regression_{self.dataset}_{start_prediction_batch}_{end_prediction_batch}_{household}_lag_{lag}_l2_penalty_{l2_penalty}_solver_{solver}_half_hour_{half_hour}_quantile_{quantile}.joblib"
                path = os.path.join(qr_dir, model_name)

                if os.path.exists(path):
                    # print(f"Model was found - Loading quantile regression model - {model_name}")

                    model = joblib.load(path)
                    invalid_time = True

                else:
                    # print(f"Model was not found - Training quantile regression model - {model_name}")

                    model = QuantileRegressor(
                        quantile=quantile,
                        alpha=self.hyperparameters["l2_penalty"],
                        solver=self.hyperparameters["solver"]
                    )

                    model.fit(training_data, target_data.iloc[:, half_hour])

                    joblib.dump(model, path)

                models.append(model)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # print(f"The model training took {elapsed_time} seconds to run.")

        if invalid_time:
            return models, np.nan
        else:
            return models, elapsed_time

    @staticmethod
    def tilted_loss(q, y, f):
        e = (y - f)
        return K.mean(K.maximum(q * e, (q - 1) * e), axis=-1)

    def get_neural_network_models(self, start_prediction_batch, end_prediction_batch, household, training_data, target_data):

        lag = self.hyperparameters["lag"]
        neurons_layer_1 = self.hyperparameters["neurons_layer_1"]
        neurons_layer_2 = self.hyperparameters["neurons_layer_2"]
        epochs = self.hyperparameters["epochs"]

        models = []

        start_time = time.perf_counter()
        invalid_time = False

        for quantile in [0.10, 0.25, 0.50, 0.75, 0.90]:

            quantile = np.round(quantile, 2)

            model_name = f"model_neural_network_{self.dataset}_{start_prediction_batch}_{end_prediction_batch}_{household}_lag_{lag}_neurons_layer_1_{neurons_layer_1}_neurons_layer_2_{neurons_layer_2}_epochs_{epochs}_quantile_{quantile}.h5"
            path = os.path.join(nn_dir, model_name)

            if os.path.exists(path):
                # print(f"Model was found - Loading neural network model - {model_name}")

                model = load_model(path, compile=False)
                invalid_time = True

            else:
                # print(f"Model was not found - Training neural network model - {model_name}")

                model = Sequential()
                model.add(Dense(units=neurons_layer_1, input_dim=training_data.shape[1], activation='relu'))
                model.add(Dense(units=neurons_layer_2, activation='relu'))
                model.add(Dense(units=TIME_POINTS_PER_DAY))

                model.compile(loss=lambda y, f: self.tilted_loss(quantile, y, f), optimizer='adadelta')
                model.fit(training_data, target_data, epochs=epochs, verbose=0)

                model.save(path)

            models.append(model)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # print(f"The model training took {elapsed_time} seconds to run.")

        if invalid_time:
            return models, np.nan
        else:
            return models, elapsed_time

    def predict_with_quantile_regression(self):

        training_data = self.database.get_training_data_local(self.household)
        target_data = self.database.get_target_data_local(self.household)

        prediction_data = extract_historical_series(
            df=self.database.df,
            end_time=self.start_prediction_time,
            nb_historical_days=self.database.lag,
            household_id=self.household
        )

        models, elapsed_time = self.get_quantile_regression_models(
            start_prediction_batch=self.start_prediction_batch,
            end_prediction_batch=self.end_prediction_batch,
            household=self.household,
            training_data=training_data,
            target_data=target_data
        )

        probabilistic_predictions = pd.DataFrame()
        prediction_data_temp = np.append(prediction_data.values.reshape(1, -1), self.start_prediction_time.dayofweek)
        pred_data = np.append(prediction_data_temp, int(prediction_data_temp[-1] == 5 or prediction_data_temp[-1] == 6))

        counter = 0
        for half_hour in range(TIME_POINTS_PER_DAY):
            predictions = []
            for _ in [0.10, 0.25, 0.50, 0.75, 0.90]:
                model = models[counter]
                prediction = model.predict(pred_data.reshape(1, -1))[0]
                predictions.append(prediction)
                counter = counter + 1

            probabilistic_predictions.loc[:, half_hour] = np.array(predictions)

        # print(f"Prediction made with quantile regression for household: {self.household}")

        return probabilistic_predictions, elapsed_time

    def predict_with_random_forest(self):

        training_data = self.database.get_training_data_local(self.household)
        target_data = self.database.get_target_data_local(self.household)

        prediction_data = extract_historical_series(
            df=self.database.df,
            end_time=self.start_prediction_time,
            nb_historical_days=self.database.lag,
            household_id=self.household
        )

        model, elapsed_time = self.get_random_forest_model(
            start_prediction_batch=self.start_prediction_batch,
            end_prediction_batch=self.end_prediction_batch,
            household=self.household,
            training_data=training_data,
            target_data=target_data
        )

        predictions = pd.DataFrame()
        prediction_data_temp = np.append(prediction_data.values.reshape(1, -1), self.start_prediction_time.dayofweek)
        pred_data = np.append(prediction_data_temp, int(prediction_data_temp[-1] == 5 or prediction_data_temp[-1] == 6))
        for tree in model.estimators_:
            tree_predictions = tree.predict(pred_data.reshape(1, -1))
            predictions = pd.concat([predictions, pd.DataFrame(tree_predictions)])

        probabilistic_predictions = pd.DataFrame()
        for i in range(TIME_POINTS_PER_DAY):
            probabilistic_predictions.loc[:, i] = predictions.iloc[:, i].quantile(self.quantiles).values

        # print(f"Prediction made with random forest for household: {self.household}")

        return probabilistic_predictions, elapsed_time

    def predict_with_historical_sampling(self):
        start_time = time.perf_counter()

        index = self.database.df[self.household].index.get_loc(self.start_prediction_time)
        predictions = pd.DataFrame()
        for i in range(TIME_POINTS_PER_DAY):
            indices = [index + i - 7*TIME_POINTS_PER_DAY*j for j in range(1, self.hyperparameters["historical_sampling_days"]+1)]
            values_list = self.database.df.iloc[indices][self.household].tolist()
            quantilesQ = np.quantile(values_list, self.quantiles)
            predictions.loc[:, i] = quantilesQ

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # print(f"Prediction made with historical sampling for household: {self.household}")

        return predictions, elapsed_time

    def predict_with_neural_network(self):

        training_data = self.database.get_training_data_local(self.household)
        target_data = self.database.get_target_data_local(self.household)

        prediction_data = extract_historical_series(
            df=self.database.df,
            end_time=self.start_prediction_time,
            nb_historical_days=self.database.lag,
            household_id=self.household
        )

        models, elapsed_time = self.get_neural_network_models(
            start_prediction_batch=self.start_prediction_batch,
            end_prediction_batch=self.end_prediction_batch,
            household=self.household,
            training_data=training_data,
            target_data=target_data
        )

        probabilistic_predictions = pd.DataFrame()
        prediction_data_temp = np.append(prediction_data.values.reshape(1, -1), self.start_prediction_time.dayofweek)
        pred_data = np.append(prediction_data_temp, int(prediction_data_temp[-1] == 5 or prediction_data_temp[-1] == 6))

        counter = 0
        for _ in [0.10, 0.25, 0.50, 0.75, 0.90]:
            model = models[counter]
            predictions = model.predict(pred_data.reshape(1, -1), verbose=0)[0]
            probabilistic_predictions = pd.concat([probabilistic_predictions, pd.DataFrame(predictions)], axis=1)
            counter = counter + 1

        probabilistic_predictions = probabilistic_predictions.transpose()

        # print(f"Prediction made with neural network for household: {self.household}")

        return probabilistic_predictions, elapsed_time
