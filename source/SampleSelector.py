import numpy as np
from sklearn.neighbors import NearestNeighbors


class SamplesSelector:

    def __init__(self, historical_series, database, algorithm_sample_selector, aggregation_method, hyperparameters):
        """
        :param historical_series:
        :param database:
        :param algorithm_sample_selector:
        :param aggregation_method:
        """
        self.historical_series = historical_series
        self.training_data_global = database.get_training_data_global_normalized().fillna(0)
        self.target_data_global = database.get_target_data_global().fillna(0)
        self.algorithm_sample_selector = algorithm_sample_selector
        self.aggregation_method = aggregation_method
        self.hyperparameters = hyperparameters
        self.indices = None

    def fit_and_apply_sample_selector(self):
        if self.algorithm_sample_selector == "knn":
            algorithm = NearestNeighbors(n_neighbors=self.hyperparameters["nb_neighbors"])
            algorithm.fit(self.training_data_global)
            _, indices = algorithm.kneighbors(np.array(self.historical_series.values).reshape(1, -1))
        else:
            print("Not implemented yet")
        self.indices = indices[0]

    def get_similar_training_samples(self):
        return self.training_data_global.iloc[self.indices, :]

    def get_similar_prediction_samples(self):
        return self.target_data_global.iloc[self.indices, :]
