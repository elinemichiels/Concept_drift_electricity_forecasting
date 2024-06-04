import numpy as np
from properscoring import crps_ensemble


class PerformanceEvaluator:

    def __init__(self, probabilistic_predictions, target_consumption, method):
        """
        :param probabilistic_predictions:
        :param target_consumption:
        :param method:
        """
        self.probabilistic_predictions = probabilistic_predictions
        self.target_consumption = target_consumption
        self.method = method

    def compute_metric(self):
        if self.method == "CRPS":
            return np.round(self.compute_CRPS(), 4)
        else:
            print("Not implemented yet")

    def extract_percentiles(self, quantiles):
        return self.probabilistic_predictions.quantile(quantiles)

    def compute_CRPS(self):
        crps_values = []

        for timestep in range(self.probabilistic_predictions.shape[1]):
            crps = crps_ensemble(self.target_consumption[timestep], self.probabilistic_predictions[timestep])
            crps_values.append(crps)

        return np.mean(crps_values)
