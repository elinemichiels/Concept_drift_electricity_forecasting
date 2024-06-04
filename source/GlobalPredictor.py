import pandas as pd
import numpy as np


class GlobalPredictor:

    def __init__(self, prediction_samples, method, quantiles):
        """
        :param prediction_samples:
        :param method:
        :param quantiles:
        """
        self.prediction_samples = prediction_samples
        self.method = method
        self.quantiles = quantiles
        self.probabilistic_predictions = self.predict()

    def predict(self):
        if self.method == "quantiles":
            return pd.DataFrame(np.nanquantile(self.prediction_samples, self.quantiles, axis=0))
        else:
            print("Not implemented yet")

    def get_probabilistic_predictions(self):
        return self.probabilistic_predictions
