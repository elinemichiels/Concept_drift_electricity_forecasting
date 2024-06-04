import pandas as pd


class PredictionDispatcher:

    def __init__(self, database, household, hyperparameter_scan, model_hyperparameter_scan, prediction_date, concept_drift_detection_method):
        """
        :param database:
        :param: household:
        """
        self.database = database
        self.household = household
        self.hyperparameter_scan = hyperparameter_scan
        self.model_hyperparameter_scan = model_hyperparameter_scan
        self.prediction_date = prediction_date
        self.concept_drift_detection_method = concept_drift_detection_method

    @staticmethod
    def round_down_list_to_multiple_of_48(numbers):
        """
        This function takes a list of numbers and returns a list where each number has been rounded down to the nearest multiple of 48.
        If the number itself is a multiple of 48, it can be returned as is.
        """

        def round_down(number):
            # Calculate the nearest multiple of 48 that is lower or equal to the number
            nearest_multiple = (number // 48) * 48
            return nearest_multiple

        return [round_down(number) for number in numbers]

    @staticmethod
    def update_page_hinkley(value, mean, cumulative_sum, min_cumulative_sum, max_cumulative_sum, observation_count,
                            delta):
        """
        Update the Page-Hinkley Test parameters for a new observation.

        Returns updated values.
        """
        observation_count += 1
        mean += (value - mean) / observation_count
        cumulative_sum += value - mean - delta
        min_cumulative_sum = min(min_cumulative_sum, cumulative_sum)
        max_cumulative_sum = max(max_cumulative_sum, cumulative_sum)

        return mean, cumulative_sum, min_cumulative_sum, max_cumulative_sum, observation_count

    @staticmethod
    def check_drift(cumulative_sum, min_cumulative_sum, max_cumulative_sum, lambda_):
        """
        Check for drift based on the current and historical cumulative sums.

        Returns True if drift is detected, else False.
        """
        if (cumulative_sum - min_cumulative_sum > lambda_) or (max_cumulative_sum - cumulative_sum > lambda_):
            return True
        return False

    @staticmethod
    def page_hinckley(data_stream, delta=0.005, lambda_=50):
        """
        Apply the Page-Hinkley Test to a data stream to detect drifts.

        Returns indices where drifts were detected.
        """
        mean = 0.0
        cumulative_sum = 0.0
        min_cumulative_sum = 0.0
        max_cumulative_sum = 0.0
        observation_count = 0
        drift_detected_indices = []

        for i, value in enumerate(data_stream):
            mean, cumulative_sum, min_cumulative_sum, max_cumulative_sum, observation_count = PredictionDispatcher.update_page_hinkley(
                value, mean, cumulative_sum, min_cumulative_sum, max_cumulative_sum, observation_count, delta)

            if PredictionDispatcher.check_drift(cumulative_sum, min_cumulative_sum, max_cumulative_sum, lambda_):
                drift_detected_indices.append(i)
                # Reset for detecting new drifts
                cumulative_sum = 0.0
                min_cumulative_sum = 0.0
                max_cumulative_sum = 0.0
                observation_count = 0
                mean = 0.0

        return PredictionDispatcher.round_down_list_to_multiple_of_48(drift_detected_indices)

    def define_optimal_model(self):
        """
        If hyperparameter scan, overwrite model with defined model. Else, let the "detector" choose the optimal model.
        """
        if self.hyperparameter_scan:
            model = self.model_hyperparameter_scan
        else:
            prediction_datetime_str = self.prediction_date + ' 00:00:00'
            prediction_datetime = pd.to_datetime(prediction_datetime_str)
            self.database.df.sort_index(inplace=True)
            position = self.database.df.index.get_loc(prediction_datetime, method='ffill')
            trimmed_df = self.database.df.iloc[:position]

            if self.concept_drift_detection_method == "Page Hinckley":
                drift_detected_indices = PredictionDispatcher.page_hinckley(trimmed_df[self.household], delta=0.005, lambda_=100)
            else:
                print("Concept drift detection method not yet implemented")
                drift_detected_indices = None
            amount_of_points = len(trimmed_df)

            if any(amount_of_points-30*48 <= number <= amount_of_points for number in drift_detected_indices):
                model = "global_method"
            else:
                model = "random_forest"
        return model
