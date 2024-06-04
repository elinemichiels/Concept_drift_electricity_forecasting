class Parameters:
    def __init__(self,
                 historical_days=7,
                 nb_neighbors=60,
                 global_prediction_method="next_day",
                 lag=10,
                 nb_trees=200,
                 max_depth=40,
                 historical_sampling_days=7,
                 l2_penalty=1,
                 solver="highs",
                 neurons_layer_1=10,
                 neurons_layer_2=10,
                 epochs=100
                 ):
        self.historical_days = historical_days
        self.nb_neighbors = nb_neighbors
        self.global_prediction_method = global_prediction_method
        self.lag = lag
        self.nb_trees = nb_trees
        self.max_depth = max_depth
        self.historical_sampling_days = historical_sampling_days
        self.l2_penalty = l2_penalty
        self.solver = solver
        self.neurons_layer_1 = neurons_layer_1
        self.neurons_layer_2 = neurons_layer_2
        self.epochs = epochs


class DateInformation:
    def __init__(self,
                 training_start,
                 prediction_start,
                 prediction_end
                 ):
        self.training_start = training_start
        self.prediction_start = prediction_start
        self.prediction_end = prediction_end
