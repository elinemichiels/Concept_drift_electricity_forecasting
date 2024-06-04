import os
from pathlib import Path


package_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = str(Path(package_dir).parent.absolute())
source_dir = os.path.join(root_dir, "source")
data_dir = os.path.join(root_dir, "data")
model_dir = os.path.join(root_dir, "models")

# Raw data
raw_dir = os.path.join(data_dir, "raw")

london_data_path = os.path.join(raw_dir, "London_dataset.csv")
irish_data_path = os.path.join(raw_dir, "Irish_dataset.csv")

# Processed data
processed_dir = os.path.join(data_dir, "processed")

# Result data
results_dir = os.path.join(data_dir, "results")

# Config path
config_file_path = os.path.join(source_dir, "config.yml")

# Households split path
households_split_path = os.path.join(source_dir, "households_split.json")

# Random forest models
rf_dir = os.path.join(model_dir, "random_forest")

# Quantile regression models
qr_dir = os.path.join(model_dir, "quantile_regression")

# Neural network models
nn_dir = os.path.join(model_dir, "neural_network")
