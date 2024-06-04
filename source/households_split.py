import pandas as pd
import numpy as np
from datetime import timedelta
import json
import copy
import matplotlib.pyplot as plt
import statistics

from helpers.paths import *
from helpers.constants import *
from helpers.functions import *
import pandas as pd
import json
import random

df = pd.read_csv(london_data_path, index_col=0).transpose()
missing_count = df.isnull().sum()

columns_to_exclude = missing_count > 100

filtered_df = df.loc[:, ~columns_to_exclude]
filled_df = filtered_df.fillna(method='ffill')
filled_df.index = pd.to_datetime(filled_df.index)
households = list(filled_df.columns)
split_index = int(len(households) * 1)

# Split the list
list_global_local = households[:split_index]
list_concept_drift = households[split_index:]

# Save the lists to a JSON file
with open('households_split.json', 'w') as f:
    json.dump({'list_global_local': list_global_local, 'list_concept_drift': list_concept_drift}, f, indent=4)

print('klaar')