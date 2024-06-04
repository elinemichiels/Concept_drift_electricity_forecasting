import numpy as np

from helpers.constants import *
from helpers.functions import *
from datetime import timedelta


class Database:

    def __init__(self, df, dataset, historical_days, lag, start_training_date, end_training_date, method):
        """
        :param historical_days:
        :param start_training_date:
        :param end_training_date:
        :param method:
        """
        self.df = df
        self.dataset = dataset
        self.historical_days = historical_days
        self.lag = lag
        self.start_training_date = start_training_date
        self.end_training_date = end_training_date
        self.method = method
        self.nb_of_training_start_days = self.get_nb_of_training_days()
        self.training_data_global = pd.DataFrame()
        self.training_data_global_normalized = pd.DataFrame()
        self.target_data_global = pd.DataFrame()
        self.training_data_local = dict()
        self.target_data_local = dict()

    def get_nb_of_training_days(self):
        return (pd.Timestamp(self.end_training_date) - pd.Timestamp(self.start_training_date)).days - self.historical_days

    def initiate_training_data_global(self):

        filename = f"training_data_{self.dataset}_global_{self.start_training_date}_{self.end_training_date}_lag{self.historical_days}days_{self.method}.csv"
        path = os.path.join(processed_dir, filename)

        if os.path.exists(path):
            print("File was found - Loading training data")

            training_data_global = pd.read_csv(path, index_col=0)
            self.training_data_global = training_data_global

        else:
            print("File was not found - Creating training data")

            dictionary_list = []

            for nb_of_training_day in range(self.nb_of_training_start_days):
                start_series, end_series = self.define_start_and_end_date(counter=nb_of_training_day, method="training")
                training_series = self.filter_date(start_series, end_series)

                training_series = training_series.transpose()
                training_series.columns = list(np.arange(0, TIME_POINTS_PER_DAY * self.historical_days))
                training_series_rows = training_series.index.map(lambda x: training_series.loc[x].to_dict())
                dictionary_list.extend(training_series_rows)

            training_data_global = pd.DataFrame.from_dict(dictionary_list)
            training_data_global.to_csv(path)
            self.training_data_global = training_data_global

    def initiate_target_data_global(self):
        filename = f"target_data_{self.dataset}_global_{self.start_training_date}_{self.end_training_date}_lag{self.historical_days}days_{self.method}.csv"
        path = os.path.join(processed_dir, filename)

        if os.path.exists(path):
            print("File was found - Loading target data")

            target_data_global = pd.read_csv(path, index_col=0)
            self.target_data_global = target_data_global

        else:
            print("File was not found - Creating target data")

            dictionary_list = []

            for nb_of_training_day in range(self.nb_of_training_start_days):
                start_series, end_series = self.define_start_and_end_date(counter=nb_of_training_day, method=self.method)
                target_series = self.filter_date(start_series, end_series)

                target_series = target_series.transpose()
                target_series.columns = list(np.arange(0, TIME_POINTS_PER_DAY))
                target_series_rows = target_series.index.map(lambda x: target_series.loc[x].to_dict())
                dictionary_list.extend(target_series_rows)

            target_data_global = pd.DataFrame.from_dict(dictionary_list)
            target_data_global.to_csv(path)
            self.target_data_global = target_data_global

    def initiate_data_local(self):
        filename_training = f"training_data_{self.dataset}_local_{self.start_training_date}_{self.end_training_date}_lag{self.lag}days.json"
        path_training = os.path.join(processed_dir, filename_training)

        filename_target = f"target_data_{self.dataset}_local_{self.start_training_date}_{self.end_training_date}_lag{self.lag}days.json"
        path_target = os.path.join(processed_dir, filename_target)

        if os.path.exists(path_training) and os.path.exists(path_target):
            print("Files were found - Loading data local")

            with open(path_training) as file:
                training_data_local = json.load(file)

            with open(path_target) as file:
                target_data_local = json.load(file)

            training_data_local = self.convert_json_to_df(training_data_local)
            target_data_local = self.convert_json_to_df(target_data_local)

            self.training_data_local = training_data_local
            self.target_data_local = target_data_local

        else:
            print("Files were not found - Creating data local")

            training_data_local = dict()
            target_data_local = dict()
            if self.dataset == "london":
                with open(households_split_path, 'r') as households_split:
                    loaded_lists = json.load(households_split)
                    households = loaded_lists['list_global_local']
            if self.dataset == "irish":
                households = self.df.columns.to_list()

            for household in households[:2]:

                training_days = (pd.to_datetime(self.end_training_date) - self.df.index[0]).days
                complete_historical_series = extract_historical_series(
                    self.df,
                    pd.to_datetime(self.end_training_date),
                    training_days,
                    household
                )

                household_training_data_local = convert_series_to_table_format(
                    series=complete_historical_series,
                    lag=TIME_POINTS_PER_DAY*self.lag,
                    operations=["add_time_variables"]
                )

                target_data_local[household] = pd.DataFrame(household_training_data_local.consumption.values.reshape(-1, 48)).to_json()

                all_features = [col for col in household_training_data_local.columns if "lag" in col]

                household_training_data_local = household_training_data_local.loc[:, all_features]
                household_training_data_local = household_training_data_local[
                    (household_training_data_local.index.hour == 0) & (household_training_data_local.index.minute == 0)
                ]
                household_training_data_local['day_of_week_int'] = household_training_data_local.index.dayofweek
                household_training_data_local['weekend_holiday'] = (
                            (household_training_data_local['day_of_week_int'] == 5) | (
                                household_training_data_local['day_of_week_int'] == 6)).astype(int)

                training_data_local[household] = household_training_data_local.to_json()

            with open(path_training, "w") as file:
                json.dump(training_data_local, file)

            with open(path_target, "w") as file:
                json.dump(target_data_local, file)

            training_data_local = self.convert_json_to_df(training_data_local)
            target_data_local = self.convert_json_to_df(target_data_local)

            self.training_data_local = training_data_local
            self.target_data_local = target_data_local

    def define_start_and_end_date(self, counter, method):
        if method == "training":
            start_series = pd.Timestamp(self.start_training_date) + timedelta(days=counter)
            end_series = start_series + timedelta(days=self.historical_days)
            return start_series, end_series
        elif method == "add_target_data":
            start_series = pd.Timestamp(self.end_training_date) + timedelta(days=counter)
            end_series = start_series + timedelta(days=1)
            return start_series, end_series
        elif method == "next_day":
            start_series = pd.Timestamp(self.start_training_date) + timedelta(days=counter) + timedelta(days=self.historical_days)
            end_series = start_series + timedelta(days=1)
            return start_series, end_series
        elif method == "last_day":
            start_series = pd.Timestamp(self.start_training_date) + timedelta(days=counter) + timedelta(days=self.historical_days) - timedelta(days=1)
            end_series = start_series + timedelta(days=1)
            return start_series, end_series
        else:
            print("Not implemented yet")

    def filter_date(self, start_series, end_series, household_selection=None):
        if household_selection:
            return self.df.loc[(self.df.index >= start_series) & (self.df.index < end_series), household_selection]
        else:
            return self.df.loc[(self.df.index >= start_series) & (self.df.index < end_series), :]

    def add_rows_to_training_data_global(self, rows):
        self.training_data_global = pd.concat([self.training_data_global] + rows).reset_index(drop=True)

    def add_rows_to_target_data_global(self, rows):
        self.target_data_global = pd.concat([self.target_data_global] + rows).reset_index(drop=True)

    def add_rows_to_training_data_global_normalized(self, rows):
        self.training_data_global_normalized = pd.concat([self.training_data_global_normalized] + rows).reset_index(drop=True)

    def add_row_to_training_data_local(self, row, household):
        self.training_data_local[household] = pd.concat([self.training_data_local[household], pd.DataFrame(row)]).reset_index(drop=True)

    def add_row_to_target_data_local(self, row, household):
        self.target_data_local[household] = pd.concat([self.target_data_local[household], pd.DataFrame(row)]).reset_index(drop=True)

    @staticmethod
    def process_series_to_database_row(row, cols):
        row_tmp = pd.DataFrame(row).reset_index(drop=True).transpose()
        row_tmp.columns = cols
        return row_tmp

    def get_training_data_global(self):
        return self.training_data_global

    def get_training_data_global_normalized(self):
        return self.training_data_global_normalized

    def get_target_data_global(self):
        return self.target_data_global

    def add_training_data_global_normalized(self, data):
        df = pd.DataFrame(data)
        df.columns = list(map(str, list(np.arange(0, TIME_POINTS_PER_DAY * self.historical_days))))
        self.training_data_global_normalized = df

    @staticmethod
    def convert_json_to_df(dictionary):
        for household in dictionary:
            dictionary[household] = pd.read_json(dictionary[household])
        return dictionary

    def get_training_data_local(self, household):
        return self.training_data_local[household]

    def get_target_data_local(self, household):
        return self.target_data_local[household]
