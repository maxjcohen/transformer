"""
FlightsDataset
"""

import calendar

import numpy as np
import pandas as pd
import seaborn as sns

from src.time_series_dataset import TimeSeriesDataset

def _make_predictor(features, number_of_training_examples):
    #pylint: disable=too-many-function-args
    return np.concatenate(features, axis=-1).reshape(
        number_of_training_examples, -1, len(features)).astype(np.float32)

def _get_labels(input_features, output_features):
    def _features_to_label_list(features):
        return [list(feature)[0] for feature in features]
    labels = {}
    labels['x'] = _features_to_label_list(input_features)
    labels['y'] = _features_to_label_list(output_features)
    return labels

class FlightsDataset(TimeSeriesDataset):
    """
    FlightsDataset class
    """
    def __init__(self):
        flights_dataset = sns.load_dataset("flights")
        passengers = flights_dataset['passengers']
        month = flights_dataset['month']
        year = flights_dataset['year']

        month_number = [list(calendar.month_name).index(_month) for _month in month]

        passengers_df = pd.DataFrame(passengers)
        month_number_df = pd.DataFrame(data={'month_number': month_number})
        year_df = pd.DataFrame(year)

        number_of_training_examples = 1
        # Store month_number and year as _x
        input_features = [month_number_df, year_df]
        _x = _make_predictor(input_features, number_of_training_examples)

        # Store passengers as _y
        output_features = [passengers_df]
        _y = _make_predictor(output_features, number_of_training_examples)

        super().__init__(_x, _y, _get_labels(input_features, output_features))
