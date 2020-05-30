"""
FlightsDataset
"""

import calendar

import numpy as np
import pandas as pd
import seaborn as sns

from src.time_series_dataset import TimeSeriesDataset


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

        passengers = pd.DataFrame(passengers)
        month_number = pd.DataFrame(month_number)
        year = pd.DataFrame(year)

        number_of_training_examples = 1
        # Store month_number and year as _x
        _x = np.concatenate([month_number, year], axis=-1)
        d_input = _x.shape[1]
        _x = _x.reshape(number_of_training_examples, -1, d_input).astype(np.float32)

        # Store passengers as _y
        d_output = 1
        _y = passengers.values.reshape(number_of_training_examples, -1, d_output).astype(np.float32)

        super().__init__(_x, _y)
