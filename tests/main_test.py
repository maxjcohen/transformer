"""
main test script
To run issue the command pytest at the root folder of the project.
"""
import time
from datetime import timedelta

import pytest
import torch
from flights_time_series_dataset import FlightsDataset
from time_series_predictor import TimeSeriesPredictor
from time_series_transformer import Transformer


def test_transformer_tsp():
    """
    Tests the TransformerTimeSeriesPredictor
    """
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA compatible GPU available to run this test")

    start = time.time()
    tsp = TimeSeriesPredictor(
        Transformer(),
        max_epochs=50,
        train_split=None,
    )

    tsp.fit(FlightsDataset())
    score = tsp.score(tsp.dataset)
    assert score > -1
    end = time.time()
    elapsed = timedelta(seconds = end - start)
    print("Fitting in GPU time delta: {}".format(elapsed))

def test_transformer_tsp_in_cpu():
    """
    Tests the TransformerTimeSeriesPredictor
    """
    start = time.time()
    tsp = TimeSeriesPredictor(
        Transformer(),
        max_epochs=50,
        train_split=None,
        device='cpu',
    )

    tsp.fit(FlightsDataset())
    score = tsp.score(tsp.dataset)
    assert score > -1
    end = time.time()
    elapsed = timedelta(seconds = end - start)
    print("Fitting in CPU time delta: {}".format(elapsed))
