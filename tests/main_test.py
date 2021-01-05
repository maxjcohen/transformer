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
from tst import Transformer

@pytest.mark.parametrize('device', ['cuda', 'cpu'])
def test_transformer_tsp(device):
    if device == 'cuda':
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
    print("Fitting in {} time delta: {}".format(device, elapsed))