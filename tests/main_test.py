"""
main test script
To run issue the command pytest at the root folder of the project.
"""
from src.lstm_tsp import LSTMTimeSeriesPredictor
from src.transformer_tsp import TransformerTimeSeriesPredictor
from .flights_dataset import FlightsDataset
import pytest

@pytest.mark.skip
def test_lstm_tsp():
    """
    Tests the LSTMTimeSeriesPredictor
    """
    tsp = LSTMTimeSeriesPredictor(epochs=50)

    tsp.fit(FlightsDataset())
    mean_loss = tsp.compute_mean_loss(tsp.dataloader)
    assert mean_loss < 0.01

def test_transformer_tsp():
    """
    Tests the TransformerTimeSeriesPredictor
    """
    tsp = TransformerTimeSeriesPredictor()

    tsp.fit(FlightsDataset())
    mean_loss = tsp.compute_mean_loss(tsp.dataloader)
    assert mean_loss < 0.03
