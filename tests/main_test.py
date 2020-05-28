"""
main test script
To run issue the command pytest at the root folder of the project.
"""
from pathlib import Path

from time_series_transformer.loss import OZELoss

from src.dataset import OzeNPZDataset
from src.lstm_tsp import LSTMTimeSeriesPredictor
from src.transformer_tsp import TransformerTimeSeriesPredictor
from src.utils import npz_check

def test_lstm_tsp():
    """
    Tests the LSTMTimeSeriesPredictor
    """
    tsp = LSTMTimeSeriesPredictor()
    dataset = OzeNPZDataset(dataset_path=npz_check(
        Path('datasets'), 'dataset'), labels_path="labels.json")

    tsp.fit(dataset, loss_function=OZELoss(alpha=0.3))
    mean_loss = tsp.compute_mean_loss(tsp.dataloader)
    assert mean_loss < 0.01

def test_transformer_tsp():
    """
    Tests the TransformerTimeSeriesPredictor
    """
    tsp = TransformerTimeSeriesPredictor()
    dataset = OzeNPZDataset(dataset_path=npz_check(
        Path('datasets'), 'dataset'), labels_path="labels.json")

    tsp.fit(dataset, loss_function=OZELoss(alpha=0.3))
    mean_loss = tsp.compute_mean_loss(tsp.dataloader)
    assert mean_loss < 0.01
