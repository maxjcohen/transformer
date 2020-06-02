"""
lstm_tsp
"""
import torch

from .model import BenchmarkLSTM
from .time_series_predictor import TimeSeriesPredictor

class LSTMTimeSeriesPredictor(TimeSeriesPredictor):
    """
    TransformerTimeSeriesPredictor
    """
    def __init__(self,
                 learning_rate=1e-2,
                 epochs=5,
                 hidden_dim=100,
                 num_layers=3):
        super().__init__(learning_rate, epochs)
        self.hidden_dim = hidden_dim    # Number of neurons in hidden layers
        self.num_layers = num_layers    # Number of layers

    # pylint: disable=arguments-differ
    def fit(self, dataset, loss_function=torch.nn.MSELoss()):
        """fit"""
        d_input = dataset.get_x_shape()[2]     # From dataset
        d_output = dataset.get_y_shape()[2]    # From dataset
        net = BenchmarkLSTM(input_dim=d_input, hidden_dim=self.hidden_dim,
                            output_dim=d_output, num_layers=self.num_layers)
        return super().fit(dataset, net, loss_function=loss_function)
