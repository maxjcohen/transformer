# coding: UTF-8

import torch
import torch.nn as nn


class LSTMBenchmark(nn.Module):
    """Benchmark LSTM.

    Parameters
    ----------
    input_dim:
        Input dimension.
    hidden_dim:
        Latent dimension.
    output_dim:
        Output dimension.
    num_layers:
        Number of LSTM layers.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int,
                 **kwargs):
        super().__init__(**kwargs)

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate input through the network.

        Parameters
        ----------
        x:
            Input tensor with shape (m, K, input_dim)

        Returns
        -------
            Output tensor with shape (m, K, output_dim)
        """
        lstm_out, _ = self.lstm(x)
        output = self.linear(lstm_out)
        return output
