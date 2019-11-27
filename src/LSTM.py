# coding: UTF-8

import torch
import torch.nn as nn


class LSTMBenchmark(nn.Module):
    """Benchmark LSTM.

    Parameters
    ----------
    input_dim: :py:class:`int`
        Input dimension.
    hidden_dim: :py:class:`int`
        Latent dimension.
    output_dim: :py:class:`int`
        Output dimension.
    num_layers: :py:class:`int`
        Number of LSTM layers.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, **kwargs):
        super().__init__(**kwargs)

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Propagate input through the network.

        Parameters
        ----------
        x: :class:`torch.Tensor`
            Input tensor with shape (m, K, input_dim)

        Returns
        -------
        output: :class:`torch.Tensor`
            Output tensor with shape (m, K, output_dim)
        """
        lstm_out, _ = self.lstm(x)
        output = self.linear(lstm_out)
        return output
