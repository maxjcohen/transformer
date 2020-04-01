import torch
import torch.nn as nn


class LSTM(nn.Module):
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
    dropout:
        Dropout value. Default is ``0``.
    bidirectional:
        If ``True``, becomes a bidirectional LSTM. Default: ``False``.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int,
                 dropout: float = 0,
                 bidirectional: bool = False,
                 **kwargs):
        super().__init__(**kwargs)

        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)

        if bidirectional:
            hidden_dim *= 2
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
        rnn_out, _ = self.rnn(x)
        output = self.linear(rnn_out)
        return output


class BiGRU(LSTM):
    """Benchmark Bidirictionnal GRU.

    Parameters
    ----------
    input_dim:
        Input dimension.
    hidden_dim:
        Latent dimension.
    output_dim:
        Output dimension.
    num_layers:
        Number of GRU layers.
    dropout:
        Dropout value. Default is ``0``.
    bidirectional:
        If ``True``, becomes a bidirectional GRU. Default: ``True``.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int,
                 dropout: float = 0,
                 bidirectional: bool = False,
                 **kwargs):
        super().__init__(input_dim, hidden_dim, output_dim, num_layers, dropout, bidirectional, **kwargs)

        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
