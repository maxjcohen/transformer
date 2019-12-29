import torch
import torch.nn as nn

from src.Encoder import Encoder
from src.Decoder import Decoder


class Transformer(nn.Module):
    """Transformer model from Attention is All You Need.

    A classic transformer model adapted for sequential data.
    Embedding has been replaced with a fully connected layer,
    the last layer softmax is now a sigmoid.

    Attributes
    ----------
    layers_encoding: :py:class:`list` of :class:`Encoder.Encoder`
        stack of Encoder layers.
    layers_decoding: :py:class:`list` of :class:`Decoder.Decoder`
        stack of Decoder layers.

    Parameters
    ----------
    d_input:
        Model input dimension.
    d_model:
        Dimension of the input vector.
    d_output:
        Model output dimension.
    q:
        Dimension of queries and keys.
    v:
        Dimension of values.
    h:
        Number of heads.
    k:
        Time length.
    N:
        Number of encoder and decoder layers to stack.
    dropout:
        Dropout probability after each MHA or PFF block.
        Default is ``0.3``.
    chunk_mode:
        Swict between different MultiHeadAttention blocks.
        One of ``'chunk'``, ``'window'`` or ``None``. Default is ``'chunk'``.
    pe:
        Type of positional encoding to add.
        Must be one of ``'original'``, ``'regular'`` or ``None``. Default is ``None``.
    """

    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 q: int,
                 v: int,
                 h: int,
                 k: int,
                 N: int,
                 dropout: float = 0.3,
                 chunk_mode: bool = True,
                 pe: str = None):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self.layers_encoding = nn.ModuleList([Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      k,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode,
                                                      pe=pe) for _ in range(N)])
        self.layers_decoding = nn.ModuleList([Decoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      k,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode,
                                                      pe=pe) for _ in range(N)])

        self._embedding = nn.Linear(d_input, d_model)
        self._linear = nn.Linear(d_model, d_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate input through transformer

        Forward input through an embedding module,
        the encoder then decoder stacks, and an output module.

        Parameters
        ----------
        x:
            :class:`torch.Tensor` of shape (batch_size, K, d_input).

        Returns
        -------
            Output tensor with shape (batch_size, K, d_output).
        """
        # Embeddin module
        encoding = self._embedding(x)

        # Encoding stack
        for layer in self.layers_encoding:
            encoding = layer(encoding)

        # Decoding stack
        decoding = encoding
        for layer in self.layers_decoding:
            decoding = layer(decoding, encoding)

        # Output module
        output = self._linear(decoding)
        output = torch.sigmoid(output)
        return output
