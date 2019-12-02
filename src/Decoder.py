import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.blocks import MultiHeadAttention, MultiHeadAttentionChunk, PositionwiseFeedForward
from src.utils import generate_original_PE, generate_regular_PE


class Decoder(nn.Module):
    """Decoder block from Attention is All You Need.

    Apply two Multi Head Attention block followed by a Point-wise Feed Forward block.
    Residual sum and normalization are applied at each step.

    Parameters
    ----------
    d_model: 
        Dimension of the input vector.
    q:
        Dimension of all query matrix.
    v:
        Dimension of all value matrix.
    h:
        Number of heads.
    k:
        Time window length.
    time_chunk:
        If True, will divide time dimension in chunks.
        Default True.
    """

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 k: int,
                 time_chunk: bool = True):
        """Initialize the Decoder block"""
        super().__init__()

        if time_chunk:
            from src.blocks import MultiHeadAttentionChunk as MultiHeadAttention
        else:
            from src.blocks import MultiHeadAttention

        self._selfAttention = MultiHeadAttention(d_model, q, v, h, k)
        self._encoderDecoderAttention = MultiHeadAttention(d_model, q, v, h, k)
        self._feedForward = PositionwiseFeedForward(d_model)

        self._layerNorm1 = nn.LayerNorm(d_model)
        self._layerNorm2 = nn.LayerNorm(d_model)
        self._layerNorm3 = nn.LayerNorm(d_model)

        self._PE = generate_regular_PE(k, d_model)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Propagate the input through the Decoder block.

        Apply the self attention block, add residual and normalize.
        Apply the encoder-decoder attention block, add residual and normalize.
        Apply the feed forward network, add residual and normalize.

        Parameters
        ----------
        x:
            Input tensor with shape (batch_size, K, d_model).
        memory:
            Memory tensor with shape (batch_size, K, d_model)
            from encoder output.

        Returns
        -------
        x:
            Output tensor with shape (batch_size, K, d_model).
        """
        # Add position encoding
        x.add_(self._PE)

        # Self attention
        residual = x
        x = self._selfAttention(query=x, key=x, value=x, mask="subsequent")
        x.add_(residual)
        x = self._layerNorm1(x)

        # Encoder-decoder attention
        residual = x
        x = self._selfAttention(query=x, key=memory, value=memory)
        x.add_(residual)
        x = self._layerNorm2(x)

        # Feed forward
        redisual = x
        x = self._feedForward(x)
        x.add_(residual)
        x = self._layerNorm3(x)

        return x
