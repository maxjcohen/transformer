import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.blocks import MultiHeadAttention, PositionwiseFeedForward
from src.utils import generate_positional_encoding

class Decoder(nn.Module):
    """Decoder block from Attention is All You Need.

    Apply two Multi Head Attention block followed by a Point-wise Feed Forward block.
    Residual sum and normalization are applied at each step.

    Parameters
    ----------
    d_model: :py:class:`int`
        Dimension of the input vector.
    q: :py:class:`int`
        Dimension of all query matrix.
    v: :py:class:`int`
        Dimension of all value matrix.
    h: :py:class:`int`
        Number of heads.
    k: :py:class:`int`
        Time window length.
    """
    def __init__(self, d_model, q, v, h, k):
        """Initialize the Decoder block"""
        super().__init__()
        
        self._selfAttention = MultiHeadAttention(d_model, q, v, h, k)
        self._encoderDecoderAttention = MultiHeadAttention(d_model, q, v, h, k)
        self._feedForward = PositionwiseFeedForward(d_model)
        
        self._layerNorm1 = nn.LayerNorm(d_model)
        self._layerNorm2 = nn.LayerNorm(d_model)
        self._layerNorm3 = nn.LayerNorm(d_model)

        self._PE = generate_positional_encoding(k, d_model)
        
    def forward(self, x, memory):
        """Propagate the input through the Decoder block.

        Apply the self attention block, add residual and normalize.
        Apply the encoder-decoder attention block, add residual and normalize.
        Apply the feed forward network, add residual and normalize.

        Parameters
        ----------
        x: :class:`torch.Tensor`
            Input tensor with shape (batch_size, K, d_model).
        memory: :class:`torch.Tensor`
            Memory tensor with shape (batch_size, K, d_model)
            from encoder output.
        
        Returns
        -------
        x: :class:`torch.Tensor`
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