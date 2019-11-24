import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.blocks import MultiHeadAttention, PositionwiseFeedForward
from src.utils import generate_positional_encoding

class Encoder(nn.Module):
    """Encoder block from Attention is All You Need.

    Apply Multi Head Attention block followed by a Point-wise Feed Forward block.
    Residual sum and normalization are applied at each step.

    Attributes
    ----------
    selfAttention: Module
        Multi Head Attention block.
    feedForward: Module
        Point-wise Feed Forward block.
    layerNorm1: LayerNorm
        First normalization layer from the paper `Layer Normalization`.
    layerNorm2: LayerNorm
        Second normalization layer from the paper `Layer Normalization`.
    PE: Tensor
        Position encoding.
    """
    def __init__(self, d_model, q, v, h, k):
        """Initialize the Encoder block

        Parameters
        ----------
        d_model: int
            Dimension of the input vector.
        q: int
            Dimension of all query matrix.
        v: int
            Dimension of all value matrix.
        h: int
            Number of heads.
        k: int
            Time window length.
        """
        super().__init__()
        
        self._selfAttention = MultiHeadAttention(d_model, q, v, h)
        self._feedForward = PositionwiseFeedForward(d_model)
        
        self._layerNorm1 = nn.LayerNorm(d_model)
        self._layerNorm2 = nn.LayerNorm(d_model)

        self._PE = generate_positional_encoding(k, d_model)
        
    def forward(self, x):
        """Propagate the input through the Encoder block.

        Apply the Multi Head Attention block, add residual and normalize.
        Apply the Point-wise Feed Forward block, add residual and normalize.

        Parameters
        ----------
        x: Tensor
            Input tensor with shape (batch_size, K, d_model).
        
        Returns
        -------
        x: Tensor
            Output tensor with shape (batch_size, K, d_model).
        """
        # Add position encoding
        x.add_(self._PE)

        # Self attention
        residual = x
        x = self._selfAttention(query=x, key=x, value=x)
        x.add_(residual)
        x = self._layerNorm1(x)
        
        # Feed forward
        redisual = x
        x = self._feedForward(x)
        x.add_(residual)
        x = self._layerNorm2(x)
        
        return x