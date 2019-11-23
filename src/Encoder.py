import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.blocks import MultiHeadAttention, PositionwiseFeedForwad

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
    """
    def __init__(self, d_model, q, v, h):
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
        """
        super().__init__()
        
        self._selfAttention = MultiHeadAttention(d_model, q, v, h)
        self._feedForward = PositionwiseFeedForwad(d_model)
        
        self._layerNorm1 = nn.LayerNorm(d_model)
        self._layerNorm2 = nn.LayerNorm(d_model)
        
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
        residual = x
        x = self._selfAttention(query=x, key=x, value=x)
        x.add_(residual)
        x = self._layerNorm1(x)
        
        redisual = x
        x = self._feedForward(x)
        x.add_(residual)
        x = self._layerNorm2(x)
        
        return x