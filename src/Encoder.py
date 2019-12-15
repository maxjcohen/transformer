from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.blocks import MultiHeadAttention, MultiHeadAttentionChunk, PositionwiseFeedForward
from src.utils import generate_original_PE, generate_regular_PE


class Encoder(nn.Module):
    """Encoder block from Attention is All You Need.

    Apply Multi Head Attention block followed by a Point-wise Feed Forward block.
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
    pe:
        Type of positional encoding to add.
        Must be one of original, regular or None. Default is None.
    """

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 k: int,
                 time_chunk: Optional[bool] = True,
                 pe: Optional[str] = None):
        """Initialize the Encoder block"""
        super().__init__()

        if time_chunk:
            from src.blocks import MultiHeadAttentionChunk as MultiHeadAttention
        else:
            from src.blocks import MultiHeadAttention

        self._selfAttention = MultiHeadAttention(d_model, q, v, h, k)
        self._feedForward = PositionwiseFeedForward(d_model)

        self._layerNorm1 = nn.LayerNorm(d_model)
        self._layerNorm2 = nn.LayerNorm(d_model)

        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._PE = pe_functions[pe](k, d_model)
        elif pe is None:
            self._PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate the input through the Encoder block.

        Apply the Multi Head Attention block, add residual and normalize.
        Apply the Point-wise Feed Forward block, add residual and normalize.

        Parameters
        ----------
        x:
            Input tensor with shape (batch_size, K, d_model).

        Returns
        -------
            Output tensor with shape (batch_size, K, d_model).
        """
        # Add position encoding
        if self._PE is not None:
            x.add_(self._PE)

        # Self attention
        residual = x
        x = self._selfAttention(query=x, key=x, value=x)
        x.add_(residual)
        x = self._layerNorm1(x)

        # Feed forward
        residual = x
        x = self._feedForward(x)
        x.add_(residual)
        x = self._layerNorm2(x)

        return x

    @property
    def attention_map(self) -> torch.Tensor:
        """Attention map after a forward propagation,
        variable `score` in the original paper.
        """
        return self._selfAttention.attention_map
