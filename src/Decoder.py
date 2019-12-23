from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.MultiHeadAttention import MultiHeadAttention, MultiHeadAttentionChunk, MultiHeadAttentionWindow
from src.PositionwiseFeedForward import PositionwiseFeedForward
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
    chunk_mode:
        Swict between different MultiHeadAttention blocks.
        One of "chunk", "window" or None, Default is 'chunk'.
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
                 chunk_mode: Union[str, None] = 'chunk',
                 pe: Optional[str] = None):
        """Initialize the Decoder block"""
        super().__init__()

        chunk_mode_modules = {
            'chunk': MultiHeadAttentionChunk,
            'window': MultiHeadAttentionWindow,
        }

        if chunk_mode in chunk_mode_modules.keys():
            MHA = chunk_mode_modules[chunk_mode]
        elif chunk_mode is None:
            MHA = MultiHeadAttention
        else:
            raise NameError(
                f'chunk_mode "{chunk_mode}" not understood. Must be one of {", ".join(chunk_mode_modules.keys())} or None.')

        self._selfAttention = MHA(d_model, q, v, h, k)
        self._encoderDecoderAttention = MHA(d_model, q, v, h, k)
        self._feedForward = PositionwiseFeedForward(d_model)

        self._layerNorm1 = nn.LayerNorm(d_model)
        self._layerNorm2 = nn.LayerNorm(d_model)
        self._layerNorm3 = nn.LayerNorm(d_model)

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
        if self._PE is not None:
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
        residual = x
        x = self._feedForward(x)
        x.add_(residual)
        x = self._layerNorm3(x)

        return x
