from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi Head Attention block from Attention is All You Need.

    Given 3 inputs of shape (batch_size, K, d_model), that will be used
    to compute query, keys and values, we output a self attention
    tensor of shape (batch_size, K, d_model).

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
    """

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 k: int):
        """Initialize the Multi Head Block."""
        super().__init__()

        self._K = k
        self._h = h

        # Query, keys and value matrices
        self._W_q = nn.Linear(d_model, q*self._h)
        self._W_k = nn.Linear(d_model, q*self._h)
        self._W_v = nn.Linear(d_model, v*self._h)

        # Output linear function
        self._W_o = nn.Linear(self._h*v, d_model)

        # Score mask for decoder
        self._scores_mask = torch.triu(torch.ones(
            (self._K, self._K)), diagonal=1).bool()

        # Score placeholder
        self._scores = None

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[str] = None) -> torch.Tensor:
        """Propagate forward the input through the MHB.

        We compute for each head the queries, keys and values matrices,
        followed by the Scaled Dot-Product. The result is concatenated 
        and returned with shape (batch_size, K, d_model).

        Parameters
        ----------
        query:
            Input tensor with shape (batch_size, K, d_model) used to compute queries.
        key:
            Input tensor with shape (batch_size, K, d_model) used to compute keys.
        value:
            Input tensor with shape (batch_size, K, d_model) used to compute values.
        mask:
            Mask to apply on scores before computing attention.
            One of "subsequent", None. Default is None.

        Returns
        -------
            Self attention tensor with shape (batch_size, K, d_model).
        """
        # Compute Q, K and V, concatenate heads on batch dimension
        queries = torch.cat(self._W_q(query).chunk(self._h, dim=-1), dim=0)
        keys = torch.cat(self._W_k(key).chunk(self._h, dim=-1), dim=0)
        values = torch.cat(self._W_v(value).chunk(self._h, dim=-1), dim=0)

        # Scaled Dot Product
        self._scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(self._K)

        # Mask scores
        if mask == "subsequent":
            self._scores = self._scores.masked_fill(self._scores_mask, float('-inf'))

        self._scores = F.softmax(self._scores, dim=-1)

        attention = torch.bmm(self._scores, values)

        # Concatenat the heads
        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)

        # Apply linear transformation W^O
        self_attention = self._W_o(attention_heads)

        return self_attention

    @property
    def attention_map(self) -> torch.Tensor:
        """Attention map after a forward propagation,
        variable `score` in the original paper.
        """
        if self._scores is None:
            raise RuntimeError(
                "Evaluate the model once to generate attention map")
        return self._scores


class MultiHeadAttentionChunk(MultiHeadAttention):
    """Multi Head Attention block with chunk.

    Given 3 inputs of shape (batch_size, K, d_model), that will be used
    to compute query, keys and values, we output a self attention
    tensor of shape (batch_size, K, d_model).
    Queries, keys and values are divided in chunks of constant size.

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
    """

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 k: int,
                 **kwargs):
        """Initialize the Multi Head Block."""
        super().__init__(d_model, q, v, h, k, **kwargs)

        self._n_chunk = self._K // 168

        # Score mask for decoder
        self._scores_mask = nn.Parameter(torch.triu(torch.ones((self._K // self._n_chunk, self._K // self._n_chunk)), diagonal=1).bool(), requires_grad=False)


    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[str] = None) -> torch.Tensor:
        """Propagate forward the input through the MHB.

        We compute for each head the queries, keys and values matrices,
        followed by the Scaled Dot-Product. The result is concatenated 
        and returned with shape (batch_size, K, d_model).

        Parameters
        ----------
        query:
            Input tensor with shape (batch_size, K, d_model) used to compute queries.
        key:
            Input tensor with shape (batch_size, K, d_model) used to compute keys.
        value:
            Input tensor with shape (batch_size, K, d_model) used to compute values.
        mask:
            Mask to apply on scores before computing attention.
            One of "subsequent", None. Default is None.

        Returns
        -------
            Self attention tensor with shape (batch_size, K, d_model).
        """
        # Compute Q, K and V, concatenate heads on batch dimension
        queries = torch.cat(torch.cat(self._W_q(query).chunk(self._h, dim=-1), dim=0).chunk(self._n_chunk, dim=1), dim=0)
        keys = torch.cat(torch.cat(self._W_k(key).chunk(self._h, dim=-1), dim=0).chunk(self._n_chunk, dim=1), dim=0)
        values = torch.cat(torch.cat(self._W_v(value).chunk(self._h, dim=-1), dim=0).chunk(self._n_chunk, dim=1), dim=0)

        # Scaled Dot Product
        self._scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(self._K)

        if mask == "subsequent":
            self._scores = self._scores.masked_fill(self._scores_mask, float('-inf'))

        self._scores = F.softmax(self._scores, dim=-1)

        attention = torch.bmm(self._scores, values)

        # Concatenat the heads
        attention_heads = torch.cat(torch.cat(attention.chunk(self._n_chunk, dim=0), dim=1).chunk(self._h, dim=0), dim=-1)

        # Apply linear transformation W^O
        self_attention = self._W_o(attention_heads)

        return self_attention