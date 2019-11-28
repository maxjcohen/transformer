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
        """Initialize the Multi Head Block."""
        super().__init__()
        
        self._W_q = [nn.Linear(d_model, q) for _ in range(h)]
        self._W_k = [nn.Linear(d_model, q) for _ in range(h)]
        self._W_v = [nn.Linear(d_model, v) for _ in range(h)]
        
        self._W_o = nn.Linear(h*v, d_model)
        
        self._K = k
    def forward(self, query, key, value, mask=None):
        """Propagate forward the input through the MHB.

        We compute for each head the queries, keys and values matrices,
        followed by the Scaled Dot-Product. The result is concatenated 
        and returned with shape (batch_size, K, d_model).

        Parameters
        ----------
        query: :class:`torch.Tensor`
            Input tensor with shape (batch_size, K, d_model) used to compute queries.
        key: :class:`torch.Tensor`
            Input tensor with shape (batch_size, K, d_model) used to compute keys.
        value: :class:`torch.Tensor`
            Input tensor with shape (batch_size, K, d_model) used to compute values.
        mask: :py:class:`str`, optional
            Mask to apply on scores before computing attention.
            One of "subsequent", None. Default is None.

        Returns
        -------
        self_attention: :class:`torch.Tensor`
            Self attention tensor with shape (batch_size, K, d_model).
        """
        attention_heads = []
        for W_q, W_k, W_v in zip(self._W_q, self._W_k, self._W_v):
            queries = W_q(query)
            keys = W_k(key)
            values = W_v(value)

            # Scaled Dot Product
            scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(self._K)

            # Mask scores
            if mask == "subsequent":
                scores_mask = torch.triu(torch.ones((self._K, self._K)), diagonal=1).bool()
                scores = scores.masked_fill(scores_mask, float('-inf'))

            scores = F.softmax(scores, dim=-1)
            
            attention = torch.bmm(scores, values)
            attention_heads.append(attention)
        
        # Concatenat the heads
        attention_heads = torch.cat(attention_heads, dim=-1)
        
        # Apply linear transformation W^O
        self_attention = self._W_o(attention_heads)
        
        return self_attention

class MultiHeadAttentionChunk(MultiHeadAttention):
    """Multi Head Attention block with chunk.

    Given 3 inputs of shape (batch_size, K, d_model), that will be used
    to compute query, keys and values, we output a self attention
    tensor of shape (batch_size, K, d_model).
    Queries, keys and values are divided in chunks of constant size.

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
    def __init__(self, d_model, q, v, h, k, **kwargs):
        """Initialize the Multi Head Block."""
        super().__init__(d_model, q, v, h, k, **kwargs)

        self._n_chunk = self._K // 24
        
    def forward(self, query, key, value, mask=None):
        """Propagate forward the input through the MHB.

        We compute for each head the queries, keys and values matrices,
        followed by the Scaled Dot-Product. The result is concatenated 
        and returned with shape (batch_size, K, d_model).

        Parameters
        ----------
        query: :class:`torch.Tensor`
            Input tensor with shape (batch_size, K, d_model) used to compute queries.
        key: :class:`torch.Tensor`
            Input tensor with shape (batch_size, K, d_model) used to compute keys.
        value: :class:`torch.Tensor`
            Input tensor with shape (batch_size, K, d_model) used to compute values.
        mask: :py:class:`str`, optional
            Mask to apply on scores before computing attention.
            One of "subsequent", None. Default is None.

        Returns
        -------
        self_attention: :class:`torch.Tensor`
            Self attention tensor with shape (batch_size, K, d_model).
        """
        attention_heads = []
        for W_q, W_k, W_v in zip(self._W_q, self._W_k, self._W_v):
            queries = torch.cat(torch.chunk(W_q(query), self._n_chunk, dim=1), dim=0)
            keys = torch.cat(torch.chunk(W_k(key), self._n_chunk, dim=1), dim=0)
            values = torch.cat(torch.chunk(W_v(value), self._n_chunk, dim=1), dim=0)

            # Scaled Dot Product
            scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(self._K)

            # Mask scores
            if mask == "subsequent":
                scores_mask = torch.triu(torch.ones((self._K // self._n_chunk, self._K // self._n_chunk)), diagonal=1).bool()
                scores = scores.masked_fill(scores_mask, float('-inf'))

            scores = F.softmax(scores, dim=-1)

            attention = torch.bmm(scores, values)            
            attention_heads.append(torch.cat(torch.chunk(attention, self._n_chunk, dim=0), dim=1))
        
        # Concatenat the heads
        attention_heads = torch.cat(attention_heads, dim=-1)
        
        # Apply linear transformation W^O
        self_attention = self._W_o(attention_heads)
        
        return self_attention

class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed Forward Network block from Attention is All You Need.

    Apply two linear transformations to each input, separately but indetically. We
    implement them as 1D convolutions. Input and output have a shape (batch_size, d_model).

    Parameters
    ----------
    d_model: :py:class:`int`
        Dimension of input tensor.
    d_dd: :py:class:`int`, optional
        Dimension of hidden layer, default is 2048.
    """
    def __init__(self, d_model, d_ff=2048):
        """Initialize the PFF block."""
        super().__init__()
        
        self._linear1 = nn.Linear(d_model, d_ff)
        self._linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        """Propagate forward the input through the PFF block.

        Apply the first linear transformation, then a relu actvation,
        and the second linear transformation.

        Parameters
        ----------
        x: :class:`torch.Tensor`
            Input tensor with shape (batch_size, K, d_model).

        Returns
        -------
        x: :class:`torch.Tensor`
            Output tensor with shape (batch_size, K, d_model).
        """
        return self._linear2(F.relu(self._linear1(x)))
