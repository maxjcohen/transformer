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
    layers_encoding: :py:class:`list` of :class:`torch.nn.Module`
        stack of Encoder layers.
    layers_decoding: :py:class:`list` of :class:`torch.nn.Module`
        stack of Decoder layers.
    embedding: :class:`torch.nn.Linear`
        Fully connected layer acting as embedding.
    linear: :class:`torch.nn.Linear`
        Fully connected output layer.

    Parameters
    ----------
    d_input: :py:class:`int`
        Model input dimension.
    d_model: :py:class:`int`
        Dimension of the input vector.
    d_output: :py:class:`int`
        Model output dimension.
    q: :py:class:`int`
        Dimension of queries and keys.
    v: :py:class:`int`
        Dimension of values.
    h: :py:class:`int`
        Number of heads.
    k: :py:class:`int`
        Time length.
    N: :py:class:`int`
        Number of encoder and decoder layers to stack.
    """
    def __init__(self, d_input, d_model, d_output, q, v, h, k, N):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()
    
        self._layers_encoding = [Encoder(d_model, q, v, h, k) for _ in range(N)]
        self._layers_decoding = [Decoder(d_model, q, v, h, k) for _ in range(N)]
        
        self._embedding = nn.Linear(d_input, d_model)
        self._linear = nn.Linear(d_model, d_output)
        
    def forward(self, x):
        """Propagate input through transformer

        Forward input through an embedding module,
        the encoder then decoder stacks, and an output module.

        Parameters
        ----------
        x: Tensor
            Tensor of shape (batch_size, K, d_input).
        
        Returns
        -------
        output: Tensor
            Output tensor with shape (batch_size, K, d_output).
        """
        # Embeddin module
        encoding = self._embedding(x)
        
        # Encoding stack
        for layer in self._layers_encoding:
            encoding = layer(encoding)
        
        # Decoding stack
        decoding = encoding
        for layer in self._layers_decoding:
            decoding = layer(decoding, encoding)
        
        # Output module
        output = self._linear(decoding)
        output = torch.sigmoid(output)
        return output