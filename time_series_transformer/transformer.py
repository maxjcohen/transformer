"""
Transformer
"""
import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder
from .utils import generate_original_PE, generate_regular_PE


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
    input_dim:
        Model input dimension.
    hidden_dim:
        Dimension of the input vector.
    output_dim:
        Model output dimension.
    q:
        Dimension of queries and keys.
    v:
        Dimension of values.
    h:
        Number of heads.
    N:
        Number of encoder and decoder layers to stack.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
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
                 input_dim: int = 1,
                 hidden_dim: int = 32,
                 output_dim: int = 1,
                 q: int = 4,
                 v: int = 4,
                 h: int = 4,
                 N: int = 4,
                 attention_size: int = 6,
                 dropout: float = 0.2,
                 chunk_mode: bool = None,
                 pe: str = None,
                 **kwargs):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._hidden_dim = hidden_dim

        self.layers_encoding = nn.ModuleList([Encoder(hidden_dim,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode,
                                                      **kwargs) for _ in range(N)])
        self.layers_decoding = nn.ModuleList([Decoder(hidden_dim,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode,
                                                      **kwargs) for _ in range(N)])

        self._embedding = nn.Linear(input_dim, hidden_dim)
        self._linear = nn.Linear(hidden_dim, output_dim)

        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
        elif pe is None:
            self._generate_PE = None
        else:
            # pylint: disable=line-too-long
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate input through transformer

        Forward input through an embedding module,
        the encoder then decoder stacks, and an output module.

        Parameters
        ----------
        x:
            :class:`torch.Tensor` of shape (batch_size, K, input_dim).

        Returns
        -------
            Output tensor with shape (batch_size, K, output_dim).
        """
        x_shape_length = len(x.shape)
        if x_shape_length > 1:
            K = x.shape[1]
        elif x_shape_length == 1:
            K = 1
        else:
            K = 0

        # Embedding module
        encoding = self._embedding(x)

        # Add position encoding
        if self._generate_PE is not None:
            positional_encoding = self._generate_PE(K, self._hidden_dim)
            positional_encoding = positional_encoding
            encoding.add_(positional_encoding)

        # Encoding stack
        for layer in self.layers_encoding:
            encoding = layer(encoding)

        # Decoding stack
        decoding = encoding

        # Add position encoding
        if self._generate_PE is not None:
            positional_encoding = self._generate_PE(K, self._hidden_dim)
            positional_encoding = positional_encoding
            decoding.add_(positional_encoding)

        for layer in self.layers_decoding:
            decoding = layer(decoding, encoding)

        # Output module
        output = self._linear(decoding)
        output = torch.sigmoid(output)
        return output
