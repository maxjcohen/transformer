import torch

def generate_positional_encoding(length, d_model):
    """Generate positional encoding as described in original paper.

    Parameters
    ----------
    length: int
        Time window length, i.e. K.
    d_model: int
        Dimension of the model vector.

    Returns
    -------
    PE: Tensor
        Tensor of shape (K, d_model).
    """
    PE = torch.zeros((length, d_model))

    pos = torch.arange(length).unsqueeze(1)
    PE[:, 0::2] = torch.sin(pos / torch.pow(1000, torch.arange(0, d_model, 2, dtype=torch.float32)/d_model))
    PE[:, 1::2] = torch.cos(pos / torch.pow(1000, torch.arange(1, d_model, 2, dtype=torch.float32)/d_model))
    
    return PE