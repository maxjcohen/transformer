import torch

def MSE(y_true, y_pred, occupation=None):
    # Compute squared difference
    diff = torch.pow(y_true-y_pred, 2)
    
    if occupation is not None:
        # Add dimension for broacasting
        occupation.unsqueeze_(-1)

        # Mask with occupation
        diff = diff * occupation
    
    # Return reduced sum
    return torch.sum(diff) / torch.prod(torch.Tensor([*y_true.shape]))