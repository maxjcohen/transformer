import torch


def MSE(y_true, y_pred, occupation=None, idx_label=None, reduction='mean'):
    # Select output labels
    idx_label = idx_label or torch.arange(y_true.shape[-1])

    # Compute squared difference
    diff = torch.pow(y_true[..., idx_label]-y_pred[..., idx_label], 2)

    if occupation is not None:
        # Add dimension for broacasting
        occupation = occupation.unsqueeze(-1)

        # Mask with occupation
        diff = diff * occupation

    if reduction == 'mean':
        return torch.mean(diff).item()
    elif reduction == 'none':
        return torch.mean(diff, dim=(1, 2)).numpy()
