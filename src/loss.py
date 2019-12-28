import torch
import torch.nn as nn


class OZELoss(nn.Module):
    """Custom loss for TRNSys metamodel
    """

    def __init__(self, alpha: float = 0.3):
        super().__init__()

        self.alpha = alpha
        self.base_loss = nn.MSELoss()

    def forward(self, y_true, y_pred):
        delta_Q = self.base_loss(y_pred[..., :-1], y_true[..., :-1])
        delta_T = self.base_loss(y_pred[..., -1], y_true[..., -1])

        return torch.log(1 + delta_T) + self.alpha * torch.log(1 + delta_Q)
