from typing import Optional, Union

import numpy as np
from matplotlib import pyplot as plt
import torch


def generate_original_PE(length: int, d_model: int) -> torch.Tensor:
    """Generate positional encoding as described in original paper.  :class:`torch.Tensor`

    Parameters
    ----------
    length:
        Time window length, i.e. K.
    d_model:
        Dimension of the model vector.

    Returns
    -------
        Tensor of shape (K, d_model).
    """
    PE = torch.zeros((length, d_model))

    pos = torch.arange(length).unsqueeze(1)
    PE[:, 0::2] = torch.sin(
        pos / torch.pow(1000, torch.arange(0, d_model, 2, dtype=torch.float32)/d_model))
    PE[:, 1::2] = torch.cos(
        pos / torch.pow(1000, torch.arange(1, d_model, 2, dtype=torch.float32)/d_model))

    return PE


def generate_regular_PE(length: int, d_model: int, period: Optional[int] = 24) -> torch.Tensor:
    """Generate positional encoding with a given period.

    Parameters
    ----------
    length:
        Time window length, i.e. K.
    d_model:
        Dimension of the model vector.
    period:
        Size of the pattern to repeat.
        Default is 24.

    Returns
    -------
        Tensor of shape (K, d_model).
    """
    PE = torch.zeros((length, d_model))

    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    PE = torch.sin(pos * 2 * np.pi / period)
    PE = PE.repeat((1, d_model))

    return PE


def visual_sample(dataloader: torch.utils.data.DataLoader,
                  net: torch.nn.Module,
                  device: Union[torch.device, str, None] = 'cpu'):
    """Plot prediction and ground truth for a dataset sample.

    Parameters
    ----------
    dataloader:
        PyTorch DataLoader containing the dataset.
    net:
        PyTorch model.
    device:
        Device to run predictions on. All other computations are done on cpu.
        Default is 'cpu'.
    """
    # Select training example
    idx = np.random.randint(0, len(dataloader.dataset))
    x, y = dataloader.dataset[idx]

    # Run predictions
    with torch.no_grad():
        netout = net(torch.Tensor(x[np.newaxis, ...]).to(device)).cpu()

    plt.figure(figsize=(30, 30))
    for idx_label, label in enumerate(dataloader.dataset.labels['X']):
        # Select real temperature
        y_true = y[:, idx_label]
        y_pred = netout[0, :, idx_label].numpy()

        # Select subplot
        plt.subplot(9, 1, idx_label+1)

        # If consumption, rescale axis
        if label.startswith('Q_'):
            plt.ylim(-0.1, 1.1)
        # If temperature, rescale output
        elif label == 'T_INT_OFFICE':
            y_true = dataloader.dataset.rescale(y_true, idx_label)
            y_pred = dataloader.dataset.rescale(y_pred, idx_label)

        # Add title, axis and legend
        plt.plot(y_true, label="Truth")
        plt.plot(y_pred, label="Prediction")
        plt.title(label)
        plt.legend()

    # Plot ambient temperature
    plt.subplot(9, 1, idx_label+2)
    t_amb = x[:, dataloader.dataset.labels["Z"].index("TAMB")]
    t_amb = dataloader.dataset.rescale(t_amb, -1)
    plt.plot(t_amb, label="TAMB", c="red")
    plt.legend()


def compute_loss(net, dataloader, loss_function, device='cpu'):
    running_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            netout = net(x.to(device)).cpu()
            running_loss += loss_function(y, netout)

    return running_loss / len(dataloader)
