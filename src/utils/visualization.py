import torch
import numpy as np
from matplotlib import pyplot as plt


def visual_sample(dataloader: torch.utils.data.DataLoader,
                  net: torch.nn.Module,
                  device: torch.device = 'cpu'):
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
    for idx_label, label in enumerate(dataloader.dataset.dataset.labels['X']):
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
            y_true = dataloader.dataset.dataset.rescale(y_true, idx_label)
            y_pred = dataloader.dataset.dataset.rescale(y_pred, idx_label)

        # Add title, axis and legend
        plt.plot(y_true, label="Truth")
        plt.plot(y_pred, label="Prediction")
        plt.title(label)
        plt.legend()

    # Plot ambient temperature
    plt.subplot(9, 1, idx_label+2)
    t_amb = x[:, dataloader.dataset.dataset.labels["Z"].index("TAMB")]
    t_amb = dataloader.dataset.dataset.rescale(t_amb, -1)
    plt.plot(t_amb, label="TAMB", c="red")
    plt.legend()
