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


def plot_error(dataset, predictions, labels, indices=None, time_limit=None):
    y_hat = predictions
    
    if indices is not None:
        x = dataset._x[indices].numpy()
        y = dataset._y[indices].numpy()
    else:
        x = dataset._x.numpy()
        y = dataset._y.numpy()

    if time_limit is None:
        time_limit = x.shape[1]
        
    occupancy = (x[..., labels["Z"].index("occupancy")].mean(axis=0)>0.5).astype(float)[:time_limit]

    fig, axes = plt.subplots(8, 1)
    fig.set_figwidth(20)
    fig.set_figheight(40)
    plt.subplots_adjust(bottom=0.05)

    for idx_label, (label, ax) in enumerate(zip(labels['X'], axes)):
        # Select output to plot
        y_true = y[:, :time_limit, idx_label]
        y_pred = y_hat[:, :time_limit, idx_label]

        # Rescale
        y_true = dataset.rescale(y_true, idx_label)
        y_pred = dataset.rescale(y_pred, idx_label)

        # Convert kJ/h to kW
        if label.startswith('Q_'):
            y_true /= 3600
            y_pred /= 3600

        # Compute delta, mean and std
        delta = np.abs(y_true - y_pred)

        mean = delta.mean(axis=0)
        std = delta.std(axis=0)

        # Plot
        # Labels for consumption and temperature
        if label.startswith('Q_'):
            y_label_unit = 'kW'
        else:
            y_label_unit = '°C'

        # Occupancy
        occupancy_idxes = np.where(np.diff(occupancy) != 0)[0]
        for idx in range(0, len(occupancy_idxes), 2):
            ax.axvspan(occupancy_idxes[idx], occupancy_idxes[idx+1], facecolor='green', alpha=.15)

        # Std
        ax.fill_between(np.arange(time_limit), (mean - std), (mean + std), alpha=.4, label='std')

        # Mean
        ax.plot(mean, label='mean')

        # Title and labels
        ax.set_title(label)
        ax.set_xlabel('time', fontsize=16)
        ax.set_ylabel(y_label_unit, fontsize=16)

        ax.legend()


def plot_values(dataset, predictions, labels, indices=None, time_limit=None):
    y_hat = predictions
    
    if indices is not None:
        x = dataset._x[indices].numpy()
        y = dataset._y[indices].numpy()
    else:
        x = dataset._x.numpy()
        y = dataset._y.numpy()
        
    if time_limit is None:
        time_limit = x.shape[1]
        
    occupancy = (x[..., labels["Z"].index("occupancy")].mean(axis=0)>0.5).astype(float)[:time_limit]
        
    fig, axes = plt.subplots(8, 1)
    fig.set_figwidth(20)
    fig.set_figheight(40)
    plt.subplots_adjust(bottom=0.05)

    for idx_label, (label, ax) in enumerate(zip(labels['X'], axes)):
        # Select output to plot
        y_true = y[:, :time_limit, idx_label]
        y_pred = y_hat[:, :time_limit, idx_label]

        # Rescale
        y_true = dataset.rescale(y_true, idx_label)
        y_pred = dataset.rescale(y_pred, idx_label)

        # Convert kJ/h to kW
        if label.startswith('Q_'):
            y_true /= 3600
            y_pred /= 3600

        # Compute mean and std
        y_true_mean = y_true.mean(axis=0)
        y_true_std = y_true.std(axis=0)
        y_pred_mean = y_pred.mean(axis=0)
        y_pred_std = y_pred.std(axis=0)

        # Plot
        # Labels for consumption and temperature
        if label.startswith('Q_'):
            y_label_unit = 'kW'
        else:
            y_label_unit = '°C'

        # Occupancy
        occupancy_idxes = np.where(np.diff(occupancy) != 0)[0]
        for idx in range(0, len(occupancy_idxes), 2):
            ax.axvspan(occupancy_idxes[idx], occupancy_idxes[idx+1], facecolor='green', alpha=.15)

        # Std
        ax.fill_between(np.arange(time_limit), (y_true_mean - y_true_std), (y_true_mean + y_true_std), alpha=.4, color='darkgreen', label='true std')
        ax.fill_between(np.arange(time_limit), (y_pred_mean - y_pred_std), (y_pred_mean + y_pred_std), alpha=.4, color='b', label='pred std')

        # Mean
        ax.plot(y_true_mean, color='darkgreen', linewidth=3, label='true mean')
        ax.plot(y_pred_mean, color='b', label='pred mean')

        # Title and labels
        ax.set_title(label)
        ax.set_xlabel('time', fontsize=16)
        ax.set_ylabel(y_label_unit, fontsize=16)

        ax.legend()