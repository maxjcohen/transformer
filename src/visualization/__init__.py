import numpy as np
from matplotlib import pyplot as plt

from .plot_functions import plot_values_distribution, plot_error_distribution, plot_errors_threshold, plot_visual_sample, plot_dataset_distribution


def map_plot_function_input(dataset, plot_function, plot_kwargs={}, dataset_indices=None, labels=None, time_limit=None):

    labels = labels or dataset.labels['Z']
    time_limit = time_limit or dataset._y.shape[1]
    if dataset_indices is not None:
        dataset_x = dataset._x[dataset_indices].numpy()
    else:
        dataset_x = dataset._x.numpy()

    # Create subplots
    fig, axes = plt.subplots(len(labels), 1)
    fig.set_figwidth(25)
    fig.set_figheight(5*len(labels))
    plt.subplots_adjust(bottom=0.05)

    # Fix for single label
    if len(labels) == 1:
        axes = [axes]

    for label, ax in zip(labels, axes):
        # Get label index from dataset
        idx_label = dataset.labels['Z'].index(label)

        # Select data for time period and label
        x = dataset_x[:, :time_limit, idx_label]

        plot_function(x, ax, **plot_kwargs)

        ax.set_title(label)

        n_ticks = time_limit // 24
        for idx, label in enumerate(ax.get_xticklabels()):
            if idx % n_ticks:
                label.set_visible(False)


def map_plot_function(dataset, predictions, plot_function, plot_kwargs={}, dataset_indices=None, labels=None, time_limit=None):

    labels = labels or dataset.labels['X']
    time_limit = time_limit or dataset._y.shape[1]
    if dataset_indices is not None:
        dataset_y = dataset._y[dataset_indices].numpy()
    else:
        dataset_y = dataset._y.numpy()

    # Create subplots
    fig, axes = plt.subplots(len(labels), 1)
    fig.set_figwidth(25)
    fig.set_figheight(5*len(labels))
    plt.subplots_adjust(bottom=0.05)

    # Fix for single label
    if len(labels) == 1:
        axes = [axes]

    for label, ax in zip(labels, axes):
        # Get label index from dataset
        idx_label = dataset.labels['X'].index(label)

        # Select data for time period and label
        y_pred = predictions[:, :time_limit, idx_label]
        y_true = dataset_y[:, :time_limit, idx_label]

        # Rescale data
        y_pred = dataset.rescale(y_pred, idx_label)
        y_true = dataset.rescale(y_true, idx_label)

        # If a consumption
        if label.startswith('Q_'):
            unit = 'kW'
        else:
            unit = '°C'

        plot_function(y_true, y_pred, ax, **plot_kwargs, unit=unit)

        ax.set_title(label)

        n_ticks = time_limit // 24
        for idx, label in enumerate(ax.get_xticklabels()):
            if idx % n_ticks:
                label.set_visible(False)
