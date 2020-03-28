import numpy as np


def plot_errorbar(y, ax, percentile=85, **kwargs):
    mean = y.mean(axis=0)
    yerr = np.stack([
        mean - np.percentile(y, 100-percentile, axis=0),
        np.percentile(y, percentile, axis=0) - mean
    ])

    ax.errorbar(np.arange(mean.shape[0]), mean, yerr=yerr, **kwargs)
