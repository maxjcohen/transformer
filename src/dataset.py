import json

import numpy as np
from torch.utils.data import Dataset
import torch

class OzeDataset(Dataset):
    """Torch dataset for Oze datachallenge.

    Load dataset from a single npz file.

    Attributes
    ----------
    labels: :py:class:`dict`
        Ordered labels list for R, Z and X.
    m: :class:`numpy.ndarray`
        Normalization constant.
    M: :class:`numpy.ndarray`
        Normalization constant.

    Parameters
    ---------
    dataset_x: :py:class:`str` or :class:`pathlib.Path`
        Path to the dataset inputs as npz.
    labels_path: :py:class:`str` or :class:`pathlib.Path`, optional
        Path to the labels, divided in R, Z and X, in json format.
        Default is "labels.json".
    normalize: :py:class:`str`, optional
        Data normalization method, one of "mean", "max".
        Default is "max".
    """

    def __init__(self, dataset_path, labels_path="labels.json", normalize="max", **kwargs):
        """Load dataset from npz."""
        super().__init__(**kwargs)

        self._load_npz(dataset_path, labels_path, normalize)

    def _load_npz(self, dataset_path, labels_path, normalize):
        """Load dataset from csv and create x_train and y_train tensors."""
        # Load dataset as csv
        dataset = np.load(dataset_path)

        # Load labels, can be found through csv or challenge description
        with open(labels_path, "r") as stream_json:
            self.labels = json.load(stream_json)

        R, X, Z = dataset['R'].astype(np.float32), dataset['X'].astype(np.float32), dataset['Z'].astype(np.float32)
        m = Z.shape[0]  # Number of training example
        K = Z.shape[-1]  # Time serie length

        R = np.tile(R[:, np.newaxis, :], (1, K, 1))
        Z = Z.transpose((0, 2, 1))
        X = X.transpose((0, 2, 1))

        # Store R, Z and X as x and y
        self._x = np.concatenate([Z, R], axis=-1)
        self._y = X

        # Normalize
        if normalize == "mean":
            mean = np.mean(self._x, axis=(0, 1))
            std = np.std(self._x, axis=(0, 1))
            self._x = (self._x - mean) / (std + np.finfo(float).eps)

            self.mean = np.mean(self._y, axis=(0, 1))
            self.std = np.std(self._y, axis=(0, 1))
            self._y = (self._y - self.mean) / (self.std + np.finfo(float).eps)
        elif normalize == "max":
            M = np.max(self._x, axis=(0, 1))
            m = np.min(self._x, axis=(0, 1))
            self._x = (self._x - m) / (M - m + np.finfo(float).eps)

            self.M = np.max(self._y, axis=(0, 1))
            self.m = np.min(self._y, axis=(0, 1))
            self._y = (self._y - self.m) / (self.M - self.m + np.finfo(float).eps)
                
        # Convert to float32
        self._x = torch.Tensor(self._x)
        self._y = torch.Tensor(self._y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (self._x[idx], self._y[idx])

    def __len__(self):
        return self._x.shape[0]

class OzeDatasetWindow(OzeDataset):
    """Torch dataset with windowed time dimension.

    Load dataset from a single npz file.

    Attributes
    ----------
    labels: :py:class:`dict`
        Ordered labels list for R, Z and X.
    m: :class:`numpy.ndarray`
        Normalization constant.
    M: :class:`numpy.ndarray`
        Normalization constant.

    
    Parameters
    ---------
    dataset_x: :py:class:`str` or :class:`pathlib.Path`
        Path to the dataset inputs as npz.
    labels_path: :py:class:`str` or :class:`pathlib.Path`, optional
        Path to the labels, divided in R, Z and X, in json format.
        Default is "labels.json".
    window_size: :py:class:`int`, optional
        Size of the window to apply on time dimension.
        Default 5.
    padding: :py:class:`int`, optional
        Padding size to apply on time dimension windowing.
        Default 1.
    """

    def __init__(self, dataset_path, labels_path="labels.json", window_size=5, padding=1, **kwargs):
        """Load dataset from npz."""
        super().__init__(dataset_path, labels_path, **kwargs)

        self._window_dataset(window_size=window_size, padding=padding)

    def _window_dataset(self, window_size=5, padding=1):
        m, K, d_input = self._x.shape
        _, _, d_output = self._y.shape

        step = window_size - 2 * padding
        n_step = (K - window_size - 1) // step + 1

        dataset_x = np.empty((m, n_step, window_size, d_input), dtype=np.float32)
        dataset_y = np.empty((m, n_step, step, d_output), dtype=np.float32)

        for idx_step, idx in enumerate(range(0, K-window_size, step)):
            dataset_x[:, idx_step, :, :] = self._x[:, idx:idx+window_size, :]
            dataset_y[:, idx_step, :, :] = self._y[:, idx+padding:idx+window_size-padding, :]
            
        self._x = dataset_x
        self._y = dataset_y