"""
time_series_dataset
"""

from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    """
    TimeSeriesDataset
    """
    def __init__(self, _x, _y, labels):
        super().__init__()
        self.labels = labels
        self._x = _x
        self._y = _y
        self.original_y = None
        self.M = None
        self.m = None
        self._normalize()

    def __getitem__(self, idx):
        return (self._x[idx], self._y[idx])

    def __len__(self):
        return self._x.shape[0]

    def get_x_shape(self):
        """get_x_shape"""
        return self._x.shape

    def get_y_shape(self):
        """get_y_shape"""
        return self._y.shape

    def _normalize(self):
        # Normalize x
        M = np.max(self._x, axis=(0, 1))
        m = np.min(self._x, axis=(0, 1))
        self._x = (self._x - m) / (M - m + np.finfo(float).eps)

        self.original_y = np.array(self._y).astype(np.float32)

        # Normalize y
        self.M = np.max(self._y, axis=(0, 1))
        self.m = np.min(self._y, axis=(0, 1))
        self._y = (self._y - self.m) / (self.M - self.m + np.finfo(float).eps)
