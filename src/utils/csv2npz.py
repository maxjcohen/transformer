"""
Convert .csv files to .npz
"""
import json
import os

import numpy as np
import pandas as pd

def csv2npz(dataset_x_path, dataset_y_path, output_path, filename, labels_path='labels.json'):
    """Load input dataset from csv and create x_train tensor."""
    # Load dataset as csv
    x = pd.read_csv(dataset_x_path)
    y = pd.read_csv(dataset_y_path)

    # Load labels, file can be found in challenge description
    with open(labels_path, "r") as stream_json:
        labels = json.load(stream_json)

    m = x.shape[0]
    K = 672  # Can be found through csv

    # Create R and Z
    R = x[labels["R"]].values
    R = R.astype(np.float32)

    X = y[[f"{var_name}_{i}" for var_name in labels["X"]
           for i in range(K)]]
    X = X.values.reshape((m, -1, K))
    X = X.astype(np.float32)

    Z = x[[f"{var_name}_{i}" for var_name in labels["Z"]
           for i in range(K)]]
    Z = Z.values.reshape((m, -1, K))
#     Z = Z.transpose((0, 2, 1))
    Z = Z.astype(np.float32)

    np.savez(os.path.join(output_path, filename+'.npz'), R=R, X=X, Z=Z)
