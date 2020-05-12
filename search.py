import itertools
import datetime
import json
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from tst import Transformer
from tst.loss import OZELoss

from src.dataset import OzeDataset
from src.utils import compute_loss, fit, Logger

# ===== user set params ====
search_params = OrderedDict({
    "d_model": [32],
    "q": [8],
    "v": [8],
    "h": [2, 4, 8],
    "N": [2],
    "attention_size": [12],
})

# Training parameters
DATASET_PATH = 'datasets/dataset.npz'
BATCH_SIZE = 4
NUM_WORKERS = 4
LR = 2e-4
EPOCHS = 30
# ===== user set params ====

# Model parameters
dropout = 0.2  # Dropout rate
pe = None  # Positional encoding
chunk_mode = None

d_input = 38  # From dataset
d_output = 8  # From dataset

# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

# Define loss function
loss_function = OZELoss(alpha=0.3)

# Load dataset
ozeDataset = OzeDataset(DATASET_PATH)

# Split between train and val
dataset_train, dataset_val = random_split(
    ozeDataset,
    (int(len(ozeDataset)*0.9), len(ozeDataset) - int(len(ozeDataset)*0.9)))

dataloader_train = DataLoader(dataset_train,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=NUM_WORKERS
                              )

dataloader_val = DataLoader(dataset_val,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=NUM_WORKERS
                            )

# Start search
n_steps = np.prod([len(search_range)
                   for search_range in search_params.values()])

logger = Logger('logs/search_log.csv', list(search_params.keys()) + ['loss'])

with tqdm(total=n_steps*EPOCHS) as pbar:
    for params in itertools.product(*search_params.values()):
        params = {key: params[idx]
                  for idx, key in enumerate(search_params.keys())}
        pbar.set_postfix(params)

        # Load transformer with Adam optimizer and MSE loss function
        net = Transformer(d_input=d_input,
                          d_output=d_output,
                          dropout=dropout,
                          chunk_mode=chunk_mode,
                          pe=pe,
                          **params).to(device)
        optimizer = optim.Adam(net.parameters(), lr=LR)

        # Fit model
        loss = fit(net, optimizer, loss_function, dataloader_train,
                   dataloader_val, epochs=EPOCHS, pbar=pbar, device=device)

        # Log
        logger.log(loss=loss, **params)
