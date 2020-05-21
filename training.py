import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import r2_score

from tst import Transformer
from tst.loss import OZELoss

from src.dataset import OzeDataset
from src.utils import compute_loss, fit, Logger, kfold
from src.benchmark import LSTM, BiGRU
from src.metrics import MSE


# Training parameters
DATASET_PATH = 'datasets/dataset_CAPT_v7.npz'
BATCH_SIZE = 8
NUM_WORKERS = 0
LR = 2e-4
EPOCHS = 30

# Model parameters
d_model = 64  # Lattent dim
q = 8  # Query size
v = 8  # Value size
h = 8  # Number of heads
N = 4  # Number of encoder and decoder to stack
attention_size = 12  # Attention window size
dropout = 0.2  # Dropout rate
pe = None  # Positional encoding
chunk_mode = None

d_input = 38  # From dataset
d_output = 8  # From dataset

# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

# Load dataset
ozeDataset = OzeDataset(DATASET_PATH)

# Split between train, validation and test
dataset_train, dataset_val, dataset_test = random_split(
    ozeDataset, (38000, 1000, 1000))

dataloader_train = DataLoader(dataset_train,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=NUM_WORKERS,
                              pin_memory=False
                              )

dataloader_val = DataLoader(dataset_val,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=NUM_WORKERS
                            )

dataloader_test = DataLoader(dataset_test,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=NUM_WORKERS
                             )

# Load transformer with Adam optimizer and MSE loss function
net = Transformer(d_input, d_model, d_output, q, v, h, N, attention_size=attention_size,
                  dropout=dropout, chunk_mode=chunk_mode, pe=pe).to(device)
optimizer = optim.Adam(net.parameters(), lr=LR)
loss_function = OZELoss(alpha=0.3)

logger = Logger(f'logs/training.csv', model_name='transformer', params=['training_loss',
                                                                        'mse_tint_total',
                                                                        'mse_cold_total',
                                                                        'mse_tint_occupation',
                                                                        'mse_cold_occupation',
                                                                        'r2_tint',
                                                                        'r2_cold'])

# Fit model
with tqdm(total=EPOCHS) as pbar:
    loss = fit(net, optimizer, loss_function, dataloader_train,
            dataloader_val, epochs=EPOCHS, device=device)

# Switch to evaluation
_ = net.eval()

# Select target values in test split
y_true = ozeDataset._y[dataloader_test.dataset.indices]

# Compute predictions
predictions = torch.empty(len(dataloader_test.dataset), 168, 8)
idx_prediction = 0
with torch.no_grad():
    for x, y in tqdm(dataloader_test, total=len(dataloader_test)):
        netout = net(x.to(device)).cpu()
        predictions[idx_prediction:idx_prediction+x.shape[0]] = netout
        idx_prediction += x.shape[0]

# Compute occupation times
occupation = ozeDataset._x[dataloader_test.dataset.indices,
                           :, ozeDataset.labels['Z'].index('occupancy')]

# Training loss
training_loss = loss_function(y_true, predictions).item()

# MSE losses
mse_tint_total = MSE(y_true, predictions, idx_label=[-1])
mse_cold_total = MSE(y_true, predictions, idx_label=[0])

mse_tint_occupation = MSE(y_true, predictions,
                          idx_label=[-1],
                          occupation=occupation)
mse_cold_occupation = MSE(y_true, predictions,
                          idx_label=[0],
                          occupation=occupation)

# R2 score
r2_tint = r2_score(y_true[..., -1], predictions[..., -1])
r2_cold = r2_score(y_true[..., 0], predictions[..., 0])

# Log
logger.log(
    training_loss=training_loss,
    mse_tint_total=mse_tint_total,
    mse_cold_total=mse_cold_total,
    mse_tint_occupation=mse_tint_occupation,
    mse_cold_occupation=mse_cold_occupation,
    r2_tint=r2_tint,
    r2_cold=r2_cold,
)
