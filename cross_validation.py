import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from tst import Transformer
from tst.loss import OZELoss

from src.dataset import OzeDataset
from src.utils import compute_loss, fit, Logger, kfold
from src.benchmark import LSTM, BiGRU

# Search parameters
CHUNKS = 5

# Training parameters
DATASET_PATH = 'datasets/dataset.npz'
BATCH_SIZE = 8
NUM_WORKERS = 4
LR = 2e-4
EPOCHS = 30

# Model parameters
d_model = 48  # Lattent dim
q = 8  # Query size
v = 8  # Value size
h = 4  # Number of heads
N = 8  # Number of encoder and decoder to stack
attention_size = 24  # Attention window size
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

# Load network
# Load transformer with Adam optimizer and MSE loss function
loss_function = OZELoss(alpha=0.3)


logger = Logger(f'logs/crossvalidation_log.csv', params=['loss'])

kfoldIterator = kfold(ozeDataset, n_chunk=CHUNKS,
                      batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

with tqdm(total=CHUNKS*EPOCHS) as pbar:
    for dataloader_train, dataloader_val in kfoldIterator:

        # Load transformer with Adam optimizer and MSE loss function
        # net = Transformer(d_input, d_model, d_output, q, v, h, N, attention_size=attention_size,
        #                   dropout=dropout, chunk_mode=chunk_mode, pe=pe).to(device)
        net = BiGRU(d_input, d_model, d_output, num_layers=N, dropout=dropout, bidirectional=True).to(device)

        optimizer = optim.Adam(net.parameters(), lr=LR)

        # Fit model
        loss = fit(net, optimizer, loss_function, dataloader_train,
                   dataloader_val, epochs=EPOCHS, pbar=pbar, device=device)

        # Log
        logger.log(loss=loss)
