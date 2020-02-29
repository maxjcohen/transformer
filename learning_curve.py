import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import seaborn as sns

from tst import Transformer
from tst.loss import OZELoss

from src.dataset import OzeDataset
from src.utils import visual_sample, compute_loss
from src.utils import compute_loss, fit, Logger, kfold, leargnin_curve

# Search parameters
PARTS = 8
VALIDATION_SPLIT = 0.3

# Training parameters
DATASET_PATH = 'datasets/dataset_random.npz'
BATCH_SIZE = 8
NUM_WORKERS = 4
LR = 2e-4
EPOCHS = 5

# Model parameters
d_model = 32  # Lattent dim
q = 8  # Query size
v = 8  # Value size
h = 2  # Number of heads
N = 2  # Number of encoder and decoder to stack
attention_size = 24  # Attention window size
dropout = 0.2  # Dropout rate
pe = None  # Positional encoding
chunk_mode = None

d_input = 38  # From dataset
d_output = 8  # From dataset

# Config
sns.set()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

# Load dataset
ozeDataset = OzeDataset(DATASET_PATH)

# Load network
# Load transformer with Adam optimizer and MSE loss function
loss_function = OZELoss(alpha=0.3)


logger = Logger('learningcurve_log.csv')

learningcurveIterator = leargnin_curve(ozeDataset, n_part=PARTS, validation_split=VALIDATION_SPLIT,
                      batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

with tqdm(total=PARTS*EPOCHS) as pbar:
    for dataloader_train, dataloader_val in learningcurveIterator:

        # Load transformer with Adam optimizer and MSE loss function
        net = Transformer(d_input, d_model, d_output, q, v, h, N, attention_size=attention_size,
                          dropout=dropout, chunk_mode=chunk_mode, pe=pe).to(device)

        optimizer = optim.Adam(net.parameters(), lr=LR)

        # Fit model
        loss = fit(net, optimizer, loss_function, dataloader_train,
                   dataloader_val, epochs=EPOCHS, pbar=pbar, device=device)

        # Log
        logger.log(loss=loss)