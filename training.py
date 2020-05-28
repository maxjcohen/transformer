"""
training
"""
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Classic

# %%
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import seaborn as sns

from time_series_transformer import Transformer
from time_series_transformer.loss import OZELoss

from src.dataset import OzeNPZDataset
from src.utils import npz_check, fit, Logger

# Training parameters
BATCH_SIZE = 8
NUM_WORKERS = 0
LR = 2e-4
EPOCHS = 30

# Model parameters
d_model = 64 # Lattent dim
q = 8 # Query size
v = 8 # Value size
h = 8 # Number of heads
N = 4 # Number of encoder and decoder to stack
attention_size = 12 # Attention window size
dropout = 0.2 # Dropout rate
pe = None # Positional encoding
chunk_mode = None

d_input = 38 # From dataset
d_output = 8 # From dataset

# Config
sns.set()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

# %% [markdown]
# ## Training
# %% [markdown]
# ### Load dataset

# %%
ozeDataset = OzeNPZDataset(dataset_path=npz_check(Path('datasets'), 'dataset'), labels_path="labels.json")


# %%
# Split between train and val
oze_dataset_length = len(ozeDataset)
total_length = 38000+1000+1000
training_length = round(38000*oze_dataset_length/total_length)
validation_length = round(1000*oze_dataset_length/total_length)
test_length = oze_dataset_length - validation_length - training_length

dataset_train, dataset_val, dataset_test = random_split(ozeDataset, (training_length, validation_length, test_length))

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

logger = Logger('logs/training.csv', params=['loss'])

with tqdm(total=EPOCHS) as pbar:
    # Fit model
    loss = fit(net, optimizer, loss_function, dataloader_train,
               dataloader_val, epochs=EPOCHS, pbar=pbar, device=device)

    # Log
    logger.log(loss=loss)
