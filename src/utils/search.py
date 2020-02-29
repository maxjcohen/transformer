import csv

import torch
import numpy as np

from src.utils.utils import compute_loss

def fit(net, optimizer, loss_function, dataloader_train, dataloader_val, epochs=10, pbar=None, device='cpu'):
    val_loss_best = np.inf

    # Prepare loss history
    for idx_epoch in range(epochs):
        for idx_batch, (x, y) in enumerate(dataloader_train):
            optimizer.zero_grad()

            # Propagate input
            netout = net(x.to(device))

            # Comupte loss
            loss = loss_function(y.to(device), netout)

            # Backpropage loss
            loss.backward()

            # Update weights
            optimizer.step()
            
        val_loss = compute_loss(net, dataloader_val, loss_function, device).item()
        
        if val_loss < val_loss_best:
            val_loss_best = val_loss

        if pbar is not None:
            pbar.update()
    
    return val_loss_best

class Logger:
    def __init__(self, csv_path, search_params=[]):
        self.csv_file = open(csv_path, 'w')
        self.writer = csv.DictWriter(self.csv_file, search_params + ['loss'])
        self.writer.writeheader()
        
    def log(self, params={}, **kwargs):
        params.update(kwargs)
        self.writer.writerow(params)
        self.csv_file.flush()
        
    def __del__(self):
        self.csv_file.close()