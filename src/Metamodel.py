import numpy as np
import torch

class Metamodel:
    def __init__(self, model_path, m_in, M_in, m_out, M_out, device='cpu'):
        self.net = torch.load(model_path, map_location=device)
        
        self.m_in = m_in
        self.M_in = M_in
        
        self.m_out = m_out
        self.M_out = M_out
        
    def predict(self, x):
        if len(x.shape) < 3:
            x = x[np.newaxis, ...]
            
        # Normalize inpue
        x = (x - self.m_in) / (self.M_in - self.m_in + np.finfo(float).eps)
        
        # Run prediction
        x = torch.Tensor(x)
        with torch.no_grad():
            netout = self.net(x).detach().numpy()
        
        # Rescale
        netout = netout * (self.M_out - self.m_out + np.finfo(float).eps) + self.m_out
        
        return netout