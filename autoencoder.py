
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class FraudAutoencoderV2(nn.Module):
    def __init__(self, input_dim=29, bottleneck=4, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),        nn.Dropout(dropout),
            nn.Linear(32, 16),        nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),        nn.Dropout(dropout),
            nn.Linear(16, 8),         nn.BatchNorm1d(8),
            nn.LeakyReLU(0.1),
            nn.Linear(8, bottleneck)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 8),  nn.BatchNorm1d(8),  nn.LeakyReLU(0.1),
            nn.Linear(8, 16),          nn.BatchNorm1d(16), nn.LeakyReLU(0.1),
            nn.Linear(16, 32),         nn.BatchNorm1d(32), nn.LeakyReLU(0.1),
            nn.Linear(32, input_dim)
        )
    def forward(self, x): return self.decoder(self.encoder(x))

def get_recon_errors(model, X, device):
    model.eval()
    loader = DataLoader(TensorDataset(torch.FloatTensor(X)), batch_size=512)
    errors = []
    with torch.no_grad():
        for (b,) in loader:
            b = b.to(device)
            r = model(b)
            errors.extend(((b-r)**2).mean(dim=1).cpu().numpy())
    return np.array(errors)
