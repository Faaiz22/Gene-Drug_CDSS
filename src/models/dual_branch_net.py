
"""Dual-branch model: drug encoder + protein encoder + fusion head.

Simple, clear, and compatible with saved model.pt from notebook.
"""
import torch
import torch.nn as nn

class DualBranchNet(nn.Module):
    def __init__(self, drug_dim: int = 135, prot_dim: int = 30, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),
            nn.ReLU()
        )
        self.prot_encoder = nn.Sequential(
            nn.Linear(prot_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),
            nn.ReLU()
        )
        # fusion size = hidden//2 + hidden//2
        self.fusion = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1),
            nn.Sigmoid()
        )

    def forward(self, drug_x, prot_x):
        d = self.drug_encoder(drug_x)
        p = self.prot_encoder(prot_x)
        fused = torch.cat([d, p], dim=1)
        out = self.fusion(fused).squeeze(-1)
        return out
