
"""Simple training loop for DualBranchNet (CPU-friendly)."""
import argparse, json
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.models.dual_branch_net import DualBranchNet
from src.utils.evaluation import compute_metrics

def train(drug_X, prot_X, labels, epochs=10, batch_size=64, lr=1e-3, out_path='artifacts/model.pt'):
    device = torch.device('cpu')
    model = DualBranchNet(drug_dim=drug_X.shape[1], prot_dim=prot_X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()
    ds = TensorDataset(torch.tensor(drug_X, dtype=torch.float32), torch.tensor(prot_X, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    best_loss = float('inf')
    for ep in range(epochs):
        model.train()
        epoch_losses = []
        for d,p,y in dl:
            opt.zero_grad()
            out = model(d,p)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss))
        val_loss = np.mean(epoch_losses)
        print(f"Epoch {ep+1:02d} | TrainLoss {val_loss:.4f}")
        # simple saving policy
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), out_path)
    # return path
    return out_path

if __name__ == '__main__':
    print('Train script placeholder. Import train() and use programmatically.')
