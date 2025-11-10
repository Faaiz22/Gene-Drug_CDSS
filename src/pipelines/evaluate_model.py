
"""Evaluate a saved model on arrays of drug/protein features and labels."""
import torch, numpy as np, json
from src.models.dual_branch_net import DualBranchNet
from src.utils.evaluation import compute_metrics

def evaluate(model_path, drug_X, prot_X, labels, out_metrics_path='artifacts/metrics_test.json'):
    model = DualBranchNet(drug_dim=drug_X.shape[1], prot_dim=prot_X.shape[1])
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        import torch as _t
        d = _t.tensor(drug_X, dtype=_t.float32)
        p = _t.tensor(prot_X, dtype=_t.float32)
        probs = model(d,p).numpy()
    metrics = compute_metrics(labels, probs)
    with open(out_metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics
