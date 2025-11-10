
import torch
from src.models.dual_branch_net import DualBranchNet

def test_forward_pass():
    model = DualBranchNet()
    d = torch.randn(2,135)
    p = torch.randn(2,30)
    out = model(d,p)
    assert out.shape == (2,)
    assert (0 <= out).all() and (out <= 1).all()
