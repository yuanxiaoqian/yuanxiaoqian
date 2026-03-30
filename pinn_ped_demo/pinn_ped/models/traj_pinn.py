import torch
from torch import nn
from .mlp import MLP


class TrajPINN(nn.Module):
    def __init__(self, hidden_dim: int = 128, depth: int = 6):
        super().__init__()
        self.backbone = MLP(2, 2, hidden_dim=hidden_dim, depth=depth)

    def forward(self, tid: torch.Tensor):
        return self.backbone(tid)
