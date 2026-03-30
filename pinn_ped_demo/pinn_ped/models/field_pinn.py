import torch
from torch import nn
from .mlp import MLP


class FieldPINN(nn.Module):
    def __init__(self, hidden_dim: int = 128, depth: int = 6):
        super().__init__()
        self.backbone = MLP(3, 3, hidden_dim=hidden_dim, depth=depth)

    def forward(self, xyt: torch.Tensor):
        out = self.backbone(xyt)
        rho = torch.relu(out[:, :1])
        u = out[:, 1:]
        return rho, u
