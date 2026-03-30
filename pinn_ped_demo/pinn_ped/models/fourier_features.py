import torch


class FourierFeatures(torch.nn.Module):
    def __init__(self, in_dim: int, n_freq: int = 32, scale: float = 10.0):
        super().__init__()
        self.register_buffer("B", torch.randn(in_dim, n_freq) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * torch.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
