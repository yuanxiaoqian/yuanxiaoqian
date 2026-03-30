import torch


def sfm_residual(model, tid: torch.Tensor, tau: float = 0.5):
    tid = tid.requires_grad_(True)
    x = model(tid)
    v = torch.autograd.grad(x.sum(), tid, create_graph=True)[0][:, 0:1]
    a = torch.autograd.grad(v.sum(), tid, create_graph=True)[0][:, 0:1]
    v_des = torch.ones_like(v)
    return a - (v_des - v) / tau
