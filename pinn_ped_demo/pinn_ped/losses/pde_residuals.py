import torch


def continuity_residual(model, xyt: torch.Tensor):
    xyt = xyt.requires_grad_(True)
    rho, u = model(xyt)

    grads_rho = torch.autograd.grad(rho.sum(), xyt, create_graph=True)[0]
    drho_dt = grads_rho[:, 2:3]

    flux_x = rho * u[:, 0:1]
    flux_y = rho * u[:, 1:2]
    dflux_x_dx = torch.autograd.grad(flux_x.sum(), xyt, create_graph=True)[0][:, 0:1]
    dflux_y_dy = torch.autograd.grad(flux_y.sum(), xyt, create_graph=True)[0][:, 1:2]

    return drho_dt + dflux_x_dx + dflux_y_dy
