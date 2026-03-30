import argparse
from pathlib import Path
import numpy as np
import torch
import yaml

from pinn_ped.models.field_pinn import FieldPINN
from pinn_ped.losses.pde_residuals import continuity_residual
from pinn_ped.losses.weighting import linear_ramp
from pinn_ped.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg.get("seed", 42))

    device = "cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu"
    data = np.load(cfg["data"]["processed_npz"])
    xyt = torch.tensor(data["xyt"], device=device)
    rho_gt = torch.tensor(data["rho"], device=device)

    model = FieldPINN(cfg["model"]["hidden_dim"], cfg["model"]["depth"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    steps = cfg["train"]["steps"]
    bs = min(cfg["train"]["batch_size"], xyt.shape[0])
    for step in range(1, steps + 1):
        idx = torch.randint(0, xyt.shape[0], (bs,), device=device)
        xb, rb = xyt[idx], rho_gt[idx]

        rho_pred, _ = model(xb)
        data_loss = torch.mean((rho_pred - rb) ** 2)
        pde = torch.mean(continuity_residual(model, xb) ** 2)
        lam = linear_ramp(step, steps, cfg["field"]["lambda_pde_start"], cfg["field"]["lambda_pde_end"])
        loss = cfg["field"]["lambda_data"] * data_loss + lam * pde

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 200 == 0:
            print(f"step={step} loss={loss.item():.6f} data={data_loss.item():.6f} pde={pde.item():.6f}")

    run_dir = Path("outputs/runs/field")
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    print(f"saved: {run_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
