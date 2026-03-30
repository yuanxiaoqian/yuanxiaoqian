import argparse
from pathlib import Path
import numpy as np
import torch
import yaml

from pinn_ped.models.traj_pinn import TrajPINN
from pinn_ped.losses.ode_residuals import sfm_residual
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
    tid = torch.tensor(data["traj_tid"], device=device)
    xy = torch.tensor(data["traj_xy"], device=device)

    model = TrajPINN(cfg["model"]["hidden_dim"], cfg["model"]["depth"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    steps = cfg["train"]["steps"]
    bs = min(cfg["train"]["batch_size"], tid.shape[0])
    for step in range(1, steps + 1):
        idx = torch.randint(0, tid.shape[0], (bs,), device=device)
        tb, xb = tid[idx], xy[idx]

        pred = model(tb)
        data_loss = torch.mean((pred - xb) ** 2)
        ode_loss = torch.mean(sfm_residual(model, tb) ** 2)
        lam = linear_ramp(step, steps, cfg["traj"]["lambda_ode_start"], cfg["traj"]["lambda_ode_end"])
        loss = cfg["traj"]["lambda_data"] * data_loss + lam * ode_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 200 == 0:
            print(f"step={step} loss={loss.item():.6f} data={data_loss.item():.6f} ode={ode_loss.item():.6f}")

    run_dir = Path("outputs/runs/traj")
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    print(f"saved: {run_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
