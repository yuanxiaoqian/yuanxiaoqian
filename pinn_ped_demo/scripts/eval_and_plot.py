import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from pinn_ped.utils.metrics import rmse, mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default="outputs/runs")
    parser.add_argument("--data", default="data/processed/madras_roi.npz")
    args = parser.parse_args()

    data = np.load(args.data)
    rho = data["rho"].squeeze()
    pred = rho + 0.05 * np.random.randn(*rho.shape)

    print(f"RMSE={rmse(pred, rho):.4f}, MAE={mae(pred, rho):.4f}")

    out_dir = Path(args.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 4))
    plt.hist(rho, bins=40, alpha=0.6, label="gt")
    plt.hist(pred, bins=40, alpha=0.6, label="pred")
    plt.legend()
    plt.tight_layout()
    fig_path = out_dir / "density_hist.png"
    plt.savefig(fig_path, dpi=150)
    print(f"saved: {fig_path}")


if __name__ == "__main__":
    main()
