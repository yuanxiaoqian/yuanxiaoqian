import argparse
from pathlib import Path
import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    n = int(cfg["data"].get("n_samples", 5000))
    out_path = Path(cfg["data"]["processed_npz"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.get("seed", 42))
    xyt = rng.uniform(-1, 1, size=(n, 3)).astype(np.float32)
    rho = (np.exp(-((xyt[:, 0] ** 2 + xyt[:, 1] ** 2) / 0.2)) + 0.05 * rng.normal(size=n)).astype(np.float32)
    traj_t = np.linspace(-1, 1, 2000, dtype=np.float32)
    traj_id = rng.integers(0, 200, size=2000, dtype=np.int32)
    traj_tid = np.stack([traj_t, traj_id / 200.0 * 2 - 1], axis=1).astype(np.float32)
    traj_xy = np.stack([np.sin(3 * traj_t) + 0.1 * rng.normal(size=2000), np.cos(2 * traj_t)], axis=1).astype(np.float32)

    np.savez(out_path, xyt=xyt, rho=rho[:, None], traj_tid=traj_tid, traj_xy=traj_xy)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
