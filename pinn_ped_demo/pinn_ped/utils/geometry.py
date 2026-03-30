import numpy as np


def in_roi(xy: np.ndarray, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0)) -> np.ndarray:
    x, y = xy[:, 0], xy[:, 1]
    return (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])


def sample_boundary(n: int) -> np.ndarray:
    t = np.random.rand(n)
    side = np.random.randint(0, 4, size=n)
    pts = np.zeros((n, 2), dtype=np.float32)
    pts[side == 0] = np.stack([-np.ones((side == 0).sum()), 2 * t[side == 0] - 1], axis=1)
    pts[side == 1] = np.stack([np.ones((side == 1).sum()), 2 * t[side == 1] - 1], axis=1)
    pts[side == 2] = np.stack([2 * t[side == 2] - 1, -np.ones((side == 2).sum())], axis=1)
    pts[side == 3] = np.stack([2 * t[side == 3] - 1, np.ones((side == 3).sum())], axis=1)
    return pts
