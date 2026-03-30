import numpy as np


def gaussian_kde_density(points: np.ndarray, grid: np.ndarray, bandwidth: float = 0.1) -> np.ndarray:
    diff = grid[:, None, :] - points[None, :, :]
    d2 = (diff ** 2).sum(axis=-1)
    k = np.exp(-0.5 * d2 / (bandwidth ** 2))
    norm = 2 * np.pi * (bandwidth ** 2) * max(points.shape[0], 1)
    return (k.sum(axis=1) / norm).astype(np.float32)
