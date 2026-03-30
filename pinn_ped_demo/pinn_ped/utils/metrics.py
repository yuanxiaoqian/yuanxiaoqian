import numpy as np


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def ade(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(pred - target, axis=-1)))


def fde(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(pred[:, -1] - target[:, -1], axis=-1)))
