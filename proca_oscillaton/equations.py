
from __future__ import annotations

import numpy as np


def reconstruct_metric(A: np.ndarray, C: np.ndarray):

    sqrtC = np.sqrt(np.maximum(C, 0.0))
    a = np.sqrt(np.maximum(A, 0.0))
    alpha = np.sqrt(np.maximum(A / C, 0.0))
    return a, alpha, sqrtC


def reconstruct_W(C: np.ndarray, E_t: np.ndarray):

    return -np.sqrt(np.maximum(C, 0.0)) * E_t

