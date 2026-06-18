"""Pointwise real Proca equations in polar-areal A,C variables."""

from __future__ import annotations

import numpy as np


def reconstruct_metric(A: np.ndarray, C: np.ndarray):
    """Return `a`, `alpha`, and `sqrtC` from `A,C`."""

    sqrtC = np.sqrt(np.maximum(C, 0.0))
    a = np.sqrt(np.maximum(A, 0.0))
    alpha = np.sqrt(np.maximum(A / C, 0.0))
    return a, alpha, sqrtC


def reconstruct_W(C: np.ndarray, E_t: np.ndarray):
    """Return `W = -sqrt(C) dot(E)` from docs Eq. (P8)."""

    return -np.sqrt(np.maximum(C, 0.0)) * E_t

