"""Diagnostics and residuals for polar-areal oscillaton profiles."""

from __future__ import annotations

import numpy as np

from .equations import hamiltonian_rhs_a, slicing_rhs_alpha


def mass_function(x: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Return `M(x) = x/2 (1 - 1/A)` from docs Eq. (F16)."""

    x = np.asarray(x, dtype=float)
    A = np.asarray(A, dtype=float)
    return 0.5 * x * (1.0 - 1.0 / A)


def total_mass(x: np.ndarray, A: np.ndarray, *, tail_fraction: float = 0.1) -> float:
    """Estimate total mass from the outer tail of the mass function."""

    M = mass_function(x, A)
    n_tail = max(1, int(np.ceil(M.size * tail_fraction)))
    return float(np.median(M[-n_tail:]))


def rmax_grr(x: np.ndarray, A: np.ndarray) -> tuple[float, float]:
    """Return `(Rmax, max(A))` from docs Eq. (F17)."""

    idx = int(np.argmax(A))
    return float(x[idx]), float(A[idx])


def hamiltonian_residual(Phi, Psi, Pi, a, x):
    """Return the Hamiltonian residual in docs Eq. (F18)."""

    da_dx = np.gradient(a, x, edge_order=2)
    return da_dx - hamiltonian_rhs_a(Phi, Psi, Pi, a, x)


def slicing_residual(Phi, a, alpha, x):
    """Return the slicing residual in docs Eq. (F18)."""

    da_dx = np.gradient(a, x, edge_order=2)
    dalpha_dx = np.gradient(alpha, x, edge_order=2)
    return dalpha_dx - slicing_rhs_alpha(Phi, a, alpha, da_dx, x)


def momentum_constraint_residual(a_t, alpha, Psi, Pi, x):
    """Return the polar-areal momentum residual in docs Eq. (F18)."""

    return a_t - 0.5 * x * alpha * Psi * Pi


def l2_norm(values: np.ndarray, x: np.ndarray | None = None) -> float:
    """Return an RMS norm, optionally with radial trapezoidal weighting."""

    values = np.asarray(values, dtype=float)
    if x is None:
        return float(np.sqrt(np.mean(values**2)))
    integral = np.trapz(values**2, x)
    length = float(x[-1] - x[0])
    return float(np.sqrt(integral / length))

