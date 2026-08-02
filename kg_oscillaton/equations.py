"""Pointwise polar-areal equations for real scalar oscillatons."""

from __future__ import annotations

import numpy as np


def _safe_x(x):
    x = np.asarray(x, dtype=float)
    return np.where(np.abs(x) < 1.0e-14, np.inf, x)


def kg_rhs_polar_areal(Phi, Psi, Pi, alpha, a, x):
    """Return the KG RHS in polar-areal variables.

    Implements docs Eq. (F6). The radial derivative in the `Pi` equation is
    computed with `numpy.gradient`, so this helper is intended for diagnostics
    and simple method-of-lines experiments rather than the Fourier BVP.
    """

    x = np.asarray(x, dtype=float)
    ratio = alpha / a
    Phi_t = ratio * Pi
    Psi_t = np.gradient(ratio * Pi, x, edge_order=2)
    flux = x**2 * ratio * Psi
    Pi_t = np.gradient(flux, x, edge_order=2) / _safe_x(x) ** 2 - a * alpha * Phi
    return Phi_t, Psi_t, Pi_t


def hamiltonian_rhs_a(Phi, Psi, Pi, a, x):
    """Return `d_x a` from docs Eq. (F7)."""

    x = np.asarray(x, dtype=float)
    a = np.asarray(a, dtype=float)
    rhs = (1.0 - a**2) / (2.0 * _safe_x(x))
    rhs += x / 4.0 * (Psi**2 + Pi**2 + a**2 * Phi**2)
    return a * rhs


def slicing_rhs_alpha(Phi, a, alpha, da_dx, x):
    """Return `d_x alpha` from docs Eq. (F8)."""

    x = np.asarray(x, dtype=float)
    rhs = da_dx / a + (a**2 - 1.0) / _safe_x(x)
    rhs += -0.5 * x * a**2 * Phi**2
    return alpha * rhs


def A_C_equations(Phi, Phi_t, Phi_x, A, C, x, C_t=None):
    """Return the pointwise `A_x`, `C_x`, and optionally `Phi_xx`.

    Implements docs Eqs. (F10a)-(F10c). If `C_t` is provided, the returned
    dictionary includes `Phi_xx`, solving (F10c) for the radial second
    derivative.
    """

    x = np.asarray(x, dtype=float)
    xx = _safe_x(x)
    A_x = A * x / 2.0 * (C * Phi_t**2 + Phi_x**2 + A * Phi**2)
    A_x += A / xx * (1.0 - A)
    C_x = 2.0 * C / xx * (1.0 - A + 0.5 * x**2 * A * Phi**2)
    out = {"A_x": A_x, "C_x": C_x}
    if C_t is not None:
        raise ValueError(
            "Phi_xx also requires Phi_tt. Use fourier_projection for BVP assembly."
        )
    return out

