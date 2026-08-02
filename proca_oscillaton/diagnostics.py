"""Diagnostics for real radial-Proca oscillaton profiles."""

from __future__ import annotations

import numpy as np

from .fourier import build_phase_grid, evaluate_modes, project_cos
from .residuals import (
    residual_S1_Aprime,
    residual_S2_Cprime,
    residual_S3_gauss,
    residual_S4_Uprime,
    residual_S5_momentum,
)


def l2_norm(values: np.ndarray) -> float:
    """Return an RMS norm."""

    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def mass_function(x: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Return ``M(<x) = x (1 - 1/A) / 2``."""

    return 0.5 * np.asarray(x, dtype=float)[:, None] * (
        1.0 - 1.0 / np.asarray(A, dtype=float)
    )


def residual_norms(profile, *, n_time: int = 96) -> dict[str, float]:
    """Return RMS residuals of the reduced radial-Proca equations."""

    theta = build_phase_grid(n_time)
    x = profile.x
    U = evaluate_modes(profile.U_modes, profile.matter_modes, theta)
    E = evaluate_modes(profile.E_modes, profile.matter_modes, theta)
    E_t = evaluate_modes(
        profile.E_modes,
        profile.matter_modes,
        theta,
        omega=profile.omega,
        time_derivative=1,
    )
    A = evaluate_modes(profile.A_modes, profile.metric_modes, theta)
    C = evaluate_modes(profile.C_modes, profile.metric_modes, theta)
    A_t = evaluate_modes(
        profile.A_modes,
        profile.metric_modes,
        theta,
        omega=profile.omega,
        time_derivative=1,
    )
    U_x = np.gradient(U, x, axis=0, edge_order=2)
    E_x = np.gradient(E, x, axis=0, edge_order=2)
    A_x = np.gradient(A, x, axis=0, edge_order=2)
    C_x = np.gradient(C, x, axis=0, edge_order=2)
    return {
        "S1_hamiltonian": l2_norm(residual_S1_Aprime(A_x, A, C, U, E, E_t, x)),
        "S2_slicing": l2_norm(residual_S2_Cprime(C_x, A, C, E, x)),
        "S3_gauss": l2_norm(residual_S3_gauss(E_x, E, C, U, x)),
        "S4_U": l2_norm(residual_S4_Uprime(U_x, A, C, E, E_t, profile.omega)),
        "S5_momentum": l2_norm(residual_S5_momentum(A_t, A, C, U, E_t, x)),
    }


def origin_regular_error(profile) -> float:
    """Return the leading radial-Proca regularity error at the inner grid point."""

    data = profile.initial_data()
    x0 = float(profile.x[0])
    if x0 == 0.0:
        x0 = float(profile.x[1])
        U0 = data["U"][1]
        E0 = data["E"][1]
        C0 = data["C"][1]
    else:
        U0 = data["U"][0]
        E0 = data["E"][0]
        C0 = data["C"][0]
    return float(E0 / x0 + np.sqrt(C0) * U0 / 3.0)


def boundary_residuals(profile, *, n_time: int = 96) -> dict[str, float]:
    """Return the coefficient-space boundary residuals used by the BVP solver."""

    theta = build_phase_grid(n_time)
    x_min = float(profile.x[0])
    x_max = float(profile.x[-1])
    U_center = evaluate_modes(
        profile.U_modes[:, 0, None], profile.matter_modes, theta
    )[0:1]
    C_center = evaluate_modes(
        profile.C_modes[:, 0, None], profile.metric_modes, theta
    )[0:1]
    regular_E = -x_min / 3.0 * project_cos(
        np.sqrt(np.maximum(C_center, 1.0e-14)) * U_center,
        theta,
        profile.matter_modes,
    )[:, 0]

    outer_matter = []
    for idx, mode in enumerate(profile.matter_modes):
        wave = abs(float(mode) * profile.omega)
        if wave < 1.0:
            decay = np.sqrt(max(1.0 - wave**2, 1.0e-14))
            outer_matter.append(
                decay**2 * profile.E_modes[idx, -1]
                - (decay + 1.0 / x_max) * profile.U_modes[idx, -1]
            )
        else:
            outer_matter.append(profile.U_modes[idx, -1])
    outer_matter = np.asarray(outer_matter, dtype=float)
    return {
        "origin_U1_error": float(profile.U_modes[0, 0] - profile.u1_center),
        "origin_E_regular_abs_max": float(
            np.max(np.abs(profile.E_modes[:, 0] - regular_E))
        ),
        "origin_A0_error": float(profile.A0[0] - 1.0),
        "outer_matter_abs_max": float(np.max(np.abs(outer_matter))),
        "outer_C0_minus_A0_squared": float(profile.C0[-1] - profile.A0[-1] ** 2),
        "outer_positive_C_abs_max": float(np.max(np.abs(profile.C_modes[1:, -1])))
        if profile.C_modes.shape[0] > 1
        else 0.0,
    }
