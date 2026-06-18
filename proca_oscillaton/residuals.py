"""Projected Proca equations and residuals."""

from __future__ import annotations

import numpy as np

from .fourier import (
    build_phase_grid,
    cos_basis,
    evaluate_modes,
    mode_set,
    project_cos,
    project_sin,
    reduced_state_size,
    spectral_dt,
)


def unpack_reduced_state(y: np.ndarray, jmax: int):
    modes = mode_set(jmax)
    nm = modes.n_matter
    u = y[:nm]
    e = y[nm : 2 * nm]
    A0 = y[2 * nm]
    C = y[2 * nm + 1 :]
    return u, e, A0, C


def residual_S1_Aprime(A_x, A, C, U, E, E_t, x):
    """Hamiltonian residual from docs Eq. (S1)."""

    xx = x[:, None]
    rhs = A * (1.0 - A) / xx
    rhs += A * xx / 2.0 * (A * E**2 + C * U**2 + C * E_t**2)
    return A_x - rhs


def residual_S2_Cprime(C_x, A, C, E, x):
    """Slicing residual from docs Eq. (S2)."""

    xx = x[:, None]
    rhs = 2.0 * C / xx * (1.0 - A + 0.5 * xx**2 * A * E**2)
    return C_x - rhs


def residual_S3_gauss(E_x, E, C, U, x):
    """Gauss residual from docs Eq. (S3)."""

    return E_x + 2.0 * E / x[:, None] + np.sqrt(np.maximum(C, 0.0)) * U


def residual_S4_Uprime(U_x, A, C, E, E_t, omega):
    """Radial-U residual from docs Eq. (S4)."""

    sqrtC = np.sqrt(np.maximum(C, 1.0e-14))
    product_t = spectral_dt(sqrtC * E_t, omega)
    return U_x + product_t + A * E / sqrtC


def residual_S5_momentum(A_t, A, C, U, E_t, x):
    """Momentum residual from docs Eq. (S5)."""

    sqrtC = np.sqrt(np.maximum(C, 0.0))
    return A_t + x[:, None] * A * sqrtC * U * E_t


def solve_A_modes_from_momentum(
    A0: np.ndarray,
    U: np.ndarray,
    E_t: np.ndarray,
    C: np.ndarray,
    x: np.ndarray,
    omega: float,
    metric_modes: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """Solve positive even `A_j` algebraically from the momentum equation."""

    positive = np.asarray(metric_modes, dtype=int)[1:]
    if positive.size == 0:
        return A0[None, :]

    sqrtC = np.sqrt(np.maximum(C, 1.0e-14))
    F = x[:, None] * sqrtC * U * E_t
    sin_pos = np.sin(np.outer(positive, theta))
    cos_pos = np.cos(np.outer(positive, theta))
    b = 2.0 * np.mean(F[:, None, :] * sin_pos[None, :, :], axis=-1)
    matrix_terms = 2.0 * np.mean(
        F[:, None, None, :]
        * sin_pos[None, :, None, :]
        * cos_pos[None, None, :, :],
        axis=-1,
    )
    matrices = -omega * np.diag(positive.astype(float))[None, :, :] + matrix_terms
    rhs = -A0[:, None] * b
    try:
        positive_A = np.linalg.solve(matrices, rhs[..., None])[..., 0].T
    except np.linalg.LinAlgError:
        positive_A = np.stack(
            [np.linalg.lstsq(matrices[i], rhs[i], rcond=None)[0] for i in range(x.size)],
            axis=1,
        )
    return np.vstack([A0[None, :], positive_A])


def reconstruct_A_modes(
    x: np.ndarray,
    y: np.ndarray,
    omega: float,
    jmax: int,
    *,
    n_time: int = 96,
) -> np.ndarray:
    modes = mode_set(jmax)
    theta = build_phase_grid(n_time)
    u, e, A0, Ccoef = unpack_reduced_state(y, jmax)
    U = evaluate_modes(u, modes.matter, theta)
    E_t = evaluate_modes(e, modes.matter, theta, omega=omega, time_derivative=1)
    C = evaluate_modes(Ccoef, modes.metric, theta)
    return solve_A_modes_from_momentum(
        A0, U, E_t, C, x, omega, modes.metric, theta
    )


def rhs_reduced(
    x: np.ndarray,
    y: np.ndarray,
    omega: float,
    jmax: int,
    *,
    n_time: int = 96,
) -> np.ndarray:
    """Reduced radial RHS for the Proca BVP."""

    if y.shape[0] != reduced_state_size(jmax):
        raise ValueError("unexpected reduced state size")

    modes = mode_set(jmax)
    theta = build_phase_grid(n_time)
    u, e, A0, Ccoef = unpack_reduced_state(y, jmax)

    U = evaluate_modes(u, modes.matter, theta)
    E = evaluate_modes(e, modes.matter, theta)
    E_t = evaluate_modes(e, modes.matter, theta, omega=omega, time_derivative=1)
    C = evaluate_modes(Ccoef, modes.metric, theta)
    sqrtC = np.sqrt(np.maximum(C, 1.0e-14))
    Acoef = solve_A_modes_from_momentum(
        A0, U, E_t, C, x, omega, modes.metric, theta
    )
    A = Acoef.T @ cos_basis(modes.metric, theta)

    xx = x[:, None]
    E_rhs = -2.0 * E / xx - sqrtC * U
    U_rhs = -spectral_dt(sqrtC * E_t, omega) - A * E / sqrtC
    A_rhs = A * (1.0 - A) / xx
    A_rhs += A * xx / 2.0 * (A * E**2 + C * U**2 + C * E_t**2)
    C_rhs = 2.0 * C / xx * (1.0 - A + 0.5 * xx**2 * A * E**2)

    dy = np.zeros_like(y)
    nm = modes.n_matter
    dy[:nm] = project_cos(U_rhs, theta, modes.matter)
    dy[nm : 2 * nm] = project_cos(E_rhs, theta, modes.matter)
    dy[2 * nm] = project_cos(A_rhs, theta, np.array([0], dtype=int))[0]
    dy[2 * nm + 1 :] = project_cos(C_rhs, theta, modes.metric)
    return dy


def projected_equation_residuals(
    x: np.ndarray,
    y: np.ndarray,
    dy_dx: np.ndarray,
    omega: float,
    jmax: int,
    *,
    n_time: int = 96,
) -> dict[str, np.ndarray]:
    """Return projected residuals for S1-S5."""

    modes = mode_set(jmax)
    theta = build_phase_grid(n_time)
    u, e, A0, Ccoef = unpack_reduced_state(y, jmax)
    du, de, dA0, dC = unpack_reduced_state(dy_dx, jmax)
    U = evaluate_modes(u, modes.matter, theta)
    E = evaluate_modes(e, modes.matter, theta)
    E_t = evaluate_modes(e, modes.matter, theta, omega=omega, time_derivative=1)
    U_x = evaluate_modes(du, modes.matter, theta)
    E_x = evaluate_modes(de, modes.matter, theta)
    C = evaluate_modes(Ccoef, modes.metric, theta)
    C_x = evaluate_modes(dC, modes.metric, theta)
    Acoef = solve_A_modes_from_momentum(A0, U, E_t, C, x, omega, modes.metric, theta)
    A = Acoef.T @ cos_basis(modes.metric, theta)
    A_x0 = dA0
    A_x = A_x0[:, None]
    A_t = Acoef.T @ (
        -omega * modes.metric[:, None] * np.sin(np.outer(modes.metric, theta))
    )

    return {
        "S1": project_cos(residual_S1_Aprime(A_x, A, C, U, E, E_t, x), theta, np.array([0])),
        "S2": project_cos(residual_S2_Cprime(C_x, A, C, E, x), theta, modes.metric),
        "S3": project_cos(residual_S3_gauss(E_x, E, C, U, x), theta, modes.matter),
        "S4": project_cos(residual_S4_Uprime(U_x, A, C, E, E_t, omega), theta, modes.matter),
        "S5": project_sin(residual_S5_momentum(A_t, A, C, U, E_t, x), theta, modes.metric[1:]),
    }

