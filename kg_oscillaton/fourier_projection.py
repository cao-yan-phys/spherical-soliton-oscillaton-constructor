"""Fourier projection helpers for the oscillaton eigenvalue problem."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModeSet:
    """Odd scalar modes and even metric modes for a truncation."""

    scalar: np.ndarray
    metric: np.ndarray

    @property
    def n_scalar(self) -> int:
        return int(self.scalar.size)

    @property
    def n_metric(self) -> int:
        return int(self.metric.size)


def mode_set(jmax: int) -> ModeSet:
    """Return odd scalar and even metric modes up to ``jmax``."""

    if jmax < 2:
        raise ValueError("jmax must be at least 2")
    return ModeSet(
        scalar=np.arange(1, jmax + 1, 2, dtype=int),
        metric=np.arange(0, jmax + 1, 2, dtype=int),
    )


def reduced_state_size(jmax: int) -> int:
    """Number of ODE variables in the reduced momentum-algebraic system."""

    modes = mode_set(jmax)
    return 2 * modes.n_scalar + 1 + modes.n_metric


def full_state_size(jmax: int) -> int:
    """Number of ODE variables if all ``A_j`` and ``C_j`` are evolved."""

    modes = mode_set(jmax)
    return 2 * modes.n_scalar + 2 * modes.n_metric


def build_time_grid(n_time: int, omega: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return Fourier phase samples and coordinate times.

    Implements docs Eq. (F12): ``theta_k = 2 pi k / N`` and
    ``t_k = theta_k / omega``.
    """

    if n_time < 2:
        raise ValueError("n_time must be at least 2")
    if omega <= 0:
        raise ValueError("omega must be positive")
    theta = 2.0 * np.pi * np.arange(n_time, dtype=float) / float(n_time)
    return theta, theta / omega


def evaluate_fourier_modes(
    modes: np.ndarray,
    theta: np.ndarray,
    coefficients: np.ndarray | None = None,
    *,
    omega: float = 1.0,
    time_derivative: int = 0,
) -> np.ndarray:
    """Evaluate cosine Fourier modes or a coefficient expansion.

    If ``coefficients`` is omitted, the basis matrix with shape
    ``(n_modes, n_time)`` is returned. Otherwise the expansion is evaluated
    along the first axis of ``coefficients`` and the result has shape
    ``coefficients.shape[1:] + (n_time,)``.
    """

    modes = np.asarray(modes, dtype=float)
    theta = np.asarray(theta, dtype=float)
    phase = np.outer(modes, theta)
    if time_derivative == 0:
        basis = np.cos(phase)
    elif time_derivative == 1:
        basis = -(modes * omega)[:, None] * np.sin(phase)
    elif time_derivative == 2:
        basis = -((modes * omega) ** 2)[:, None] * np.cos(phase)
    else:
        raise ValueError("time_derivative must be 0, 1, or 2")

    if coefficients is None:
        return basis
    coefficients = np.asarray(coefficients, dtype=float)
    return np.tensordot(coefficients, basis, axes=(0, 0))


def project_cos_coefficients(
    values: np.ndarray, theta: np.ndarray, j_list: np.ndarray
) -> np.ndarray:
    """Project samples onto cosine coefficients.

    Implements docs Eq. (F12). ``values`` must have the time axis last.
    The returned array has shape ``(len(j_list),) + values.shape[:-1]``.
    """

    values = np.asarray(values, dtype=float)
    theta = np.asarray(theta, dtype=float)
    coeffs = []
    for j in np.asarray(j_list, dtype=int):
        if j == 0:
            coeffs.append(np.mean(values, axis=-1))
        else:
            coeffs.append(2.0 * np.mean(values * np.cos(j * theta), axis=-1))
    return np.stack(coeffs, axis=0)


def project_sin_coefficients(
    values: np.ndarray, theta: np.ndarray, j_list: np.ndarray
) -> np.ndarray:
    """Project samples onto sine coefficients with the time axis last."""

    values = np.asarray(values, dtype=float)
    theta = np.asarray(theta, dtype=float)
    coeffs = [
        2.0 * np.mean(values * np.sin(int(j) * theta), axis=-1)
        for j in np.asarray(j_list, dtype=int)
    ]
    return np.stack(coeffs, axis=0)


def unpack_reduced_state(y: np.ndarray, jmax: int):
    """Split reduced state into ``phi, q, A0, C`` arrays."""

    modes = mode_set(jmax)
    ns = modes.n_scalar
    phi = y[:ns]
    q = y[ns : 2 * ns]
    A0 = y[2 * ns]
    C = y[2 * ns + 1 :]
    return phi, q, A0, C


def unpack_full_state(y: np.ndarray, jmax: int):
    """Split full state into ``phi, q, A, C`` arrays."""

    modes = mode_set(jmax)
    ns = modes.n_scalar
    nm = modes.n_metric
    phi = y[:ns]
    q = y[ns : 2 * ns]
    A = y[2 * ns : 2 * ns + nm]
    C = y[2 * ns + nm : 2 * ns + 2 * nm]
    return phi, q, A, C


def solve_metric_modes_from_momentum(
    A0: np.ndarray,
    Phi_t: np.ndarray,
    Phi_x: np.ndarray,
    x: np.ndarray,
    omega: float,
    metric_modes: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """Compute positive even ``A_j`` from the momentum constraint.

    This implements docs Eq. (F13). The returned array includes ``A0`` as
    its first row and has shape ``(n_metric, n_x)``.
    """

    x = np.asarray(x, dtype=float)
    A0 = np.asarray(A0, dtype=float)
    metric_modes = np.asarray(metric_modes, dtype=int)
    positive = metric_modes[1:]
    if positive.size == 0:
        return A0[None, :]

    F = x[:, None] * Phi_t * Phi_x
    sin_pos = np.sin(np.outer(positive, theta))
    cos_pos = np.cos(np.outer(positive, theta))

    b = 2.0 * np.mean(F[:, None, :] * sin_pos[None, :, :], axis=-1)
    matrix_terms = 2.0 * np.mean(
        F[:, None, None, :]
        * sin_pos[None, :, None, :]
        * cos_pos[None, None, :, :],
        axis=-1,
    )
    diag = omega * np.diag(positive.astype(float))
    matrices = -diag[None, :, :] - matrix_terms
    rhs = A0[:, None] * b

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
    n_time: int = 128,
) -> np.ndarray:
    """Reconstruct all metric ``A_j`` from reduced state variables."""

    modes = mode_set(jmax)
    theta, _ = build_time_grid(n_time, omega)
    phi, q, A0, _ = unpack_reduced_state(y, jmax)
    cos_s = evaluate_fourier_modes(modes.scalar, theta)
    sin_t = evaluate_fourier_modes(
        modes.scalar, theta, omega=omega, time_derivative=1
    )
    Phi_t = phi.T @ sin_t
    Phi_x = q.T @ cos_s
    return solve_metric_modes_from_momentum(
        A0, Phi_t, Phi_x, x, omega, modes.metric, theta
    )


def fourier_rhs_reduced(
    x: np.ndarray,
    y: np.ndarray,
    omega: float,
    jmax: int,
    *,
    n_time: int = 128,
) -> np.ndarray:
    """Reduced radial RHS using docs Eqs. (F10)-(F13)."""

    modes = mode_set(jmax)
    theta, _ = build_time_grid(n_time, omega)
    phi, q, A0, Ccoef = unpack_reduced_state(y, jmax)

    cos_s = evaluate_fourier_modes(modes.scalar, theta)
    Phi = phi.T @ cos_s
    Phi_x = q.T @ cos_s
    Phi_t = phi.T @ evaluate_fourier_modes(
        modes.scalar, theta, omega=omega, time_derivative=1
    )
    Phi_tt = phi.T @ evaluate_fourier_modes(
        modes.scalar, theta, omega=omega, time_derivative=2
    )

    Acoef = solve_metric_modes_from_momentum(
        A0, Phi_t, Phi_x, x, omega, modes.metric, theta
    )
    cos_m = evaluate_fourier_modes(modes.metric, theta)
    C_t_basis = evaluate_fourier_modes(
        modes.metric, theta, omega=omega, time_derivative=1
    )
    A = Acoef.T @ cos_m
    C = Ccoef.T @ cos_m
    C_t = Ccoef.T @ C_t_basis

    xx = x[:, None]
    A_rhs = A * xx / 2.0 * (C * Phi_t**2 + Phi_x**2 + A * Phi**2)
    A_rhs += A / xx * (1.0 - A)
    C_rhs = 2.0 * C / xx * (1.0 - A + 0.5 * xx**2 * A * Phi**2)
    Phi_xx = C * Phi_tt + 0.5 * C_t * Phi_t
    Phi_xx += -Phi_x * (2.0 / xx - C_rhs / (2.0 * C)) + A * Phi

    dy = np.zeros_like(y)
    ns = modes.n_scalar
    dy[:ns] = q
    dy[ns : 2 * ns] = project_cos_coefficients(
        Phi_xx, theta, modes.scalar
    )
    dy[2 * ns] = project_cos_coefficients(
        A_rhs, theta, np.array([0], dtype=int)
    )[0]
    dy[2 * ns + 1 :] = project_cos_coefficients(
        C_rhs, theta, modes.metric
    )
    return dy


def fourier_rhs_full(
    x: np.ndarray,
    y: np.ndarray,
    omega: float,
    jmax: int,
    *,
    n_time: int = 128,
) -> np.ndarray:
    """Full ``A_j,C_j`` radial RHS from docs Eq. (F10).

    This is useful for residual checks. The production BVP solver uses
    :func:`fourier_rhs_reduced`.
    """

    modes = mode_set(jmax)
    theta, _ = build_time_grid(n_time, omega)
    phi, q, Acoef, Ccoef = unpack_full_state(y, jmax)

    cos_s = evaluate_fourier_modes(modes.scalar, theta)
    Phi = phi.T @ cos_s
    Phi_x = q.T @ cos_s
    Phi_t = phi.T @ evaluate_fourier_modes(
        modes.scalar, theta, omega=omega, time_derivative=1
    )
    Phi_tt = phi.T @ evaluate_fourier_modes(
        modes.scalar, theta, omega=omega, time_derivative=2
    )
    cos_m = evaluate_fourier_modes(modes.metric, theta)
    A = Acoef.T @ cos_m
    C = Ccoef.T @ cos_m
    C_t = Ccoef.T @ evaluate_fourier_modes(
        modes.metric, theta, omega=omega, time_derivative=1
    )

    xx = x[:, None]
    A_rhs = A * xx / 2.0 * (C * Phi_t**2 + Phi_x**2 + A * Phi**2)
    A_rhs += A / xx * (1.0 - A)
    C_rhs = 2.0 * C / xx * (1.0 - A + 0.5 * xx**2 * A * Phi**2)
    Phi_xx = C * Phi_tt + 0.5 * C_t * Phi_t
    Phi_xx += -Phi_x * (2.0 / xx - C_rhs / (2.0 * C)) + A * Phi

    dy = np.zeros_like(y)
    ns = modes.n_scalar
    nm = modes.n_metric
    dy[:ns] = q
    dy[ns : 2 * ns] = project_cos_coefficients(
        Phi_xx, theta, modes.scalar
    )
    dy[2 * ns : 2 * ns + nm] = project_cos_coefficients(
        A_rhs, theta, modes.metric
    )
    dy[2 * ns + nm :] = project_cos_coefficients(
        C_rhs, theta, modes.metric
    )
    return dy


def residual_fourier_odes(
    y: np.ndarray,
    dy_dx: np.ndarray,
    x: np.ndarray,
    omega: float,
    jmax: int,
    *,
    n_time: int = 128,
) -> np.ndarray:
    """Residual ``dy_dx - RHS`` for the full Fourier ODE system."""

    return dy_dx - fourier_rhs_full(x, y, omega, jmax, n_time=n_time)

