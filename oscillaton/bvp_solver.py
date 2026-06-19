"""Boundary-value solver for the Fourier oscillaton eigenproblem."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_bvp

from .fourier_projection import (
    fourier_rhs_reduced,
    mode_set,
    reconstruct_A_modes,
    reduced_state_size,
)
from .profiles import OscillatonProfile
from .sp_ground_state import solve_sp_ground_state


SP_BASE_CLOUD_MASS = 1.734128080715


def _initial_guess(jmax: int, x: np.ndarray, phi1_center: float) -> np.ndarray:
    modes = mode_set(jmax)
    y = np.zeros((reduced_state_size(jmax), x.size))
    radius = 4.0 + 8.0 / (1.0 + 10.0 * phi1_center)

    for idx, mode in enumerate(modes.scalar):
        amp = phi1_center if mode == 1 else ((-1.0) ** idx) * phi1_center * 0.03**idx
        y[idx] = amp * np.exp(-(x / radius) ** 2)
        y[modes.n_scalar + idx] = y[idx] * (-2.0 * x / radius**2)

    A0_idx = 2 * modes.n_scalar
    y[A0_idx] = 1.0 + 0.15 * (x / radius) ** 2 * np.exp(-(x / (1.5 * radius)) ** 2)
    y[A0_idx, 0] = 1.0

    C_start = A0_idx + 1
    y[C_start] = 1.03 + 0.25 * np.exp(-(x / radius) ** 2)
    for idx in range(1, modes.n_metric):
        y[C_start + idx] = 0.01 * ((-1.0) ** idx) * np.exp(-(x / radius) ** 2)
    return y


def kappa_from_phi1_center(phi1_center: float) -> float:
    """Return the SP scaling kappa matching phi_1(0) ~= sqrt(2) kappa^2."""

    if phi1_center <= 0.0:
        raise ValueError("phi1_center must be positive")
    return float(np.sqrt(phi1_center / np.sqrt(2.0)))


def phi1_center_from_sp_mass(target_mass: float) -> float:
    """Estimate ``phi_1(0)`` for a weak scalar oscillaton ADM mass.

    In the SP limit, ``mu M ~= SP_BASE_CLOUD_MASS * kappa`` and
    ``phi_1(0) ~= sqrt(2) kappa^2``.
    """

    if target_mass <= 0.0:
        raise ValueError("target_mass must be positive")
    kappa = target_mass / SP_BASE_CLOUD_MASS
    return float(np.sqrt(2.0) * kappa**2)


def sp_mass_estimate_from_phi1_center(phi1_center: float) -> float:
    """Estimate the weak-field ADM mass from ``phi_1(0)``."""

    return float(SP_BASE_CLOUD_MASS * kappa_from_phi1_center(phi1_center))


def _sp_initial_guess(
    jmax: int,
    x: np.ndarray,
    phi1_center: float,
    *,
    sp_y_max: float | None = None,
    sp_tol: float = 1.0e-6,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Build a weak-field Fourier-BVP guess from the SP ground state."""

    modes = mode_set(jmax)
    y = np.zeros((reduced_state_size(jmax), x.size))
    kappa = kappa_from_phi1_center(phi1_center)
    z_max = float(kappa * x[-1])
    sp_profile = solve_sp_ground_state(
        y_max=sp_y_max or max(40.0, z_max),
        n_grid=max(500, min(1800, int(25 * max(40.0, z_max)))),
        tol=sp_tol,
    )

    z = kappa * x
    F = np.interp(z, sp_profile.y, sp_profile.F)
    dF_dz = np.interp(z, sp_profile.y, sp_profile.dF)
    V = kappa**2 * np.interp(z, sp_profile.y, sp_profile.V)
    dV_dx = kappa**3 * np.interp(z, sp_profile.y, sp_profile.dV)
    V_infinity = kappa**2 * sp_profile.V_infinity

    phi1 = np.sqrt(2.0) * kappa**2 * F
    dphi1 = np.sqrt(2.0) * kappa**3 * dF_dz
    y[0] = phi1
    y[modes.n_scalar] = dphi1

    enclosed_over_x = x * dV_dx
    A0_idx = 2 * modes.n_scalar
    y[A0_idx] = 1.0 + enclosed_over_x

    C_start = A0_idx + 1
    y[C_start] = 1.0 + enclosed_over_x + V_infinity - V
    for idx in range(1, modes.n_metric):
        y[C_start + idx] = 0.0

    omega = 1.0 - 0.5 * V_infinity
    metadata = {
        "sp_kappa": kappa,
        "sp_z_max": z_max,
        "sp_mass_estimate": sp_profile.dimensionless_cloud_mass * kappa,
        "sp_omega_estimate": omega,
        "sp_phi1_center_estimate": float(phi1[0]),
    }
    return y, omega, metadata


def _expand_guess(previous: OscillatonProfile, jmax: int, x: np.ndarray) -> np.ndarray:
    old = mode_set(previous.jmax)
    new = mode_set(jmax)
    y = np.zeros((reduced_state_size(jmax), x.size))

    old_phi = np.vstack([previous.phi, previous.dphi])
    for old_idx, mode in enumerate(old.scalar):
        if mode in new.scalar:
            new_idx = int(np.where(new.scalar == mode)[0][0])
            y[new_idx] = np.interp(x, previous.x, old_phi[old_idx])
            y[new.n_scalar + new_idx] = np.interp(
                x, previous.x, old_phi[old.n_scalar + old_idx]
            )

    y[2 * new.n_scalar] = np.interp(x, previous.x, previous.A0)
    old_C_start = 2 * old.n_scalar + 1
    for old_idx, mode in enumerate(old.metric):
        if mode in new.metric:
            new_idx = int(np.where(new.metric == mode)[0][0])
            y[2 * new.n_scalar + 1 + new_idx] = np.interp(
                x, previous.x, previous.C[old_idx]
            )

    radius = 6.0
    for idx, mode in enumerate(new.scalar):
        if mode not in old.scalar:
            y[idx] = 0.003 * ((-1.0) ** idx) * np.exp(-(x / radius) ** 2)
            y[new.n_scalar + idx] = y[idx] * (-2.0 * x / radius**2)
    for idx, mode in enumerate(new.metric):
        if mode not in old.metric:
            y[2 * new.n_scalar + 1 + idx] = (
                0.001 * ((-1.0) ** idx) * np.exp(-(x / radius) ** 2)
            )
    return y


def _make_bc(jmax: int, phi1_center: float):
    modes = mode_set(jmax)
    ns = modes.n_scalar

    def bc(ya, yb, p):
        A0_outer = yb[2 * ns]
        C_outer = yb[2 * ns + 1 :]
        return np.r_[
            ya[0] - phi1_center,
            ya[ns : 2 * ns],
            ya[2 * ns] - 1.0,
            yb[:ns],
            C_outer[0] - A0_outer**2,
            C_outer[1:],
        ]

    return bc


def _solve_single(
    phi1_center: float,
    jmax: int,
    *,
    x_max: float,
    n_grid: int,
    n_time: int,
    tol: float,
    previous: OscillatonProfile | None,
    initial_y_guess: np.ndarray | None = None,
    omega_guess: float | None = None,
    seed_metadata: dict | None = None,
    verbose: int,
) -> OscillatonProfile:
    x = np.linspace(1.0e-4, x_max, n_grid)
    if initial_y_guess is not None:
        y_guess = initial_y_guess
        omega_guess_array = np.array([0.92 if omega_guess is None else omega_guess])
    elif previous is None:
        y_guess = _initial_guess(jmax, x, phi1_center)
        omega_guess_array = np.array([0.92])
    else:
        y_guess = _expand_guess(previous, jmax, x)
        omega_guess_array = np.array([previous.omega])

    def rhs(x_eval, y_eval, p_eval):
        return fourier_rhs_reduced(
            x_eval, y_eval, float(p_eval[0]), jmax, n_time=n_time
        )

    solution = solve_bvp(
        rhs,
        _make_bc(jmax, phi1_center),
        x,
        y_guess,
        p=omega_guess_array,
        tol=tol,
        bc_tol=tol,
        max_nodes=max(10000, 40 * n_grid),
        verbose=verbose,
    )

    modes = mode_set(jmax)
    A_modes = reconstruct_A_modes(
        solution.x, solution.y, float(solution.p[0]), jmax, n_time=n_time
    )
    ns = modes.n_scalar
    C_start = 2 * ns + 1
    metadata = {
        "success": bool(solution.success),
        "message": solution.message,
        "n_nodes": int(solution.x.size),
        "max_rms_residual": float(np.max(solution.rms_residuals))
        if solution.rms_residuals.size
        else np.nan,
        "status": int(solution.status),
    }
    if seed_metadata:
        metadata.update(seed_metadata)
    return OscillatonProfile(
        x=solution.x,
        scalar_modes=modes.scalar,
        metric_modes=modes.metric,
        phi=solution.y[:ns],
        dphi=solution.y[ns : 2 * ns],
        A=A_modes,
        C=solution.y[C_start : C_start + modes.n_metric],
        omega=float(solution.p[0]),
        jmax=jmax,
        phi1_center=phi1_center,
        metadata=metadata,
    )


def solve_profile(
    phi1_center: float,
    jmax: int = 6,
    x_max: float = 80.0,
    n_grid: int = 800,
    *,
    n_time: int = 128,
    tol: float = 1.0e-4,
    continuation: bool = True,
    previous: OscillatonProfile | None = None,
    verbose: int = 0,
) -> OscillatonProfile:
    """Solve the Fourier eigenvalue problem for a central scalar amplitude.

    The implementation follows docs Eq. (F13) and uses continuation through
    `jmax = 2, 4, ..., target` by default.
    """

    if phi1_center <= 0:
        raise ValueError("phi1_center must be positive")
    if jmax % 2 != 0:
        raise ValueError("jmax must be even")

    if not continuation or previous is not None or jmax == 2:
        return _solve_single(
            phi1_center,
            jmax,
            x_max=x_max,
            n_grid=n_grid,
            n_time=n_time,
            tol=tol,
            previous=previous,
            verbose=verbose,
        )

    profile = None
    for current_jmax in range(2, jmax + 1, 2):
        profile = _solve_single(
            phi1_center,
            current_jmax,
            x_max=x_max,
            n_grid=n_grid,
            n_time=n_time,
            tol=tol,
            previous=profile,
            verbose=verbose,
        )
    assert profile is not None
    return profile


def solve_profile_sp_seeded(
    phi1_center: float,
    jmax: int = 2,
    x_max: float | None = None,
    n_grid: int = 900,
    *,
    n_time: int = 96,
    tol: float = 5.0e-5,
    sp_y_max: float | None = None,
    sp_tol: float = 1.0e-6,
    verbose: int = 0,
) -> OscillatonProfile:
    """Solve a weak-field scalar oscillaton using the SP ground state as seed.

    This is intended for the nonrelativistic branch where the ordinary
    Gaussian initial guess becomes inefficient.  The input is still the
    relativistic Fourier-BVP parameter ``phi1_center``.
    """

    if phi1_center <= 0:
        raise ValueError("phi1_center must be positive")
    if jmax % 2 != 0:
        raise ValueError("jmax must be even")
    if x_max is None:
        kappa = kappa_from_phi1_center(phi1_center)
        x_max = max(80.0, 45.0 / kappa)

    x = np.linspace(1.0e-4, x_max, n_grid)
    y_guess, omega_guess, seed_metadata = _sp_initial_guess(
        jmax,
        x,
        phi1_center,
        sp_y_max=sp_y_max,
        sp_tol=sp_tol,
    )
    return _solve_single(
        phi1_center,
        jmax,
        x_max=x_max,
        n_grid=n_grid,
        n_time=n_time,
        tol=tol,
        previous=None,
        initial_y_guess=y_guess,
        omega_guess=omega_guess,
        seed_metadata=seed_metadata,
        verbose=verbose,
    )


def solve_family(
    phi1_values,
    jmax: int = 6,
    *,
    x_max: float = 80.0,
    n_grid: int = 800,
    n_time: int = 128,
    tol: float = 1.0e-4,
    verbose: int = 0,
) -> list[OscillatonProfile]:
    """Solve a family of profiles using amplitude continuation."""

    profiles: list[OscillatonProfile] = []
    previous = None
    for value in phi1_values:
        profile = solve_profile(
            float(value),
            jmax=jmax,
            x_max=x_max,
            n_grid=n_grid,
            n_time=n_time,
            tol=tol,
            previous=previous,
            continuation=False,
            verbose=verbose,
        )
        profiles.append(profile)
        previous = profile
    return profiles
