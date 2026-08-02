"""High-level builders for scalar and radial-vector oscillaton comparisons."""

from __future__ import annotations

import copy
from functools import lru_cache

import numpy as np

from kg_oscillaton import (
    kappa_from_phi1_center,
    phi1_center_from_sp_mass,
    solve_profile_sp_seeded,
    solve_sp_ground_state,
)
from proca_oscillaton import (
    solve_profile as solve_vector_profile,
    solve_profile_scaled_seeded,
    solve_radial_proca_nr_ground_state,
)


@lru_cache(maxsize=16)
def _cached_sp_ground_state(y_max: float, n_grid: int, tol: float):
    return solve_sp_ground_state(y_max=y_max, n_grid=n_grid, tol=tol)


@lru_cache(maxsize=8)
def _cached_radial_proca_nr_ground_state(y_max: float, n_grid: int, tol: float):
    return solve_radial_proca_nr_ground_state(y_max=y_max, n_grid=n_grid, tol=tol)


def epsilon_from_omega(omega: float) -> float:
    """Return epsilon = sqrt(1 - omega^2) for dimensionless omega."""

    return float(np.sqrt(max(0.0, 1.0 - float(omega) ** 2)))


def zero_mode_mass(profile) -> float:
    """Return the ADM mass read from the outer zero-mode mass plateau."""

    A0 = np.asarray(profile.A0)
    mass = 0.5 * profile.x * (1.0 - 1.0 / A0)
    tail = mass[-max(1, int(0.1 * mass.size)) :]
    return float(np.median(tail))


def zero_mode_metric_arrays(profile, x: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate A0, C0, and M(<x) on a requested radial grid."""

    x = np.asarray(x, dtype=float)
    A0 = np.interp(x, profile.x, profile.A0)
    C0 = np.interp(x, profile.x, profile.C0)
    return {
        "A0": A0,
        "C0": C0,
        "M0": 0.5 * x * (1.0 - 1.0 / A0),
    }


def mass_tail_diagnostics(profile, *, tail_fraction: float = 0.1) -> dict[str, float]:
    """Summarize the outer plateau of ``M(<x)`` read from ``A0``."""

    if not (0.0 < tail_fraction <= 1.0):
        raise ValueError("tail_fraction must be in (0, 1]")
    mass = 0.5 * profile.x * (1.0 - 1.0 / np.asarray(profile.A0))
    tail_size = max(1, int(np.ceil(tail_fraction * mass.size)))
    tail = mass[-tail_size:]
    median = float(np.median(tail))
    scale = max(abs(median), np.finfo(float).tiny)
    return {
        "mass_tail_median": median,
        "mass_tail_mean": float(np.mean(tail)),
        "mass_tail_std": float(np.std(tail)),
        "mass_tail_relative_std": float(np.std(tail) / scale),
        "mass_tail_min": float(np.min(tail)),
        "mass_tail_max": float(np.max(tail)),
    }


def boundary_diagnostics(profile, *, n_time: int = 96) -> dict[str, float]:
    """Return scalar or radial-Proca BVP boundary residual diagnostics."""

    if hasattr(profile, "phi"):
        positive_C = profile.C[1:, -1]
        return {
            "origin_phi1_error": float(profile.phi[0, 0] - profile.phi1_center),
            "origin_dphi_abs_max": float(np.max(np.abs(profile.dphi[:, 0]))),
            "origin_A0_error": float(profile.A0[0] - 1.0),
            "outer_phi_abs_max": float(np.max(np.abs(profile.phi[:, -1]))),
            "outer_C0_minus_A0_squared": float(profile.C0[-1] - profile.A0[-1] ** 2),
            "outer_positive_C_abs_max": float(np.max(np.abs(positive_C)))
            if positive_C.size
            else 0.0,
        }
    if hasattr(profile, "U_modes"):
        from proca_oscillaton.diagnostics import boundary_residuals

        return boundary_residuals(profile, n_time=n_time)
    raise TypeError("profile must be a scalar or radial-Proca oscillaton profile")


def profile_diagnostics(profile, *, n_time: int = 96) -> dict[str, float | bool | str]:
    """Return a compact quality summary for a scalar or radial-Proca profile."""

    metadata = dict(getattr(profile, "metadata", {}))
    out: dict[str, float | bool | str] = {
        "success": bool(metadata.get("success", True)),
        "message": str(metadata.get("message", "")),
        "status": int(metadata.get("status", 0)),
        "max_rms_residual": float(metadata.get("max_rms_residual", np.nan)),
        "jmax": int(profile.jmax),
        "omega": float(profile.omega),
        "epsilon": epsilon_from_omega(profile.omega),
        "x_max": float(profile.x[-1]),
        "n_nodes": int(profile.x.size),
        "mass": float(profile.mass),
        "zero_mode_mass": zero_mode_mass(profile),
    }
    out.update(mass_tail_diagnostics(profile))
    out.update(boundary_diagnostics(profile, n_time=n_time))
    return out


def metric_mode(profile, family: str, mode: int, x: np.ndarray) -> np.ndarray:
    """Evaluate a Fourier metric mode A_mode or C_mode on a requested grid."""

    if family == "A":
        coeffs = getattr(profile, "A", None)
        if coeffs is None:
            coeffs = profile.A_modes
    elif family == "C":
        coeffs = getattr(profile, "C", None)
        if coeffs is None:
            coeffs = profile.C_modes
    else:
        raise ValueError("family must be 'A' or 'C'")

    matches = np.where(profile.metric_modes == mode)[0]
    if matches.size == 0:
        return np.zeros_like(x, dtype=float)
    return np.interp(x, profile.x, coeffs[int(matches[0])])


def construct_scalar_oscillaton(
    target_mass: float,
    *,
    jmax: int = 6,
    n_grid: int = 1000,
    n_time: int = 96,
    tol: float = 1.0e-6,
    mass_tol: float = 5.0e-5,
    require_success: bool = True,
):
    """Construct a scalar oscillaton tuned to a target dimensionless ADM mass."""

    if target_mass <= 0.0:
        raise ValueError("target_mass must be positive")
    if jmax < 2 or jmax % 2:
        raise ValueError("jmax must be an even integer >= 2")

    def solve(phi1_center: float):
        kappa = kappa_from_phi1_center(phi1_center)
        return solve_profile_sp_seeded(
            phi1_center,
            jmax=jmax,
            x_max=max(80.0, 45.0 / kappa),
            n_grid=n_grid,
            n_time=n_time,
            tol=tol,
            require_success=require_success,
        )

    phi = phi1_center_from_sp_mass(target_mass)
    profile = solve(phi)
    if abs(profile.mass - target_mass) / target_mass < mass_tol:
        return profile

    if profile.mass < target_mass:
        lo_phi, lo_profile = phi, profile
        hi_phi = phi * 1.2
        hi_profile = solve(hi_phi)
        for _ in range(8):
            if hi_profile.mass >= target_mass:
                break
            lo_phi, lo_profile = hi_phi, hi_profile
            hi_phi *= 1.2
            hi_profile = solve(hi_phi)
    else:
        hi_phi, hi_profile = phi, profile
        lo_phi = phi / 1.2
        lo_profile = solve(lo_phi)
        for _ in range(8):
            if lo_profile.mass <= target_mass:
                break
            hi_phi, hi_profile = lo_phi, lo_profile
            lo_phi /= 1.2
            lo_profile = solve(lo_phi)

    best = lo_profile if abs(lo_profile.mass - target_mass) < abs(hi_profile.mass - target_mass) else hi_profile
    for _ in range(10):
        mid_phi = 0.5 * (lo_phi + hi_phi)
        mid_profile = solve(mid_phi)
        if abs(mid_profile.mass - target_mass) < abs(best.mass - target_mass):
            best = mid_profile
        if abs(mid_profile.mass - target_mass) / target_mass < mass_tol:
            return mid_profile
        if mid_profile.mass < target_mass:
            lo_phi, lo_profile = mid_phi, mid_profile
        else:
            hi_phi, hi_profile = mid_phi, mid_profile
    return best


def _construct_vector_reference_uncached():
    """Build the weak bound reference used by the scaled vector solver."""

    values = [0.02, 0.012, 0.006, 0.003, 0.0015, 0.001, 0.0007, 0.0005, 0.00035, 0.000334]
    previous = None
    for value in values:
        previous = solve_vector_profile(
            value,
            jmax=2,
            x_max=360.0,
            n_grid=900,
            n_time=64,
            tol=5.0e-4,
            continuation=False,
            previous=previous,
        )
    return previous


@lru_cache(maxsize=1)
def _cached_vector_reference():
    return _construct_vector_reference_uncached()


def construct_vector_reference(*, cache: bool = True):
    """Build the weak bound reference used by the scaled vector solver."""

    if cache:
        return copy.deepcopy(_cached_vector_reference())
    return _construct_vector_reference_uncached()


def construct_vector_oscillaton(
    target_mass: float,
    *,
    jmax: int = 6,
    n_grid: int = 1000,
    n_time: int = 96,
    tol: float = 1.0e-6,
    mass_tol: float = 5.0e-5,
    reference=None,
    require_success: bool = True,
):
    """Construct a radial real Proca oscillaton at a target ADM mass."""

    if target_mass <= 0.0:
        raise ValueError("target_mass must be positive")
    if jmax < 2 or jmax % 2:
        raise ValueError("jmax must be an even integer >= 2")

    profile = construct_vector_reference() if reference is None else reference
    u1_center = profile.u1_center * (target_mass / profile.mass) ** 3
    profile = solve_profile_scaled_seeded(
        u1_center,
        profile,
        jmax=2,
        n_grid=n_grid,
        n_time=64,
        tol=tol,
        require_success=require_success,
    )
    for _ in range(3):
        u1_center *= (target_mass / profile.mass) ** 3
        profile = solve_profile_scaled_seeded(
            u1_center,
            profile,
            jmax=2,
            n_grid=n_grid,
            n_time=64,
            tol=tol,
            require_success=require_success,
        )
    if jmax > 2:
        profile = solve_profile_scaled_seeded(
            u1_center,
            profile,
            jmax=jmax,
            x_max=profile.x[-1],
            n_grid=n_grid,
            n_time=n_time,
            tol=tol,
            require_success=require_success,
        )
    for _ in range(4):
        if abs(profile.mass - target_mass) / target_mass < mass_tol:
            break
        u1_center *= (target_mass / profile.mass) ** 3
        profile = solve_profile_scaled_seeded(
            u1_center,
            profile,
            jmax=jmax,
            n_grid=n_grid,
            n_time=n_time,
            tol=tol,
            require_success=require_success,
        )
    return profile


def scalar_sp_metric_arrays(x: np.ndarray, target_mass: float):
    """Return the scalar SP weak-field metric arrays at the requested ADM mass."""

    if target_mass <= 0.0:
        raise ValueError("target_mass must be positive")
    x = np.asarray(x, dtype=float)
    sp_base = _cached_sp_ground_state(45.0, 1200, 1.0e-7)
    kappa_sp = target_mass / sp_base.dimensionless_cloud_mass
    z = kappa_sp * x
    sp = _cached_sp_ground_state(max(45.0, float(z[-1])), 1200, 1.0e-7)
    V = kappa_sp**2 * np.interp(z, sp.y, sp.V)
    dV_dx = kappa_sp**3 * np.interp(z, sp.y, sp.dV)
    V_inf = kappa_sp**2 * sp.V_infinity
    arrays = {
        "A0": 1.0 + x * dV_dx,
        "C0": 1.0 + x * dV_dx + V_inf - V,
        "M0": 0.5 * x**2 * dV_dx,
    }
    return arrays, copy.deepcopy(sp.scaled(kappa_sp))


def radial_proca_sp_metric_arrays(x: np.ndarray, target_mass: float):
    """Return radial Proca SP metric arrays at the requested ADM mass."""

    if target_mass <= 0.0:
        raise ValueError("target_mass must be positive")
    profile = _cached_radial_proca_nr_ground_state(50.0, 900, 1.0e-7)
    scale = target_mass / profile.dimensionless_cloud_mass
    scaled = copy.deepcopy(profile.scaled(scale))
    return scaled.as_metric_arrays(np.asarray(x, dtype=float)), scaled
