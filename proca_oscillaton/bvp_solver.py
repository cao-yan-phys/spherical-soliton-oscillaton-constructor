"""BVP solver for real Proca oscillatons."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_bvp

from .fourier import build_phase_grid, evaluate_modes, mode_set, project_cos, reduced_state_size
from .profiles import ProcaOscillatonProfile
from .residuals import reconstruct_A_modes, rhs_reduced


def _initial_guess(jmax: int, x: np.ndarray, u1_center: float) -> np.ndarray:
    modes = mode_set(jmax)
    y = np.zeros((reduced_state_size(jmax), x.size))
    radius = 12.0 / max(1.0, np.sqrt(u1_center / 0.02))

    U1 = u1_center * np.exp(-(x / radius) ** 2)
    E1 = -u1_center * x / 3.0 * np.exp(-(x / radius) ** 2)
    y[0] = U1
    y[modes.n_matter] = E1

    for idx, mode in enumerate(modes.matter[1:], start=1):
        amp = ((-1.0) ** idx) * u1_center * 0.02**idx
        y[idx] = amp * np.exp(-(x / radius) ** 2)
        y[modes.n_matter + idx] = -amp * x / 3.0 * np.exp(-(x / radius) ** 2)

    A0_idx = 2 * modes.n_matter
    y[A0_idx] = 1.0 + 0.002 * (x / radius) ** 2 * np.exp(-(x / (1.5 * radius)) ** 2)
    y[A0_idx, 0] = 1.0
    C_start = A0_idx + 1
    y[C_start] = 1.0 + 0.005 * np.exp(-(x / radius) ** 2)
    for idx in range(1, modes.n_metric):
        y[C_start + idx] = 0.001 * ((-1.0) ** idx) * np.exp(-(x / radius) ** 2)
    return y


def _expand_guess(previous: ProcaOscillatonProfile, jmax: int, x: np.ndarray) -> np.ndarray:
    old = mode_set(previous.jmax)
    new = mode_set(jmax)
    y = np.zeros((reduced_state_size(jmax), x.size))
    for old_idx, mode in enumerate(old.matter):
        if mode in new.matter:
            new_idx = int(np.where(new.matter == mode)[0][0])
            y[new_idx] = np.interp(x, previous.x, previous.U_modes[old_idx])
            y[new.n_matter + new_idx] = np.interp(
                x, previous.x, previous.E_modes[old_idx]
            )
    y[2 * new.n_matter] = np.interp(x, previous.x, previous.A0)
    for old_idx, mode in enumerate(old.metric):
        if mode in new.metric:
            new_idx = int(np.where(new.metric == mode)[0][0])
            y[2 * new.n_matter + 1 + new_idx] = np.interp(
                x, previous.x, previous.C_modes[old_idx]
            )
    radius = 10.0
    for idx, mode in enumerate(new.matter):
        if mode not in old.matter:
            amp = 0.001 * ((-1.0) ** idx)
            y[idx] = amp * np.exp(-(x / radius) ** 2)
            y[new.n_matter + idx] = -amp * x / 3.0 * np.exp(-(x / radius) ** 2)
    for idx, mode in enumerate(new.metric):
        if mode not in old.metric:
            y[2 * new.n_matter + 1 + idx] = (
                0.0005 * ((-1.0) ** idx) * np.exp(-(x / radius) ** 2)
            )
    return y


def _interp_tail(
    x_new: np.ndarray,
    x_old: np.ndarray,
    values: np.ndarray,
    *,
    right: float,
) -> np.ndarray:
    return np.interp(x_new, x_old, values, left=float(values[0]), right=right)


def _scaled_weak_field_guess(
    reference: ProcaOscillatonProfile,
    u1_center: float,
    jmax: int,
    x: np.ndarray,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Scale a weak Proca profile to a smaller central amplitude.

    In the nonrelativistic zero-magnetic branch, the dominant electric field
    scales like kappa**2 while U is one derivative smaller at the origin and
    scales like kappa**3.  Thus lambda = kappa/kappa_ref is inferred from
    U_1(0).
    """

    if reference.u1_center <= 0.0:
        raise ValueError("reference.u1_center must be positive")
    if u1_center <= 0.0:
        raise ValueError("u1_center must be positive")

    lam = float((u1_center / reference.u1_center) ** (1.0 / 3.0))
    x_ref = lam * x
    old = mode_set(reference.jmax)
    new = mode_set(jmax)
    y = np.zeros((reduced_state_size(jmax), x.size))

    for new_idx, mode in enumerate(new.matter):
        if mode in old.matter:
            old_idx = int(np.where(old.matter == mode)[0][0])
            y[new_idx] = lam**3 * _interp_tail(
                x_ref, reference.x, reference.U_modes[old_idx], right=0.0
            )
            y[new.n_matter + new_idx] = lam**2 * _interp_tail(
                x_ref, reference.x, reference.E_modes[old_idx], right=0.0
            )

    A0_idx = 2 * new.n_matter
    y[A0_idx] = 1.0 + lam**2 * (
        _interp_tail(x_ref, reference.x, reference.A0, right=1.0) - 1.0
    )
    C_start = A0_idx + 1
    for new_idx, mode in enumerate(new.metric):
        if mode in old.metric:
            old_idx = int(np.where(old.metric == mode)[0][0])
            if mode == 0:
                y[C_start + new_idx] = 1.0 + lam**2 * (
                    _interp_tail(
                        x_ref, reference.x, reference.C_modes[old_idx], right=1.0
                    )
                    - 1.0
                )
            else:
                y[C_start + new_idx] = lam**2 * _interp_tail(
                    x_ref, reference.x, reference.C_modes[old_idx], right=0.0
                )

    omega_guess = np.sqrt(max(1.0 - lam**2 * (1.0 - reference.omega**2), 1.0e-14))
    metadata = {
        "scaled_seed_lambda": lam,
        "scaled_seed_reference_u1": float(reference.u1_center),
        "scaled_seed_reference_mass": float(reference.mass),
        "scaled_seed_reference_omega": float(reference.omega),
        "scaled_seed_mass_estimate": float(reference.mass * lam),
        "scaled_seed_omega_estimate": float(omega_guess),
    }
    return y, float(omega_guess), metadata


def _make_bc(jmax: int, u1_center: float, x_min: float, x_max: float, n_time: int):
    modes = mode_set(jmax)
    nm = modes.n_matter
    theta = build_phase_grid(n_time)

    def bc(ya, yb, p):
        U_center = evaluate_modes(ya[:nm, None], modes.matter, theta)[0:1]
        C_center = evaluate_modes(ya[2 * nm + 1 :, None], modes.metric, theta)[0:1]
        regular_E = -x_min / 3.0 * project_cos(
            np.sqrt(np.maximum(C_center, 1.0e-14)) * U_center,
            theta,
            modes.matter,
        )[:, 0]
        omega = float(p[0])
        outer_matter = []
        for idx, mode in enumerate(modes.matter):
            wave = abs(float(mode) * omega)
            if wave < 1.0:
                decay = np.sqrt(max(1.0 - wave**2, 1.0e-14))
                outer_matter.append(
                    decay**2 * yb[nm + idx] - (decay + 1.0 / x_max) * yb[idx]
                )
            else:
                outer_matter.append(yb[idx])
        A0_outer = yb[2 * nm]
        C_outer = yb[2 * nm + 1 :]
        return np.r_[
            ya[0] - u1_center,
            ya[nm : 2 * nm] - regular_E,
            ya[2 * nm] - 1.0,
            outer_matter,
            C_outer[0] - A0_outer**2,
            C_outer[1:],
        ]

    return bc


def _solve_single(
    u1_center: float,
    jmax: int,
    *,
    x_max: float,
    n_grid: int,
    n_time: int,
    tol: float,
    previous: ProcaOscillatonProfile | None,
    initial_y_guess: np.ndarray | None = None,
    omega_guess: float | None = None,
    seed_metadata: dict | None = None,
    verbose: int,
) -> ProcaOscillatonProfile:
    x = np.linspace(1.0e-4, x_max, n_grid)
    if initial_y_guess is not None:
        y_guess = initial_y_guess
        omega_guess_array = np.array([0.97 if omega_guess is None else omega_guess])
    elif previous is None:
        y_guess = _initial_guess(jmax, x, u1_center)
        omega_guess_array = np.array([0.97])
    else:
        y_guess = _expand_guess(previous, jmax, x)
        omega_guess_array = np.array([previous.omega])

    def rhs(x_eval, y_eval, p_eval):
        return rhs_reduced(x_eval, y_eval, float(p_eval[0]), jmax, n_time=n_time)

    solution = solve_bvp(
        rhs,
        _make_bc(jmax, u1_center, x[0], x[-1], n_time),
        x,
        y_guess,
        p=omega_guess_array,
        tol=tol,
        bc_tol=tol,
        max_nodes=max(10000, 40 * n_grid),
        verbose=verbose,
    )
    modes = mode_set(jmax)
    nm = modes.n_matter
    C_start = 2 * nm + 1
    A_modes = reconstruct_A_modes(
        solution.x, solution.y, float(solution.p[0]), jmax, n_time=n_time
    )
    return ProcaOscillatonProfile(
        x=solution.x,
        matter_modes=modes.matter,
        metric_modes=modes.metric,
        U_modes=solution.y[:nm],
        E_modes=solution.y[nm : 2 * nm],
        A_modes=A_modes,
        C_modes=solution.y[C_start : C_start + modes.n_metric],
        omega=float(solution.p[0]),
        jmax=jmax,
        u1_center=u1_center,
        metadata={
            "success": bool(solution.success),
            "message": solution.message,
            "status": int(solution.status),
            "n_nodes": int(solution.x.size),
            "max_rms_residual": float(np.max(solution.rms_residuals))
            if solution.rms_residuals.size
            else np.nan,
            **(seed_metadata or {}),
        },
    )


def solve_profile(
    u1_center: float,
    jmax: int = 2,
    x_max: float = 80.0,
    n_grid: int = 500,
    *,
    n_time: int = 96,
    tol: float = 1.0e-4,
    continuation: bool = True,
    previous: ProcaOscillatonProfile | None = None,
    verbose: int = 0,
) -> ProcaOscillatonProfile:
    """Solve a real Proca oscillaton candidate."""

    if u1_center <= 0:
        raise ValueError("u1_center must be positive")
    if previous is not None or not continuation or jmax == 2:
        return _solve_single(
            u1_center,
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
            u1_center,
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


def solve_profile_scaled_seeded(
    u1_center: float,
    reference: ProcaOscillatonProfile,
    jmax: int | None = None,
    x_max: float | None = None,
    n_grid: int = 1000,
    *,
    n_time: int = 96,
    tol: float = 1.0e-6,
    verbose: int = 0,
) -> ProcaOscillatonProfile:
    """Solve a weak Proca oscillaton using a scaled weak-field reference."""

    if u1_center <= 0:
        raise ValueError("u1_center must be positive")
    jmax = reference.jmax if jmax is None else int(jmax)
    if jmax < 2 or jmax % 2:
        raise ValueError("jmax must be an even integer >= 2")
    lam = float((u1_center / reference.u1_center) ** (1.0 / 3.0))
    if x_max is None:
        x_max = reference.x[-1] / lam

    x = np.linspace(1.0e-4, x_max, n_grid)
    y_guess, omega, seed_metadata = _scaled_weak_field_guess(
        reference, u1_center, jmax, x
    )
    return _solve_single(
        u1_center,
        jmax,
        x_max=x_max,
        n_grid=n_grid,
        n_time=n_time,
        tol=tol,
        previous=None,
        initial_y_guess=y_guess,
        omega_guess=omega,
        seed_metadata=seed_metadata,
        verbose=verbose,
    )


def solve_family(
    u1_values,
    jmax: int = 2,
    *,
    x_max: float = 80.0,
    n_grid: int = 500,
    n_time: int = 96,
    tol: float = 1.0e-4,
    verbose: int = 0,
) -> list[ProcaOscillatonProfile]:
    profiles = []
    previous = None
    for value in u1_values:
        profile = solve_profile(
            float(value),
            jmax=jmax,
            x_max=x_max,
            n_grid=n_grid,
            n_time=n_time,
            tol=tol,
            continuation=False,
            previous=previous,
            verbose=verbose,
        )
        profiles.append(profile)
        previous = profile
    return profiles
