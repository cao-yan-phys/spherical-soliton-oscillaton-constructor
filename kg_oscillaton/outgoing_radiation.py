"""Quasi-isotropic scalar-oscillaton outgoing-radiation construction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.integrate import solve_bvp

from .bvp_solver import solve_profile as solve_polar_profile
from .fourier_projection import (
    evaluate_fourier_modes,
    mode_set,
    project_cos_coefficients,
)


@dataclass(frozen=True)
class _RadiationModes:
    scalar: np.ndarray
    metric: np.ndarray

    @property
    def n_scalar(self) -> int:
        return int(self.scalar.size)

    @property
    def n_metric(self) -> int:
        return int(self.metric.size)


@dataclass(frozen=True)
class ScalarOutgoingRadiationResult:
    """Numerical output of the scalar outgoing-radiation construction."""

    omega: float
    epsilon: float
    mass: float
    phi1_center: float
    delta3: float
    scalar_jmax: int
    r_max: float
    c3_standing: float
    c3_outgoing: float
    mass_loss_rate: float
    full_max_rms_residual: float
    response_max_rms_residual: float
    continuation_radii: np.ndarray
    continuation_omega: np.ndarray
    continuation_mass: np.ndarray
    continuation_c3: np.ndarray
    r: np.ndarray
    phi3_standing: np.ndarray
    phi3_outgoing: np.ndarray


def _radiation_modes(jmax: int) -> _RadiationModes:
    modes = mode_set(jmax)
    return _RadiationModes(scalar=modes.scalar, metric=modes.metric)


def _split_state(y: np.ndarray, jmax: int):
    modes = _radiation_modes(jmax)
    ns = modes.n_scalar
    nm = modes.n_metric
    phi = y[:ns]
    dphi = y[ns : 2 * ns]
    a = y[2 * ns : 2 * ns + nm]
    b = y[2 * ns + nm : 2 * ns + 2 * nm]
    db = y[2 * ns + 2 * nm : 2 * ns + 3 * nm]
    return phi, dphi, a, b, db


def _state_size(jmax: int) -> int:
    modes = _radiation_modes(jmax)
    return 2 * modes.n_scalar + 3 * modes.n_metric


def _time_grid(n_time: int) -> np.ndarray:
    return 2.0 * np.pi * np.arange(n_time, dtype=float) / float(n_time)


def _omega_from_eta(eta: float) -> float:
    omega_floor = 1.0 / 3.0 + 1.0e-6
    if eta >= 0.0:
        exp_neg = math.exp(-eta)
        sigmoid = 1.0 / (1.0 + exp_neg)
    else:
        exp_pos = math.exp(eta)
        sigmoid = exp_pos / (1.0 + exp_pos)
    return omega_floor + (1.0 - omega_floor) * sigmoid


def _eta_from_omega(omega: float) -> float:
    omega_floor = 1.0 / 3.0 + 1.0e-6
    fraction = (omega - omega_floor) / (1.0 - omega_floor)
    fraction = min(max(fraction, 1.0e-12), 1.0 - 1.0e-12)
    return math.log(fraction / (1.0 - fraction))


def _project_cos_complex(
    values: np.ndarray,
    theta: np.ndarray,
    modes: np.ndarray,
) -> np.ndarray:
    coefficients = []
    for mode in np.asarray(modes, dtype=int):
        if mode == 0:
            coefficients.append(np.mean(values, axis=-1))
        else:
            coefficients.append(
                2.0 * np.mean(values * np.cos(mode * theta), axis=-1)
            )
    return np.stack(coefficients, axis=0)


def _isotropic_rhs(
    r: np.ndarray,
    y: np.ndarray,
    omega: float,
    jmax: int,
    *,
    n_time: int,
) -> np.ndarray:
    modes = _radiation_modes(jmax)
    theta = _time_grid(n_time)
    phi, dphi, a_modes, b_modes, db_modes = _split_state(y, jmax)

    scalar_basis = evaluate_fourier_modes(modes.scalar, theta)
    metric_basis = evaluate_fourier_modes(modes.metric, theta)
    field = phi.T @ scalar_basis
    field_r = dphi.T @ scalar_basis
    field_t = phi.T @ evaluate_fourier_modes(
        modes.scalar, theta, omega=omega, time_derivative=1
    )
    field_tt = phi.T @ evaluate_fourier_modes(
        modes.scalar, theta, omega=omega, time_derivative=2
    )

    metric_a = 1.0 + a_modes.T @ metric_basis
    metric_a_t = a_modes.T @ evaluate_fourier_modes(
        modes.metric, theta, omega=omega, time_derivative=1
    )
    metric_b = 1.0 + b_modes.T @ metric_basis
    metric_b_t = b_modes.T @ evaluate_fourier_modes(
        modes.metric, theta, omega=omega, time_derivative=1
    )
    metric_b_tt = b_modes.T @ evaluate_fourier_modes(
        modes.metric, theta, omega=omega, time_derivative=2
    )
    metric_b_r = db_modes.T @ metric_basis

    radius = r[:, None]
    inverse_radius = 1.0 / radius
    source_tt = (
        field_t**2
        + (metric_a / metric_b) * field_r**2
        + metric_a * field**2
    )
    remaining_tt = 2.0 * (
        0.75 * (metric_b_t / metric_b) ** 2
        - 2.0 * metric_a * metric_b_r * inverse_radius / metric_b**2
        + 0.75 * metric_a * metric_b_r**2 / metric_b**3
    )
    metric_b_rr = -(metric_b**2 / (2.0 * metric_a)) * (
        source_tt - remaining_tt
    )

    source_rr = (
        field_r**2
        + (metric_b / metric_a) * field_t**2
        - metric_b * field**2
    )
    remaining_rr = (
        metric_b_r * inverse_radius / metric_b
        + 0.25 * (metric_b_r / metric_b) ** 2
        + 0.25 * metric_b_t**2 / (metric_a * metric_b)
        - metric_b_tt / metric_a
        + 0.5 * metric_a_t * metric_b_t / metric_a**2
    )
    coefficient_a_r = (
        2.0 / (metric_a * radius) + metric_b_r / (metric_a * metric_b)
    )
    metric_a_r = (source_rr - 2.0 * remaining_rr) / coefficient_a_r

    field_rr = metric_b * (
        field_tt / metric_a
        - 2.0 * field_r * inverse_radius / metric_b
        - 0.5 * field_r * metric_a_r / (metric_a * metric_b)
        - 0.5 * metric_b_r * field_r / metric_b**2
        + 1.5 * field_t * metric_b_t / (metric_a * metric_b)
        - 0.5 * metric_a_t * field_t / metric_a**2
        + field
    )

    projector = _project_cos_complex if np.iscomplexobj(y) else project_cos_coefficients
    derivative = np.zeros_like(y)
    ns = modes.n_scalar
    nm = modes.n_metric
    derivative[:ns] = dphi
    derivative[ns : 2 * ns] = projector(
        field_rr, theta, modes.scalar
    )
    derivative[2 * ns : 2 * ns + nm] = projector(
        metric_a_r, theta, modes.metric
    )
    derivative[2 * ns + nm : 2 * ns + 2 * nm] = db_modes
    derivative[2 * ns + 2 * nm : 2 * ns + 3 * nm] = (
        projector(metric_b_rr, theta, modes.metric)
    )
    return derivative


def _matching_basis(
    r_outer: float,
    omega: float,
    delta3: float,
    mass: float,
) -> tuple[float, float]:
    wave_number = math.sqrt((3.0 * omega) ** 2 - 1.0)
    phase_coefficient = mass * (2.0 * wave_number**2 + 1.0) / wave_number
    phase = (
        wave_number * r_outer
        + phase_coefficient * math.log(r_outer)
        + delta3
    )
    cosine = math.cos(phase)
    sine = math.sin(phase)
    value = cosine / r_outer
    derivative = (
        -(wave_number + phase_coefficient / r_outer) * sine / r_outer
        - cosine / r_outer**2
    )
    return value, derivative


def _standing_wave_amplitude(
    phi3: float,
    dphi3: float,
    r_outer: float,
    omega: float,
    mass: float,
) -> float:
    wave_number = math.sqrt((3.0 * omega) ** 2 - 1.0)
    phase_coefficient = mass * (2.0 * wave_number**2 + 1.0) / wave_number
    cosine_part = r_outer * phi3
    sine_part = -(
        r_outer * dphi3 + phi3
    ) / (wave_number + phase_coefficient / r_outer)
    return float(math.hypot(cosine_part, sine_part))


def _polar_seed(
    jmax: int,
    r: np.ndarray,
    phi1_center: float,
    *,
    n_time: int,
    tol: float,
) -> tuple[np.ndarray, float]:
    modes = _radiation_modes(jmax)
    ns = modes.n_scalar
    nm = modes.n_metric
    y = np.zeros((_state_size(jmax), r.size))
    seed_jmax = jmax + 1 if jmax % 2 else jmax
    seed = solve_polar_profile(
        phi1_center,
        jmax=max(4, seed_jmax),
        x_max=max(90.0, float(r[-1]) + 2.0),
        n_grid=min(800, max(450, r.size)),
        n_time=max(64, n_time),
        tol=max(tol, 5.0e-5),
        continuation=True,
        verbose=0,
    )

    x_seed = seed.x
    mass_profile = 0.5 * x_seed * (1.0 - 1.0 / seed.A0)
    outer_size = max(20, int(0.1 * mass_profile.size))
    mass = float(np.median(mass_profile[-outer_size:]))
    x_outer = float(x_seed[-1])
    isotropic_outer = 0.5 * (
        x_outer - mass + math.sqrt(x_outer * (x_outer - 2.0 * mass))
    )
    integrand = np.sqrt(np.maximum(seed.A0, 1.0e-14)) / x_seed
    log_isotropic_radius = np.empty_like(x_seed)
    log_isotropic_radius[-1] = math.log(isotropic_outer)
    for index in range(x_seed.size - 2, -1, -1):
        dx = x_seed[index + 1] - x_seed[index]
        log_isotropic_radius[index] = log_isotropic_radius[index + 1] - 0.5 * (
            integrand[index] + integrand[index + 1]
        ) * dx
    isotropic_seed = np.maximum.accumulate(np.exp(log_isotropic_radius))
    origin_scale = x_seed[0] / max(isotropic_seed[0], 1.0e-14)
    x_at_r = np.interp(r, isotropic_seed, x_seed, left=np.nan, right=x_seed[-1])
    origin = np.isnan(x_at_r)
    x_at_r[origin] = origin_scale * r[origin]

    for index, mode in enumerate(modes.scalar):
        matches = np.where(seed.scalar_modes == mode)[0]
        if not matches.size:
            continue
        seed_index = int(matches[0])
        y[index] = np.interp(
            x_at_r,
            x_seed,
            seed.phi[seed_index],
            left=seed.phi[seed_index, 0],
            right=0.0,
        )
        dphi_dx = np.interp(
            x_at_r,
            x_seed,
            seed.dphi[seed_index],
            left=0.0,
            right=0.0,
        )
        dr_dx = np.interp(
            x_at_r,
            x_seed,
            isotropic_seed * integrand,
            left=1.0 / max(origin_scale, 1.0e-14),
            right=1.0,
        )
        y[ns + index] = dphi_dx / np.maximum(dr_dx, 1.0e-14)

    theta = _time_grid(max(64, n_time))
    polar_fields = seed.evaluate(theta)
    lapse_squared = polar_fields["A"] / polar_fields["C"]
    a_seed = project_cos_coefficients(
        lapse_squared - 1.0, theta, modes.metric
    )
    b0_seed = (x_at_r / np.maximum(r, 1.0e-14)) ** 2 - 1.0
    for index in range(nm):
        y[2 * ns + index] = np.interp(
            x_at_r,
            x_seed,
            a_seed[index],
            left=a_seed[index, 0],
            right=0.0,
        )
        if index == 0:
            y[2 * ns + nm + index] = b0_seed
        y[2 * ns + 2 * nm + index] = np.gradient(
            y[2 * ns + nm + index], r, edge_order=2
        )
        y[2 * ns + 2 * nm + index, 0] = 0.0
    return y, float(seed.omega)


def _extend_solution(
    old_r: np.ndarray,
    old_y: np.ndarray,
    old_omega: float,
    r: np.ndarray,
    jmax: int,
) -> np.ndarray:
    guess = np.vstack(
        [np.interp(r, old_r, row, left=row[0], right=0.0) for row in old_y]
    )
    outer = float(old_r[-1])
    mask = r > outer
    if not np.any(mask):
        return guess

    modes = _radiation_modes(jmax)
    ns = modes.n_scalar
    nm = modes.n_metric
    radius = r[mask]
    a0 = 2 * ns
    b0 = 2 * ns + nm
    db0 = 2 * ns + 2 * nm
    guess[a0, mask] = old_y[a0, -1] * outer / radius
    guess[b0, mask] = old_y[b0, -1] * outer / radius
    guess[db0, mask] = -old_y[b0, -1] * outer / radius**2
    epsilon = math.sqrt(max(0.0, 1.0 - old_omega**2))
    phi1 = old_y[0, -1] * outer / radius * np.exp(-epsilon * (radius - outer))
    guess[0, mask] = phi1
    guess[ns, mask] = -(epsilon + 1.0 / radius) * phi1
    return guess


def _standing_boundary_conditions(
    phi1_center: float,
    jmax: int,
    delta3: float,
    r_outer: float,
    *,
    n_time: int,
):
    modes = _radiation_modes(jmax)
    ns = modes.n_scalar
    nm = modes.n_metric

    def boundary(ya: np.ndarray, yb: np.ndarray, parameter: np.ndarray):
        omega = _omega_from_eta(float(parameter[0]))
        _, _, a_outer, b_outer, db_outer = _split_state(yb[:, None], jmax)
        a_outer = a_outer[:, 0]
        b_outer = b_outer[:, 0]
        db_outer = db_outer[:, 0]
        rhs_outer = _isotropic_rhs(
            np.array([r_outer]),
            yb[:, None],
            omega,
            jmax,
            n_time=n_time,
        )[:, 0]
        da_outer = rhs_outer[2 * ns : 2 * ns + nm]
        mass = 0.5 * r_outer * b_outer[0]
        scalar_outer = []
        for index, mode in enumerate(modes.scalar):
            value = float(yb[index])
            derivative = float(yb[ns + index])
            if mode == 1:
                epsilon = math.sqrt(max(0.0, 1.0 - omega**2))
                scalar_outer.append(
                    derivative + (epsilon + 1.0 / r_outer) * value
                )
            elif mode == 3:
                basis, basis_derivative = _matching_basis(
                    r_outer, omega, delta3, mass
                )
                scalar_outer.append(
                    value * basis_derivative - derivative * basis
                )
            else:
                scalar_outer.append(value)
        return np.r_[
            ya[0] - phi1_center,
            ya[ns : 2 * ns],
            ya[2 * ns + 2 * nm : 2 * ns + 3 * nm],
            np.asarray(scalar_outer),
            a_outer[0] + r_outer * da_outer[0],
            a_outer[1:],
            b_outer[0] + r_outer * db_outer[0],
            b_outer[1:],
        ]

    return boundary


def _solve_standing_wave(
    phi1_center: float,
    delta3: float,
    *,
    jmax: int,
    r_max: float,
    n_grid: int,
    n_time: int,
    tol: float,
    previous,
):
    r = np.linspace(1.0e-4, r_max, n_grid)
    if previous is None:
        guess, omega_guess = _polar_seed(
            jmax,
            r,
            phi1_center,
            n_time=n_time,
            tol=tol,
        )
    else:
        old_r, old_y, omega_guess = previous
        guess = _extend_solution(old_r, old_y, omega_guess, r, jmax)

    solution = solve_bvp(
        lambda radius, state, parameter: _isotropic_rhs(
            radius,
            state,
            _omega_from_eta(float(parameter[0])),
            jmax,
            n_time=n_time,
        ),
        _standing_boundary_conditions(
            phi1_center,
            jmax,
            delta3,
            float(r[-1]),
            n_time=n_time,
        ),
        r,
        guess,
        p=np.array([_eta_from_omega(omega_guess)]),
        tol=tol,
        bc_tol=tol,
        max_nodes=max(15000, 60 * n_grid),
        verbose=0,
    )
    solution.eta = solution.p.copy()
    solution.p = np.array([_omega_from_eta(float(solution.eta[0]))])
    if not solution.success:
        residual = float(np.max(solution.rms_residuals))
        raise RuntimeError(
            "scalar standing-wave solve failed: "
            f"status={solution.status}, max_rms_residual={residual:.3e}, "
            f"message={solution.message}"
        )
    return solution


def _standing_summary(solution, jmax: int) -> dict[str, float]:
    modes = _radiation_modes(jmax)
    ns = modes.n_scalar
    omega = float(solution.p[0])
    r_outer = float(solution.x[-1])
    _, _, a_modes, b_modes, _ = _split_state(solution.y, jmax)
    mass_a = -0.5 * r_outer * a_modes[0, -1]
    mass_b = 0.5 * r_outer * b_modes[0, -1]
    i3 = int(np.where(modes.scalar == 3)[0][0])
    c3 = _standing_wave_amplitude(
        float(solution.y[i3, -1]),
        float(solution.y[ns + i3, -1]),
        r_outer,
        omega,
        mass_b,
    )
    return {
        "omega": omega,
        "mass": float(0.5 * (mass_a + mass_b)),
        "mass_b": float(mass_b),
        "c3": c3,
        "rms": float(np.max(solution.rms_residuals)),
    }


@dataclass(frozen=True)
class _ComplexBackground:
    full_solution: object
    omega: float
    jmax: int
    selected: np.ndarray
    i3: int
    ns: int
    nm: int

    @classmethod
    def from_solution(cls, solution, jmax: int) -> "_ComplexBackground":
        modes = _radiation_modes(jmax)
        i3 = int(np.where(modes.scalar == 3)[0][0])
        i5 = int(np.where(modes.scalar == 5)[0][0])
        ns = modes.n_scalar
        nm = modes.n_metric
        selected = [i3, ns + i3, i5, ns + i5]
        for metric_index in range(1, nm):
            selected.extend(
                [
                    2 * ns + metric_index,
                    2 * ns + nm + metric_index,
                    2 * ns + 2 * nm + metric_index,
                ]
            )
        return cls(
            full_solution=solution,
            omega=float(solution.p[0]),
            jmax=jmax,
            selected=np.asarray(selected, dtype=int),
            i3=i3,
            ns=ns,
            nm=nm,
        )

    def total_state(self, r: np.ndarray, response: np.ndarray) -> np.ndarray:
        state = np.asarray(self.full_solution.sol(r), dtype=complex)
        state[self.selected] = response
        return state

    def initial_response(self, r: np.ndarray) -> np.ndarray:
        state = np.asarray(self.full_solution.sol(r), dtype=float)
        return state[self.selected].astype(complex)


def _pack_complex(values: np.ndarray) -> np.ndarray:
    return np.vstack([values.real, values.imag])


def _unpack_complex(values: np.ndarray) -> np.ndarray:
    half = values.shape[0] // 2
    return values[:half] + 1j * values[half:]


def _outgoing_log_derivative(
    r_outer: float,
    omega: float,
    mass: float,
    *,
    use_mass_phase: bool,
) -> complex:
    wave_number = math.sqrt((3.0 * omega) ** 2 - 1.0)
    phase_coefficient = 0.0
    if use_mass_phase:
        phase_coefficient = mass * (2.0 * wave_number**2 + 1.0) / wave_number
    return 1j * (wave_number + phase_coefficient / r_outer) - 1.0 / r_outer


def _outgoing_amplitude(
    phi3: complex,
    r_outer: float,
    omega: float,
    mass: float,
    *,
    use_mass_phase: bool,
) -> complex:
    wave_number = math.sqrt((3.0 * omega) ** 2 - 1.0)
    phase_coefficient = 0.0
    if use_mass_phase:
        phase_coefficient = mass * (2.0 * wave_number**2 + 1.0) / wave_number
    phase = wave_number * r_outer + phase_coefficient * math.log(r_outer)
    return phi3 * r_outer * np.exp(-1j * phase)


def _solve_outgoing_response(
    background: _ComplexBackground,
    *,
    n_time: int,
    tol: float,
    use_mass_phase: bool,
):
    r = np.asarray(background.full_solution.x)
    guess = _pack_complex(background.initial_response(r))
    r_outer = float(r[-1])
    full_outer = np.asarray(
        background.full_solution.sol(np.array([r_outer])), dtype=float
    )
    mass = 0.5 * r_outer * float(
        full_outer[2 * background.ns + background.nm, 0]
    )
    log_derivative = _outgoing_log_derivative(
        r_outer,
        background.omega,
        mass,
        use_mass_phase=use_mass_phase,
    )

    def rhs(radius: np.ndarray, packed_response: np.ndarray):
        response = _unpack_complex(packed_response)
        total = background.total_state(radius, response)
        total_rhs = _isotropic_rhs(
            radius,
            total,
            background.omega,
            background.jmax,
            n_time=n_time,
        )
        return _pack_complex(total_rhs[background.selected])

    def boundary_complex(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        residuals = [left[1], left[3]]
        offset = 4
        for _ in range(1, background.nm):
            residuals.append(left[offset + 2])
            offset += 3
        residuals.append(right[1] - log_derivative * right[0])
        residuals.append(right[2])
        offset = 4
        for _ in range(1, background.nm):
            residuals.extend([right[offset], right[offset + 1]])
            offset += 3
        return np.asarray(residuals, dtype=complex)

    def boundary(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        residual = boundary_complex(
            _unpack_complex(left[:, None])[:, 0],
            _unpack_complex(right[:, None])[:, 0],
        )
        return np.r_[residual.real, residual.imag]

    solution = solve_bvp(
        rhs,
        boundary,
        r,
        guess,
        tol=tol,
        bc_tol=tol,
        max_nodes=max(18000, 80 * r.size),
        verbose=0,
    )
    solution.mass = mass
    if not solution.success:
        residual = float(np.max(solution.rms_residuals))
        raise RuntimeError(
            "scalar outgoing-radiation response solve failed: "
            f"status={solution.status}, max_rms_residual={residual:.3e}, "
            f"message={solution.message}"
        )
    return solution


def _validated_radii(r_max: float | Iterable[float]) -> np.ndarray:
    if np.isscalar(r_max):
        radii = np.array([float(r_max)])
    else:
        radii = np.asarray(tuple(r_max), dtype=float)
    if radii.ndim != 1 or radii.size == 0:
        raise ValueError("r_max must be a positive radius or a nonempty sequence")
    if np.any(~np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("all r_max values must be finite and positive")
    if np.any(np.diff(radii) <= 0.0):
        raise ValueError("r_max continuation values must be strictly increasing")
    return radii


def solve_scalar_outgoing_radiation(
    phi1_center: float = 0.53137,
    delta3: float = -0.8,
    *,
    scalar_jmax: int = 5,
    r_max: float | Iterable[float] = (60.0, 80.0),
    n_grid: int = 520,
    n_time: int = 96,
    tol: float = 5.0e-5,
    response_tol: float = 1.0e-7,
    use_mass_phase: bool = True,
) -> ScalarOutgoingRadiationResult:
    """Construct the standing scalar core and its complex outgoing radiation.

    ``scalar_jmax`` is the largest retained odd Klein-Gordon harmonic. Metric
    modes are the even harmonics below it. The default parameters reproduce
    the published scalar benchmark near ``omega=0.86``.
    """

    if not np.isfinite(phi1_center) or phi1_center <= 0.0:
        raise ValueError("phi1_center must be finite and positive")
    if (
        not isinstance(scalar_jmax, (int, np.integer))
        or scalar_jmax < 5
        or scalar_jmax % 2 == 0
    ):
        raise ValueError("scalar_jmax must be an odd integer at least 5")
    if n_grid < 100:
        raise ValueError("n_grid must be at least 100")
    if n_time < 2 * scalar_jmax + 2:
        raise ValueError("n_time must be at least 2*scalar_jmax+2")
    if tol <= 0.0 or response_tol <= 0.0:
        raise ValueError("tol and response_tol must be positive")
    radii = _validated_radii(r_max)

    previous = None
    standing_solution = None
    omega_history = []
    mass_history = []
    c3_history = []
    for radius in radii:
        standing_solution = _solve_standing_wave(
            phi1_center,
            delta3,
            jmax=scalar_jmax,
            r_max=float(radius),
            n_grid=n_grid,
            n_time=n_time,
            tol=tol,
            previous=previous,
        )
        summary = _standing_summary(standing_solution, scalar_jmax)
        omega_history.append(summary["omega"])
        mass_history.append(summary["mass"])
        c3_history.append(summary["c3"])
        previous = (
            standing_solution.x,
            standing_solution.y,
            summary["omega"],
        )

    assert standing_solution is not None
    final_summary = _standing_summary(standing_solution, scalar_jmax)
    background = _ComplexBackground.from_solution(standing_solution, scalar_jmax)
    response_solution = _solve_outgoing_response(
        background,
        n_time=n_time,
        tol=response_tol,
        use_mass_phase=use_mass_phase,
    )
    response = _unpack_complex(response_solution.y)
    r = np.asarray(response_solution.x).copy()
    outgoing_phi3 = response[0].copy()
    standing_state = np.asarray(standing_solution.sol(r), dtype=float)
    standing_phi3 = standing_state[background.i3].copy()
    amplitude = _outgoing_amplitude(
        outgoing_phi3[-1],
        float(r[-1]),
        background.omega,
        response_solution.mass,
        use_mass_phase=use_mass_phase,
    )
    c3_outgoing = float(abs(amplitude))
    wave_number = math.sqrt((3.0 * background.omega) ** 2 - 1.0)
    mass_loss_rate = (
        -0.75 * c3_outgoing**2 * background.omega * wave_number
    )
    return ScalarOutgoingRadiationResult(
        omega=background.omega,
        epsilon=math.sqrt(max(0.0, 1.0 - background.omega**2)),
        mass=final_summary["mass"],
        phi1_center=float(phi1_center),
        delta3=float(delta3),
        scalar_jmax=int(scalar_jmax),
        r_max=float(r[-1]),
        c3_standing=final_summary["c3"],
        c3_outgoing=c3_outgoing,
        mass_loss_rate=float(mass_loss_rate),
        full_max_rms_residual=final_summary["rms"],
        response_max_rms_residual=float(
            np.max(response_solution.rms_residuals)
        ),
        continuation_radii=radii.copy(),
        continuation_omega=np.asarray(omega_history),
        continuation_mass=np.asarray(mass_history),
        continuation_c3=np.asarray(c3_history),
        r=r,
        phi3_standing=standing_phi3,
        phi3_outgoing=outgoing_phi3,
    )


__all__ = ["ScalarOutgoingRadiationResult", "solve_scalar_outgoing_radiation"]
