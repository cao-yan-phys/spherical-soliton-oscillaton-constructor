"""Nonrelativistic radial Proca ground state."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_bvp


@dataclass
class RadialProcaNRProfile:
    """Ground state of the radial Proca NR equation.

    The equations are

        F'' + 2 F'/y - 2 F/y^2 = V F,
        V'' + 2 V'/y = F^2.

    The base solution fixes ``F'(0)=1``.  The scaling symmetry is
    ``F_lambda(y)=lambda**2 F(lambda y)`` and
    ``V_lambda(y)=lambda**2 V(lambda y)``.
    """

    y: np.ndarray
    F: np.ndarray
    dF: np.ndarray
    V: np.ndarray
    dV: np.ndarray
    scale: float = 1.0
    metadata: dict = field(default_factory=dict)

    @property
    def enclosed_integral(self) -> np.ndarray:
        return self.y**2 * self.dV

    @property
    def mass_integral(self) -> float:
        return float(self.enclosed_integral[-1])

    @property
    def dimensionless_cloud_mass(self) -> float:
        return 0.5 * self.mass_integral

    @property
    def V_infinity(self) -> float:
        return float(self.V[-1] + self.y[-1] * self.dV[-1])

    @property
    def epsilon(self) -> float:
        return float(np.sqrt(max(self.V_infinity, 0.0)))

    @property
    def mass_over_epsilon(self) -> float:
        return self.dimensionless_cloud_mass / self.epsilon

    @property
    def tail_robin_residual(self) -> float:
        y = self.y[-1]
        k = self.epsilon
        return float(self.dF[-1] + (k + 1.0 / y + 1.0 / (y * (k * y + 1.0))) * self.F[-1])

    def scaled(self, scale: float) -> "RadialProcaNRProfile":
        if scale <= 0.0:
            raise ValueError("scale must be positive")
        if np.isclose(scale, self.scale):
            return self
        ratio = scale / self.scale
        return RadialProcaNRProfile(
            y=self.y / ratio,
            F=ratio**2 * self.F,
            dF=ratio**3 * self.dF,
            V=ratio**2 * self.V,
            dV=ratio**3 * self.dV,
            scale=float(scale),
            metadata={**self.metadata, "scaled_from": self.scale},
        )

    def as_metric_arrays(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate the weak-field metric and mass on a physical x grid."""

        x = np.asarray(x, dtype=float)
        V = np.interp(x, self.y, self.V, left=float(self.V[0]), right=float(self.V[-1]))
        dV_dx = np.interp(
            x, self.y, self.dV, left=float(self.dV[0]), right=float(self.dV[-1])
        )
        V_inf = self.V_infinity
        A0 = 1.0 + x * dV_dx
        C0 = 1.0 + x * dV_dx + V_inf - V
        return {
            "A0": A0,
            "C0": C0,
            "M0": 0.5 * x**2 * dV_dx,
        }


def _rhs(y: np.ndarray, state: np.ndarray) -> np.ndarray:
    F, dF, V, dV = state
    safe_y = np.maximum(y, 1.0e-14)
    return np.vstack(
        (
            dF,
            V * F - 2.0 * dF / safe_y + 2.0 * F / safe_y**2,
            dV,
            F**2 - 2.0 * dV / safe_y,
        )
    )


def solve_radial_proca_nr_ground_state(
    *,
    slope: float = 1.0,
    y_max: float = 50.0,
    y_min: float = 1.0e-5,
    n_grid: int = 800,
    tol: float = 1.0e-6,
    initial_radius: float = 3.0,
    initial_V_center: float = -1.5,
    initial_V_infinity: float = 2.0,
    max_nodes: int | None = None,
    verbose: int = 0,
    raise_on_fail: bool = True,
) -> RadialProcaNRProfile:
    """Solve the radial Proca NR ground state."""

    if slope <= 0.0:
        raise ValueError("slope must be positive")
    if y_min <= 0.0:
        raise ValueError("y_min must be positive")
    if y_max <= y_min:
        raise ValueError("y_max must be larger than y_min")

    y = np.linspace(y_min, y_max, n_grid)
    width = 1.5 * initial_radius
    F = slope * y * np.exp(-((y / initial_radius) ** 2))
    dF = slope * np.exp(-((y / initial_radius) ** 2)) * (
        1.0 - 2.0 * y**2 / initial_radius**2
    )
    bridge = np.exp(-((y / width) ** 2))
    V = initial_V_infinity - (initial_V_infinity - initial_V_center) * bridge
    dV = (initial_V_infinity - initial_V_center) * bridge * (2.0 * y / width**2)

    def bc(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        V_inf = yb[2] + y_max * yb[3]
        decay = np.sqrt(np.maximum(V_inf, 1.0e-14))
        tail = yb[1] + (
            decay + 1.0 / y_max + 1.0 / (y_max * (decay * y_max + 1.0))
        ) * yb[0]
        V0 = ya[2]
        return np.array(
            (
                ya[0] - (slope * y_min + V0 * slope * y_min**3 / 10.0),
                ya[1] - (slope + 3.0 * V0 * slope * y_min**2 / 10.0),
                ya[3] - slope**2 * y_min**3 / 5.0,
                tail,
            )
        )

    solution = solve_bvp(
        _rhs,
        bc,
        y,
        np.vstack((F, dF, V, dV)),
        tol=tol,
        bc_tol=tol,
        max_nodes=max_nodes or max(10000, 40 * n_grid),
        verbose=verbose,
    )
    if raise_on_fail and not solution.success:
        raise RuntimeError(solution.message)

    V0 = solution.y[2, 0] - slope**2 * y_min**4 / 20.0
    profile = RadialProcaNRProfile(
        y=np.r_[0.0, solution.x],
        F=np.r_[0.0, solution.y[0]],
        dF=np.r_[slope, solution.y[1]],
        V=np.r_[V0, solution.y[2]],
        dV=np.r_[0.0, solution.y[3]],
        metadata={
            "success": bool(solution.success),
            "message": solution.message,
            "status": int(solution.status),
            "n_nodes": int(solution.x.size),
            "max_rms_residual": float(np.max(solution.rms_residuals))
            if solution.rms_residuals.size
            else np.nan,
            "slope": float(slope),
            "y_min": float(y_min),
            "y_max": float(y_max),
            "tol": float(tol),
        },
    )
    return profile
