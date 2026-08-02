"""Schrodinger-Poisson ground state in the Appendix-A normalization."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_bvp


@dataclass
class SPGroundState:
    """A spherical SP ground-state profile with alpha = gamma = 0.

    The profile uses the Appendix-A variables

        y = mu r,
        F = sqrt(8 pi / mu) f,
        V = 2 (Phi_c - E_NR / mu).

    The default normalization is F(0) = 1.  The scaling symmetry is
    F_kappa(y) = kappa**2 F_1(kappa y) and
    V_kappa(y) = kappa**2 V_1(kappa y).
    """

    y: np.ndarray
    F: np.ndarray
    dF: np.ndarray
    V: np.ndarray
    dV: np.ndarray
    kappa: float = 1.0
    metadata: dict = field(default_factory=dict)

    @property
    def enclosed_integral(self) -> np.ndarray:
        """Return N(y) = integral_0^y s^2 F(s)^2 ds."""

        return self.y**2 * self.dV

    @property
    def mass_integral(self) -> float:
        """Return N_infinity = integral y^2 F^2 dy."""

        return float(self.enclosed_integral[-1])

    @property
    def dimensionless_cloud_mass(self) -> float:
        """Return mu M_c = N_infinity / 2 in the Appendix-A convention."""

        return 0.5 * self.mass_integral

    @property
    def V_infinity(self) -> float:
        """Asymptotic constant in V(y) = V_infinity - N/y + O(y^-2)."""

        return float(self.V[-1] + self.y[-1] * self.dV[-1])

    @property
    def binding_energy_over_mu(self) -> float:
        """Return E_NR / mu when Phi_c(infinity) is set to zero."""

        return -0.5 * self.V_infinity

    @property
    def tail_robin_residual(self) -> float:
        """Residual of F' + (sqrt(V_infinity) + 1/y) F = 0 at y_max."""

        k = np.sqrt(max(self.V_infinity, 0.0))
        return float(self.dF[-1] + (k + 1.0 / self.y[-1]) * self.F[-1])

    @property
    def poisson_mass_tail(self) -> float:
        """Return y_max^2 V'(y_max), equal to N_infinity for a good tail."""

        return float(self.y[-1] ** 2 * self.dV[-1])

    def node_count(self, atol: float = 1.0e-10) -> int:
        """Count sign changes away from numerical zero."""

        mask = np.abs(self.F) > atol
        if np.count_nonzero(mask) < 2:
            return 0
        signs = np.sign(self.F[mask])
        return int(np.count_nonzero(signs[1:] * signs[:-1] < 0))

    def scaled(self, kappa: float) -> "SPGroundState":
        """Return the same solution rescaled by the Appendix-A symmetry."""

        if kappa <= 0.0:
            raise ValueError("kappa must be positive")
        if np.isclose(kappa, self.kappa):
            return self
        ratio = kappa / self.kappa
        return SPGroundState(
            y=self.y / ratio,
            F=ratio**2 * self.F,
            dF=ratio**3 * self.dF,
            V=ratio**2 * self.V,
            dV=ratio**3 * self.dV,
            kappa=float(kappa),
            metadata={**self.metadata, "scaled_from_kappa": self.kappa},
        )

    def as_columns(self) -> np.ndarray:
        """Return columns useful for saving to CSV."""

        N = self.enclosed_integral
        return np.column_stack((self.y, self.F, self.dF, self.V, self.dV, N, 0.5 * N))


def _initial_guess(
    y: np.ndarray,
    *,
    radius: float,
    V_center: float,
    V_infinity: float,
) -> np.ndarray:
    F = np.exp(-((y / radius) ** 2))
    dF = -2.0 * y * F / radius**2
    width = 1.5 * radius
    bridge = np.exp(-((y / width) ** 2))
    V = V_infinity - (V_infinity - V_center) * bridge
    dV = (V_infinity - V_center) * bridge * (2.0 * y / width**2)
    return np.vstack((F, dF, V, dV))


def _rhs(y: np.ndarray, state: np.ndarray) -> np.ndarray:
    F, dF, V, dV = state
    safe_y = np.maximum(y, 1.0e-14)
    return np.vstack(
        (
            dF,
            V * F - 2.0 * dF / safe_y,
            dV,
            F**2 - 2.0 * dV / safe_y,
        )
    )


def solve_sp_ground_state(
    *,
    kappa: float = 1.0,
    y_max: float = 40.0,
    y_min: float = 1.0e-5,
    n_grid: int = 800,
    tol: float = 1.0e-6,
    initial_radius: float = 2.0,
    initial_V_center: float = -1.0,
    initial_V_infinity: float = 1.0,
    max_nodes: int | None = None,
    verbose: int = 0,
    raise_on_fail: bool = True,
) -> SPGroundState:
    """Solve the alpha = gamma = 0 spherical SP ground-state BVP.

    The equations are

        F'' + 2 F'/y = V F,
        V'' + 2 V'/y = F^2,

    with regular origin, F(0)=1, and an outgoing Yukawa/Robin tail.
    The constant part of V is the eigenvalue selected by the tail condition.
    """

    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    if y_min <= 0.0:
        raise ValueError("y_min must be positive")
    if y_max <= y_min:
        raise ValueError("y_max must be larger than y_min")
    if n_grid < 20:
        raise ValueError("n_grid must be at least 20")

    y = np.linspace(y_min, y_max, n_grid)
    guess = _initial_guess(
        y,
        radius=initial_radius,
        V_center=initial_V_center,
        V_infinity=initial_V_infinity,
    )

    def bc(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        V_inf = yb[2] + y_max * yb[3]
        decay = np.sqrt(np.maximum(V_inf, 1.0e-14))
        return np.array(
            (
                ya[0] - 1.0,
                ya[1] - ya[2] * ya[0] * y_min / 3.0,
                ya[3] - ya[0] ** 2 * y_min / 3.0,
                yb[1] + (decay + 1.0 / y_max) * yb[0],
            )
        )

    solution = solve_bvp(
        _rhs,
        bc,
        y,
        guess,
        tol=tol,
        bc_tol=tol,
        max_nodes=max_nodes or max(10000, 40 * n_grid),
        verbose=verbose,
    )
    if raise_on_fail and not solution.success:
        raise RuntimeError(solution.message)

    y_sol = np.r_[0.0, solution.x]
    F_sol = np.r_[1.0, solution.y[0]]
    dF_sol = np.r_[0.0, solution.y[1]]
    V0 = solution.y[2, 0] - y_min**2 / 6.0
    V_sol = np.r_[V0, solution.y[2]]
    dV_sol = np.r_[0.0, solution.y[3]]

    metadata = {
        "success": bool(solution.success),
        "message": solution.message,
        "status": int(solution.status),
        "n_nodes": int(solution.x.size),
        "max_rms_residual": float(np.max(solution.rms_residuals))
        if solution.rms_residuals.size
        else np.nan,
        "y_min": float(y_min),
        "y_max": float(y_max),
        "tol": float(tol),
    }
    base = SPGroundState(
        y=y_sol,
        F=F_sol,
        dF=dF_sol,
        V=V_sol,
        dV=dV_sol,
        kappa=1.0,
        metadata=metadata,
    )
    return base.scaled(kappa)
