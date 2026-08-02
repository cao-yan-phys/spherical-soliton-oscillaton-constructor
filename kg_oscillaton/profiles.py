"""Profile containers and reconstruction helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .diagnostics import mass_function, rmax_grr, total_mass
from .fourier_projection import evaluate_fourier_modes, mode_set


@dataclass
class OscillatonProfile:
    """A solved Fourier oscillaton profile."""

    x: np.ndarray
    scalar_modes: np.ndarray
    metric_modes: np.ndarray
    phi: np.ndarray
    dphi: np.ndarray
    A: np.ndarray
    C: np.ndarray
    omega: float
    jmax: int
    phi1_center: float
    metadata: dict = field(default_factory=dict)

    def evaluate(self, theta: np.ndarray | float) -> dict[str, np.ndarray]:
        """Evaluate the rescaled `Phi`, `Phi_t`, `A`, `C`, `a`, and `alpha`."""

        theta_arr = np.atleast_1d(np.asarray(theta, dtype=float))
        Phi = evaluate_fourier_modes(self.scalar_modes, theta_arr, self.phi)
        Phi_t = evaluate_fourier_modes(
            self.scalar_modes,
            theta_arr,
            self.phi,
            omega=self.omega,
            time_derivative=1,
        )
        Phi_x = evaluate_fourier_modes(self.scalar_modes, theta_arr, self.dphi)
        A = evaluate_fourier_modes(self.metric_modes, theta_arr, self.A)
        C = evaluate_fourier_modes(self.metric_modes, theta_arr, self.C)
        a = np.sqrt(A)
        alpha = np.sqrt(A / C)
        return {
            "theta": theta_arr,
            "Phi": Phi,
            "Phi_t": Phi_t,
            "Phi_x": Phi_x,
            "A": A,
            "C": C,
            "a": a,
            "alpha": alpha,
        }

    def initial_data(self) -> dict[str, np.ndarray]:
        """Return the `t=0` phase described in docs Eq. (F15)."""

        data = self.evaluate(0.0)
        squeeze = {}
        for key, value in data.items():
            if key == "theta":
                continue
            squeeze[key] = np.asarray(value)[..., 0]
        squeeze["Pi"] = np.zeros_like(self.x)
        squeeze["Psi"] = squeeze["Phi_x"]
        return squeeze

    @property
    def A0(self) -> np.ndarray:
        return self.A[0]

    @property
    def C0(self) -> np.ndarray:
        return self.C[0]

    @property
    def mass_profile(self) -> np.ndarray:
        return mass_function(self.x, self.initial_data()["A"])

    @property
    def mass(self) -> float:
        return total_mass(self.x, self.initial_data()["A"])

    @property
    def rmax(self) -> tuple[float, float]:
        return rmax_grr(self.x, self.initial_data()["A"])


def empty_profile_like_grid(jmax: int, x: np.ndarray, phi1_center: float):
    """Return zero arrays with the correct profile shapes."""

    modes = mode_set(jmax)
    return {
        "x": x,
        "scalar_modes": modes.scalar,
        "metric_modes": modes.metric,
        "phi": np.zeros((modes.n_scalar, x.size)),
        "dphi": np.zeros((modes.n_scalar, x.size)),
        "A": np.zeros((modes.n_metric, x.size)),
        "C": np.zeros((modes.n_metric, x.size)),
        "omega": np.nan,
        "jmax": jmax,
        "phi1_center": phi1_center,
    }
