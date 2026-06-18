"""Profile container for real Proca oscillatons."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fourier import build_phase_grid, evaluate_modes


@dataclass
class ProcaOscillatonProfile:
    x: np.ndarray
    matter_modes: np.ndarray
    metric_modes: np.ndarray
    U_modes: np.ndarray
    E_modes: np.ndarray
    A_modes: np.ndarray
    C_modes: np.ndarray
    omega: float
    jmax: int
    u1_center: float
    metadata: dict = field(default_factory=dict)

    def evaluate(self, theta: np.ndarray | float) -> dict[str, np.ndarray]:
        theta_arr = np.atleast_1d(np.asarray(theta, dtype=float))
        U = evaluate_modes(self.U_modes, self.matter_modes, theta_arr)
        E = evaluate_modes(self.E_modes, self.matter_modes, theta_arr)
        E_t = evaluate_modes(
            self.E_modes,
            self.matter_modes,
            theta_arr,
            omega=self.omega,
            time_derivative=1,
        )
        A = evaluate_modes(self.A_modes, self.metric_modes, theta_arr)
        C = evaluate_modes(self.C_modes, self.metric_modes, theta_arr)
        sqrtC = np.sqrt(np.maximum(C, 0.0))
        return {
            "theta": theta_arr,
            "U": U,
            "E": E,
            "E_t": E_t,
            "W": -sqrtC * E_t,
            "A": A,
            "C": C,
            "a": np.sqrt(np.maximum(A, 0.0)),
            "alpha": np.sqrt(np.maximum(A / C, 0.0)),
        }

    def initial_data(self) -> dict[str, np.ndarray]:
        data = self.evaluate(0.0)
        return {key: value[..., 0] for key, value in data.items() if key != "theta"}

    @property
    def A0(self) -> np.ndarray:
        return self.A_modes[0]

    @property
    def C0(self) -> np.ndarray:
        return self.C_modes[0]

    @property
    def mass_profile(self) -> np.ndarray:
        A = self.initial_data()["A"]
        return 0.5 * self.x * (1.0 - 1.0 / A)

    @property
    def mass(self) -> float:
        tail = self.mass_profile[-max(1, int(0.1 * self.x.size)) :]
        return float(np.median(tail))

    def phase_grid(self, n_time: int = 96) -> np.ndarray:
        return build_phase_grid(n_time)

