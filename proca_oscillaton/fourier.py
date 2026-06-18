"""Fourier-collocation utilities for real Proca oscillatons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModeSet:
    matter: np.ndarray
    metric: np.ndarray

    @property
    def n_matter(self) -> int:
        return int(self.matter.size)

    @property
    def n_metric(self) -> int:
        return int(self.metric.size)


def mode_set(jmax: int) -> ModeSet:
    if jmax < 2 or jmax % 2:
        raise ValueError("jmax must be an even integer >= 2")
    return ModeSet(
        matter=np.arange(1, jmax + 1, 2, dtype=int),
        metric=np.arange(0, jmax + 1, 2, dtype=int),
    )


def reduced_state_size(jmax: int) -> int:
    modes = mode_set(jmax)
    return 2 * modes.n_matter + 1 + modes.n_metric


def build_phase_grid(n_time: int) -> np.ndarray:
    if n_time < 4:
        raise ValueError("n_time must be at least 4")
    return 2.0 * np.pi * np.arange(n_time, dtype=float) / float(n_time)


def cos_basis(modes: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return np.cos(np.outer(np.asarray(modes, dtype=float), theta))


def sin_basis(modes: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return np.sin(np.outer(np.asarray(modes, dtype=float), theta))


def evaluate_modes(
    coeffs: np.ndarray,
    modes: np.ndarray,
    theta: np.ndarray,
    *,
    omega: float = 1.0,
    time_derivative: int = 0,
) -> np.ndarray:
    modes = np.asarray(modes, dtype=float)
    phase = np.outer(modes, theta)
    if time_derivative == 0:
        basis = np.cos(phase)
    elif time_derivative == 1:
        basis = -(modes * omega)[:, None] * np.sin(phase)
    elif time_derivative == 2:
        basis = -((modes * omega) ** 2)[:, None] * np.cos(phase)
    else:
        raise ValueError("time_derivative must be 0, 1, or 2")
    return np.asarray(coeffs).T @ basis


def project_cos(values: np.ndarray, theta: np.ndarray, modes: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    coeffs = []
    for j in np.asarray(modes, dtype=int):
        if j == 0:
            coeffs.append(np.mean(values, axis=-1))
        else:
            coeffs.append(2.0 * np.mean(values * np.cos(j * theta), axis=-1))
    return np.stack(coeffs, axis=0)


def project_sin(values: np.ndarray, theta: np.ndarray, modes: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.stack(
        [2.0 * np.mean(values * np.sin(int(j) * theta), axis=-1) for j in modes],
        axis=0,
    )


def spectral_dt(values: np.ndarray, omega: float) -> np.ndarray:
    """Return `d_t values` for samples on the uniform phase grid."""

    n_time = values.shape[-1]
    wave_numbers = np.fft.fftfreq(n_time, d=1.0 / n_time)
    derivative = np.fft.ifft(
        1j * wave_numbers * np.fft.fft(values, axis=-1), axis=-1
    ).real
    return omega * derivative

