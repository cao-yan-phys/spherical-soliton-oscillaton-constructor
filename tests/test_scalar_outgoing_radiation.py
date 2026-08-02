import os

import numpy as np
import pytest

from kg_oscillaton import solve_scalar_outgoing_radiation
from kg_oscillaton.outgoing_radiation import (
    _matching_basis,
    _outgoing_amplitude,
    _outgoing_log_derivative,
    _standing_wave_amplitude,
    _validated_radii,
)


def test_standing_wave_amplitude_reconstruction():
    r_outer = 80.0
    omega = 0.86
    mass = 0.60535
    amplitude = 7.3e-7
    basis, derivative = _matching_basis(r_outer, omega, -0.72, mass)
    recovered = _standing_wave_amplitude(
        amplitude * basis,
        amplitude * derivative,
        r_outer,
        omega,
        mass,
    )
    assert recovered == pytest.approx(amplitude, rel=1.0e-12)


def test_outgoing_boundary_and_amplitude():
    r_outer = 80.0
    omega = 0.86
    mass = 0.60535
    amplitude = 5.1e-7 * np.exp(0.37j)
    wave_number = np.sqrt((3.0 * omega) ** 2 - 1.0)
    phase_coefficient = mass * (2.0 * wave_number**2 + 1.0) / wave_number
    phase = wave_number * r_outer + phase_coefficient * np.log(r_outer)
    phi3 = amplitude * np.exp(1j * phase) / r_outer
    recovered = _outgoing_amplitude(
        phi3,
        r_outer,
        omega,
        mass,
        use_mass_phase=True,
    )
    derivative = _outgoing_log_derivative(
        r_outer,
        omega,
        mass,
        use_mass_phase=True,
    )
    assert recovered == pytest.approx(amplitude, rel=1.0e-12)
    assert derivative == pytest.approx(
        1j * (wave_number + phase_coefficient / r_outer) - 1.0 / r_outer
    )


def test_radii_validation():
    assert np.array_equal(_validated_radii(60.0), np.array([60.0]))
    assert np.array_equal(_validated_radii([60.0, 80.0]), np.array([60.0, 80.0]))
    with pytest.raises(ValueError):
        _validated_radii([80.0, 60.0])


def test_public_solver_rejects_even_scalar_truncation():
    with pytest.raises(ValueError, match="scalar_jmax"):
        solve_scalar_outgoing_radiation(scalar_jmax=6)


@pytest.mark.skipif(
    os.environ.get("RUN_SLOW_RADIATION") != "1",
    reason="set RUN_SLOW_RADIATION=1 to run the literature benchmark",
)
@pytest.mark.parametrize(
    ("phi1_center", "delta3", "literature_omega", "literature_mass", "literature_c3", "options"),
    [
        (0.58468, -0.28, 0.85, 0.60399, 1.2793e-6, {}),
        (0.53137, -0.80, 0.86, 0.60535, 5.1183e-7, {}),
        (
            0.43380,
            1.53,
            0.88,
            0.60093,
            6.2311e-8,
            {
                "r_max": (60.0, 80.0, 100.0),
                "n_grid": 700,
                "n_time": 128,
                "tol": 2.0e-6,
                "response_tol": 1.0e-9,
            },
        ),
    ],
)
def test_scalar_outgoing_radiation_literature_benchmark(
    phi1_center,
    delta3,
    literature_omega,
    literature_mass,
    literature_c3,
    options,
):
    result = solve_scalar_outgoing_radiation(
        phi1_center=phi1_center,
        delta3=delta3,
        **options,
    )
    assert result.omega == pytest.approx(literature_omega, abs=1.0e-3)
    assert result.mass == pytest.approx(literature_mass, rel=1.0e-2)
    assert result.c3_outgoing == pytest.approx(literature_c3, rel=5.0e-2)
    assert result.full_max_rms_residual < 1.0e-3
    assert result.response_max_rms_residual < 1.0e-5
