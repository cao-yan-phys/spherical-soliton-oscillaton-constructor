"""Minimal real Proca oscillaton construction API."""

from .bvp_solver import solve_family, solve_profile, solve_profile_scaled_seeded
from .diagnostics import boundary_residuals, origin_regular_error, residual_norms
from .nr_radial import RadialProcaNRProfile, solve_radial_proca_nr_ground_state
from .profiles import ProcaOscillatonProfile

__all__ = [
    "ProcaOscillatonProfile",
    "RadialProcaNRProfile",
    "boundary_residuals",
    "origin_regular_error",
    "residual_norms",
    "solve_family",
    "solve_profile",
    "solve_profile_scaled_seeded",
    "solve_radial_proca_nr_ground_state",
]
