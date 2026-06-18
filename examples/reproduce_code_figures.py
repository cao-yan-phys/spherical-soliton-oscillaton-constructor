"""Reproduce the comparison figures shipped with the code."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


CASES = [
    (
        "2.0e-3",
        "figures/scalar_proca_radial_nr_same_mass_m2e-3.csv",
        "figures/scalar_proca_radial_nr_same_mass_m2e-3.png",
    ),
    (
        "2.0e-1",
        "figures/scalar_proca_radial_nr_same_mass_m2e-1.csv",
        "figures/scalar_proca_radial_nr_same_mass_m2e-1.png",
    ),
    (
        "6.0e-1",
        "figures/scalar_proca_radial_nr_same_mass_m6e-1.csv",
        "figures/scalar_proca_radial_nr_same_mass_m6e-1.png",
    ),
]


def main() -> None:
    script = Path(__file__).with_name("compare_scalar_proca_sp_seeded.py")
    for target_mass, csv_path, plot_path in CASES:
        command = [
            sys.executable,
            str(script),
            "--target-mass",
            target_mass,
            "--output-csv",
            csv_path,
            "--plot",
            plot_path,
        ]
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True)

    local_script = Path(__file__).with_name("compare_scalar_local_estimate.py")
    local_command = [
        sys.executable,
        str(local_script),
        "--output-csv",
        "figures/scalar_poisson_potentials_vs_local_m1e-1.csv",
        "--plot",
        "figures/scalar_poisson_potentials_vs_local_m1e-1.png",
    ]
    print(" ".join(local_command), flush=True)
    subprocess.run(local_command, check=True)


if __name__ == "__main__":
    main()
