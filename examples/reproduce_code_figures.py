
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


CASES = [
    (
        "2.0e-3",
        "figures/kg_proca_radial_nr_same_mass_m2e-3.csv",
        "figures/kg_proca_radial_nr_same_mass_m2e-3.png",
    ),
    (
        "2.0e-1",
        "figures/kg_proca_radial_nr_same_mass_m2e-1.csv",
        "figures/kg_proca_radial_nr_same_mass_m2e-1.png",
    ),
    (
        "6.0e-1",
        "figures/kg_proca_radial_nr_same_mass_m6e-1.csv",
        "figures/kg_proca_radial_nr_same_mass_m6e-1.png",
    ),
]


def main() -> None:
    script = Path(__file__).with_name("compare_kg_proca_sp_seeded.py")
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

    local_script = Path(__file__).with_name("compare_kg_local_estimate.py")
    local_command = [
        sys.executable,
        str(local_script),
        "--output-csv",
        "figures/kg_poisson_potentials_vs_local_m1e-1.csv",
        "--plot",
        "figures/kg_poisson_potentials_vs_local_m1e-1.png",
    ]
    print(" ".join(local_command), flush=True)
    subprocess.run(local_command, check=True)


if __name__ == "__main__":
    main()
