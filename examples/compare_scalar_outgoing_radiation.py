import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kg_oscillaton import solve_scalar_outgoing_radiation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/scalar_outgoing_radiation_benchmark_omega086.png"),
    )
    args = parser.parse_args()

    result = solve_scalar_outgoing_radiation()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    outer = result.r >= result.r_max - 18.0
    scale = 1.0e7
    figure, axis = plt.subplots(figsize=(7.6, 4.7), constrained_layout=True)
    axis.plot(
        result.r[outer],
        scale * result.r[outer] * result.phi3_outgoing[outer].real,
        color="#1769aa",
        linewidth=1.8,
        label="outgoing",
        zorder=3,
    )
    axis.plot(
        result.r[outer],
        scale * result.r[outer] * result.phi3_standing[outer],
        color="black",
        linestyle="--",
        linewidth=1.6,
        label="standing wave",
        zorder=4,
    )
    literature_c3 = 5.1183e-7
    axis.axhline(
        scale * literature_c3,
        color="#b33a3a",
        linestyle=":",
        linewidth=1.8,
        label=r"arXiv:1107.2791",
        zorder=2,
    )
    axis.axhline(
        -scale * literature_c3,
        color="#b33a3a",
        linestyle=":",
        linewidth=1.8,
        zorder=2,
    )
    axis.set_xlabel(r"$r$")
    axis.set_ylabel(r"$10^7 r\Phi_3$")
    axis.set_ylim(-5.65, 5.65)
    axis.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.8,
        facecolor="0.94",
        edgecolor="0.65",
    )
    axis.grid(color="0.88", linewidth=0.8)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
    axis.set_title(
        rf"Scalar oscillaton outgoing radiation: $\omega={result.omega:.7f}$, "
        rf"$\mu M_{{\mathrm{{ADM}}}}={result.mass:.7f}$",
        fontsize=13,
    )
    figure.savefig(args.output, dpi=220)
    plt.close(figure)

    table = np.column_stack(
        [
            result.continuation_radii,
            result.continuation_omega,
            result.continuation_mass,
            result.continuation_c3,
        ]
    )
    np.savetxt(
        args.output.with_suffix(".csv"),
        table,
        delimiter=",",
        header="r_max,omega,mass,c3_standing",
        comments="",
    )
    print(f"omega={result.omega:.12e}")
    print(f"epsilon={result.epsilon:.12e}")
    print(f"mass={result.mass:.12e}")
    print(f"c3_standing={result.c3_standing:.12e}")
    print(f"c3_outgoing={result.c3_outgoing:.12e}")
    print(f"mass_loss_rate={result.mass_loss_rate:.12e}")
    print(f"full_max_rms_residual={result.full_max_rms_residual:.12e}")
    print(f"response_max_rms_residual={result.response_max_rms_residual:.12e}")
    print(f"figure={args.output}")


if __name__ == "__main__":
    main()
