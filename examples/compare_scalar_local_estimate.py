from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oscillaton import (  # noqa: E402
    kappa_from_phi1_center,
    phi1_center_from_sp_mass,
    solve_sp_ground_state,
    solve_profile_sp_seeded,
)
from oscillaton_builders import (  # noqa: E402
    epsilon_from_omega,
    metric_mode,
    zero_mode_mass,
)


def cumulative_trapezoid(x: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values)
    out[1:] = np.cumsum(0.5 * (values[:-1] + values[1:]) * np.diff(x))
    return out


def tail_integral(x: np.ndarray, values: np.ndarray) -> np.ndarray:
    cumulative = cumulative_trapezoid(x, values)
    return cumulative[-1] - cumulative


def isotropic_outer_radius(areal_radius: float, mass: float) -> float:
    if mass <= 0.0:
        return areal_radius
    discriminant = max(areal_radius * (areal_radius - 2.0 * mass), 0.0)
    return 0.5 * (areal_radius - mass + np.sqrt(discriminant))


def static_isotropic_radius(x: np.ndarray, A0: np.ndarray, mass: float) -> np.ndarray:
    r_outer = float(x[-1])
    R_outer = isotropic_outer_radius(r_outer, mass)
    correction = tail_integral(x, (np.sqrt(A0) - 1.0) / x)
    return R_outer * (x / r_outer) * np.exp(-correction)


def solve_linear_radial_shift(
    x: np.ndarray,
    A0: np.ndarray,
    A2: np.ndarray,
    *,
    rtol: float,
    atol: float,
    max_step_divisor: float,
):
    A0_spline = CubicSpline(x, A0)
    A2_spline = CubicSpline(x, A2)
    dA0 = A0_spline.derivative()
    x_outer = float(x[-1])
    x_inner = float(x[0])
    max_step = (x_outer - x_inner) / max_step_divisor

    def rhs(radius: float, state: np.ndarray) -> list[float]:
        a0 = float(A0_spline(radius))
        coefficient = 1.0 / radius - 0.5 * float(dA0(radius)) / a0
        source = -0.5 * float(A2_spline(radius)) / a0
        return [coefficient * state[0] + source]

    return solve_ivp(
        rhs,
        (x_outer, x_inner),
        [0.0],
        rtol=rtol,
        atol=atol,
        dense_output=True,
        max_step=max_step,
    )


def solve_time_shift(
    x: np.ndarray,
    C0: np.ndarray,
    L_solution,
    omega: float,
    *,
    rtol: float,
    atol: float,
    max_step_divisor: float,
):
    C0_spline = CubicSpline(x, C0)
    x_outer = float(x[-1])
    x_inner = float(x[0])
    max_step = (x_outer - x_inner) / max_step_divisor
    metric_frequency = 2.0 * omega

    def rhs(radius: float, state: np.ndarray) -> list[float]:
        L = float(L_solution.sol(radius)[0])
        return [-metric_frequency * float(C0_spline(radius)) * L]

    return solve_ivp(
        rhs,
        (x_outer, x_inner),
        [0.0],
        rtol=rtol,
        atol=atol,
        dense_output=True,
        max_step=max_step,
    )


def construct_stable_poisson_overlap(
    profile,
    *,
    ode_rtol: float,
    ode_atol: float,
    max_step_divisor: float,
):
    x = np.asarray(profile.x, dtype=float)
    A0 = metric_mode(profile, "A", 0, x)
    C0 = metric_mode(profile, "C", 0, x)
    A2 = metric_mode(profile, "A", 2, x)
    C2 = metric_mode(profile, "C", 2, x)
    B0 = A0 / C0
    B2 = A2 / C0 - A0 * C2 / C0**2

    L_solution = solve_linear_radial_shift(
        x,
        A0,
        A2,
        rtol=ode_rtol,
        atol=ode_atol,
        max_step_divisor=max_step_divisor,
    )
    S_solution = solve_time_shift(
        x,
        C0,
        L_solution,
        profile.omega,
        rtol=ode_rtol,
        atol=ode_atol,
        max_step_divisor=max_step_divisor,
    )

    L = L_solution.sol(x)[0]
    S = S_solution.sol(x)[0]
    dB0 = CubicSpline(x, B0).derivative()(x)
    h00_2 = -(B2 + L * dB0) - 4.0 * profile.omega * B0 * S

    R = static_isotropic_radius(x, A0, zero_mode_mass(profile))
    chi_0 = (x / R) ** 2
    minus_psi_0 = 0.5 * (1.0 - B0)
    minus_phi_0 = 0.5 * (chi_0 - 1.0)
    chi_2 = 2.0 * x * L / R**2
    metric_phi_2 = -0.5 * chi_2

    scalar_index = int(np.where(profile.scalar_modes == 1)[0][0])
    phi1 = profile.phi[scalar_index]
    dphi1 = profile.dphi[scalar_index]
    phi1_pg = phi1 + 0.5 * L * dphi1 - 0.5 * profile.omega * S * phi1

    return {
        "R": R,
        "L": L,
        "S": S,
        "minus_psi_0": minus_psi_0,
        "minus_phi_0": minus_phi_0,
        "h00_2": h00_2,
        "chi_2": chi_2,
        "metric_phi_2": metric_phi_2,
        "phi1_pg": phi1_pg,
        "L_nfev": L_solution.nfev,
        "S_nfev": S_solution.nfev,
        "L_success": L_solution.success,
        "S_success": S_solution.success,
    }


def solve_scalar_profile(args):
    phi1_center = phi1_center_from_sp_mass(args.target_mass)
    kappa = kappa_from_phi1_center(phi1_center)
    return solve_profile_sp_seeded(
        phi1_center,
        jmax=args.jmax,
        x_max=max(args.x_min, args.rho_max / kappa),
        n_grid=args.n_grid,
        n_time=args.n_time,
        tol=args.tol,
    )


def scalar_sp_newtonian_potential(R: np.ndarray, kappa: float) -> np.ndarray:
    y = kappa * R
    sp = solve_sp_ground_state(
        y_max=max(40.0, float(y[-1])),
        n_grid=max(500, min(1800, int(25 * max(40.0, float(y[-1]))))),
        tol=1.0e-6,
    )
    V = np.interp(y, sp.y, sp.V)
    return 0.5 * kappa**2 * (V - sp.V_infinity)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-mass", type=float, default=0.1)
    parser.add_argument("--jmax", type=int, default=2)
    parser.add_argument("--n-grid", type=int, default=1200)
    parser.add_argument("--n-time", type=int, default=96)
    parser.add_argument("--tol", type=float, default=1.0e-8)
    parser.add_argument("--x-min", type=float, default=80.0)
    parser.add_argument("--rho-max", type=float, default=45.0)
    parser.add_argument("--plot-rho-min", type=float, default=0.05)
    parser.add_argument("--plot-rho-max", type=float, default=8.0)
    parser.add_argument("--ode-rtol", type=float, default=3.0e-10)
    parser.add_argument("--ode-atol", type=float, default=1.0e-13)
    parser.add_argument("--max-step-divisor", type=float, default=800.0)
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("figures/scalar_poisson_potentials_vs_local_m1e-1.png"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("figures/scalar_poisson_potentials_vs_local_m1e-1.csv"),
    )
    args = parser.parse_args()

    profile = solve_scalar_profile(args)
    phi1_center = phi1_center_from_sp_mass(args.target_mass)
    kappa = kappa_from_phi1_center(phi1_center)
    result = construct_stable_poisson_overlap(
        profile,
        ode_rtol=args.ode_rtol,
        ode_atol=args.ode_atol,
        max_step_divisor=args.max_step_divisor,
    )
    epsilon = epsilon_from_omega(profile.omega)
    rho = epsilon * result["R"]
    minus_psi_2 = 0.5 * result["h00_2"]
    local_estimate = result["phi1_pg"] ** 2 / 16.0
    minus_sp_newtonian = -scalar_sp_newtonian_potential(result["R"], kappa)

    data = np.column_stack(
        (
            result["R"],
            rho,
            minus_psi_2,
            result["metric_phi_2"],
            local_estimate,
            result["phi1_pg"],
            result["h00_2"],
            result["chi_2"],
            result["minus_psi_0"],
            result["minus_phi_0"],
            minus_sp_newtonian,
        )
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.output_csv,
        data,
        delimiter=",",
        header=(
            "R,rho,minus_Psi2_num,Phi2_num,phi1_pg_sq_over_16,"
            "phi1_pg,h00_2_num,chi_2_num,"
            "minus_Psi0_PG,minus_Phi0_PG,minus_PhiN_SP"
        ),
    )

    mask = (
        (rho >= args.plot_rho_min)
        & (rho <= args.plot_rho_max)
        & (local_estimate > local_estimate.max() * 1.0e-10)
        & (minus_psi_2 > 0.0)
        & (result["metric_phi_2"] > 0.0)
    )
    static_mask = (
        (rho >= args.plot_rho_min)
        & (rho <= args.plot_rho_max)
        & (result["minus_psi_0"] > 0.0)
        & (result["minus_phi_0"] > 0.0)
        & (minus_sp_newtonian > 0.0)
    )

    plt.rcParams.update({"font.size": 12, "mathtext.fontset": "dejavusans"})
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 4.5), sharex=True)

    axes[0].plot(
        rho[mask],
        minus_psi_2[mask],
        lw=2.3,
        color="#245f9e",
        label=r"$-\Psi_2^{\mathrm{overlap}}$",
    )
    axes[0].plot(
        rho[mask],
        local_estimate[mask],
        lw=2.0,
        color="k",
        ls="--",
        label=r"$\phi_{1,\mathrm{PG}}^2/16$",
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$\tilde{\rho}=\epsilon R$")
    axes[0].set_ylabel(r"$-\Psi_2$")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=10)

    axes[1].plot(
        rho[mask],
        result["metric_phi_2"][mask],
        lw=2.3,
        color="#b2442c",
        label=r"$\Phi_2^{\mathrm{overlap}}$",
    )
    axes[1].plot(
        rho[mask],
        local_estimate[mask],
        lw=2.0,
        color="k",
        ls="--",
        label=r"$\phi_{1,\mathrm{PG}}^2/16$",
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$\tilde{\rho}=\epsilon R$")
    axes[1].set_ylabel(r"$\Phi_2$")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=10)

    axes[2].plot(
        rho[static_mask],
        result["minus_psi_0"][static_mask],
        lw=2.3,
        color="#245f9e",
        label=r"$-\Psi_0^{\mathrm{PG}}$",
    )
    axes[2].plot(
        rho[static_mask],
        result["minus_phi_0"][static_mask],
        lw=2.3,
        color="#b2442c",
        label=r"$-\Phi_0^{\mathrm{PG}}$",
    )
    axes[2].plot(
        rho[static_mask],
        minus_sp_newtonian[static_mask],
        lw=2.0,
        color="k",
        ls="--",
        label=r"$-\Phi_N^{\mathrm{SP}}$",
    )
    axes[2].set_yscale("log")
    axes[2].set_xlabel(r"$\tilde{\rho}=\epsilon R$")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(fontsize=10)

    fig.suptitle(
        rf"Stable scalar Poisson-gauge overlap vs local estimate, "
        rf"$\mu M_{{\mathrm{{ADM}}}}={zero_mode_mass(profile):.9g}$"
    )
    fig.tight_layout()
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=190)
    plt.close(fig)

    summary_mask = (
        (rho >= 0.05)
        & (rho <= 3.0)
        & (local_estimate > local_estimate.max() * 1.0e-8)
    )
    static_summary_mask = (
        (rho >= 0.05)
        & (rho <= 3.0)
        & (minus_sp_newtonian > minus_sp_newtonian.max() * 1.0e-8)
    )
    print(f"zero_mode_mass:                 {zero_mode_mass(profile):.15e}")
    print(f"omega:                          {profile.omega:.15e}")
    print(f"epsilon:                        {epsilon:.15e}")
    print(
        "median_minus_psi_ratio_0p05_3: "
        f"{np.median(minus_psi_2[summary_mask] / local_estimate[summary_mask]):.15e}"
    )
    print(
        "median_phi_ratio_0p05_3:       "
        f"{np.median(result['metric_phi_2'][summary_mask] / local_estimate[summary_mask]):.15e}"
    )
    print(
        "median_static_psi_ratio_0p05_3:"
        f" {np.median(result['minus_psi_0'][static_summary_mask] / minus_sp_newtonian[static_summary_mask]):.15e}"
    )
    print(
        "median_static_phi_ratio_0p05_3:"
        f" {np.median(result['minus_phi_0'][static_summary_mask] / minus_sp_newtonian[static_summary_mask]):.15e}"
    )
    print(f"L_ode_success:                  {result['L_success']}")
    print(f"S_ode_success:                  {result['S_success']}")
    print(f"L_ode_nfev:                     {result['L_nfev']}")
    print(f"S_ode_nfev:                     {result['S_nfev']}")
    print(f"plot:                           {args.plot}")
    print(f"csv:                            {args.output_csv}")


if __name__ == "__main__":
    main()
