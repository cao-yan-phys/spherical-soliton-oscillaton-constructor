"""Compare a scalar oscillaton with the local Poisson-gauge estimate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oscillaton import (  # noqa: E402
    kappa_from_phi1_center,
    phi1_center_from_sp_mass,
    solve_profile_sp_seeded,
)
from oscillaton_builders import epsilon_from_omega, zero_mode_mass  # noqa: E402


def spectral_tau_derivative(values: np.ndarray, omega: float) -> np.ndarray:
    """Return d/dtau of periodic samples with theta = omega tau."""

    n_theta = values.size
    modes = np.fft.fftfreq(n_theta, d=1.0 / n_theta)
    return omega * np.fft.ifft(1j * modes * np.fft.fft(values)).real


def interpolate_profile(xp: np.ndarray, fp: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.interp(x, xp, fp, left=float(fp[0]), right=float(fp[-1]))


def evaluate_metric(profile, radius: np.ndarray, theta: np.ndarray):
    A = np.zeros_like(radius)
    C = np.zeros_like(radius)
    for mode, A_mode, C_mode in zip(profile.metric_modes, profile.A, profile.C):
        A += interpolate_profile(profile.x, A_mode, radius) * np.cos(mode * theta)
        C += interpolate_profile(profile.x, C_mode, radius) * np.cos(mode * theta)
    return A, C


def evaluate_scalar(profile, radius: np.ndarray, theta: np.ndarray):
    field = np.zeros_like(radius)
    for mode, phi_mode in zip(profile.scalar_modes, profile.phi):
        field += interpolate_profile(profile.x, phi_mode, radius) * np.cos(mode * theta)
    return field


def isotropic_outer_radius(areal_radius: float, mass: float) -> float:
    """Schwarzschild isotropic radius matching a large areal radius."""

    if mass <= 0.0:
        return areal_radius
    discriminant = max(areal_radius * (areal_radius - 2.0 * mass), 0.0)
    return 0.5 * (areal_radius - mass + np.sqrt(discriminant))


def gauge_rhs(profile, radius: float, state: np.ndarray, theta: np.ndarray):
    """Return d_R [L(theta), T(theta)] from the exact spherical conditions."""

    n_theta = theta.size
    L = state[:n_theta]
    T = state[n_theta:]
    L_tau = spectral_tau_derivative(L, profile.omega)
    T_tau = spectral_tau_derivative(T, profile.omega)

    r_old = radius + L
    theta_old = theta + profile.omega * T
    A, C = evaluate_metric(profile, r_old, theta_old)
    lapse_squared = A / C
    t_tau = 1.0 + T_tau
    r_tau = L_tau

    denominator = 1.0 - A * r_tau**2 / (lapse_squared * t_tau**2)
    denominator = np.maximum(denominator, 1.0e-14)
    r_R = (r_old / radius) / np.sqrt(A * denominator)
    t_R = A * r_tau * r_R / (lapse_squared * t_tau)
    return np.r_[r_R - 1.0, t_R]


def rk4_step(profile, radius: float, state: np.ndarray, step: float, theta: np.ndarray):
    k1 = gauge_rhs(profile, radius, state, theta)
    k2 = gauge_rhs(profile, radius + 0.5 * step, state + 0.5 * step * k1, theta)
    k3 = gauge_rhs(profile, radius + 0.5 * step, state + 0.5 * step * k2, theta)
    k4 = gauge_rhs(profile, radius + step, state + step * k3, theta)
    return state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def construct_spherical_poisson_gauge(profile, *, n_theta: int, n_radial: int):
    """Construct the spherical isotropic, or Poisson-like, gauge."""

    mass = zero_mode_mass(profile)
    r_outer = float(profile.x[-1])
    R_outer = isotropic_outer_radius(r_outer, mass)
    R_inner = float(profile.x[0])
    radius_desc = np.linspace(R_outer, R_inner, n_radial)
    theta = 2.0 * np.pi * np.arange(n_theta) / n_theta

    state = np.r_[np.full(n_theta, r_outer - R_outer), np.zeros(n_theta)]
    h00_2 = np.zeros(n_radial)
    chi_2 = np.zeros(n_radial)
    metric_phi_2 = np.zeros(n_radial)
    phi1_pg = np.zeros(n_radial)
    spatial_residual_max = 0.0
    shift_residual_max = 0.0

    for idx, radius in enumerate(radius_desc):
        n = n_theta
        L = state[:n]
        T = state[n:]
        derivatives = gauge_rhs(profile, radius, state, theta)
        L_R = derivatives[:n]
        T_R = derivatives[n:]
        L_tau = spectral_tau_derivative(L, profile.omega)
        T_tau = spectral_tau_derivative(T, profile.omega)

        r_old = radius + L
        theta_old = theta + profile.omega * T
        A, C = evaluate_metric(profile, r_old, theta_old)
        lapse_squared = A / C
        g_tt = -lapse_squared
        g_rr = A

        g_tau_R = g_tt * (1.0 + T_tau) * T_R + g_rr * L_tau * (1.0 + L_R)
        g_RR = g_tt * T_R**2 + g_rr * (1.0 + L_R) ** 2
        spatial_chi = (r_old / radius) ** 2
        spatial_residual_max = max(
            spatial_residual_max, float(np.max(np.abs(g_RR - spatial_chi)))
        )
        shift_residual_max = max(shift_residual_max, float(np.max(np.abs(g_tau_R))))

        g_tau_tau = g_tt * (1.0 + T_tau) ** 2 + g_rr * L_tau**2
        h00_2[idx] = 2.0 * np.mean((g_tau_tau + 1.0) * np.cos(2.0 * theta))
        chi_2[idx] = 2.0 * np.mean((spatial_chi - 1.0) * np.cos(2.0 * theta))
        metric_phi_2[idx] = -0.5 * chi_2[idx]
        field = evaluate_scalar(profile, r_old, theta_old)
        phi1_pg[idx] = 2.0 * np.mean(field * np.cos(theta))

        if idx < n_radial - 1:
            step = radius_desc[idx + 1] - radius
            state = rk4_step(profile, radius, state, step, theta)

    return {
        "R": radius_desc[::-1],
        "h00_2": h00_2[::-1],
        "chi_2": chi_2[::-1],
        "metric_phi_2": metric_phi_2[::-1],
        "phi1_pg": phi1_pg[::-1],
        "spatial_residual_max": spatial_residual_max,
        "shift_residual_max": shift_residual_max,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-mass", type=float, default=0.1)
    parser.add_argument("--jmax", type=int, default=2)
    parser.add_argument("--n-grid", type=int, default=1200)
    parser.add_argument("--n-time", type=int, default=96)
    parser.add_argument("--tol", type=float, default=1.0e-8)
    parser.add_argument("--x-min", type=float, default=80.0)
    parser.add_argument("--rho-max", type=float, default=45.0)
    parser.add_argument("--theta-samples", type=int, default=192)
    parser.add_argument("--radial-samples", type=int, default=1200)
    parser.add_argument("--plot-rho-min", type=float, default=0.05)
    parser.add_argument("--plot-rho-max", type=float, default=8.0)
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
    result = construct_spherical_poisson_gauge(
        profile, n_theta=args.theta_samples, n_radial=args.radial_samples
    )
    epsilon = epsilon_from_omega(profile.omega)
    rho = epsilon * result["R"]
    minus_psi_2 = 0.5 * result["h00_2"]
    local_estimate = result["phi1_pg"] ** 2 / 16.0

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
        )
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.output_csv,
        data,
        delimiter=",",
        header=(
            "R,rho,minus_Psi2_num,Phi2_num,phi1_pg_sq_over_16,"
            "phi1_pg,h00_2_num,chi_2_num"
        ),
    )

    mask = (
        (rho >= args.plot_rho_min)
        & (rho <= args.plot_rho_max)
        & (local_estimate > local_estimate.max() * 1.0e-10)
        & (minus_psi_2 > 0.0)
        & (result["metric_phi_2"] > 0.0)
    )

    plt.rcParams.update({"font.size": 12, "mathtext.fontset": "dejavusans"})
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), sharex=True)

    axes[0].plot(
        rho[mask],
        minus_psi_2[mask],
        lw=2.3,
        color="#245f9e",
        label=r"$-\Psi_2^{\mathrm{num}}$",
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
        label=r"$\Phi_2^{\mathrm{num}}$",
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

    fig.suptitle(
        rf"Poisson-gauge scalar metric potentials vs local estimate, "
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
    print(f"spatial_residual_max:           {result['spatial_residual_max']:.15e}")
    print(f"shift_residual_max:             {result['shift_residual_max']:.15e}")
    print(f"plot:                           {args.plot}")
    print(f"csv:                            {args.output_csv}")


if __name__ == "__main__":
    main()
