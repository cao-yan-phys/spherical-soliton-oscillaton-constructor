"""Compare weak scalar and Proca oscillatons at the same ADM mass."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oscillaton import (
    kappa_from_phi1_center,
    phi1_center_from_sp_mass,
    solve_profile_sp_seeded as solve_scalar_sp_seeded,
    solve_sp_ground_state,
)
from proca_oscillaton import (
    solve_profile as solve_proca_profile,
    solve_profile_scaled_seeded as solve_proca_scaled_seeded,
    solve_radial_proca_nr_ground_state,
)


def solve_bound_proca_reference():
    """Return a moderately weak bound Proca profile for NR rescaling."""

    values = [0.02, 0.012, 0.006, 0.003, 0.0015, 0.001, 0.0007, 0.0005, 0.00035, 0.000334]
    previous = None
    for value in values:
        previous = solve_proca_profile(
            value,
            jmax=2,
            x_max=360.0,
            n_grid=900,
            n_time=64,
            tol=5.0e-4,
            continuation=False,
            previous=previous,
        )
    return previous


def solve_proca_for_mass(target_mass: float, *, jmax: int, n_grid: int, n_time: int, tol: float):
    ref = solve_bound_proca_reference()
    u1_center = ref.u1_center * (target_mass / ref.mass) ** 3
    profile = solve_proca_scaled_seeded(
        u1_center,
        ref,
        jmax=2,
        n_grid=n_grid,
        n_time=64,
        tol=tol,
    )
    for _ in range(3):
        u1_center *= (target_mass / profile.mass) ** 3
        profile = solve_proca_scaled_seeded(
            u1_center,
            profile,
            jmax=2,
            n_grid=n_grid,
            n_time=64,
            tol=tol,
        )
    if jmax > 2:
        profile = solve_proca_scaled_seeded(
            u1_center,
            profile,
            jmax=jmax,
            x_max=profile.x[-1],
            n_grid=n_grid,
            n_time=n_time,
            tol=tol,
        )
    for _ in range(4):
        if abs(profile.mass - target_mass) / target_mass < 5.0e-5:
            break
        u1_center *= (target_mass / profile.mass) ** 3
        profile = solve_proca_scaled_seeded(
            u1_center,
            profile,
            jmax=jmax,
            n_grid=n_grid,
            n_time=n_time,
            tol=tol,
        )
    return profile


def solve_scalar_for_mass(
    target_mass: float,
    *,
    jmax: int,
    n_grid: int,
    n_time: int,
    tol: float,
    mass_tol: float,
):
    """Tune phi_1(0) so the full scalar BVP, not just SP, hits target_mass."""

    def solve(phi1_center: float):
        kappa = kappa_from_phi1_center(phi1_center)
        return solve_scalar_sp_seeded(
            phi1_center,
            jmax=jmax,
            x_max=max(80.0, 45.0 / kappa),
            n_grid=n_grid,
            n_time=n_time,
            tol=tol,
        )

    phi = phi1_center_from_sp_mass(target_mass)
    profile = solve(phi)
    if abs(profile.mass - target_mass) / target_mass < mass_tol:
        return profile

    if profile.mass < target_mass:
        lo_phi, lo_profile = phi, profile
        hi_phi = phi * 1.2
        hi_profile = solve(hi_phi)
        for _ in range(8):
            if hi_profile.mass >= target_mass:
                break
            lo_phi, lo_profile = hi_phi, hi_profile
            hi_phi *= 1.2
            hi_profile = solve(hi_phi)
    else:
        hi_phi, hi_profile = phi, profile
        lo_phi = phi / 1.2
        lo_profile = solve(lo_phi)
        for _ in range(8):
            if lo_profile.mass <= target_mass:
                break
            hi_phi, hi_profile = lo_phi, lo_profile
            lo_phi /= 1.2
            lo_profile = solve(lo_phi)

    best = lo_profile if abs(lo_profile.mass - target_mass) < abs(hi_profile.mass - target_mass) else hi_profile
    for _ in range(10):
        mid_phi = 0.5 * (lo_phi + hi_phi)
        mid_profile = solve(mid_phi)
        if abs(mid_profile.mass - target_mass) < abs(best.mass - target_mass):
            best = mid_profile
        if abs(mid_profile.mass - target_mass) / target_mass < mass_tol:
            return mid_profile
        if mid_profile.mass < target_mass:
            lo_phi, lo_profile = mid_phi, mid_profile
        else:
            hi_phi, hi_profile = mid_phi, mid_profile
    return best


def zero_mode_arrays(profile, x: np.ndarray) -> dict[str, np.ndarray]:
    A0 = np.interp(x, profile.x, profile.A0)
    C0 = np.interp(x, profile.x, profile.C0)
    return {
        "A0": A0,
        "C0": C0,
        "M0": 0.5 * x * (1.0 - 1.0 / A0),
    }


def zero_mode_mass(profile) -> float:
    A0 = np.asarray(profile.A0)
    mass = 0.5 * profile.x * (1.0 - 1.0 / A0)
    tail = mass[-max(1, int(0.1 * mass.size)) :]
    return float(np.median(tail))


def relative_l2_residual(
    rho: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    offset: float,
    rho_min: float = 0.05,
    rho_max: float = 35.0,
) -> float:
    mask = (rho >= rho_min) & (rho <= min(rho_max, float(rho[-1])))
    scale = np.sqrt(np.mean((reference[mask] - offset) ** 2))
    return float(np.sqrt(np.mean((reference[mask] - candidate[mask]) ** 2)) / max(scale, 1.0e-300))


def metric_mode(profile, family: str, mode: int, x: np.ndarray) -> np.ndarray:
    if family == "A":
        coeffs = getattr(profile, "A", None)
        if coeffs is None:
            coeffs = profile.A_modes
    elif family == "C":
        coeffs = getattr(profile, "C", None)
        if coeffs is None:
            coeffs = profile.C_modes
    else:
        raise ValueError("family must be 'A' or 'C'")

    matches = np.where(profile.metric_modes == mode)[0]
    if matches.size == 0:
        return np.zeros_like(x, dtype=float)
    return np.interp(x, profile.x, coeffs[int(matches[0])])


def scalar_sp_arrays_for_mass(x: np.ndarray, target_mass: float):
    sp_base = solve_sp_ground_state(y_max=45.0, n_grid=1200, tol=1.0e-7)
    kappa_sp = target_mass / sp_base.dimensionless_cloud_mass
    z = kappa_sp * x
    sp = solve_sp_ground_state(y_max=max(45.0, float(z[-1])), n_grid=1200, tol=1.0e-7)
    V = kappa_sp**2 * np.interp(z, sp.y, sp.V)
    dV_dx = kappa_sp**3 * np.interp(z, sp.y, sp.dV)
    V_inf = kappa_sp**2 * sp.V_infinity
    arrays = {
        "A0": 1.0 + x * dV_dx,
        "C0": 1.0 + x * dV_dx + V_inf - V,
        "M0": 0.5 * x**2 * dV_dx,
    }
    return arrays, sp.scaled(kappa_sp)


def radial_proca_nr_arrays(x: np.ndarray, target_mass: float):
    profile = solve_radial_proca_nr_ground_state(y_max=50.0, n_grid=900, tol=1.0e-7)
    scale = target_mass / profile.dimensionless_cloud_mass
    scaled = profile.scaled(scale)
    return scaled.as_metric_arrays(x), scaled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-mass", type=float, default=2.0e-3)
    parser.add_argument("--scalar-jmax", type=int, default=6)
    parser.add_argument("--proca-jmax", type=int, default=6)
    parser.add_argument("--n-grid", type=int, default=1000)
    parser.add_argument("--n-time", type=int, default=96)
    parser.add_argument("--tol", type=float, default=1.0e-6)
    parser.add_argument("--mass-tol", type=float, default=5.0e-5)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("figures/scalar_proca_radial_nr_same_mass_m2e-3.csv"),
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("figures/scalar_proca_radial_nr_same_mass_m2e-3.png"),
    )
    args = parser.parse_args()

    scalar = solve_scalar_for_mass(
        args.target_mass,
        jmax=args.scalar_jmax,
        n_grid=args.n_grid,
        n_time=args.n_time,
        tol=args.tol,
        mass_tol=args.mass_tol,
    )
    kappa = kappa_from_phi1_center(scalar.phi1_center)
    proca = solve_proca_for_mass(
        args.target_mass,
        jmax=args.proca_jmax,
        n_grid=args.n_grid,
        n_time=args.n_time,
        tol=args.tol,
    )

    rho_max = min(45.0, kappa * scalar.x[-1], kappa * proca.x[-1])
    rho = np.linspace(0.0, rho_max, 1400)
    x = np.maximum(rho / kappa, 1.0e-4)
    scalar_arrays = zero_mode_arrays(scalar, x)
    proca_arrays = zero_mode_arrays(proca, x)
    scalar_sp, scalar_sp_profile = scalar_sp_arrays_for_mass(x, zero_mode_mass(scalar))
    proca_nr, proca_nr_profile = radial_proca_nr_arrays(x, zero_mode_mass(proca))
    A2_scalar = metric_mode(scalar, "A", 2, x)
    C2_scalar = metric_mode(scalar, "C", 2, x)
    A2_proca = metric_mode(proca, "A", 2, x)
    C2_proca = metric_mode(proca, "C", 2, x)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.output_csv,
        np.column_stack(
            (
                x,
                rho,
                scalar_arrays["A0"],
                scalar_arrays["C0"],
                scalar_arrays["M0"],
                proca_arrays["A0"],
                proca_arrays["C0"],
                proca_arrays["M0"],
                proca_nr["A0"],
                proca_nr["C0"],
                proca_nr["M0"],
                scalar_sp["A0"],
                scalar_sp["C0"],
                scalar_sp["M0"],
                A2_scalar,
                C2_scalar,
                A2_proca,
                C2_proca,
            )
        ),
        delimiter=",",
        header=(
            "x,rho,A0_scalar,C0_scalar,M0_scalar,A0_proca,C0_proca,M0_proca,"
            "A0_proca_nr_radial,C0_proca_nr_radial,M0_proca_nr_radial,"
            "A0_scalar_sp,C0_scalar_sp,M0_scalar_sp,A2_scalar,C2_scalar,A2_proca,C2_proca"
        ),
    )

    colors = {
        "scalar": "#245f9e",
        "proca": "#b2442c",
        "nr": "#000000",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.0))
    fields = [
        ("A0", r"$A_0(\rho)-1$"),
        ("C0", r"$C_0(\rho)-1$"),
        ("M0", r"$M(<\rho)$"),
    ]
    for ax, (field, ylabel) in zip(axes.flat[:3], fields):
        offset = 1.0 if field != "M0" else 0.0
        ax.plot(
            rho,
            scalar_arrays[field] - offset,
            color=colors["scalar"],
            lw=2.0,
            label=r"$\mathrm{scalar}$",
        )
        ax.plot(
            rho,
            proca_arrays[field] - offset,
            color=colors["proca"],
            lw=2.0,
            label=r"$\mathrm{vector}$",
        )
        ax.plot(
            rho,
            proca_nr[field] - offset,
            color=colors["nr"],
            lw=2.0,
            ls="-.",
            label=r"$\mathrm{Proca\ radial\ SP}$",
        )
        ax.plot(
            rho,
            scalar_sp[field] - offset,
            color=colors["nr"],
            lw=2.0,
            ls="--",
            label=r"$\mathrm{scalar\ SP}$",
        )
        ax.set_xlabel(r"$\rho=\kappa x$")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    ax = axes.flat[3]
    ax.plot(rho, A2_scalar, color=colors["scalar"], lw=2.0, label=r"$A_2^{\rm scalar}$")
    ax.plot(
        rho,
        C2_scalar,
        color=colors["scalar"],
        lw=2.0,
        ls="--",
        label=r"$C_2^{\rm scalar}$",
    )
    ax.plot(rho, A2_proca, color=colors["proca"], lw=2.0, label=r"$A_2^{\rm vector}$")
    ax.plot(
        rho,
        C2_proca,
        color=colors["proca"],
        lw=2.0,
        ls="--",
        label=r"$C_2^{\rm vector}$",
    )
    ax.set_xlabel(r"$\rho=\kappa x$")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    scalar_eps = np.sqrt(max(0.0, 1.0 - scalar.omega**2))
    proca_eps = np.sqrt(max(0.0, 1.0 - proca.omega**2))
    residuals = {}
    for field in ("A0", "C0", "M0"):
        offset = 1.0 if field != "M0" else 0.0
        residuals[(field, "proca_nr_vs_full")] = relative_l2_residual(
            rho, proca_arrays[field], proca_nr[field], offset=offset
        )
        residuals[(field, "scalar_sp_vs_full")] = relative_l2_residual(
            rho, scalar_arrays[field], scalar_sp[field], offset=offset
        )
    fig.suptitle(
        rf"$\mu M_{{\rm ADM}}^{{\rm scalar}}={scalar.mass:.9g},\ "
        rf"\mu M_{{\rm ADM}}^{{\rm Proca}}={proca.mass:.9g},\ "
        rf"\epsilon_{{\rm scalar}}={scalar_eps:.4g},\ "
        rf"\epsilon_{{\rm Proca}}={proca_eps:.4g}\ "
        rf"(\epsilon=\sqrt{{1-\omega^2}})$"
    )
    fig.tight_layout()
    fig.savefig(args.plot, dpi=180)
    plt.close(fig)

    print(f"scalar_phi1_center:      {scalar.phi1_center:.15e}")
    print(f"scalar_mu_madm:          {scalar.mass:.15e}")
    print(f"scalar_zero_mode_mass:   {zero_mode_mass(scalar):.15e}")
    print(f"scalar_omega:            {scalar.omega:.15f}")
    print(f"scalar_epsilon:          {scalar_eps:.15e}")
    print(f"scalar_tail_rel_std:     {np.std(scalar.mass_profile[-100:]) / abs(scalar.mass):.3e}")
    print(f"proca_u1_center:         {proca.u1_center:.15e}")
    print(f"proca_mu_madm:           {proca.mass:.15e}")
    print(f"proca_zero_mode_mass:    {zero_mode_mass(proca):.15e}")
    print(f"proca_omega:             {proca.omega:.15f}")
    print(f"proca_epsilon:           {proca_eps:.15e}")
    print(f"proca_tail_rel_std:      {np.std(proca.mass_profile[-100:]) / abs(proca.mass):.3e}")
    print(f"relative_mass_mismatch:  {(proca.mass - scalar.mass) / scalar.mass:.3e}")
    print(f"zero_mode_mass_mismatch: {(zero_mode_mass(proca) - zero_mode_mass(scalar)) / zero_mode_mass(scalar):.3e}")
    print(f"epsilon_ratio_s_over_p:  {scalar_eps / proca_eps:.15e}")
    print(f"scalar_sp_mass:          {scalar_sp_profile.dimensionless_cloud_mass:.15e}")
    print(f"scalar_sp_kappa:         {scalar_sp_profile.kappa:.15e}")
    print(f"proca_nr_mass:           {proca_nr_profile.dimensionless_cloud_mass:.15e}")
    print(f"proca_nr_epsilon:        {proca_nr_profile.epsilon:.15e}")
    print(f"proca_nr_mass_over_eps:  {proca_nr_profile.mass_over_epsilon:.15e}")
    print(f"proca_full_mass_over_eps:{zero_mode_mass(proca) / proca_eps:.15e}")
    for field in ("A0", "C0", "M0"):
        print(f"residual_{field}_scalar_sp_full:{residuals[(field, 'scalar_sp_vs_full')]:.15e}")
        print(f"residual_{field}_proca_radial_sp_full:{residuals[(field, 'proca_nr_vs_full')]:.15e}")
    print(f"plot:                    {args.plot}")
    print(f"csv:                     {args.output_csv}")


if __name__ == "__main__":
    main()
