# Spherical Soliton & Oscillaton Constructor

Numerical constructors for spherically symmetric real scalar and Proca oscillatons in polar-areal gauge. The code also includes scalar SP and radial-vector SP reference profiles for weak-field comparisons.

<p align="center">
  <img src="figures/proca_g00_m6e-1.gif" alt="Time-dependent density plot of the Proca oscillaton g00 profile in the phi=0 plane" width="420" style="border-radius:50%;">
</p>
<p align="center"><i>g</i><sub>00</sub>(t,x) of a spherical Proca oscillaton, with parameters: <code>mu=1, G=1, mu*M_ADM=0.600014874, omega=0.978575355, epsilon=0.205888987, jmax=6</code></p>

## Scope

The vector constructor is restricted to the strictly spherical zero-magnetic sector,

$$
X_\mu dx^\mu = U(t,x)\,dt + W(t,x)\,dx .
$$

Here zero-magnetic means

$$
B^i=\frac{1}{2}\epsilon^{ijk}F_{jk}=0 .
$$

The tensor $\epsilon^{ijk}$ is the Levi-Civita tensor on the spatial slice, with

$$
\epsilon^{ijk}=\frac{[ijk]}{\sqrt{\gamma}},
$$

where $[ijk]$ is the Levi-Civita symbol and $\gamma=\det\gamma_{ij}$. In polar-areal coordinates,

$$
\gamma_{ij}dx^idx^j=A(t,x)\,dx^2+x^2d\Omega^2,
\qquad
\sqrt{\gamma}=\sqrt{A(t,x)}\,x^2\sin\theta .
$$

For the ansatz above, $F_{x\theta}=F_{\theta\phi}=F_{\phi x}=0$, so $B^i=0$ identically.

It is not a generic 3D Proca solver. Note that in the nonrelativistic limit this branch reduces to the radial-vector SP equation, not to the scalar SP equation.

## Units

The dimensionless coordinates are

$$
x=\mu r_{\mathrm{phys}}, \qquad t=\mu t_{\mathrm{phys}}.
$$

The code uses $\mu=1$ and $G=1$. The reported mass is the dimensionless ADM mass $\mu M_{\mathrm{ADM}}$.

The binding parameter is

$$
\epsilon=\sqrt{1-\left(\frac{\omega}{\mu}\right)^2},
$$

which becomes $\epsilon=\sqrt{1-\omega^2}$ in the code units.

## Metric

Both constructors use polar-areal gauge,

$$
ds^2=-\alpha(t,x)^2dt^2+a(t,x)^2dx^2+x^2d\Omega^2 .
$$

The numerical metric variables are

$$
A(t,x)=a(t,x)^2, \qquad C(t,x)=\left(\frac{a(t,x)}{\alpha(t,x)}\right)^2 .
$$

Therefore

$$
a=\sqrt{A}, \qquad \alpha=\sqrt{\frac{A}{C}}, \qquad g_{00}=-\alpha^2=-\frac{A}{C}.
$$

The ADM mass function is

$$
M(t,x)=\frac{x}{2}\left(1-\frac{1}{A(t,x)}\right),
$$

The code also uses the enclosed-mass notation built from the zero Fourier mode of $A$,

$$
M({<}x)=\frac{x}{2}\left(1-\frac{1}{A_0(x)}\right).
$$

This is the quantity stored as `M0` in the examples. It is not the Fourier zero mode of $M(t,x)$ itself, because $[1/A]_0\ne1/A_0$ in general.

## Spherical Poisson-Gauge Convention

Some scalar metric-potential diagnostics compare the polar-areal solutions to a local weak-field estimate after a spherical coordinate transformation. The usual Poisson gauge is a linear perturbation-theory metric,

$$
ds^2=-(1+2\Psi)d\tau^2+(1-2\Phi_{\mathrm{metric}})\delta_{ij}dX^i dX^j .
$$

For a strictly spherical solution, the corresponding non-perturbative construction is an isotropic, Poisson-like gauge. Starting from

$$
ds^2=-\frac{A}{C}dt^2+A\,dx^2+x^2d\Omega^2,
$$

define

$$
t=\tau+T(\tau,R),\qquad x=R+L(\tau,R),
$$

and impose

$$
g_{\tau R}=0,\qquad g_{RR}=\frac{g_{\theta\theta}}{R^2}\equiv \chi(\tau,R).
$$

These are exact spherical conditions. They become the usual Cartesian Poisson gauge only after expanding around flat space, where

$$
g_{\tau\tau}^{\mathrm{PG}}=-(1+2\Psi+\cdots),\qquad \chi=1-2\Phi_{\mathrm{metric}}+\cdots .
$$

For any periodic quantity $Q(\tau,R)$, write the Fourier series at fixed $R$ as

$$
Q(\tau,R)=Q_0(R)+\sum_{n\ge1}\left[Q_{n,c}(R)\cos(n\omega\tau)+Q_{n,s}(R)\sin(n\omega\tau)\right].
$$

The coefficient used below is the cosine coefficient at frequency $2\omega$,

$$
[Q]_{2\omega}\equiv Q_{2,c}(R)=\frac{\omega}{\pi}\int_{\tau_0}^{\tau_0+2\pi/\omega}Q(\tau,R)\cos(2\omega\tau)\,d\tau .
$$

The numerical potentials are then

$$
-\Psi_2^{\mathrm{num}}=\frac{1}{2}[g_{\tau\tau}^{\mathrm{PG}}+1]_{2\omega},
$$

$$
\Phi_2^{\mathrm{num}}=-\frac{1}{2}[\chi-1]_{2\omega}.
$$

The transformed scalar fundamental mode is the cosine coefficient at frequency $\omega$,

$$
\phi_{1,\mathrm{PG}}(R)=\frac{\omega}{\pi}\int_{\tau_0}^{\tau_0+2\pi/\omega}
\Phi_{\mathrm{code}}(t(\tau,R),x(\tau,R))\cos(\omega\tau)\,d\tau .
$$

With $\Phi_{\mathrm{code}}=\sqrt{8\pi G}\,\varphi_{\mathrm{phys}}$, the local real-scalar estimate gives

$$
\phi_{1,\mathrm{PG}}^2=16\pi G\frac{\varrho_{\mathrm{loc}}}{\mu^2},
$$

where $\varrho_{\mathrm{loc}}$ is the local physical energy density entering the weak-field estimate, not the scaled plotting radius `rho`. Therefore

$$
-\Psi_2^{\mathrm{local}}=\Phi_2^{\mathrm{local}}=\frac{\phi_{1,\mathrm{PG}}^2}{16}.
$$

Equivalently, $h_{00}=g_{\tau\tau}^{\mathrm{PG}}+1=-2\Psi+\cdots$ has $h_{00,2}^{\mathrm{local}}=\phi_{1,\mathrm{PG}}^2/8$. When plotting the potentials $-\Psi_2$ and $\Phi_2$ themselves, the reference curve is $\phi_{1,\mathrm{PG}}^2/16$.

The natural scaled radius for this transformed comparison is

$$
\rho_{\mathrm{plot}}=\epsilon R,\qquad \epsilon=\sqrt{1-(\omega/\mu)^2}.
$$

In the weak scalar SP limit, $\epsilon\simeq\kappa$ and $R\simeq x$, so this agrees with the older SP-scaled radius $\kappa x$.

## Scalar Field

The real scalar field has particle mass $\mu=1$ and no self-interaction. The Fourier ansatz is

$$
\Phi(t,x)=\sum_{\substack{j\ge 1\\ j\ \mathrm{odd}}}\phi_j(x)\cos(j\omega t),
$$

$$
A(t,x)=\sum_{\substack{j\ge 0\\ j\ \mathrm{even}}}A_j(x)\cos(j\omega t), \qquad C(t,x)=\sum_{\substack{j\ge 0\\ j\ \mathrm{even}}}C_j(x)\cos(j\omega t).
$$

The scalar central input parameter in the API is `phi1_center`, the value of the first scalar Fourier coefficient at the origin.

For weak scalar oscillatons the constructor uses the scalar SP ground state as the initial guess.

## Vector Field

The massive vector field is

$$
X_\mu dx^\mu = U(t,x)\,dt+W(t,x)\,dx,
$$

with

$$
F_{tx}=\dot W-U', \qquad E=\frac{F_{tx}}{\alpha a}.
$$

The first-order Proca relations used by the constructor are

$$
(x^2E)'=-x^2\sqrt{C}\,U,
$$

$$
\dot E=-\frac{W}{\sqrt{C}}, \qquad W=-\sqrt{C}\,\dot E,
$$

$$
U'=-\partial_t\!\left(\sqrt{C}\,\dot E\right)-\frac{AE}{\sqrt{C}}.
$$

The Fourier ansatz is

$$
U(t,x)=\sum_{\substack{j\ge 1\\ j\ \mathrm{odd}}}u_j(x)\cos(j\omega t), \qquad E(t,x)=\sum_{\substack{j\ge 1\\ j\ \mathrm{odd}}}e_j(x)\cos(j\omega t),
$$

$$
W(t,x)=\sum_{\substack{j\ge 1\\ j\ \mathrm{odd}}}w_j(x)\sin(j\omega t),
$$

$$
A(t,x)=\sum_{\substack{j\ge 0\\ j\ \mathrm{even}}}A_j(x)\cos(j\omega t), \qquad C(t,x)=\sum_{\substack{j\ge 0\\ j\ \mathrm{even}}}C_j(x)\cos(j\omega t).
$$

The projected Einstein-Proca system is

$$
\begin{aligned}
A'&=\frac{A(1-A)}{x}+\frac{Ax}{2}\left(AE^2+CU^2+C\dot{E}^{\,2}\right),\\
C'&=\frac{2C}{x}\left(1-A+\frac{x^2AE^2}{2}\right),\\
E'&=-\frac{2E}{x}-\sqrt{C}\,U,\\
U'&=-\partial_t\!\left(\sqrt{C}\,\dot E\right)-\frac{AE}{\sqrt{C}},\\
\dot A&=-xA\sqrt{C}\,U\dot E .
\end{aligned}
$$

The vector central input parameter in the API is `u1_center`, the value of the first vector Fourier coefficient at the origin.

## Boundary Conditions

The Fourier BVP is solved on a finite interval $x_{\min}\le x\le x_{\max}$. The solvers use `x_min=1.0e-4`, while `x_max` is user-controlled. The frequency `omega` is solved as an eigenvalue. The retained matter modes are odd and the retained metric modes are even:

$$
\mathcal J_{\mathrm{m}}=\{j\in\mathbb N:\ j\ \mathrm{odd},\ 1\le j\le j_{\max}\},
\qquad
\mathcal J_{\mathrm{g}}=\{j\in\mathbb N_0:\ j\ \mathrm{even},\ 0\le j\le j_{\max}\}.
$$

For the scalar constructor, let $\phi_{1,c}$ denote the input `phi1_center`. The inner boundary conditions are

$$
\begin{aligned}
\phi_1(x_{\min})&=\phi_{1,c},\\
\phi_j'(x_{\min})&=0\qquad (j\in\mathcal J_{\mathrm{m}}),\\
A_0(x_{\min})&=1.
\end{aligned}
$$

The scalar outer boundary conditions are

$$
\begin{aligned}
\phi_j(x_{\max})&=0\qquad (j\in\mathcal J_{\mathrm{m}}),\\
C_0(x_{\max})&=A_0(x_{\max})^2,\\
C_j(x_{\max})&=0\qquad (j\in\mathcal J_{\mathrm{g}},\ j\ge2).
\end{aligned}
$$

For the Proca constructor, let $u_{1,c}$ denote the input `u1_center`. The inner boundary conditions are

$$
\begin{aligned}
u_1(x_{\min})&=u_{1,c},\\
A_0(x_{\min})&=1,\\
e_j(x_{\min})&=-\frac{x_{\min}}{3}\,[\sqrt{C}\,U]_j(x_{\min})\qquad (j\in\mathcal J_{\mathrm{m}}).
\end{aligned}
$$

Here $[\sqrt{C}\,U]_j$ is the cosine Fourier coefficient of $\sqrt{C(t,x_{\min})}\,U(t,x_{\min})$ at odd matter mode $j$. This is the regular-origin Gauss-law condition for the radial electric coefficients.

The Proca outer matter condition is applied mode by mode for $j\in\mathcal J_{\mathrm{m}}$. For $j\omega<1$, define

$$
\gamma_j=\sqrt{1-j^2\omega^2}.
$$

The outer matter condition is

$$
\begin{cases}
\gamma_j^2 e_j(x_{\max})-\left(\gamma_j+\dfrac{1}{x_{\max}}\right)u_j(x_{\max})=0, & j\omega<1,\\
u_j(x_{\max})=0, & j\omega\ge1.
\end{cases}
$$

The first line is the finite-radius Yukawa Robin condition obtained from $u_j\propto e^{-\gamma_j x}/x$ and the asymptotic linear Proca relation $u_j'=-\gamma_j^2 e_j$. The second line is the finite-domain closure used by this code when the retained mode does not have a Yukawa decay constant. The code does not impose an outgoing-wave condition.

The Proca metric outer boundary conditions are the same as in the scalar constructor,

$$
\begin{aligned}
C_0(x_{\max})&=A_0(x_{\max})^2,\\
C_j(x_{\max})&=0\qquad (j\in\mathcal J_{\mathrm{g}},\ j\ge2).
\end{aligned}
$$

The inner conditions enforce regularity at the origin in polar-areal variables. The metric outer conditions set the oscillating metric modes to zero at the boundary and use $C_0=A_0^2$ as the finite-radius Schwarzschild/asymptotic time-normalization condition. The matter outer conditions are finite-domain closure conditions for the Fourier BVP; they are not time-evolution radiation boundary conditions.

Basic numerical checks are: `profile.metadata["success"]` should be true, `profile.metadata["max_rms_residual"]` should be comparable to the requested `tol`, the outer tail of `profile.mass_profile` should be nearly flat, and the plotted profiles should be stable when `x_max`, `n_grid`, and the Fourier truncation are increased. The comparison example prints `scalar_tail_rel_std` and `proca_tail_rel_std`, which are the relative standard deviations of the outer mass tail.

## Nonrelativistic References

The variables $F$ and $V$ are nonrelativistic limit variables. They are defined from a weak-field one-parameter family of relativistic solutions, not by an independent convention. Primes in the SP equations below mean derivatives with respect to the scaled radius $y$.

For the scalar branch, define the scaled radius by

$$
y=\kappa x
$$

and take the weak-field limit $\kappa\to0$ with $y$ fixed. The scalar SP wavefunction $F$ is the rescaled first scalar Fourier coefficient,

$$
F(y)=\lim_{\kappa\to0}
\frac{\phi_1(y/\kappa)}{\sqrt{2}\,\kappa^2}.
$$

The scalar SP potential $V$ is the effective Schrodinger-Poisson potential including the frequency eigenvalue shift. In the code's polar-areal variables it is defined by

$$
V_\infty=\lim_{\kappa\to0}\frac{1-\omega^2}{\kappa^2},
\qquad
V(y)=V_\infty-\lim_{\kappa\to0}
\frac{C_0(y/\kappa)-A_0(y/\kappa)}{\kappa^2}.
$$

Equivalently, if $\mathcal V_\kappa(x)=\kappa^2V(\kappa x)$ and primes on $\mathcal V_\kappa$ denote $d/dx$, the weak-field zero-mode metric profiles obey

$$
A_0(x)=1+x\mathcal V_\kappa'(x)+O(\kappa^4),
$$

$$
C_0(x)=1+x\mathcal V_\kappa'(x)+\kappa^2V_\infty-\mathcal V_\kappa(x)+O(\kappa^4),
$$

$$
M({<}x)=\frac{x^2}{2}\mathcal V_\kappa'(x)+O(\kappa^3).
$$

With these definitions, the scalar SP ground state solves

$$
F''+\frac{2}{y}F'=VF, \qquad V''+\frac{2}{y}V'=F^2.
$$

The inverse relation for the relativistic scalar profile is

$$
\phi_1(x)=\sqrt{2}\,\kappa^2F(\kappa x)+O(\kappa^4).
$$

The weak-field frequency and total mass scale as

$$
\omega=1-\frac{1}{2}\kappa^2V_\infty+O(\kappa^4),
\qquad
\epsilon=\kappa\sqrt{V_\infty}+O(\kappa^3),
$$

$$
\mu M_{\mathrm{ADM}}=\kappa M_{\mathrm{SP}}+O(\kappa^3).
$$

The scalar SP mass convention is

$$
M_{\mathrm{SP}}=\frac{1}{2}\int_0^\infty y^2F(y)^2\,dy.
$$

For the strict radial-vector branch, define the scaled radius by

$$
y=\lambda x
$$

and take the weak-field limit $\lambda\to0$ with $y$ fixed. The radial-vector SP wavefunction $F$ is defined from the first Fourier coefficient of the radial electric field,

$$
F(y)=-\lim_{\lambda\to0}\frac{e_1(y/\lambda)}{\lambda^2}.
$$

The minus sign is the code convention for positive `u1_center`. The first Proca potential coefficient is then fixed at leading order by the flat-space Gauss relation,

$$
u_1(x)=\lambda^3
\left[
F'(\lambda x)+\frac{2F(\lambda x)}{\lambda x}
\right]+O(\lambda^5).
$$

Thus, for the normalization $F'(0)=1$, one has $u_1(0)=3\lambda^3+O(\lambda^5)$. The radial-vector SP potential $V$ is defined from the metric in the same way as above,

$$
V_\infty=\lim_{\lambda\to0}\frac{1-\omega^2}{\lambda^2},
\qquad
V(y)=V_\infty-\lim_{\lambda\to0}
\frac{C_0(y/\lambda)-A_0(y/\lambda)}{\lambda^2}.
$$

The strict radial-vector SP ground state solves

$$
F''+\frac{2}{y}F'-\frac{2}{y^2}F=VF, \qquad V''+\frac{2}{y}V'=F^2.
$$

The total mass and binding scale as

$$
\mu M_{\mathrm{ADM}}=\lambda M_{\mathrm{SP}}+O(\lambda^3),
\qquad
\epsilon=\lambda\sqrt{V_\infty}+O(\lambda^3).
$$

Note that there is an extra $-2F/y^2$ term in the radial-vector equation.

## Main Parameters

| Name | Meaning |
| --- | --- |
| `target_mass` | Target dimensionless ADM mass $\mu M_{\mathrm{ADM}}$. |
| `omega` | Eigenfrequency stored by the profile object; mathematically this is $\omega/\mu$, and the code uses $\mu=1$. |
| `epsilon` | Binding parameter returned by `epsilon_from_omega`; mathematically $\epsilon=\sqrt{1-(\omega/\mu)^2}$. |
| `jmax` | Largest retained Fourier index. Matter modes are odd and metric modes are even. |
| `n_grid` | Number of radial collocation points. |
| `n_time` | Number of Fourier phase collocation points. |
| `tol` | Boundary-value solver tolerance. |
| `mass_tol` | Relative tolerance used by the mass-tuning wrapper. |
| `x_max` | Outer radial boundary in dimensionless radius $x$. |
| `rho` | Scaled plotting radius. The scalar/vector/SP comparison uses $\rho=\kappa x$; the Poisson-gauge local-estimate comparison uses $\rho=\epsilon R$. |
| `A0`, `C0` | Code arrays storing the zero Fourier modes of `A` and `C`. |
| `A2`, `C2` | Code arrays storing the second Fourier modes of `A` and `C`. |
| `M0` | Code array storing $M({<}x)=x(1-1/A_0)/2$, the enclosed-mass function built from the zero Fourier mode of `A`. |
| `phi1_center` | Scalar central amplitude used as the scalar BVP input. |
| `u1_center` | Vector central amplitude used as the vector BVP input. |

## Function Usage

```python
from oscillaton_builders import (
    construct_scalar_oscillaton,
    construct_vector_oscillaton,
    epsilon_from_omega,
    zero_mode_metric_arrays,
)

target_mass = 2.0e-3

scalar = construct_scalar_oscillaton(target_mass)
vector = construct_vector_oscillaton(target_mass)

print(scalar.mass, scalar.omega, epsilon_from_omega(scalar.omega))
print(vector.mass, vector.omega, epsilon_from_omega(vector.omega))

scalar_initial_data = scalar.initial_data()
vector_initial_data = vector.initial_data()

scalar_metric = zero_mode_metric_arrays(scalar, scalar.x)
vector_metric = zero_mode_metric_arrays(vector, vector.x)
```

Profile objects expose `x`, `omega`, `mass`, `A0`, `C0`, `mass_profile`, `evaluate(theta)`, and `initial_data()`.

For scalar profiles, `evaluate(theta)` returns `Phi`, `Phi_t`, `Phi_x`, `A`, `C`, `a`, and `alpha`.

For vector profiles, `evaluate(theta)` returns `U`, `E`, `E_t`, `W`, `A`, `C`, `a`, and `alpha`.

The SP references are available through

```python
from oscillaton import solve_sp_ground_state
from proca_oscillaton import solve_radial_proca_nr_ground_state

scalar_sp = solve_sp_ground_state()
radial_vector_sp = solve_radial_proca_nr_ground_state()
```

## Example Figures

Regenerate all code figures with

```bash
python examples/reproduce_code_figures.py
```

Run a single scalar/vector/SP comparison case with

```bash
python examples/compare_scalar_proca_sp_seeded.py \
  --target-mass 6.0e-1 \
  --output-csv figures/scalar_proca_radial_nr_same_mass_m6e-1.csv \
  --plot figures/scalar_proca_radial_nr_same_mass_m6e-1.png
```

The plotted curves are `scalar` as blue solid, `vector` as red solid, `scalar SP` as black dashed, and `Proca radial SP` as black dash-dot. The fourth panel shows $A_2^{\mathrm{scalar}}$, $C_2^{\mathrm{scalar}}$, $A_2^{\mathrm{vector}}$, and $C_2^{\mathrm{vector}}$.

| Figure | Target $\mu M_{\mathrm{ADM}}$ | Scalar mass | Vector mass | Scalar $\epsilon$ | Vector $\epsilon$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| `scalar_proca_radial_nr_same_mass_m2e-3.png` | `2.0e-3` | `1.999997391e-3` | `2.000000870e-3` | `1.141118690e-3` | `6.579194612e-4` |
| `scalar_proca_radial_nr_same_mass_m2e-1.png` | `2.0e-1` | `2.000034037e-1` | `2.000036979e-1` | `1.160004602e-1` | `6.606791229e-2` |
| `scalar_proca_radial_nr_same_mass_m6e-1.png` | `6.0e-1` | `6.000092822e-1` | `6.000148738e-1` | `4.695710613e-1` | `2.058889873e-1` |

Run the scalar local-estimate comparison with

```bash
python examples/compare_scalar_local_estimate.py \
  --target-mass 1.0e-1 \
  --output-csv figures/scalar_poisson_potentials_vs_local_m1e-1.csv \
  --plot figures/scalar_poisson_potentials_vs_local_m1e-1.png
```

This example first transforms the scalar oscillaton to the spherical Poisson-like gauge described above, then plots $-\Psi_2^{\mathrm{num}}$ and $\Phi_2^{\mathrm{num}}$ against the local estimate $\phi_{1,\mathrm{PG}}^2/16$.
