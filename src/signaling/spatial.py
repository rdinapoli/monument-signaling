"""Spatial dynamics of bistable monument adoption.

Implements the bistable reaction-diffusion equation that extends the
mean-field replicator dynamics of the bistable-emergence analysis
(emergence.py) to spatially structured populations. The PDE is

    d phi / d t  =  phi (1 - phi) [w_MB(phi) - w_NB(phi)]  +  D * Laplacian(phi)

with phi(x, y, t) in [0, 1] representing the local fraction of signaling
groups. This is a Nagumo-type bistable reaction-diffusion equation, in
the class studied since Nagumo (1962), Aronson and Weinberger (1975),
and Fife (1979). It is distinct from the monostable Fisher-KPP class
that Ammerman and Cavalli-Sforza (1971, 1984) used for the Neolithic
wave of advance: bistable dynamics require a critical nucleus to
propagate and can pin at parameter discontinuities; monostable dynamics
propagate from arbitrary perturbations and do not pin.

The reaction term is the mean-field replicator under payoff-biased
imitation: a group switches to monument-building at a rate proportional to
the local building fraction phi, the local non-building fraction (1 - phi),
and the realized payoff difference w_MB - w_NB. Read as cultural
transmission, this is payoff-biased imitation of successful (high-payoff)
builders, with honesty enforced by the cost of the signal (the cost enters
w_MB - w_NB directly). The bistable RD structure is invariant under the
transmission-rule label and also follows from conformist transmission; see
the main-text discussion "Bistable spread and the pattern of independent
traditions". The diffusion term D *
Laplacian(phi) couples neighboring locations: groups also copy the strategies
of their spatial neighbors. Spread is therefore cultural transmission, not
demographic replacement, which is why it operates on the decadal-to-
centennial timescale of the archaeological record rather than the slower
timescale of cultural group selection (see the main-text discussion
"Bistable spread and the pattern of independent traditions").

The PDE is the mean-field spatial description of the discrete network
dynamics at the lattice scale (a strict continuum limit would need an
additional weak-selection or split-rate scaling; see the supplementary
scoping discussion). The module provides PDE integration and a stochastic
lattice simulation implementing the SI's displayed imitation rule; the
lattice is a qualitative counterpart used for the origination-flux check,
not a validation of the PDE.

See the main-text discussion "Bistable spread and the pattern of
independent traditions" for the manuscript treatment and Fife (1979)
Ch. 4 for the mathematical background.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from signaling.emergence import rare_builder_fitness_advantage


# =====================================================================
# Reaction term
# =====================================================================


def _build_advantage_table(
    sigma: float,
    lambda_W: float,
    mode: str,
    use_self_consistent_cost: bool,
    C_exogenous: float,
    n_phi: int = 101,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Precompute advantage values on a phi grid for interpolation."""
    phi_grid = np.linspace(1e-4, 1.0 - 1e-4, n_phi)
    adv_grid = np.zeros_like(phi_grid)
    for i, p in enumerate(phi_grid):
        res = rare_builder_fitness_advantage(
            sigma=sigma, lambda_W=lambda_W, frac_signalers=float(p),
            mode=mode,
            use_self_consistent_cost=use_self_consistent_cost,
            C_exogenous=C_exogenous,
        )
        adv_grid[i] = res["advantage"]
    return phi_grid, adv_grid


def reaction_term(
    phi: float | NDArray[np.float64],
    sigma: float,
    lambda_W: float = 0.68,
    mode: str = "multiplicative",
    use_self_consistent_cost: bool = True,
    C_exogenous: float = 0.35,
    n_phi_table: int = 101,
) -> float | NDArray[np.float64]:
    r"""Reaction term f(phi) for the spatial replicator PDE (payoff-biased imitation).

    f(phi) = phi (1 - phi) [w_MB(phi) - w_NB(phi)]

    This is the replicator under proportional imitation: the local building
    fraction phi grows where the realized payoff of building, w_MB(phi),
    exceeds that of not building, w_NB(phi). Read as cultural transmission it
    is payoff-biased imitation of successful builders, with honesty enforced
    by the signal cost. The payoff difference w_MB(phi) - w_NB(phi) is computed by
    :func:`rare_builder_fitness_advantage` with the local neighbor
    fraction equal to phi. For efficiency on large grids, advantage
    values are precomputed on a phi grid (n_phi_table points) and
    interpolated linearly. The interpolation error is negligible for
    n_phi_table >= 101.

    Parameters
    ----------
    phi : float or array
        Local fraction of signaling groups, in [0, 1].
    sigma : float
        Environmental uncertainty.
    lambda_W : float
        Within-group social reward (default 0.68, empirical anchor).
    mode : "multiplicative" or "mixed"
        Fitness composition.
    use_self_consistent_cost : bool
        If True, C = C_model(lambda_W). Default True.
    C_exogenous : float
        Exogenous cost if not using self-consistent.
    n_phi_table : int
        Number of phi grid points for the precomputed advantage table.

    Returns
    -------
    float or array
        Reaction term value(s), same shape as phi.
    """
    phi_grid, adv_grid = _build_advantage_table(
        sigma=sigma, lambda_W=lambda_W, mode=mode,
        use_self_consistent_cost=use_self_consistent_cost,
        C_exogenous=C_exogenous, n_phi=n_phi_table,
    )
    phi_arr = np.atleast_1d(np.asarray(phi, dtype=np.float64))
    phi_clipped = np.clip(phi_arr, 1e-4, 1.0 - 1e-4)
    adv_interp = np.interp(phi_clipped, phi_grid, adv_grid)
    out = phi_arr * (1.0 - phi_arr) * adv_interp
    # Zero out near the boundaries (where phi*(1-phi) is already small).
    out = np.where((phi_arr <= 0.0) | (phi_arr >= 1.0), 0.0, out)
    if np.isscalar(phi) or (hasattr(phi, "ndim") and phi.ndim == 0):
        return float(out.ravel()[0])
    return out


def reaction_integral(
    sigma: float,
    lambda_W: float = 0.68,
    mode: str = "multiplicative",
    n_quad: int = 200,
) -> float:
    r"""Compute the integral of f(phi) over phi in [0, 1].

    The sign of this integral determines wave direction in a bistable
    reaction-diffusion equation: positive integral means the signaling
    state invades into the non-signaling state; negative means the
    non-signaling state invades into signaling.

    The magnitude of the integral relates to the wave velocity through
    a variational principle (Fife 1979 Theorem 4.2). For the Nagumo
    cubic, the velocity is proportional to (1 - 2 phi_star) where
    phi_star is the unstable saddle.

    Parameters
    ----------
    sigma, lambda_W, mode : as in reaction_term.
    n_quad : int
        Quadrature points for Simpson integration.

    Returns
    -------
    float
        Integral of f from 0 to 1.
    """
    phi_grid = np.linspace(0.0, 1.0, n_quad + 1)
    f_vals = reaction_term(phi_grid, sigma=sigma, lambda_W=lambda_W, mode=mode)
    # Trapezoidal rule (np.trapezoid in NumPy >= 2.0)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(f_vals, phi_grid))
    dx = phi_grid[1] - phi_grid[0]
    return float(0.5 * dx * (f_vals[0] + f_vals[-1] + 2.0 * f_vals[1:-1].sum()))


def maxwell_point_sigma(
    lambda_W: float = 0.68,
    mode: str = "multiplicative",
    sigma_bracket: tuple[float, float] = (0.2, 1.0),
    n_quad: int = 200,
) -> float:
    r"""Maxwell point :math:`\sigma_M`: the environmental uncertainty at which
    the spatial front is stationary (the reaction integral changes sign).

    :math:`\sigma_M` is the root of :func:`reaction_integral` in sigma. Below
    :math:`\sigma_M` the integral is negative and the front retreats: the build
    state is metastable (locally stable but not advancing, so spread proceeds by
    payoff-biased imitation, not by front advance). Above :math:`\sigma_M`
    the integral is
    positive and the build front advances. The sign-change at :math:`\sigma_M`
    is the central testable prediction of the spatial model (main-text
    discussion "Bistable spread and the pattern of independent traditions"):
    for :math:`\sigma < \sigma_M` monuments persist and spread only by
    copying, while for :math:`\sigma > \sigma_M` they advance as a front. The
    reproductive cost is the self-consistent default of :func:`reaction_term`.

    Parameters
    ----------
    lambda_W, mode, n_quad : as in :func:`reaction_integral`.
    sigma_bracket : (low, high)
        Bracketing interval for the root; reaction_integral must change sign
        across it. The default (0.2, 1.0) brackets :math:`\sigma_M \approx 0.6`
        at the empirical lambda_W = 0.68.

    Returns
    -------
    float
        :math:`\sigma_M`, the Maxwell-point environmental uncertainty.

    Raises
    ------
    ValueError
        If reaction_integral does not change sign across sigma_bracket (no
        Maxwell point there; the front does not reverse within the interval).
    """
    lo, hi = sigma_bracket

    def _integral(s: float) -> float:
        return reaction_integral(s, lambda_W=lambda_W, mode=mode, n_quad=n_quad)

    f_lo, f_hi = _integral(lo), _integral(hi)
    if f_lo == 0.0:
        return float(lo)
    if f_hi == 0.0:
        return float(hi)
    if np.sign(f_lo) == np.sign(f_hi):
        raise ValueError(
            f"reaction_integral does not change sign on {sigma_bracket}: "
            f"f({lo})={f_lo:.4g}, f({hi})={f_hi:.4g}; no Maxwell point in the bracket."
        )
    return float(brentq(_integral, lo, hi))


# =====================================================================
# Diffusion (Laplacian)
# =====================================================================


def laplacian_2d(
    phi_field: NDArray[np.float64],
    dx: float = 1.0,
    boundary: str = "neumann",
) -> NDArray[np.float64]:
    r"""5-point finite-difference Laplacian on a 2D grid.

    Lap(phi)[i,j] = (phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1]
                    - 4 phi[i,j]) / dx^2

    Parameters
    ----------
    phi_field : 2D array
        Field values on a regular grid.
    dx : float
        Grid spacing (assumed isotropic).
    boundary : "neumann" or "periodic"
        Boundary condition. Neumann (zero-flux) for finite domains;
        periodic for waves on a torus.

    Returns
    -------
    2D array
        Laplacian of phi_field with the same shape.
    """
    if boundary == "periodic":
        up = np.roll(phi_field, -1, axis=0)
        down = np.roll(phi_field, 1, axis=0)
        right = np.roll(phi_field, -1, axis=1)
        left = np.roll(phi_field, 1, axis=1)
        return (up + down + left + right - 4.0 * phi_field) / (dx * dx)
    elif boundary == "neumann":
        # Use symmetric padding (reflect at edges) for zero-flux BC.
        padded = np.pad(phi_field, 1, mode="edge")
        up = padded[2:, 1:-1]
        down = padded[:-2, 1:-1]
        right = padded[1:-1, 2:]
        left = padded[1:-1, :-2]
        return (up + down + left + right - 4.0 * phi_field) / (dx * dx)
    else:
        raise ValueError(f"Unknown boundary: {boundary}")


def divergence_diffusion_2d(
    phi_field: NDArray[np.float64],
    D_field: NDArray[np.float64],
    dx: float = 1.0,
    boundary: str = "neumann",
) -> NDArray[np.float64]:
    r"""Conservative finite-volume discretization of :math:`\nabla\cdot(D(\phi)\nabla\phi)`.

    NOTE ON OPERATOR CHOICE. The mean-field lattice expansion of the
    asymmetric imitation rule produces the NON-divergence operator
    :math:`D(\phi)\,\nabla^2\phi` (supplementary section "The bistable RD
    equation from discrete network dynamics: derivation and scope", the
    branch-dependent RD expansion; implemented in
    :func:`nondivergence_diffusion_2d`, the default operator of
    :func:`solve_pde_2d_degenerate`). The conservative divergence form
    computed here differs from the derived operator by
    :math:`D'(\phi)\,|\nabla\phi|^2`, a term of the same spatial order, and
    is retained as an additional robustness variant (the standard form when
    a conserved flux, rather than a local update rate, is modeled). The two
    operators coincide when :math:`D' = 0` (constant :math:`D`).

    This routine uses a second-order conservative scheme with arithmetic
    face-averaged diffusivities,

    .. math::
        [\nabla\cdot(D\nabla\phi)]_{ij} = \frac{1}{\Delta x^2}\big[
            D_{i+1/2,j}(\phi_{i+1,j}-\phi_{ij})
          - D_{i-1/2,j}(\phi_{ij}-\phi_{i-1,j})
          + D_{i,j+1/2}(\phi_{i,j+1}-\phi_{ij})
          - D_{i,j-1/2}(\phi_{ij}-\phi_{i,j-1})\big],

    with :math:`D_{i+1/2,j}=\tfrac12(D_{ij}+D_{i+1,j})`. With a constant
    ``D_field`` this reduces exactly to ``D * laplacian_2d`` (a correctness
    check enforced in the test suite). The state-dependent coefficient it is
    used with is the branch-dependent :math:`D(\phi)` of
    :func:`degenerate_diffusion_table`.

    Parameters
    ----------
    phi_field : 2D array
        Field values on a regular grid.
    D_field : 2D array
        Diffusion coefficient at each cell (same shape as ``phi_field``).
    dx : float
        Grid spacing (isotropic).
    boundary : "neumann" or "periodic"
        Zero-flux (Neumann, via edge padding so the boundary face flux is zero)
        or periodic.

    Returns
    -------
    2D array
        :math:`\nabla\cdot(D\nabla\phi)` with the same shape as ``phi_field``.
    """
    if boundary == "periodic":
        pad_mode = "wrap"
    elif boundary == "neumann":
        pad_mode = "edge"
    else:
        raise ValueError(f"Unknown boundary: {boundary}")
    phi_p = np.pad(phi_field, 1, mode=pad_mode)
    D_p = np.pad(D_field, 1, mode=pad_mode)
    phic = phi_p[1:-1, 1:-1]
    Dc = D_p[1:-1, 1:-1]
    # Arithmetic face averages of D between the cell and each neighbor.
    D_ip = 0.5 * (Dc + D_p[2:, 1:-1])
    D_im = 0.5 * (Dc + D_p[:-2, 1:-1])
    D_jp = 0.5 * (Dc + D_p[1:-1, 2:])
    D_jm = 0.5 * (Dc + D_p[1:-1, :-2])
    flux = (
        D_ip * (phi_p[2:, 1:-1] - phic)
        - D_im * (phic - phi_p[:-2, 1:-1])
        + D_jp * (phi_p[1:-1, 2:] - phic)
        - D_jm * (phic - phi_p[1:-1, :-2])
    )
    return flux / (dx * dx)


def nondivergence_diffusion_2d(
    phi_field: NDArray[np.float64],
    D_field: NDArray[np.float64],
    dx: float = 1.0,
    boundary: str = "neumann",
) -> NDArray[np.float64]:
    r"""Non-divergence diffusion operator :math:`D(\phi)\,\nabla^2\phi`.

    This is the diffusion operator the mean-field lattice expansion of the
    asymmetric imitation rule actually produces (supplementary section "The
    bistable RD equation from discrete network dynamics: derivation and
    scope": expanding the full transition rate to :math:`O(h^2)` yields
    :math:`\partial_t\phi = f(\phi) + D(\phi)\,\nabla^2\phi` with the
    branch-dependent coefficient of :func:`degenerate_diffusion_table`). It
    is the default operator of :func:`solve_pde_2d_degenerate`.

    The conservative divergence form :math:`\nabla\cdot(D\nabla\phi)`
    (:func:`divergence_diffusion_2d`) differs from this operator by
    :math:`D'(\phi)\,|\nabla\phi|^2`, a term of the same spatial order that
    the lattice expansion does not produce; the two coincide exactly for
    constant :math:`D`.

    Parameters
    ----------
    phi_field : 2D array
        Field values on a regular grid.
    D_field : 2D array
        Diffusion coefficient at each cell (same shape as ``phi_field``).
    dx : float
        Grid spacing (isotropic).
    boundary : "neumann" or "periodic"
        Boundary condition, as in :func:`laplacian_2d`.

    Returns
    -------
    2D array
        :math:`D(\phi)\,\nabla^2\phi` with the same shape.
    """
    return D_field * laplacian_2d(phi_field, dx=dx, boundary=boundary)


# =====================================================================
# PDE integration
# =====================================================================


@dataclass
class PDEResult:
    """Result of solving the spatial PDE on a 2D grid."""
    t_array: NDArray[np.float64]
    phi_history: NDArray[np.float64]  # shape: (n_timesteps, Ny, Nx)
    sigma_field: NDArray[np.float64]
    D: float
    dx: float
    dt: float
    lambda_W: float


def solve_pde_2d(
    phi_initial: NDArray[np.float64],
    t_max: float,
    sigma_field: float | NDArray[np.float64],
    lambda_W: float = 0.68,
    D: float = 1.0,
    dx: float = 1.0,
    dt: float | None = None,
    n_record: int = 21,
    mode: str = "multiplicative",
    boundary: str = "neumann",
) -> PDEResult:
    r"""Integrate the bistable reaction-diffusion PDE on a 2D grid.

    Uses explicit Euler time-stepping. CFL stability requires
    dt <= dx^2 / (4 D); if dt is None, it is set to half the CFL limit.

    Parameters
    ----------
    phi_initial : 2D array
        Initial phi field, in [0, 1].
    t_max : float
        Final simulation time.
    sigma_field : float or 2D array
        Environmental uncertainty (spatially uniform if scalar, else a
        field of the same shape as phi_initial).
    lambda_W : float
        Within-group social reward.
    D : float
        Diffusion coefficient.
    dx : float
        Grid spacing.
    dt : float or None
        Time step. If None, computed from CFL.
    n_record : int
        Number of snapshots to record (uniformly spaced including t=0).
    mode : "multiplicative" or "mixed"
        Fitness composition.
    boundary : "neumann" or "periodic"
        Boundary condition.

    Returns
    -------
    PDEResult
        Dataclass with t_array, phi_history (n_record, Ny, Nx),
        sigma_field, D, dx, dt, lambda_W.
    """
    phi = phi_initial.copy().astype(np.float64)
    Ny, Nx = phi.shape

    # Sigma field as 2D array
    sig_field = np.broadcast_to(sigma_field, phi.shape).astype(np.float64)

    # CFL-stable time step
    if dt is None:
        dt = 0.4 * dx * dx / (4.0 * D)  # 0.4 safety factor
    elif dt > dx * dx / (4.0 * D):
        # An explicit dt above the CFL limit integrates unstably and returns
        # a clipped-garbage field that LOOKS like a valid phi configuration:
        # refuse rather than return silently wrong output.
        raise ValueError(
            f"Explicit dt={dt} exceeds the CFL stability limit "
            f"dx^2/(4D) = {dx * dx / (4.0 * D):.6g} for the explicit Euler "
            "scheme; pass a smaller dt or dt=None for the automatic step."
        )

    if n_record < 2:
        raise ValueError(f"n_record must be >= 2, got {n_record}")
    n_steps = int(np.ceil(t_max / dt))
    dt = t_max / n_steps
    record_every = max(1, n_steps // (n_record - 1))

    phi_history = [phi.copy()]
    t_recorded = [0.0]

    # Precompute advantage tables for each unique sigma value once.
    unique_sigs = np.unique(sig_field)
    adv_tables: dict[float, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    for s_val in unique_sigs:
        adv_tables[float(s_val)] = _build_advantage_table(
            sigma=float(s_val), lambda_W=lambda_W, mode=mode,
            use_self_consistent_cost=True, C_exogenous=0.35,
        )

    def _reaction(phi_field: NDArray[np.float64]) -> NDArray[np.float64]:
        out = np.zeros_like(phi_field)
        for s_val in unique_sigs:
            phi_grid, adv_grid = adv_tables[float(s_val)]
            mask = sig_field == s_val
            local_phi = phi_field[mask]
            local_phi_clipped = np.clip(local_phi, 1e-4, 1.0 - 1e-4)
            adv = np.interp(local_phi_clipped, phi_grid, adv_grid)
            out[mask] = local_phi * (1.0 - local_phi) * adv
        return out

    for step in range(1, n_steps + 1):
        reaction = _reaction(phi)
        diff = laplacian_2d(phi, dx=dx, boundary=boundary)
        phi = phi + dt * (reaction + D * diff)
        phi = np.clip(phi, 0.0, 1.0)

        if step % record_every == 0 or step == n_steps:
            phi_history.append(phi.copy())
            t_recorded.append(step * dt)

    return PDEResult(
        t_array=np.asarray(t_recorded),
        phi_history=np.asarray(phi_history),
        sigma_field=sig_field,
        D=D,
        dx=dx,
        dt=dt,
        lambda_W=lambda_W,
    )


def degenerate_diffusion_table(
    sigma: float,
    lambda_W: float = 0.68,
    mode: str = "multiplicative",
    D0: float = 1.0,
    n_phi: int = 201,
    use_self_consistent_cost: bool = True,
    C_exogenous: float = 0.35,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""State-dependent diffusion coefficient :math:`D(\phi)` from the asymmetric imitation rule.

    The asymmetric ``copy-a-random-builder-neighbor'' imitation rule yields, in
    the mean-field spatial expansion, a *degenerate*, state-dependent diffusion coefficient.
    Carrying the full O(h^2) expansion (without freezing the imitation prefactor)
    gives a coefficient that is branch-dependent in the sign of the rare-builder
    advantage :math:`\Delta w(\phi)=w_{MB}(\phi)-w_{NB}(\phi)` (the ``adv`` factor
    of :func:`reaction_term`):

    - :math:`\Delta w(\phi) > 0`:  :math:`D \propto (1-\phi)\,\tfrac{d}{d\phi}[\phi\,\Delta w]
      = (1-\phi)[\Delta w + \phi\,\Delta w']`;
    - :math:`\Delta w(\phi) < 0`:  :math:`D \propto \phi\,\tfrac{d}{d\phi}[(1-\phi)\,\Delta w]
      = \phi[(1-\phi)\,\Delta w' - \Delta w]`.

    The two branches agree at the saddle :math:`\phi^*` (where :math:`\Delta w=0`,
    both reduce to :math:`\phi^*(1-\phi^*)\,\Delta w'(\phi^*)`), so :math:`D` is
    *continuous* there with a slope kink; it is positive on :math:`(0,1)` and
    degenerate (vanishes) at the stable states :math:`\phi=0,1`. This supersedes
    the earlier frozen-prefactor simplification :math:`\phi(1-\phi)|\Delta w'|`,
    which dropped a same-order term. The main text's primary spatial operator is a
    constant-:math:`D` Laplacian (adopted as the standard bistable-RD form,
    corresponding to a symmetric pairwise-comparison update); this degenerate
    coefficient is used only as a robustness check on the critical-nucleus
    barrier (supplementary sections "The bistable RD equation from discrete
    network dynamics: derivation and scope" and "Spatial dynamics: variational identity, Allen-Cahn
    nucleus, and the formal droplet barrier").

    The table is normalized to :math:`D(\tfrac12)=D_0`, so the degenerate run is
    directly comparable to the constant-:math:`D` run at the same :math:`D_0`.

    Parameters
    ----------
    sigma, lambda_W, mode, use_self_consistent_cost, C_exogenous :
        As in :func:`reaction_term`; set the advantage profile.
    D0 : float
        Reference diffusivity (the value at the front midpoint :math:`\phi=1/2`).
    n_phi : int
        Number of phi grid points (>= 201 recommended for the derivative).

    Returns
    -------
    (phi_grid, D_grid) : tuple of 1D arrays
        Diffusion-coefficient table for interpolation; ``D_grid >= 0``.
    """
    phi_grid, adv_grid = _build_advantage_table(
        sigma=sigma, lambda_W=lambda_W, mode=mode,
        use_self_consistent_cost=use_self_consistent_cost,
        C_exogenous=C_exogenous, n_phi=n_phi,
    )
    advp = np.gradient(adv_grid, phi_grid)  # Delta w'(phi)
    # Asymmetric imitation-rule continuum diffusivity, carrying the FULL prefactor
    # expansion (not the frozen-prefactor simplification phi(1-phi)|Delta w'|).
    # Branch-dependent in sign(Delta w):
    #   Delta w > 0:  D ~ (1-phi) d/dphi[phi*Delta w]   = (1-phi)(Delta w + phi*Delta w')
    #   Delta w < 0:  D ~ phi    d/dphi[(1-phi)*Delta w] = phi((1-phi)*Delta w' - Delta w)
    # Continuous at the saddle phi* (both -> phi*(1-phi*)Delta w'(phi*)), slope kink there;
    # positive on (0,1); degenerate (vanishes) at phi = 0, 1.
    g = np.where(
        adv_grid > 0.0,
        (1.0 - phi_grid) * (adv_grid + phi_grid * advp),
        phi_grid * ((1.0 - phi_grid) * advp - adv_grid),
    )
    # Guard against negligible negative gradient artifacts at the grid endpoints
    # (the coefficient is non-negative; physical degeneracy means g -> 0 there).
    g = np.maximum(g, 0.0)
    g_half = float(np.interp(0.5, phi_grid, g))
    if g_half <= 0.0:
        raise ValueError("degenerate D normalization failed at phi=1/2")
    D_grid = D0 * g / g_half
    return phi_grid, D_grid


def solve_pde_2d_degenerate(
    phi_initial: NDArray[np.float64],
    t_max: float,
    sigma_field: float | NDArray[np.float64],
    lambda_W: float = 0.68,
    D0: float = 1.0,
    dx: float = 1.0,
    dt: float | None = None,
    n_record: int = 21,
    mode: str = "multiplicative",
    boundary: str = "neumann",
    operator: str = "nondivergence",
) -> PDEResult:
    r"""Integrate the bistable RD equation with the *degenerate* state-dependent
    diffusion coefficient :math:`D(\phi)` of :func:`degenerate_diffusion_table`
    (normalized to :math:`D(\tfrac12)=D_0`).

    Identical to :func:`solve_pde_2d` except for the diffusion operator, chosen
    by ``operator``:

    - ``"nondivergence"`` (default): :math:`\partial_t\phi = f(\phi) +
      D(\phi)\,\nabla^2\phi` (:func:`nondivergence_diffusion_2d`). This is the
      operator the mean-field lattice expansion of the asymmetric imitation
      rule actually derives (the branch-dependent RD expansion of the
      supplementary continuum-limit section), and therefore the canonical
      degenerate robustness check.
    - ``"divergence"``: the conservative form :math:`\partial_t\phi = f(\phi)
      + \nabla\cdot(D(\phi)\nabla\phi)` (:func:`divergence_diffusion_2d`),
      which adds the same-order term :math:`D'(\phi)|\nabla\phi|^2` that the
      lattice expansion does not produce; retained as a further variant.

    The explicit-Euler step is set from the CFL limit using the peak
    diffusivity over the field (the same bound governs both operators).
    ``D0`` replaces ``D`` as the reference diffusivity at the front midpoint;
    other parameters are as in :func:`solve_pde_2d`.
    """
    if operator not in ("nondivergence", "divergence"):
        raise ValueError(
            f"Unknown operator: {operator!r}. Use 'nondivergence' or 'divergence'."
        )
    phi = phi_initial.copy().astype(np.float64)
    sig_field = np.broadcast_to(sigma_field, phi.shape).astype(np.float64)
    unique_sigs = np.unique(sig_field)

    adv_tables: dict[float, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    D_tables: dict[float, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    D_max = 0.0
    for s_val in unique_sigs:
        adv_tables[float(s_val)] = _build_advantage_table(
            sigma=float(s_val), lambda_W=lambda_W, mode=mode,
            use_self_consistent_cost=True, C_exogenous=0.35,
        )
        pg, Dg = degenerate_diffusion_table(
            sigma=float(s_val), lambda_W=lambda_W, mode=mode, D0=D0,
        )
        D_tables[float(s_val)] = (pg, Dg)
        D_max = max(D_max, float(np.max(Dg)))

    # CFL-stable explicit step from the peak diffusivity over the field.
    if dt is None:
        dt = 0.4 * dx * dx / (4.0 * max(D_max, 1e-12))
    elif dt > dx * dx / (4.0 * max(D_max, 1e-12)):
        raise ValueError(
            f"Explicit dt={dt} exceeds the CFL stability limit "
            f"dx^2/(4 max D) = {dx * dx / (4.0 * max(D_max, 1e-12)):.6g}; "
            "pass a smaller dt or dt=None for the automatic step."
        )
    if n_record < 2:
        raise ValueError(f"n_record must be >= 2, got {n_record}")
    n_steps = int(np.ceil(t_max / dt))
    dt = t_max / n_steps
    record_every = max(1, n_steps // (n_record - 1))

    phi_history = [phi.copy()]
    t_recorded = [0.0]

    def _reaction(phi_field: NDArray[np.float64]) -> NDArray[np.float64]:
        out = np.zeros_like(phi_field)
        for s_val in unique_sigs:
            phi_grid, adv_grid = adv_tables[float(s_val)]
            mask = sig_field == s_val
            local_phi = phi_field[mask]
            local_phi_clipped = np.clip(local_phi, 1e-4, 1.0 - 1e-4)
            adv = np.interp(local_phi_clipped, phi_grid, adv_grid)
            out[mask] = local_phi * (1.0 - local_phi) * adv
        return out

    def _D_field(phi_field: NDArray[np.float64]) -> NDArray[np.float64]:
        out = np.zeros_like(phi_field)
        for s_val in unique_sigs:
            phi_grid, D_grid = D_tables[float(s_val)]
            mask = sig_field == s_val
            out[mask] = np.interp(
                np.clip(phi_field[mask], 1e-4, 1.0 - 1e-4), phi_grid, D_grid
            )
        return out

    _diffusion_op = (
        nondivergence_diffusion_2d if operator == "nondivergence"
        else divergence_diffusion_2d
    )
    for step in range(1, n_steps + 1):
        reaction = _reaction(phi)
        diff = _diffusion_op(phi, _D_field(phi), dx=dx, boundary=boundary)
        phi = phi + dt * (reaction + diff)
        phi = np.clip(phi, 0.0, 1.0)
        if step % record_every == 0 or step == n_steps:
            phi_history.append(phi.copy())
            t_recorded.append(step * dt)

    return PDEResult(
        t_array=np.asarray(t_recorded),
        phi_history=np.asarray(phi_history),
        sigma_field=sig_field,
        D=D0,
        dx=dx,
        dt=dt,
        lambda_W=lambda_W,
    )


# =====================================================================
# Analytical results: traveling wave velocity
# =====================================================================


def traveling_wave_velocity_dimensional_estimate(
    sigma: float,
    lambda_W: float = 0.68,
    D: float = 1.0,
    mode: str = "multiplicative",
    n_quad: int = 200,
) -> float:
    r"""Dimensional estimate of bistable RD wave velocity (NOT the AW result).

    Returns `v = sign(I) * sqrt(2 D |I|)` where `I = integral of f` from 0 to 1.
    This formula has the correct sign, the correct √D scaling, and the
    correct sign of `I` dependence, but the proportionality constant is
    an ad-hoc dimensional choice that does NOT match the Aronson-Weinberger
    formula for any bistable RD class.

    For the standard Nagumo cubic `f = phi(1-phi)(phi - alpha)` the
    Aronson-Weinberger result is

    .. math::
        v_{\mathrm{AW}} = \sqrt{D/2} \cdot (1 - 2\alpha)

    whereas this dimensional estimate gives

    .. math::
        v_{\mathrm{dim}} = \sqrt{2 D \cdot |1 - 2\alpha|/12}
        = \sqrt{D |1 - 2\alpha| / 6}

    which disagrees with AW by roughly 20-25% for typical alpha.

    The framework's actual predicted wave velocity is computed by direct
    numerical integration of the PDE, via
    :func:`traveling_wave_velocity_pde`. This dimensional estimate is
    retained only for legacy comparison purposes and is no longer the
    canonical prediction.

    Parameters
    ----------
    sigma, lambda_W, mode : as in reaction_term.
    D : float
        Diffusion coefficient.
    n_quad : int
        Quadrature points for the integral of f.

    Returns
    -------
    float
        Dimensional-estimate wave velocity.
    """
    integral = reaction_integral(
        sigma=sigma, lambda_W=lambda_W, mode=mode, n_quad=n_quad,
    )
    if integral == 0.0:
        return 0.0
    return float(np.sign(integral) * np.sqrt(2.0 * D * abs(integral)))


# Legacy alias for backward compatibility. Tests previously referenced
# `traveling_wave_velocity`; preserve that name as an alias for the
# dimensional estimate while the canonical entry point is now
# `traveling_wave_velocity_pde`.
traveling_wave_velocity = traveling_wave_velocity_dimensional_estimate


def traveling_wave_velocity_pde(
    sigma: float,
    lambda_W: float = 0.68,
    D: float = 1.0,
    mode: str = "multiplicative",
    grid_size: int = 240,
    t_max: float = 120.0,
    dx: float = 1.0,
) -> float:
    r"""Predicted traveling-wave velocity from direct PDE integration.

    .. note::
       The defaults ``grid_size=240``, ``t_max=120`` return the *asymptotic*
       front speed. The velocity is measured by regression over the second
       half of the run, and the grid is large enough that the front stays
       clear of the boundary over that window. A shorter run (the former
       ``t_max=25``) returns a transient that under-measures the speed by
       roughly 9\% (0.216 vs 0.234 sqrt(D) at the empirical anchor
       sigma = 0.8); the value is converged to three significant figures by
       ``t_max=120``.

    This is the framework's canonical wave-velocity prediction: it is
    the speed of the asymptotic traveling front in numerical integration
    of the bistable reaction-diffusion equation

    .. math::
        \frac{\partial \phi}{\partial t}
        = \phi (1-\phi)\,[w_{MB}(\phi) - w_{NB}(\phi)] + D \nabla^2 \phi.

    We initialize a 2D domain with the left half at phi = 1 and the
    right half at phi = 0, integrate to t = t_max, and measure the
    half-max contour position over the second half of the simulation
    via linear regression. Sign convention: positive velocity means the
    signaling state invades into the non-signaling state.

    Compared to the dimensional estimate
    :func:`traveling_wave_velocity_dimensional_estimate`, the PDE-measured
    velocity is the actual wave speed of the framework's spatial
    dynamics. There is no closed-form analytical approximation; the
    velocity is a direct output of the PDE.

    The heteroclinic-front existence required by the asymptotic-velocity
    variational identity (supplementary section "Spatial dynamics:
    variational identity, Allen-Cahn nucleus, and the formal droplet barrier")
    is guaranteed by Fife and McLeod (1977) for bistable reaction
    terms with simple zeros at 0, phi_star, 1 and sign changes
    consistent with bistability. The framework's f(phi) satisfies
    these hypotheses in the bistable interior of sigma, so the
    asymptotic front measured here is the unique (up to translation)
    traveling-wave solution.

    Parameters
    ----------
    sigma : float
        Environmental uncertainty.
    lambda_W : float
        Within-group social reward (default 0.68 empirical anchor).
    D : float
        Diffusion coefficient.
    mode : "multiplicative" or "mixed"
        Fitness composition.
    grid_size : int
        Lattice side length for the 2D PDE solver. Larger grids reduce
        boundary effects but are slower.
    t_max : float
        Integration time. Should be long enough for the front to
        propagate well past initial transients.
    dx : float
        Grid spacing.

    Returns
    -------
    float
        Asymptotic traveling-wave velocity, in lattice units per unit
        time. Positive: signaling state invades. Negative: signaling
        state retreats.
    """
    G = grid_size
    phi_init = np.zeros((G, G))
    phi_init[:, : G // 2] = 1.0
    res = solve_pde_2d(
        phi_initial=phi_init,
        t_max=t_max,
        sigma_field=sigma,
        lambda_W=lambda_W,
        D=D,
        dx=dx,
        n_record=21,
        mode=mode,
    )
    positions: list[float] = []
    times: list[float] = []
    for t_idx in range(len(res.t_array)):
        row = res.phi_history[t_idx, G // 2, :]
        above = np.where(row >= 0.5)[0]
        if len(above) == 0:
            continue
        i = int(above[-1])
        # Skip boundary contamination (front near domain edges)
        if i < 2 or i > G - 3:
            continue
        # Subgrid-interpolate the half-max crossing for accurate
        # asymptotic velocity (avoids integer-position quantization bias).
        if i + 1 < G and row[i] >= 0.5 and row[i + 1] < 0.5:
            frac = (row[i] - 0.5) / (row[i] - row[i + 1])
            pos = float(i) + float(frac)
        else:
            pos = float(i)
        positions.append(pos)
        times.append(float(res.t_array[t_idx]))
    if len(positions) < 3:
        # If the wave moves too slowly or too quickly to capture, return
        # a sign-only estimate from the integral. (Edge case for sigma
        # near the maintenance threshold.)
        integral = reaction_integral(
            sigma=sigma, lambda_W=lambda_W, mode=mode,
        )
        return 0.0 if integral == 0.0 else float(np.sign(integral) * 1e-3)
    # Linear regression over the last half of available snapshots, to
    # filter out the early-transient regime.
    half = len(positions) // 2
    t_arr = np.asarray(times[half:])
    x_arr = np.asarray(positions[half:])
    slope = float(np.polyfit(t_arr, x_arr, 1)[0])
    return slope


# =====================================================================
# Critical nucleus
# =====================================================================


def potential_W(
    phi: float | NDArray[np.float64],
    sigma: float,
    lambda_W: float = 0.68,
    mode: str = "multiplicative",
    n_quad: int = 200,
) -> float | NDArray[np.float64]:
    r"""The Allen-Cahn potential :math:`W(\phi) = -\int_0^{\phi} f(s)\, ds`.

    Defined so that :math:`-W'(\phi) = f(\phi)`, where f is the reaction
    term of the bistable PDE. Both phases (:math:`\phi = 0` and
    :math:`\phi = 1`) are local minima of W; the unstable saddle of f
    is a local maximum of W (the energy barrier between the phases).

    For bistable reaction-diffusion with the signaling phase preferred
    (positive reaction integral I), W(1) < W(0); their difference
    :math:`\Delta E = W(0) - W(1) = -I` is the bulk energy advantage
    of the signaling state.

    Parameters
    ----------
    phi : float or array
        Field value(s) in [0, 1].
    sigma, lambda_W, mode : as in reaction_term.
    n_quad : int
        Quadrature points for the cumulative integral.

    Returns
    -------
    float or array
        W(phi) value(s), with W(0) = 0 by convention.
    """
    phi_arr = np.atleast_1d(np.asarray(phi, dtype=np.float64))
    phi_full_grid = np.linspace(0.0, 1.0, n_quad + 1)
    f_vals = reaction_term(
        phi_full_grid, sigma=sigma, lambda_W=lambda_W, mode=mode,
    )
    dx = phi_full_grid[1] - phi_full_grid[0]
    # Cumulative integral of f from 0 to phi (trapezoidal)
    cumW_neg = np.zeros_like(phi_full_grid)
    for i in range(1, len(phi_full_grid)):
        cumW_neg[i] = cumW_neg[i - 1] + 0.5 * dx * (f_vals[i - 1] + f_vals[i])
    W_full = -cumW_neg
    out = np.interp(phi_arr, phi_full_grid, W_full)
    if np.isscalar(phi) or (hasattr(phi, "ndim") and phi.ndim == 0):
        return float(out[0])
    return out


def interfacial_tension(
    sigma: float,
    lambda_W: float = 0.68,
    D: float = 1.0,
    mode: str = "multiplicative",
    n_quad: int = 200,
    at_coexistence: bool = True,
) -> float:
    r"""Allen-Cahn interfacial tension for a 1D static front profile.

    .. math::
        \sigma_{\mathrm{int}} = \sqrt{D} \int_0^1 \sqrt{2 (W_M(\phi) - W_{M,\min})}\, d\phi

    where :math:`W_M` is the potential (:func:`potential_W`) evaluated AT
    THE MAXWELL POINT :math:`\sigma_M` (coexistence, where the two wells
    are equal and a stationary planar front exists). This is the standard
    surface tension of thin-wall nucleation theory: tension is a
    coexistence property, and the departure of the operating :math:`\sigma`
    from :math:`\sigma_M` enters the droplet analysis only through the bulk
    tilt :math:`\Delta E` (see :func:`critical_nucleus_radius_allen_cahn`).
    Integrating the tilted potential at the operating sigma instead is
    not a valid tension: there the integrand does not vanish at the
    higher-energy endpoint and the first integral of a stationary front
    cannot connect both phases. That alternative is retained via
    ``at_coexistence=False`` for comparison only.

    Parameters
    ----------
    sigma, lambda_W, D, mode : as in :func:`reaction_term`. ``sigma`` is
        used only when ``at_coexistence=False``; the coexistence tension
        depends on :math:`\sigma_M(\lambda_W)`, not on the operating sigma.
    n_quad : int
        Quadrature points.
    at_coexistence : bool
        If True (default), evaluate the potential at the Maxwell point.
        If False, legacy tilted-potential integral at the operating sigma.

    Returns
    -------
    float
        Interfacial tension :math:`\sigma_{\mathrm{int}}` in lattice
        units (depends on D in the convention used).
    """
    sigma_eval = sigma
    if at_coexistence:
        sigma_eval = maxwell_point_sigma(lambda_W=lambda_W, mode=mode)
    phi_grid = np.linspace(0.0, 1.0, n_quad + 1)
    W_vals = potential_W(
        phi_grid, sigma=sigma_eval, lambda_W=lambda_W, mode=mode, n_quad=n_quad,
    )
    W_min = float(np.min(W_vals))
    integrand = np.sqrt(np.maximum(0.0, 2.0 * (W_vals - W_min)))
    if hasattr(np, "trapezoid"):
        integral = float(np.trapezoid(integrand, phi_grid))
    else:
        dx = phi_grid[1] - phi_grid[0]
        integral = float(
            0.5 * dx * (integrand[0] + integrand[-1] + 2.0 * integrand[1:-1].sum())
        )
    return float(np.sqrt(D) * integral)


def critical_nucleus_radius_allen_cahn(
    sigma: float,
    lambda_W: float = 0.68,
    D: float = 1.0,
    mode: str = "multiplicative",
    n_quad: int = 200,
) -> float:
    r"""Critical nucleus radius from Allen-Cahn theory (thin-interface limit).

    For a 2D circular nucleus of the stable phase surrounded by the
    metastable phase, the critical radius is

    .. math::
        R_c^{\mathrm{AC}} = \frac{\sigma_{\mathrm{int}}}{|\Delta E|}

    where :math:`\sigma_{\mathrm{int}}` is the COEXISTENCE interfacial
    tension (:func:`interfacial_tension`, evaluated at the Maxwell point)
    and :math:`|\Delta E| = |W(0) - W(1)|` is the bulk energy difference
    between the two phases at the operating sigma (the thin-wall bulk
    tilt).

    This formula is asymptotically exact in the thin-interface limit
    (interface width small compared to R_c). For the framework's
    operating regime where the interface width
    :math:`\sim \sqrt{D / |f'(\phi^*)|}` and R_c are comparable, the
    thin-wall formula is not quantitatively reliable: with the coexistence
    tension it underestimates the converged PDE radius at the advancing
    anchor (~3.7 vs ~5.2 at sigma = 0.8). The framework's actual predicted
    R_c is computed via direct PDE simulation
    (:func:`critical_nucleus_radius_pde`).

    Parameters
    ----------
    sigma, lambda_W, D, mode : as in :func:`reaction_term`.
    n_quad : int
        Quadrature points.

    Returns
    -------
    float
        Allen-Cahn critical radius. NaN if reaction integral is zero
        (at the maintenance threshold; R_c diverges).
    """
    sigma_int = interfacial_tension(
        sigma=sigma, lambda_W=lambda_W, D=D, mode=mode, n_quad=n_quad,
    )
    Delta_E = abs(
        reaction_integral(
            sigma=sigma, lambda_W=lambda_W, mode=mode, n_quad=n_quad,
        )
    )
    if Delta_E < 1e-12:
        return float("nan")
    return float(sigma_int / Delta_E)


def critical_nucleus_radius_dimensional_estimate(
    sigma: float,
    lambda_W: float = 0.68,
    D: float = 1.0,
    mode: str = "multiplicative",
    n_quad: int = 200,
) -> float:
    r"""Front-thickness dimensional estimate of the critical nucleus radius.

    .. math::
        R_c^{\mathrm{dim}} = \sqrt{D / |f'(\phi^*)|}

    where :math:`\phi^*` is the unstable saddle of f. This is the
    correlation length of the front profile, not the actual critical
    nucleus radius. For asymmetric bistable RD in the thin-interface
    limit (:math:`R_c \gg` interface width), the Allen-Cahn formula
    :math:`R_c = \sigma_{\mathrm{int}} / |\Delta E|` is the correct
    expression. In the framework's empirical regime, where R_c and the
    interface width are comparable, neither formula is exact; PDE
    simulation gives the true R_c.

    Retained as a legacy comparison. The canonical entry point for
    framework predictions is :func:`critical_nucleus_radius_pde`.

    Parameters
    ----------
    sigma, lambda_W, D, mode : as in :func:`reaction_term`.
    n_quad : int
        Quadrature points (unused; reserved for full integral form).

    Returns
    -------
    float
        Dimensional estimate. NaN if no interior saddle exists.
    """
    # Locate phi_star (unstable interior root of f)
    phi_grid = np.linspace(1e-4, 1.0 - 1e-4, 1001)
    f_vals = np.asarray(
        [reaction_term(p, sigma=sigma, lambda_W=lambda_W, mode=mode)
         for p in phi_grid]
    )
    signs = np.sign(f_vals)
    sign_changes = np.where(np.diff(signs) != 0)[0]
    if len(sign_changes) == 0:
        return float("nan")
    idx = sign_changes[0]
    phi_star = 0.5 * (phi_grid[idx] + phi_grid[idx + 1])

    eps = 1e-4
    f_plus = reaction_term(
        phi_star + eps, sigma=sigma, lambda_W=lambda_W, mode=mode,
    )
    f_minus = reaction_term(
        phi_star - eps, sigma=sigma, lambda_W=lambda_W, mode=mode,
    )
    f_prime = (f_plus - f_minus) / (2 * eps)
    if abs(f_prime) < 1e-10:
        return float("inf")
    return float(np.sqrt(D / abs(f_prime)))


# Legacy alias: previous code called this `critical_nucleus_radius` and
# treated it as a dimensional estimate. Preserve the name for backward
# compatibility while the canonical entry point is now
# `critical_nucleus_radius_pde`.
critical_nucleus_radius = critical_nucleus_radius_dimensional_estimate


def critical_nucleus_radius_pde(
    sigma: float,
    lambda_W: float = 0.68,
    D: float = 1.0,
    mode: str = "multiplicative",
    R_search: tuple[float, float] = (1.0, 15.0),
    R_resolution: float = 0.5,
    grid_size: int = 60,
    t_max: float = 120.0,
    dx: float = 1.0,
) -> float:
    r"""Critical nucleus radius from direct PDE simulation (binary search).

    This is the framework's canonical R_c prediction: the radius
    threshold separating shrinking from growing circular nuclei in
    numerical integration of the bistable RD equation.

    Algorithm: for each candidate R in the search range, initialize a
    circular nucleus of radius R at the grid center, integrate the PDE
    to t_max, and check whether the area of the :math:`\phi > 0.5`
    region grew or shrank. Use bisection on R to find the threshold.

    Compared to the Allen-Cahn formula
    (:func:`critical_nucleus_radius_allen_cahn`), which is exact in the
    thin-interface limit, the PDE-measured R_c is the actual radius at
    the given parameters. In the framework's operating regime, the two
    can differ substantially because the interface width and R_c are
    comparable.

    Parameters
    ----------
    sigma : float
        Environmental uncertainty.
    lambda_W : float
        Within-group social reward.
    D : float
        Diffusion coefficient.
    mode : str
        Fitness composition.
    R_search : tuple of (R_min, R_max)
        Search range for R_c.
    R_resolution : float
        Bisection terminates when the search interval is smaller than
        this. Default 0.5 lattice units.
    grid_size : int
        Lattice side length. Should be larger than 4 * R_max to avoid
        boundary effects.
    t_max : float
        Simulation time for each candidate R.
    dx : float
        Grid spacing.

    Returns
    -------
    float
        Critical nucleus radius. NaN if no transition is found within
        the search range.
    """
    def _grows(R: float) -> bool:
        G = grid_size
        cy, cx = G // 2, G // 2
        ys, xs = np.indices((G, G))
        phi_init = np.zeros((G, G))
        mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= R ** 2
        phi_init[mask] = 1.0
        res = solve_pde_2d(
            phi_initial=phi_init,
            t_max=t_max,
            sigma_field=sigma,
            lambda_W=lambda_W,
            D=D,
            dx=dx,
            n_record=2,
            mode=mode,
        )
        final_area = int((res.phi_history[-1] > 0.5).sum())
        initial_area = int((phi_init > 0.5).sum())
        return final_area > initial_area

    R_lo, R_hi = R_search
    if _grows(R_lo):
        return float("nan")  # Even the smallest nucleus grows
    if not _grows(R_hi):
        return float("nan")  # Even the largest nucleus shrinks
    while R_hi - R_lo > R_resolution:
        R_mid = 0.5 * (R_lo + R_hi)
        if _grows(R_mid):
            R_hi = R_mid
        else:
            R_lo = R_mid
    return float(0.5 * (R_lo + R_hi))


def critical_nucleus_radius_pde_degenerate(
    sigma: float,
    lambda_W: float = 0.68,
    D0: float = 1.0,
    mode: str = "multiplicative",
    R_search: tuple[float, float] = (1.0, 15.0),
    R_resolution: float = 0.5,
    grid_size: int = 60,
    t_max: float = 120.0,
    dx: float = 1.0,
    operator: str = "nondivergence",
) -> float:
    r"""Critical nucleus radius under the *degenerate* diffusion :math:`D(\phi)`.

    Identical bisection to :func:`critical_nucleus_radius_pde`, but each
    candidate nucleus is evolved by :func:`solve_pde_2d_degenerate` with the
    state-dependent, boundary-vanishing :math:`D(\phi)` of the mean-field
    expansion (normalized to :math:`D(\tfrac12)=D_0`), under the diffusion
    operator selected by ``operator``: ``"nondivergence"`` (default) is the
    operator the lattice expansion derives, :math:`D(\phi)\nabla^2\phi`;
    ``"divergence"`` is the conservative variant
    :math:`\nabla\cdot(D\nabla\phi)`. A finite, positive return demonstrates
    that the critical-nucleus barrier (and hence the spatial test F1a)
    survives the degenerate diffusion, not only the constant-:math:`D`
    approximation used in the main text. Returns NaN if no transition is
    bracketed in ``R_search`` (every nucleus grows, or every nucleus shrinks).

    Parameters are as in :func:`critical_nucleus_radius_pde`, with ``D0`` the
    reference diffusivity at the front midpoint :math:`\phi=1/2`.
    """
    def _grows(R: float) -> bool:
        G = grid_size
        cy, cx = G // 2, G // 2
        ys, xs = np.indices((G, G))
        phi_init = np.zeros((G, G))
        mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= R ** 2
        phi_init[mask] = 1.0
        res = solve_pde_2d_degenerate(
            phi_initial=phi_init,
            t_max=t_max,
            sigma_field=sigma,
            lambda_W=lambda_W,
            D0=D0,
            dx=dx,
            n_record=2,
            mode=mode,
            operator=operator,
        )
        final_area = int((res.phi_history[-1] > 0.5).sum())
        initial_area = int((phi_init > 0.5).sum())
        return final_area > initial_area

    R_lo, R_hi = R_search
    if _grows(R_lo):
        return float("nan")  # Even the smallest nucleus grows
    if not _grows(R_hi):
        return float("nan")  # Even the largest nucleus shrinks
    while R_hi - R_lo > R_resolution:
        R_mid = 0.5 * (R_lo + R_hi)
        if _grows(R_mid):
            R_hi = R_mid
        else:
            R_lo = R_mid
    return float(0.5 * (R_lo + R_hi))


# =====================================================================
# Critical-droplet action (formal thin-wall energy barrier)
# =====================================================================


def critical_droplet_action(
    sigma: float,
    lambda_W: float = 0.68,
    D: float = 1.0,
    mode: str = "multiplicative",
    n_quad: int = 200,
) -> float:
    r"""Action of the critical 2D droplet (thin-interface limit).

    For a 2D circular nucleus of radius R, the energy functional is

    .. math::
        S(R) = 2\pi R \sigma_{\mathrm{int}} - \pi R^2 |\Delta E|

    where :math:`\sigma_{\mathrm{int}}` is the interfacial tension
    (:func:`interfacial_tension`) and :math:`|\Delta E|` is the bulk
    energy difference. The critical droplet sits at the saddle of S(R)
    given by :math:`R_c = \sigma_{\mathrm{int}} / |\Delta E|`, and the
    action at the saddle is

    .. math::
        S^* = S(R_c) = \pi \sigma_{\mathrm{int}}^2 / |\Delta E|

    This is a FORMAL thin-wall droplet free-energy barrier. A
    Freidlin-Wentzell large-deviations action for stochastic escape from
    the metastable :math:`\phi = 0` phase (cf. Freidlin & Wentzell 1998,
    Ch. 6) would additionally require a specified stochastic process and
    noise covariance, which the framework does not supply; no rate or
    likelihood claim is made from this quantity in the manuscript.

    Important caveat (fat-interface regime). Like the Allen-Cahn
    formula (:func:`critical_nucleus_radius_allen_cahn`), this formula
    is the thin-interface-limit result, valid only when the interface
    width :math:`\ell` is much smaller than the critical-nucleus radius
    :math:`R_c`. The framework's operating regime is *outside* that
    limit: at the empirical anchor (sigma in the bistable interior,
    lambda_W = 0.68, D = 1), :math:`\ell / R_c \approx 0.85`, so the
    sharp-interface assumption underlying :math:`S^* = \pi
    \sigma_{\mathrm{int}}^2 / |\Delta E|` is violated. In the
    fat-interface regime the actual action at the saddle of the PDE
    energy is smaller than the Allen-Cahn :math:`S^*`, sometimes by
    an order of magnitude. The Allen-Cahn :math:`S^*` therefore gives
    an upper bound on the corresponding barrier, but the specific
    quantitative value returned by this function is not quantitatively
    reliable. It is qualitatively informative for trends with sigma
    and for cross-channel comparisons, but headline numbers (e.g.,
    a specific numerical :math:`S^*`, which is sigma-dependent) should not be quoted as the framework's
    quantitative prediction. See the supplementary section "Spatial dynamics:
    variational identity, Allen-Cahn nucleus, and the formal droplet barrier"
    (Supplementary Information) for the manuscript-side discussion
    of the fat-interface validity boundary.

    Parameters
    ----------
    sigma, lambda_W, D, mode, n_quad : as in :func:`interfacial_tension`.

    Returns
    -------
    float
        Critical-droplet action :math:`S^*`, in dimensionless units.
        NaN if reaction integral is zero (at the maintenance threshold).
    """
    sigma_int = interfacial_tension(
        sigma=sigma, lambda_W=lambda_W, D=D, mode=mode, n_quad=n_quad,
    )
    Delta_E = abs(
        reaction_integral(
            sigma=sigma, lambda_W=lambda_W, mode=mode, n_quad=n_quad,
        )
    )
    if Delta_E < 1e-12:
        return float("nan")
    return float(np.pi * sigma_int * sigma_int / Delta_E)


def nucleation_rate_estimate(
    sigma: float,
    lambda_W: float = 0.68,
    D: float = 1.0,
    noise_temperature: float = 1.0,
    mode: str = "multiplicative",
    prefactor: float = 1.0,
) -> float:
    r"""Order-of-magnitude nucleation rate estimate (Kramers / Arrhenius form).

    .. math::
        \mathrm{rate} \approx A \exp\!\left(-\frac{S^*}{T_{\mathrm{eff}}}\right)

    where :math:`S^*` is the critical-droplet action
    (:func:`critical_droplet_action`), :math:`T_{\mathrm{eff}}` is the
    effective noise temperature (related to the per-region population
    fluctuation scale), and A is a Kramers prefactor depending on the
    curvature of the action functional at the saddle.

    For order-of-magnitude estimates at the framework's empirical
    anchor, take :math:`T_{\mathrm{eff}} = 1/N` where N is the
    per-region group population. The prefactor is set to 1 unless
    specified.

    The rate units depend on the noise-temperature convention. This
    function returns the dimensionless ratio :math:`\mathrm{rate} / A`
    in lattice-cell-per-time-step units; calibration to archaeological
    units (events per region per generation) requires specifying the
    lattice-spacing and time-step conventions in the empirical
    application (future empirical work).

    Important caveat (fat-interface regime). The
    framework's spatial dynamics operate outside the thin-interface
    limit: at the empirical anchor :math:`\ell / R_c \approx 0.85`,
    where :math:`\ell` is the front thickness and :math:`R_c` is the
    critical-nucleus radius. The thin-wall formulas used to construct
    :math:`S^*` (and hence this rate
    estimate) assume :math:`\ell \ll R_c`; the rate itself is an
    ILLUSTRATIVE, NON-CANONICAL utility (no stochastic process or noise
    covariance is specified, so no derived rate exists), and it is not
    used to support any manuscript claim, including any qualitative claim
    that nucleation is rare: without a noise model even the order of
    magnitude is undefined. They are qualitatively
    informative for trends in sigma and comparative ordering across
    parameter regimes of the FORMAL barrier, but support no rate
    conclusion, qualitative or quantitative: absent a noise model,
    figures such as :math:`\mathrm{rate} \sim \exp(-39)` per region
    per generation are formal outputs of this utility, not estimates
    of anything. The canonical nucleation-rate estimate for the
    framework is a direct numerical measurement on the stochastic
    lattice using rare-event sampling (forward-flux or transition-
    interface sampling), which is deferred to future empirical
    paper. See the supplementary section "Spatial dynamics: variational
    identity, Allen-Cahn nucleus, and the formal droplet barrier"
    (Supplementary Information) for the manuscript-side
    disclosure of this validity boundary.

    Parameters
    ----------
    sigma, lambda_W, D, mode : as in :func:`critical_droplet_action`.
    noise_temperature : float
        Effective noise temperature :math:`T_{\mathrm{eff}}`. Default 1.0
        gives the un-rescaled exp(-S*) factor; pass :math:`1/N` for a
        per-region-population calibration.
    prefactor : float
        Kramers prefactor A. Default 1.0.

    Returns
    -------
    float
        Estimated nucleation rate. Returns 0.0 if S* is infinite or
        NaN if S* cannot be computed.
    """
    S_star = critical_droplet_action(
        sigma=sigma, lambda_W=lambda_W, D=D, mode=mode,
    )
    if not np.isfinite(S_star):
        return float("nan")
    return float(prefactor * np.exp(-S_star / noise_temperature))


# =====================================================================
# Pinning condition
# =====================================================================


def pinning_at_discontinuity(
    sigma_left: float,
    sigma_right: float,
    lambda_W: float = 0.68,
    D: float = 1.0,
    mode: str = "multiplicative",
) -> dict[str, float | bool]:
    r"""Diagnose whether a traveling wave pins at a sigma discontinuity.

    At a spatial step in sigma, a bistable wave pins if the integral
    of f changes sign between the two sides: positive integral on one
    side and negative on the other means the front cannot propagate in
    either direction.

    Parameters
    ----------
    sigma_left, sigma_right : float
        Values of sigma on either side of the discontinuity.
    lambda_W : float
        Within-group social reward.
    D : float
        Diffusion coefficient (used for velocity scaling).
    mode : str
        Fitness composition.

    Returns
    -------
    dict with keys:
        v_left, v_right : traveling-wave velocities on each side
        integral_left, integral_right : reaction integrals
        pinned : True iff the velocities have opposite sign, so the
            wave cannot propagate in either direction
    """
    v_l = traveling_wave_velocity(
        sigma=sigma_left, lambda_W=lambda_W, D=D, mode=mode,
    )
    v_r = traveling_wave_velocity(
        sigma=sigma_right, lambda_W=lambda_W, D=D, mode=mode,
    )
    int_l = reaction_integral(
        sigma=sigma_left, lambda_W=lambda_W, mode=mode,
    )
    int_r = reaction_integral(
        sigma=sigma_right, lambda_W=lambda_W, mode=mode,
    )
    return {
        "v_left": v_l,
        "v_right": v_r,
        "integral_left": int_l,
        "integral_right": int_r,
        "pinned": bool(v_l * v_r < 0),
    }


# =====================================================================
# Stochastic lattice (qualitative counterpart of the SI's discrete model)
# =====================================================================


def stochastic_lattice_simulation(
    grid_size: int,
    t_max: int,
    sigma_field: float | NDArray[np.float64],
    lambda_W: float = 0.68,
    rate_scale: float = 1.0,
    initial_nucleus_radius: float | None = None,
    neighborhood_radius: int = 1,
    seed: int | None = None,
    mode: str = "multiplicative",
    n_record: int = 5,
    origination_rate: float = 0.0,
) -> dict[str, Any]:
    r"""Stochastic lattice implementing the SI's discrete imitation model.

    Each lattice site holds a binary signaling state. At each timestep,
    each site computes its local phi over its von Neumann neighborhood
    (the SI's nearest-neighbor kernel) and flips by the payoff-biased
    imitation rule the SI displays: adoption at rate
    phi_local * max(0, dW), abandonment at rate
    (1 - phi_local) * max(0, -dW), whose mean-field law is the replicator
    reaction phi(1 - phi) dW. Implementation choices not in the displayed
    model: phi_local is binned (17 levels) for speed, flips are synchronous,
    and the payoff is clipped away from phi in {0, 1} when evaluated. This
    is a QUALITATIVE stochastic counterpart of the discrete model, used for
    the origination-flux robustness check; it is not used to validate the
    constant-D PDE, and a quantitative lattice-to-PDE convergence experiment
    (requiring the appropriate rate scaling) is deferred. The full ABM
    treatment is deferred to a follow-up paper.

    With ``origination_rate = 0`` (default) the transition rates vanish at
    ``phi_local = 0``, so the all-non-building state is exactly absorbing:
    the deterministic model omits the within-group origination flux by
    construction. Because origination is individually favored within any
    single group (within-group positional selection, beta_0 > 0), the
    physically complete model adds a small spontaneous 0 -> 1 rate. Isolated
    originators sit at a group-level fitness disadvantage and revert via
    payoff-biased imitation, so origination events supply the physical
    attempts from which supercritical clusters must assemble rather than
    eroding bistability; :func:`origination_robustness_sweep` measures the rate
    above which the non-building state erodes. See the SI continuum-limit
    section (origination-flux paragraph).

    Parameters
    ----------
    grid_size : int
        Lattice side length (square grid).
    t_max : int
        Number of timesteps.
    sigma_field : float or 2D array
        Environmental uncertainty field.
    lambda_W : float
        Within-group social reward.
    rate_scale : float
        Overall rate factor for the per-step flip probability.
    initial_nucleus_radius : float or None
        If supplied, initialize with a circular nucleus of signalers
        of this radius at the grid center. If None, initialize with
        a uniform random configuration with phi=0.05.
    neighborhood_radius : int
        Von Neumann neighborhood radius (|dy| + |dx| <= r; 1 = 4 neighbors,
        the SI's nearest-neighbor kernel).
    seed : int or None
        Random seed.
    mode : "multiplicative" or "mixed"
        Fitness composition.
    n_record : int
        Number of snapshots to record.
    origination_rate : float
        Per-site per-step spontaneous 0 -> 1 probability representing
        within-group origination of the building practice, independent of
        neighbors' states. Default 0.0 (the deterministic imitation-only
        model). Must be in [0, 1].

    Returns
    -------
    dict with keys: snapshots, t_recorded, sigma_field, grid_size.
    """
    if not 0.0 <= origination_rate <= 1.0:
        raise ValueError(
            f"origination_rate must be in [0, 1], got {origination_rate}"
        )
    rng = np.random.default_rng(seed)
    G = grid_size

    sig_arr = np.broadcast_to(sigma_field, (G, G)).astype(np.float64)

    # Initial state
    state = np.zeros((G, G), dtype=np.int8)
    if initial_nucleus_radius is None:
        state = (rng.random((G, G)) < 0.05).astype(np.int8)
    else:
        cy, cx = G // 2, G // 2
        ys, xs = np.indices((G, G))
        mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= initial_nucleus_radius ** 2
        state[mask] = 1

    snapshots = [state.copy()]
    t_recorded = [0]
    record_every = max(1, t_max // (n_record - 1))

    # Precompute neighborhood offsets: von Neumann kernel (|dy| + |dx| <= r,
    # excluding the origin), matching the SI discrete model's nearest-neighbor
    # set (size 4 at r = 1). A Moore neighborhood is not the kernel the
    # derivation states.
    r = neighborhood_radius
    offsets = [
        (dy, dx)
        for dy in range(-r, r + 1)
        for dx in range(-r, r + 1)
        if (abs(dy) + abs(dx) <= r) and not (dy == 0 and dx == 0)
    ]
    n_neighbors = len(offsets)

    for step in range(1, t_max + 1):
        # Compute local phi for each cell from its neighborhood
        phi_local = np.zeros((G, G), dtype=np.float64)
        for dy, dx in offsets:
            phi_local += np.roll(state, shift=(dy, dx), axis=(0, 1))
        phi_local = phi_local / n_neighbors

        # Per-cell fitness advantage at local sigma and phi
        # For efficiency: precompute unique (sigma, phi) bin -> advantage
        advantage = np.zeros_like(phi_local)
        flat_sigma = sig_arr.ravel()
        flat_phi = phi_local.ravel()
        # Bin phi to reduce calls to rare_builder_fitness_advantage
        phi_bins = np.round(flat_phi * 16) / 16  # 17 bins
        unique_pairs = set(zip(flat_sigma.tolist(), phi_bins.tolist()))
        adv_lookup = {}
        for sig_val, phi_val in unique_pairs:
            phi_val_clipped = float(np.clip(phi_val, 1e-3, 1.0 - 1e-3))
            res = rare_builder_fitness_advantage(
                sigma=float(sig_val), lambda_W=lambda_W,
                frac_signalers=phi_val_clipped, mode=mode,
            )
            adv_lookup[(sig_val, phi_val)] = res["advantage"]

        flat_adv = np.array(
            [adv_lookup[(float(s), float(p))] for s, p in zip(flat_sigma, phi_bins)]
        )
        advantage = flat_adv.reshape(G, G)

        # Per-step flip probabilities implementing the SI discrete model's
        # imitation-replicator rule exactly:
        #   P(0->1) = rate * phi_local * max(0, +advantage)   (must meet a
        #             builder to copy; payoff-biased imitation)
        #   P(1->0) = rate * (1 - phi_local) * max(0, -advantage)
        # plus the spontaneous within-group origination flux on 0->1
        # (neighbor-independent). The phi_local / (1 - phi_local) imitation
        # prefactors are essential: omitting them implements a different
        # reaction law at leading order.
        flip_to_1 = (
            rate_scale * phi_local * np.maximum(0.0, advantage)
            + origination_rate
        ) * (1 - state)
        flip_to_0 = (
            rate_scale * (1.0 - phi_local) * np.maximum(0.0, -advantage)
        ) * state
        # Clip to valid probability range
        flip_to_1 = np.clip(flip_to_1, 0.0, 1.0)
        flip_to_0 = np.clip(flip_to_0, 0.0, 1.0)

        rand = rng.random((G, G))
        new_state = state.copy()
        # Adoption
        adoption_mask = (state == 0) & (rand < flip_to_1)
        new_state[adoption_mask] = 1
        # Abandonment
        abandon_mask = (state == 1) & (rand < flip_to_0)
        new_state[abandon_mask] = 0
        state = new_state

        if step % record_every == 0 or step == t_max:
            snapshots.append(state.copy())
            t_recorded.append(step)

    return {
        "snapshots": snapshots,
        "t_recorded": t_recorded,
        "sigma_field": sig_arr,
        "grid_size": grid_size,
    }


def origination_robustness_sweep(
    rates: tuple[float, ...] | list[float],
    sigma: float,
    grid_size: int = 64,
    t_max: int = 200,
    lambda_W: float = 0.68,
    rate_scale: float = 2.0,
    seed: int = 0,
    n_seeds: int = 3,
    mode: str = "multiplicative",
) -> dict[float, dict[str, Any]]:
    r"""Robustness of the non-building state to a spontaneous origination flux.

    For each origination rate, runs :func:`stochastic_lattice_simulation`
    from the all-non-building state (radius-0 nucleus, i.e. a single seed
    site that immediately reverts) and records the long-run builder fraction
    phi. Bistability is robust to a given rate when the long-run phi stays
    near zero: isolated originators revert via payoff-biased imitation
    before assembling a supercritical cluster (the Allen-Cahn critical
    nucleus). The erosion rate is the smallest rate at which phi escapes
    toward 1 within the horizon, i.e. where origination outruns reversion
    and supercritical clusters form faster than they dissolve.

    This is the numerical check backing the SI continuum-limit
    origination-flux paragraph: the
    deterministic RD equation omits the origination flux by construction,
    so its phi = 0 state is exactly stable; with the flux restored, the
    state is metastable, and the critical-nucleus barrier survives
    realistic origination rates.

    Parameters
    ----------
    rates : sequence of float
        Origination rates (per site per step) to test.
    sigma : float
        Environmental uncertainty (uniform field). Choose between the
        maintenance threshold and above the Maxwell point to probe the
        metastable and nucleation regimes.
    grid_size, t_max, lambda_W, rate_scale, mode :
        Passed to :func:`stochastic_lattice_simulation`.
    seed : int
        Base random seed; run j uses seed + j.
    n_seeds : int
        Independent replicates per rate; phi_final is their mean.

    Returns
    -------
    dict mapping rate -> dict with keys:
        'phi_final' : mean long-run builder fraction across replicates
        'phi_final_per_seed' : list of per-replicate values
        'phi_trajectory' : per-snapshot mean phi of the first replicate
    """
    out: dict[float, dict[str, Any]] = {}
    for rate in rates:
        finals: list[float] = []
        traj: list[float] = []
        for j in range(n_seeds):
            res = stochastic_lattice_simulation(
                grid_size=grid_size, t_max=t_max, sigma_field=sigma,
                lambda_W=lambda_W, rate_scale=rate_scale,
                initial_nucleus_radius=0.0, seed=seed + j, mode=mode,
                origination_rate=float(rate),
            )
            finals.append(float(res["snapshots"][-1].mean()))
            if j == 0:
                traj = [float(s.mean()) for s in res["snapshots"]]
        out[float(rate)] = {
            "phi_final": float(np.mean(finals)),
            "phi_final_per_seed": finals,
            "phi_trajectory": traj,
        }
    return out
