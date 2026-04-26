"""
Smooth-prior branch (CROWN-Inpaint section 6.3) -- corrected implementation.

The specification (Section 6.3.1) describes K_s steps of *harmonic* Jacobi
relaxation as the inner-loop smoother.  Section 8.2 separately claims that
in the smooth-only limit (r identically 0) the algorithm "cannot get
strictly worse than biharmonic by more than the fusion-induced perturbation."

That guarantee is FALSE for harmonic Jacobi.  Concretely, on a quadratic
profile  u(i, j) = i^2  one Jacobi step of the 4-neighbour mean produces

    u_new(i, j) = ((i - 1)^2 + (i + 1)^2 + 2 i^2) / 4 = i^2 + 1/2

so the iterate drifts away from the biharmonic completion by 0.5 per step.
Repeated harmonic relaxation converges to the *harmonic* completion of the
boundary data, which differs from the biharmonic completion whenever the
boundary is not affine.

The fix is the BIHARMONIC Jacobi update (5 x 5 stencil for `nabla^4 u = 0`):

    u(i, j) = (1/20) [
                  8 (N + S + E + W)
                - 2 (NE + NW + SE + SW)
                -    (NN + SS + EE + WW)
              ]

where N, S, E, W are the 4-neighbours, NE/NW/SE/SW are the diagonal
neighbours, and NN/SS/EE/WW are the 2-step axial neighbours (the discrete
Laplacian-of-Laplacian stencil).  Sanity checks:

    constant         -> u_new = u   (kernel of constants, sum of stencil = 0)
    affine           -> u_new = u   (analytical verification)
    quadratic i^2    -> u_new = i^2 (analytical verification)

so this Jacobi iteration has biharmonic-completion as its fixed point and
restores the no-regression guarantee from Section 8.2.

Biharmonic Jacobi is ~3x more expensive per step than harmonic Jacobi
(thirteen-point vs. five-point stencil), but the inner cost is still
trivial compared to the sparse-coding step.

The function `harmonic_relax` is kept as a thin alias for callers that
want the spec-literal harmonic update (e.g. ablation studies); the
default exported name is `biharmonic_relax`.
"""

from __future__ import annotations

import numpy as np
from skimage.restoration import inpaint_biharmonic


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def biharmonic_init(y: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    One-time biharmonic completion of the hole (Chan-Shen 2002).

    Returns the unique minimiser of bending energy with Dirichlet boundary
    conditions equal to y on Omega.  Observed pixels are preserved exactly.
    """
    if y.shape != M.shape:
        raise ValueError(f"shape mismatch: y={y.shape}, M={M.shape}")

    bh_mask = (M == 0).astype(bool)
    u0 = inpaint_biharmonic(y, bh_mask)
    u0 = np.clip(u0, 0.0, 1.0)
    u0 = np.where(M == 1, y, u0)
    return u0


# ---------------------------------------------------------------------------
# Inner-loop smoother: BIHARMONIC Jacobi relaxation
# ---------------------------------------------------------------------------

def biharmonic_relax(
    u: np.ndarray,
    y: np.ndarray,
    M: np.ndarray,
    K_s: int = 10,
    omega: float = 0.5,
) -> np.ndarray:
    """
    K_s damped-Jacobi iterations of the discrete biharmonic equation on the hole.

    Update rule
    -----------
    For each missing pixel i (M_i = 0), the undamped Jacobi step is

        v^(l+1)(i) = (1/20) [
                       8 (N + S + E + W)
                     - 2 (NE + NW + SE + SW)
                     -    (NN + SS + EE + WW)
                     ]

    and the damped iterate is

        u^(l+1)(i) = (1 - omega) u^(l)(i) + omega * v^(l+1)(i).

    Observed pixels (M_i = 1) are held fixed at y_i every step.  Image-
    border neighbours are obtained by reflection padding.

    Why damped Jacobi (and not plain Jacobi)
    ----------------------------------------
    At the highest-frequency mode  (kx, ky) = (pi, pi)  the discrete
    biharmonic operator has Fourier symbol

        20 - 16(cos kx + cos ky) + 8 cos kx cos ky + 2(cos 2kx + cos 2ky)
        = 20 - 16(-2) + 8(1) + 2(2) = 64.

    The plain Jacobi iteration matrix's spectral radius therefore has the
    bound  |1 - 64 / 20| = 2.2  which is greater than 1: high-frequency
    modes are amplified each iteration and the iterate diverges in the
    K_s -> infinity limit.  Damped Jacobi with omega in (0, 0.625) is
    unconditionally stable; we default to omega = 0.5, balancing
    convergence speed against the safety margin.

    Properties
    ----------
    * Fixed point on quadratic and affine profiles (verified analytically).
    * Observed pixels bit-exactly equal y at every step (hard Dirichlet).
    * Converges to the biharmonic completion of y on Omega^c as K_s -> inf
      (no high-frequency instability), recovering the no-regression
      guarantee of Section 8.2.

    Parameters
    ----------
    u : (H, W) float64    Current image.
    y : (H, W) float64    Observed image (Dirichlet target on Omega).
    M : (H, W) float64    Binary mask (1 observed, 0 missing).
    K_s : int             Number of iterations (>= 0).
    omega : float         Damping parameter; convergent for omega in (0, 0.625).

    Returns
    -------
    u_smooth : (H, W) float64    Updated image, clipped to [0, 1].
    """
    if u.shape != y.shape or u.shape != M.shape:
        raise ValueError(
            f"shape mismatch: u={u.shape}, y={y.shape}, M={M.shape}"
        )
    if K_s < 0:
        raise ValueError(f"K_s must be >= 0, got {K_s}")
    if not (0.0 < omega < 0.625):
        raise ValueError(
            f"omega must be in (0, 0.625) for damped-Jacobi stability; got {omega}"
        )
    if K_s == 0:
        return np.clip(np.where(M == 1, y, u), 0.0, 1.0)

    obs    = (M == 1)
    u_curr = np.where(obs, y, u).astype(np.float64, copy=True)

    for _ in range(K_s):
        # Pad by 2 with reflection so 2-step neighbours are defined at borders.
        up = np.pad(u_curr, 2, mode="reflect")            # (H+4, W+4)

        # Five-point neighbour samples (centre is up[2:-2, 2:-2])
        N  = up[1:-3, 2:-2]
        S  = up[3:-1, 2:-2]
        E  = up[2:-2, 3:-1]
        W  = up[2:-2, 1:-3]

        NE = up[1:-3, 3:-1]
        NW = up[1:-3, 1:-3]
        SE = up[3:-1, 3:-1]
        SW = up[3:-1, 1:-3]

        NN = up[0:-4, 2:-2]
        SS = up[4:,   2:-2]
        EE = up[2:-2, 4:]
        WW = up[2:-2, 0:-4]

        # Undamped Jacobi candidate value
        v = (1.0 / 20.0) * (
              8.0 * (N + S + E + W)
            - 2.0 * (NE + NW + SE + SW)
            -        (NN + SS + EE + WW)
        )

        # Damped update on hole; observed pixels stay at y.
        u_new  = (1.0 - omega) * u_curr + omega * v
        u_curr = np.where(obs, y, u_new)

    return np.clip(u_curr, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Optional spec-literal harmonic Jacobi (for ablation studies)
# ---------------------------------------------------------------------------

def harmonic_relax(
    u: np.ndarray,
    y: np.ndarray,
    M: np.ndarray,
    K_s: int = 10,
) -> np.ndarray:
    """
    Spec-literal 4-neighbour harmonic Jacobi (Section 6.3.1).

    Provided for ablation comparison; CROWN-Inpaint defaults to
    `biharmonic_relax` because the harmonic Jacobi limit is the harmonic
    completion, not the biharmonic completion (see module docstring).

    Note: the no-regression guarantee in spec Section 8.2 fails when this
    operator is used in the smooth-only limit on non-affine boundary data.
    """
    if u.shape != y.shape or u.shape != M.shape:
        raise ValueError(
            f"shape mismatch: u={u.shape}, y={y.shape}, M={M.shape}"
        )
    if K_s < 0:
        raise ValueError(f"K_s must be >= 0, got {K_s}")
    obs    = (M == 1)
    u_curr = np.where(obs, y, u).astype(np.float64, copy=True)

    for _ in range(K_s):
        up = np.zeros_like(u_curr); up[1:, :]   = u_curr[:-1, :]
        dn = np.zeros_like(u_curr); dn[:-1, :]  = u_curr[1:, :]
        lf = np.zeros_like(u_curr); lf[:, 1:]   = u_curr[:, :-1]
        rt = np.zeros_like(u_curr); rt[:, :-1]  = u_curr[:, 1:]
        n_up = np.zeros_like(u_curr); n_up[1:, :]   = 1.0
        n_dn = np.zeros_like(u_curr); n_dn[:-1, :]  = 1.0
        n_lf = np.zeros_like(u_curr); n_lf[:, 1:]   = 1.0
        n_rt = np.zeros_like(u_curr); n_rt[:, :-1]  = 1.0
        nbr_sum   = up + dn + lf + rt
        nbr_count = n_up + n_dn + n_lf + n_rt
        safe_n    = np.where(nbr_count < 0.5, 1.0, nbr_count)
        u_new     = nbr_sum / safe_n
        u_curr    = np.where(obs, y, u_new)

    return np.clip(u_curr, 0.0, 1.0)
