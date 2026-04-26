"""
Smooth-prior branch (CROWN-Inpaint section 6.3).

Two operators:
  * biharmonic_init(y, M)  -- one-time global biharmonic completion (Chan-Shen
    2002), used to initialise u^0.  Provably the unique minimiser of bending
    energy with Dirichlet boundary conditions.
  * harmonic_relax(u, M, K_s)  -- K_s steps of in-place Jacobi relaxation of
    the harmonic equation (4-neighbour mean) inside the hole, observed pixels
    held fixed.  Cheap inner-loop smooth update for outer iterations after the
    first.

Both operators preserve observed pixels exactly: u_i = y_i for i in Omega.
"""

from __future__ import annotations

import numpy as np
from skimage.restoration import inpaint_biharmonic


def biharmonic_init(y: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    One-time biharmonic completion of the hole.

    Parameters
    ----------
    y : (H, W) float64
        Observed image with arbitrary placeholder values inside the hole.
    M : (H, W) float64
        Binary mask, 1 = observed, 0 = missing.

    Returns
    -------
    u0 : (H, W) float64
        Image with hole filled by biharmonic interpolation, clipped to [0, 1].
        Observed pixels are preserved exactly (skimage.inpaint_biharmonic
        modifies only masked pixels; we additionally re-project for safety).
    """
    if y.shape != M.shape:
        raise ValueError(f"shape mismatch: y={y.shape}, M={M.shape}")

    # skimage convention: True = pixel to inpaint
    bh_mask = (M == 0).astype(bool)
    u0 = inpaint_biharmonic(y, bh_mask)
    u0 = np.clip(u0, 0.0, 1.0)

    # Re-project observed pixels exactly (skimage already preserves them, but
    # the np.clip above could in principle modify a value that was outside
    # [0,1].  In practice y is in [0,1] so this is a no-op; kept for safety).
    u0 = np.where(M == 1, y, u0)
    return u0


def harmonic_relax(
    u: np.ndarray,
    y: np.ndarray,
    M: np.ndarray,
    K_s: int = 10,
) -> np.ndarray:
    """
    K_s Jacobi iterations of harmonic relaxation on the hole.

    For each missing pixel i (M_i = 0):
        u^(l+1)(i) = mean of 4-neighbours of u^(l) at i.
    For observed pixels (M_i = 1):
        u^(l+1)(i) = y_i  (held fixed).

    Boundary pixels of the image use only the in-image neighbours (i.e. no
    wrap-around).  Implemented vectorially via padding + slicing.

    Parameters
    ----------
    u : (H, W) float64
        Current image; hole pixels carry the current estimate.
    y : (H, W) float64
        Original observed image (used to enforce u_i = y_i on Omega).
    M : (H, W) float64
        Binary mask (1 observed, 0 missing).
    K_s : int
        Number of Jacobi iterations.

    Returns
    -------
    u_smooth : (H, W) float64, clipped to [0, 1]
        Updated image after K_s relaxation steps.
    """
    if u.shape != y.shape or u.shape != M.shape:
        raise ValueError(
            f"shape mismatch: u={u.shape}, y={y.shape}, M={M.shape}"
        )
    if K_s < 0:
        raise ValueError(f"K_s must be >= 0, got {K_s}")

    u_curr = u.copy()
    H, W = u_curr.shape
    obs = (M == 1)

    # Always start from observed pixels carrying their true values.
    u_curr = np.where(obs, y, u_curr)

    for _ in range(K_s):
        # 4-neighbour mean using zero-padding boundary.  For interior pixels
        # the padded neighbours are real image values; on the image border we
        # treat outside as zero AND we count only the in-image neighbours so
        # the mean is unbiased.
        up    = np.zeros_like(u_curr); up[1:, :]   = u_curr[:-1, :]
        dn    = np.zeros_like(u_curr); dn[:-1, :]  = u_curr[1:, :]
        lf    = np.zeros_like(u_curr); lf[:, 1:]   = u_curr[:, :-1]
        rt    = np.zeros_like(u_curr); rt[:, :-1]  = u_curr[:, 1:]

        n_up    = np.zeros_like(u_curr); n_up[1:, :]   = 1.0
        n_dn    = np.zeros_like(u_curr); n_dn[:-1, :]  = 1.0
        n_lf    = np.zeros_like(u_curr); n_lf[:, 1:]   = 1.0
        n_rt    = np.zeros_like(u_curr); n_rt[:, :-1]  = 1.0

        nbr_sum   = up + dn + lf + rt
        nbr_count = n_up + n_dn + n_lf + n_rt
        # nbr_count is in {2, 3, 4}; never zero on a >=2x2 image, but guard.
        safe_n    = np.where(nbr_count < 0.5, 1.0, nbr_count)
        u_new     = nbr_sum / safe_n

        # Hold observed pixels fixed
        u_curr = np.where(obs, y, u_new)

    return np.clip(u_curr, 0.0, 1.0)
