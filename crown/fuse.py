"""
Regime-aware fusion + hard observation projection (CROWN-Inpaint section 6.7).

    bar_u       = (1 - r) . u_smooth + r . u_sparse        # convex combo
    u_{t+1}     = M . y + (1 - M) . bar_u                  # hard Dirichlet

Properties (spec section 6.7):
1. Pixel-wise convex combination -- if both branches are clipped to [0, 1]
   then bar_u in [0, 1].
2. Hard projection bit-exactly preserves observed pixels at every iteration.
3. The smooth and sparse branches never compete over observed pixels.
"""

from __future__ import annotations

import numpy as np


def fuse_and_project(
    u_smooth: np.ndarray,
    u_sparse: np.ndarray,
    r: np.ndarray,
    M: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Combine the smooth and sparse branches according to the regime map and
    enforce the hard observation constraint.

    Parameters
    ----------
    u_smooth : (H, W) float64       Output of the smooth-prior branch.
    u_sparse : (H, W) float64       Output of the sparse-coding branch.
    r : (H, W) float64
        Regime map in [0, 1].  r = 1 selects the sparse branch (textured),
        r = 0 selects the smooth branch (low texture).
    M : (H, W) float64              Binary mask (1 = observed, 0 = missing).
    y : (H, W) float64              Observed image.

    Returns
    -------
    u_new : (H, W) float64    Fused, projected image, clipped to [0, 1].
    """
    if not (u_smooth.shape == u_sparse.shape == r.shape == M.shape == y.shape):
        raise ValueError(
            f"shape mismatch: u_smooth={u_smooth.shape}, u_sparse={u_sparse.shape},"
            f" r={r.shape}, M={M.shape}, y={y.shape}"
        )
    # Be defensive about r being out of range -- the upstream regime map is
    # already clipped, but a programming error here would silently move us
    # outside the convex hull.
    r_clip   = np.clip(r, 0.0, 1.0)
    bar_u    = (1.0 - r_clip) * u_smooth + r_clip * u_sparse
    bar_u    = np.clip(bar_u, 0.0, 1.0)
    u_new    = np.where(M == 1, y, bar_u)
    return u_new
