"""
Confidence map (CROWN-Inpaint section 6.1).

For every hole pixel we estimate a scalar c in [0, 1] saying how much the
sparse-coding step should *trust* the current image estimate at that pixel.

The map is the product of three factors:

    c^var (i)    = exp( -v(i)  / tau_v^2 )       overlap-variance term
    c^geom(i)    = exp( -d(i)  / rho     )       boundary-distance term
    c^sched      = min( 1, c0 + beta * t )       iteration schedule

For observed pixels (M_i = 1) we set c_i = 1 by definition: the hard
projection in the outer loop will preserve those values regardless, so
the sparse coder may trust them fully.

NOTE: the spec is consistent with the operational reading that v(i) is
computed from the *previous* iteration's alpha (the most recent set of
patch reconstructions available before the current sparse-coding step).
Iteration 1 has no previous alpha, so we pass `var=None` and skip the
variance factor for that iteration.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from sklearn.feature_extraction.image import reconstruct_from_patches_2d


# ---------------------------------------------------------------------------
# Per-pixel statistics from current sparse codes
# ---------------------------------------------------------------------------

def overlap_mean_and_variance(
    D: np.ndarray,
    Alpha: np.ndarray,
    image_shape: tuple,
    patch_shape: tuple = (8, 8),
) -> tuple:
    """
    For each image pixel, compute:

        mean(i) = (1 / |P_i|) * sum_{p in P_i}    (D alpha_p)[local(i, p)]
        var(i)  = (1 / |P_i|) * sum_{p in P_i}    (D alpha_p)[local(i, p)]^2  -  mean(i)^2

    where P_i is the set of overlapping patches that cover pixel i, and
    local(i, p) is the position of i within patch p.

    The per-pixel mean is exactly what `reconstruct_from_patches_2d` returns
    (it averages overlapping patches).  We obtain E[X^2] in the same way by
    feeding the squared patches into `reconstruct_from_patches_2d`, then
    Var = E[X^2] - (E[X])^2 with the usual numerical floor at zero.

    Parameters
    ----------
    D : (n, K) float64      Dictionary.
    Alpha : (K, N) float64  Sparse codes for every overlapping patch.
    image_shape : (H, W)    Target image shape.
    patch_shape : (p, p)    Spatial extent of each patch.

    Returns
    -------
    mean_img : (H, W) float64    Per-pixel mean of patch reconstructions.
    var_img  : (H, W) float64    Per-pixel variance, clipped to >=0.
    """
    N = Alpha.shape[1]
    n = D.shape[0]
    if n != patch_shape[0] * patch_shape[1]:
        raise ValueError(
            f"patch_shape {patch_shape} inconsistent with dictionary dim {n}"
        )

    patches_flat = D @ Alpha                                    # (n, N)
    patches_T    = patches_flat.T                                # (N, n)
    patches_2d   = patches_T.reshape(N, patch_shape[0], patch_shape[1])

    mean_img = reconstruct_from_patches_2d(patches_2d, image_shape)
    e_x2_img = reconstruct_from_patches_2d(patches_2d ** 2, image_shape)
    var_img  = np.clip(e_x2_img - mean_img ** 2, 0.0, None)

    return mean_img, var_img


# ---------------------------------------------------------------------------
# Confidence map
# ---------------------------------------------------------------------------

def compute_confidence_map(
    M: np.ndarray,
    t: int,
    var: np.ndarray = None,
    c0: float = 0.05,
    beta: float = 0.15,
    rho: float = 4.0,
    tau_v: float = None,
) -> np.ndarray:
    """
    Build the confidence map c in [0, 1]^{H x W} for iteration t.

    Parameters
    ----------
    M : (H, W) float64
        Binary mask (1 = observed, 0 = missing).
    t : int
        Outer-iteration index (1-based).  Used only by the schedule term.
    var : (H, W) float64 or None
        Per-pixel overlap variance from the previous iteration's sparse codes.
        If None, the variance factor is skipped (c_var := 1 everywhere).
    c0 : float, default 0.05
        Initial schedule value.  Lower => more conservative early on.
    beta : float, default 0.15
        Schedule growth rate per outer iteration.
    rho : float, default 4.0 (= patch radius for p = 8)
        Length scale of the boundary-distance decay.  Larger => trust
        deeper into the hole sooner.
    tau_v : float or None
        Variance scale.  If None and `var` is provided, an adaptive value
        is computed as the median of sqrt(var) over the hole; this auto-
        adjusts to the magnitude of patch disagreement at this iteration.

    Returns
    -------
    c : (H, W) float64    Confidence in [0, 1].

    Properties
    ----------
    * c_i = 1 on every observed pixel.
    * Each factor is in [0, 1], so the product is in [0, 1].
    * Multiplicative form: any single signal collapsing to zero collapses
      the product, which is the desired conservative behaviour
      (Section 6.1.3).
    """
    H, W = M.shape
    obs  = (M == 1)
    hole = ~obs

    # Schedule (scalar)
    c_sched = min(1.0, c0 + beta * float(t))

    # Boundary distance.  distance_transform_edt(input) returns, for each
    # FOREGROUND pixel (nonzero in `input`), the distance to the nearest
    # BACKGROUND pixel (zero in `input`).  Passing (M == 0):
    #   foreground = hole, background = observed
    # so the result at hole pixels is the distance to the nearest observed
    # pixel; at observed pixels the distance is 0 (they ARE background).
    d_img = distance_transform_edt((M == 0).astype(np.uint8)).astype(np.float64)
    c_geom = np.exp(-d_img / max(rho, 1e-6))

    # Variance term
    if var is None:
        c_var = np.ones_like(M, dtype=np.float64)
    else:
        if var.shape != M.shape:
            raise ValueError(f"var shape {var.shape} != M shape {M.shape}")
        # Adaptive tau_v: median of sqrt(var) over hole pixels.
        if tau_v is None:
            if hole.any():
                med = float(np.median(np.sqrt(np.clip(var[hole], 0.0, None))))
                tau_v = max(med, 1e-4)
            else:
                tau_v = 1.0
        c_var = np.exp(-var / (tau_v ** 2 + 1e-12))

    # Combine, then enforce c = 1 on Omega
    c = c_var * c_geom * c_sched
    c = np.where(obs, 1.0, c)

    return np.clip(c, 0.0, 1.0)
