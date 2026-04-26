"""
Optional stochastic manifold correction (CROWN-Inpaint section 6.8).

Every k outer iterations we inject a small amount of Gaussian noise inside
the hole, denoise the result, and re-project observed pixels.  This is
inspired by manifold-constraint diffusion solvers (Chung et al. 2022) and
stochastic plug-and-play (Park et al. 2026), but does NOT require any
pretrained network.

Update (eq. 6.8.2):

    eta             ~ N(0, sigma_t^2 I)        sampled on hole pixels only
    u_noisy         = u + (1 - M) . eta
    u_denoised      = T_{sigma_t}( u_noisy )
    u_new           = M . y + (1 - M) . u_denoised

The denoising operator T defaults to total-variation (Chambolle), which
is parameterless besides the regularisation strength and is widely
available in scikit-image.  Schedule:

    sigma_t = sigma_0 * gamma^(t - 1)

with sigma_0 = 0.04, gamma = 0.7 by default (Section 10.1).

Safety properties (Section 8.7):
* Observed pixels are bit-exact preserved by the final projection.
* Noise variance decays geometrically, so the correction step becomes
  asymptotically mild.
"""

from __future__ import annotations

import numpy as np
from skimage.restoration import denoise_tv_chambolle


def manifold_correct(
    u: np.ndarray,
    y: np.ndarray,
    M: np.ndarray,
    sigma_t: float,
    rng: np.random.Generator,
    denoiser: str = "tv",
    tv_weight: float = None,
) -> np.ndarray:
    """
    One stochastic manifold-correction step.

    Parameters
    ----------
    u : (H, W) float64    Current image (hole pixels carry the estimate).
    y : (H, W) float64    Observed image (used to project observed pixels).
    M : (H, W) float64    Binary mask (1 = observed, 0 = missing).
    sigma_t : float       Noise standard deviation at the current iteration.
    rng : np.random.Generator
        Reproducible random source.
    denoiser : {"tv"}
        Denoising operator to apply after noise injection.  Currently only
        "tv" (total variation, Chambolle) is supported because it is
        deterministic, parameter-light, and free of external dependencies.
        The interface is left open for future plug-in denoisers.
    tv_weight : float or None
        TV regularisation strength.  If None, set proportional to sigma_t
        (`tv_weight = sigma_t`), which is a standard plug-and-play heuristic.

    Returns
    -------
    u_new : (H, W) float64    Image after correction, clipped to [0, 1],
                              with observed pixels set to y exactly.
    """
    if u.shape != y.shape or u.shape != M.shape:
        raise ValueError(
            f"shape mismatch: u={u.shape}, y={y.shape}, M={M.shape}"
        )
    if sigma_t < 0:
        raise ValueError(f"sigma_t must be >= 0, got {sigma_t}")
    if sigma_t == 0:
        # No noise to inject; still apply the projection for safety.
        return np.where(M == 1, y, np.clip(u, 0.0, 1.0))

    # 1. Inject Gaussian noise on the hole only
    eta     = rng.standard_normal(u.shape) * sigma_t
    u_noisy = u + (1.0 - M) * eta
    # The denoiser expects a roughly [0, 1] image; TV-Chambolle handles
    # excursions but clip to a sensible range first.
    u_noisy = np.clip(u_noisy, -0.1, 1.1)

    # 2. Denoise
    if denoiser == "tv":
        if tv_weight is None:
            tv_weight = float(sigma_t)
        u_den = denoise_tv_chambolle(u_noisy, weight=tv_weight)
    else:
        raise ValueError(f"unknown denoiser '{denoiser}'")

    # 3. Re-project observed pixels and clip
    u_den = np.clip(u_den, 0.0, 1.0)
    return np.where(M == 1, y, u_den)


def schedule_sigma(t: int, sigma0: float = 0.04, gamma: float = 0.7) -> float:
    """
    Geometric schedule sigma_t = sigma0 * gamma^(t - 1).

    Parameters
    ----------
    t : int      1-based iteration counter.
    sigma0 : float    Initial noise level.
    gamma  : float    Decay factor in (0, 1].

    Returns
    -------
    sigma_t : float
    """
    if t < 1:
        return sigma0
    return sigma0 * (gamma ** (t - 1))
