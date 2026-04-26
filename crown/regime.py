"""
Regime map (CROWN-Inpaint section 6.2).

For each pixel we estimate how textured its local neighbourhood is, on a
[0, 1] scale where 0 = smooth and 1 = textured.  The regime map gates the
fusion between the smooth-prior branch and the sparse-coding branch.

Three local features are combined:
    g_i  : mean gradient magnitude in a window  (texture amplitude)
    a_i  : structure-tensor anisotropy           (oriented vs. isotropic)
    h_i  : windowed-DCT spectral entropy         (frequency richness)

The combined raw map is then propagated from the observed set Omega into
the hole Omega^c via biharmonic extension and Gaussian-smoothed.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter, binary_erosion
from skimage.restoration import inpaint_biharmonic
from skimage.util import view_as_windows
from scipy.fft import dctn


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

def _gradient_energy(y: np.ndarray, window: int) -> np.ndarray:
    """
    Mean gradient magnitude in a (window x window) neighbourhood at each pixel.

        g_i = (1 / |W_i|) * sum_{j in W_i} ||grad y_j||_2

    Implemented by computing |grad y| pixel-wise then box-averaging.
    """
    gx, gy = np.gradient(y)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    return uniform_filter(grad_mag, size=window, mode="reflect")


def _structure_anisotropy(y: np.ndarray, window: int) -> np.ndarray:
    """
    Structure-tensor anisotropy in a (window x window) neighbourhood.

        J = sum_j  grad y_j (grad y_j)^T  (smoothed in the window)
        eigenvalues lambda_1 >= lambda_2 >= 0
        a_i = (lambda_1 - lambda_2) / (lambda_1 + lambda_2 + eps)

    For a symmetric 2x2 matrix [[Jxx, Jxy], [Jxy, Jyy]] the eigenvalues are
        lambda_{1,2} = tr/2 +- sqrt((tr/2)^2 - det)
    so we compute Jxx, Jyy, Jxy by box-smoothing the corresponding gradient
    products and then evaluate the closed-form anisotropy in one shot.

    a_i lies in [0, 1] by construction (zero on isotropic patches, one on
    perfect 1-D edges).  No additional normalisation is needed.
    """
    gx, gy = np.gradient(y)
    Jxx = uniform_filter(gx * gx, size=window, mode="reflect")
    Jyy = uniform_filter(gy * gy, size=window, mode="reflect")
    Jxy = uniform_filter(gx * gy, size=window, mode="reflect")

    tr   = Jxx + Jyy
    det  = Jxx * Jyy - Jxy * Jxy
    disc = np.sqrt(np.maximum(0.25 * tr * tr - det, 0.0))
    lam1 = 0.5 * tr + disc
    lam2 = 0.5 * tr - disc
    eps  = 1e-10
    return (lam1 - lam2) / (lam1 + lam2 + eps)


def _spectral_entropy(y: np.ndarray, window: int) -> np.ndarray:
    """
    Shannon entropy of the windowed-DCT magnitude spectrum, normalised to [0, 1].

    For each pixel the surrounding (window x window) patch is 2-D DCT'd
    (orthonormal), squared to obtain a power spectrum, normalised to a
    probability distribution, and Shannon-entropy is computed.  The result
    is divided by log(window^2) so that the output is in [0, 1].

    Smooth patches concentrate energy in the DC + low-frequency coefficients
    => low entropy.  Noise-like / textured patches spread energy across
    coefficients => high entropy.

    Memory: for an (H x W) image with window p, view_as_windows + DCT
    materialises a (H x W x p x p) tensor.  For 128 x 128 with p = 9 that
    is ~10 MB at float64 -- comfortably fits.
    """
    pad   = window // 2
    yp    = np.pad(y, pad, mode="reflect")               # (H+2p, W+2p)
    wins  = view_as_windows(yp, (window, window))        # (H, W, p, p)
    coef  = dctn(wins, axes=(-2, -1), norm="ortho")      # (H, W, p, p)
    power = coef * coef                                   # already non-neg

    Z      = power.sum(axis=(-2, -1), keepdims=True)
    safe_Z = np.where(Z < 1e-12, 1.0, Z)
    p_dist = power / safe_Z                              # (H, W, p, p)

    # 0 * log(0) := 0.  Use np.log(np.maximum(...)) so the log is never
    # evaluated on a zero (np.where would compute log(0) before selecting,
    # which fires a divide-by-zero warning).  Multiplying by p_dist
    # subsequently drops the unused branch to zero.
    log_p = np.log(np.maximum(p_dist, 1e-12))
    H_pix = -(p_dist * log_p).sum(axis=(-2, -1))         # (H, W)

    return np.clip(H_pix / np.log(window * window), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_regime_map(
    y: np.ndarray,
    M: np.ndarray,
    window: int = 9,
    weights: tuple = (0.4, 0.3, 0.3),
    sigma_smooth: float = 1.0,
) -> np.ndarray:
    """
    Compute the regime map r in [0, 1]^{H x W}.

    Pipeline (Section 6.2):
        1. Compute g, a, h on the (zero-filled) image.
        2. Normalise g to [0, 1] via min-max; a and h are already in [0, 1].
        3. Combine    r_raw = w_g * g_n + w_a * a + w_h * h
                       (weights renormalised to sum to one).
        4. Propagate r_raw from Omega into Omega^c via biharmonic extension
           (Section 6.2.3 explicitly suggests inpaint-style harmonic extension).
        5. Smooth with a Gaussian kernel of width sigma_smooth (Section 6.2.4).

    Parameters
    ----------
    y : (H, W) float64
        Image; hole pixels carry placeholder values (typically zero).
    M : (H, W) float64
        Binary mask (1 = observed, 0 = missing).
    window : int, default 9
        Side length of the local feature window.  Must be odd; a value
        comparable to twice the patch radius is appropriate.
    weights : tuple of three floats, default (0.4, 0.3, 0.3)
        (w_g, w_a, w_h) feature weights.  Internally renormalised to sum 1.
    sigma_smooth : float, default 1.0
        Gaussian smoothing sigma applied to the propagated map.

    Returns
    -------
    r : (H, W) float64
        Regime map clipped to [0, 1].
    """
    if y.shape != M.shape:
        raise ValueError(f"shape mismatch: y={y.shape}, M={M.shape}")
    if window < 3 or window % 2 == 0:
        raise ValueError(f"window must be odd and >=3, got {window}")

    # 1. Compute features on the whole (zero-filled) image.
    #
    # Features computed at observed pixels NEAR a hole are biased: the
    # window centred on such a pixel may receive contributions from hole
    # pixels, producing a spurious gradient/anisotropy/entropy spike.  We
    # therefore build a TRUST MASK that marks observed pixels whose entire
    # *effective feature support* lies inside Omega.
    #
    # Effective support per feature
    # -----------------------------
    #   spectral entropy : direct (window x window) DCT over y
    #                       -> needs window x window of observed y.
    #   gradient energy  : uniform_filter(|grad y|, window) where
    #                      grad uses ±1 finite differences
    #                       -> needs (window + 2) x (window + 2) of observed y.
    #   structure tensor : uniform_filter(grad y . grad y, window)
    #                       -> needs (window + 2) x (window + 2) of observed y.
    #
    # We take the worst case (window + 2) so that NO feature at a trusted
    # pixel can have its computation reach a hole pixel, even via the
    # gradient stencil's ±1 extension.
    g = _gradient_energy(y, window)
    a = _structure_anisotropy(y, window)
    h = _spectral_entropy(y, window)

    trust_size   = window + 2
    trust_struct = np.ones((trust_size, trust_size), dtype=bool)
    trust = binary_erosion(M.astype(bool), structure=trust_struct)

    if not trust.any():
        # The hole is so large that no observed pixel has a fully-observed
        # window.  Fall back to using all observed pixels (still better than
        # using the hole pixels).
        trust = M.astype(bool)

    # 2. Normalise the gradient feature using ONLY the trust region.
    # The structure-tensor anisotropy and DCT entropy are already in [0, 1]
    # by construction, so they are not re-normalised.
    g_trust = g[trust]
    g_min   = float(g_trust.min())
    g_max   = float(g_trust.max())
    g_n     = np.clip((g - g_min) / (g_max - g_min + 1e-10), 0.0, 1.0)

    # 3. Linear combination with renormalised weights
    w_g, w_a, w_h = weights
    w_sum = w_g + w_a + w_h
    if w_sum <= 0:
        raise ValueError("weights must sum to a positive number")
    r_raw = (w_g / w_sum) * g_n + (w_a / w_sum) * a + (w_h / w_sum) * h
    r_raw = np.clip(r_raw, 0.0, 1.0)

    # 4. Propagate from the TRUST region into everything else (hole + near-
    # boundary observed band).  The trust complement is the union of
    # Omega^c and the boundary band; biharmonic extension fills it from
    # the trusted boundary inward.
    propagate_mask = ~trust                          # True wherever r_raw is unreliable
    if propagate_mask.any():
        r_prop = inpaint_biharmonic(r_raw, propagate_mask)
        # Restore the trusted-region values exactly (biharmonic does not
        # modify them, but be explicit).
        r_prop = np.where(trust, r_raw, r_prop)
    else:
        r_prop = r_raw

    # 5. Gaussian smoothing avoids sharp seams in the fusion gate
    r_smooth = gaussian_filter(r_prop, sigma=sigma_smooth)

    return np.clip(r_smooth, 0.0, 1.0)
