"""
CROWN-Inpaint outer loop and dictionary trainer (sections 7 and 9).

Public functions:
    train_dictionary(y, M, ...)             learn D from observed patches
    run_crown_inpaint(y, M, D, ...)         full iterative inpainting

Both follow the algorithm in Section 7 of the specification verbatim.

This module re-uses the existing project's K-SVD primitives (`masked_ksvd`,
`extract_and_translate_patches`, `reconstruct_image_from_codes`) to keep the
training and patch-handling pipeline consistent with Phases 1-4 of the
project.
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d

# Project root on sys.path so we can import the existing modules
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_THIS_DIR)
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from masked_ksvd import masked_ksvd                                  # noqa: E402
from inpainting_multiscale_masked_ksvd import (                      # noqa: E402
    extract_and_translate_patches,
    reconstruct_image_from_codes,
    PATCH_SIZE,
    PATCH_DIM,
    N_ATOMS,
    SPARSITY,
    ALS_ITERS,
)

# CROWN sub-modules
from crown.smooth            import biharmonic_init, harmonic_relax
from crown.regime            import compute_regime_map
from crown.confidence        import compute_confidence_map, overlap_mean_and_variance
from crown.weighted_omp      import build_weight_image, weighted_omp_batch
from crown.nonlocal_coupling import nonlocal_refine
from crown.fuse              import fuse_and_project
from crown.manifold          import manifold_correct, schedule_sigma


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_features_first(image: np.ndarray,
                            patch_shape: tuple = PATCH_SIZE) -> np.ndarray:
    """
    Extract all overlapping patches and return a (p*p, N) array.

    Equivalent to the (X_filled output of `extract_and_translate_patches`
    when called with an all-ones mask), but avoids the redundant NaN-
    injection branch.  Reuses sklearn's patch extractor for consistency
    with the rest of the codebase.
    """
    raw = extract_patches_2d(image, patch_shape)                # (N, p, p)
    N   = raw.shape[0]
    return raw.reshape(N, -1).T                                  # (p*p, N)


def _hard_project(u_bar: np.ndarray, y: np.ndarray, M: np.ndarray) -> np.ndarray:
    return np.where(M == 1, y, u_bar)


def _safe_div(a, b, eps=1e-12):
    return a / (b + eps)


def _mse_hole(img_a: np.ndarray, img_b: np.ndarray, M: np.ndarray) -> float:
    hole = (M == 0)
    if not hole.any():
        return 0.0
    return float(np.mean((img_a[hole] - img_b[hole]) ** 2))


def _hole_psnr(img_true: np.ndarray, img_pred: np.ndarray, M: np.ndarray) -> float:
    mse = _mse_hole(img_true, img_pred, M)
    return 10.0 * np.log10(_safe_div(1.0, mse, 1e-12))


# ---------------------------------------------------------------------------
# Dictionary training (Section 7.2 step 1)
# ---------------------------------------------------------------------------

def train_dictionary(
    y: np.ndarray,
    M: np.ndarray,
    n_train: int = 3000,
    n_iter: int = 25,
    K: int = N_ATOMS,
    sparsity: int = SPARSITY,
    als_iters: int = ALS_ITERS,
    seed: int = 42,
    verbose: bool = True,
) -> tuple:
    """
    Learn a unit-norm dictionary D from observed patches via masked K-SVD.

    The dictionary must learn from REAL pixels only.  We extract patches from
    a zero-filled corrupted image, inject NaN at missing positions, and pass
    them with the binary mask to `masked_ksvd`, which ignores those entries
    entirely (Phases 1-2 of the project).

    Parameters
    ----------
    y : (H, W) float64    Observed image (placeholder values inside hole).
    M : (H, W) float64    Binary mask.
    n_train, n_iter, K, sparsity, als_iters, seed : training hyperparameters.

    Returns
    -------
    D         : (n, K) float64    Unit-norm dictionary.
    train_err : (n_iter,)         Masked-RMSE convergence curve.
    """
    img_zero = y * M
    _, X_nan_all, M_all, N = extract_and_translate_patches(img_zero, M)

    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=min(n_train, N), replace=False)
    X_train = X_nan_all[:, idx]
    M_train = M_all[:, idx]

    if verbose:
        print(
            f"  [CROWN] training dictionary on {X_train.shape[1]} patches, "
            f"K={K}, s={sparsity}, n_iter={n_iter}"
        )

    D, _, train_err = masked_ksvd(
        X_train, M_train,
        K=K, s=sparsity,
        n_iter=n_iter, als_iters=als_iters,
        seed=seed, label="CROWN dict",
    )

    col_norms = np.linalg.norm(D, axis=0)
    if not np.allclose(col_norms, 1.0, atol=1e-5):
        raise RuntimeError(
            f"dictionary not unit-norm: min={col_norms.min():.6f}, "
            f"max={col_norms.max():.6f}"
        )

    return D, train_err


# ---------------------------------------------------------------------------
# Outer loop (Section 7.3)
# ---------------------------------------------------------------------------

def run_crown_inpaint(
    y: np.ndarray,
    M: np.ndarray,
    D: np.ndarray,
    *,
    # Iteration control
    T: int = 5,
    eps_stop: float = 1e-4,
    # Sparse branch
    sparsity: int = SPARSITY,
    # Smooth branch
    K_s: int = 10,
    # Confidence map
    c0: float = 0.05,
    beta: float = 0.15,
    rho: float = 4.0,
    # Regime map
    regime_window: int = 9,
    regime_weights: tuple = (0.4, 0.3, 0.3),
    regime_sigma_smooth: float = 1.0,
    # Nonlocal coupling
    nonlocal_enabled: bool = True,
    K_nl: int = 10,
    nonlocal_lambda: float = 0.1,
    nonlocal_n_ista: int = 20,
    nonlocal_cadence: int = 2,
    # Manifold correction
    manifold_enabled: bool = False,
    manifold_sigma0: float = 0.04,
    manifold_gamma: float = 0.7,
    manifold_cadence: int = 2,
    manifold_seed: int = 0,
    # Logging
    img_clean: np.ndarray = None,
    verbose: bool = True,
) -> dict:
    """
    Run the full CROWN-Inpaint algorithm.

    Inputs
    ------
    y : (H, W) float64    Observed image (any placeholder inside hole).
    M : (H, W) float64    Binary mask, 1 = observed.
    D : (n, K) float64    Pre-trained dictionary with unit-norm columns.

    Iteration control
    -----------------
    T          : maximum number of outer iterations.
    eps_stop   : relative-change threshold for early stopping (Section 7.4).

    Branch hyperparameters
    ----------------------
    See Section 10.1 of the specification for defaults; we adopt them
    here as keyword-argument defaults.

    Optional manifold correction (Section 6.8)
    ------------------------------------------
    Disabled by default (`manifold_enabled=False`).  When enabled, every
    `manifold_cadence` outer iterations a small Gaussian noise is injected
    inside the hole and a TV-Chambolle denoising step is applied, followed
    by hard projection.

    Diagnostic logging (Section 7.3 step 7)
    ---------------------------------------
    If `img_clean` is provided, full-image PSNR / SSIM, hole-only PSNR,
    and the relative change `||u^t - u^{t-1}|| / ||u^{t-1}||` are logged
    per iteration.

    Returns
    -------
    out : dict with keys
        u           : (H, W) final reconstruction
        history     : list of dicts (one per iteration; iter 0 = init)
        u_iter      : list of (H, W) images per iteration
        regime      : (H, W) regime map
        confidence  : list of (H, W) confidence maps per iteration
        last_alpha  : (K, N) sparse codes at the final iteration
        timings     : dict with per-iteration runtime breakdown
    """
    # ---- 0. Validate inputs ------------------------------------------------
    if y.shape != M.shape:
        raise ValueError(f"y.shape {y.shape} != M.shape {M.shape}")
    if D.shape[0] != PATCH_DIM:
        raise ValueError(
            f"dictionary dim {D.shape[0]} != PATCH_DIM {PATCH_DIM}"
        )
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    H, W = y.shape

    rng = np.random.default_rng(manifold_seed)

    # ---- A. Setup (Section 7.2) -------------------------------------------
    if verbose:
        print("\n  [CROWN]  Phase A: setup")
        print("  --------------------------")

    t0_setup = time.time()

    # A.1 biharmonic init (one global solve)
    u = biharmonic_init(y, M)
    # A.2 regime map computed on observed image
    r_map = compute_regime_map(
        y * M, M,
        window=regime_window,
        weights=regime_weights,
        sigma_smooth=regime_sigma_smooth,
    )
    setup_time = time.time() - t0_setup
    if verbose:
        print(f"  [CROWN]  setup done in {setup_time:.2f}s   "
              f"(biharmonic init + regime map)")
        print(f"  [CROWN]  regime map  : mean={r_map.mean():.3f}, "
              f"std={r_map.std():.3f}, in-hole mean="
              f"{r_map[M==0].mean() if (M==0).any() else float('nan'):.3f}")

    # ---- Diagnostic logging helper ----------------------------------------
    history = []
    confidence_iters = []
    u_iter = [u.copy()]

    def _log(t_idx, u_curr, prev_u, label, extra):
        """Compute and append metrics for one iteration."""
        rec = {"iter": t_idx, "label": label}
        rec.update(extra)
        if img_clean is not None:
            from skimage.metrics import peak_signal_noise_ratio as psnr_fn
            from skimage.metrics import structural_similarity  as ssim_fn
            uc = np.clip(u_curr, 0.0, 1.0)
            rec["psnr"]      = float(psnr_fn(img_clean, uc, data_range=1.0))
            rec["ssim"]      = float(ssim_fn(img_clean, uc, data_range=1.0))
            rec["hole_psnr"] = float(_hole_psnr(img_clean, uc, M))
        if prev_u is not None:
            denom = float(np.linalg.norm(prev_u))
            num   = float(np.linalg.norm(u_curr - prev_u))
            rec["rel_change"] = num / max(denom, 1e-12)
        history.append(rec)
        if verbose:
            parts = [f"  [CROWN] {label}"]
            for k in ("psnr", "ssim", "hole_psnr"):
                if k in rec:
                    parts.append(f"{k}={rec[k]:.4f}")
            for k in ("rel_change",):
                if k in rec:
                    parts.append(f"{k}={rec[k]:.5f}")
            for k, v in extra.items():
                parts.append(f"{k}={v}")
            print("   ".join(parts))

    _log(0, u, None, "iter 0 (biharmonic init)", {})

    # ---- B. Outer iterations (Section 7.3) ---------------------------------
    if verbose:
        print(f"\n  [CROWN]  Phase B: {T} outer iterations")
        print("  -----------------------------------")

    Alpha_prev = None      # (K, N) cached for variance computation at t >= 2
    last_alpha = None
    timings    = []

    for t in range(1, T + 1):
        t_start = time.time()

        # ---- B.1 confidence map (uses previous Alpha for variance) -------
        if Alpha_prev is None:
            var_prev = None
        else:
            _, var_prev = overlap_mean_and_variance(
                D, Alpha_prev, image_shape=u.shape, patch_shape=PATCH_SIZE,
            )
        c_map = compute_confidence_map(
            M=M, t=t, var=var_prev,
            c0=c0, beta=beta, rho=rho, tau_v=None,
        )
        confidence_iters.append(c_map.copy())

        # ---- B.2 sparse branch -------------------------------------------
        # Build the per-pixel weight image and extract patches.
        W_img = build_weight_image(M, c_map)
        X_pat = _extract_features_first(u,     PATCH_SIZE)              # (n, N)
        W_pat = _extract_features_first(W_img, PATCH_SIZE)              # (n, N)

        Alpha = weighted_omp_batch(
            X_pat, D, W_pat, sparsity,
        )

        # ---- B.3 nonlocal coupling (cadence) -----------------------------
        used_nl = False
        if (
            nonlocal_enabled
            and t > 1
            and (t % nonlocal_cadence == 0)
        ):
            Alpha = nonlocal_refine(
                X_filled=X_pat, Alpha=Alpha,
                W_patches=W_pat, D=D,
                K_nl=K_nl, h=None,
                lam=nonlocal_lambda, n_ista=nonlocal_n_ista,
            )
            used_nl = True

        u_sparse = reconstruct_image_from_codes(D, Alpha, u.shape)

        # ---- B.4 smooth branch -------------------------------------------
        u_smooth = harmonic_relax(u, y, M, K_s=K_s)

        # ---- B.5 fuse + project ------------------------------------------
        u_bar = fuse_and_project(u_smooth, u_sparse, r_map, M, y)

        # ---- B.6 manifold correction (optional, cadence) -----------------
        used_mc = False
        sigma_t = 0.0
        if (
            manifold_enabled
            and (t % manifold_cadence == 0)
        ):
            sigma_t = schedule_sigma(t, sigma0=manifold_sigma0, gamma=manifold_gamma)
            u_bar = manifold_correct(
                u_bar, y, M,
                sigma_t=sigma_t, rng=rng, denoiser="tv",
            )
            used_mc = True

        # ---- B.7 update + diagnostics ------------------------------------
        u_prev = u
        u = u_bar
        u_iter.append(u.copy())
        Alpha_prev = Alpha
        last_alpha = Alpha

        dt = time.time() - t_start
        timings.append(dt)
        extra = {
            "dt": f"{dt:.2f}s",
            "nl": "Y" if used_nl else "n",
            "mc": "Y" if used_mc else "n",
            "sigma": f"{sigma_t:.4f}",
            "c_mean_hole": (
                f"{c_map[M==0].mean():.3f}" if (M == 0).any() else "n/a"
            ),
            "avg_nnz": f"{(Alpha != 0).sum() / max(Alpha.shape[1], 1):.2f}",
        }
        _log(t, u, u_prev, f"iter {t}", extra)

        # ---- B.8 stopping criterion --------------------------------------
        rel = history[-1].get("rel_change", float("inf"))
        if rel < eps_stop:
            if verbose:
                print(f"  [CROWN] early stop at iter {t}: rel_change={rel:.5f}")
            break

    return {
        "u":          u,
        "history":    history,
        "u_iter":     u_iter,
        "regime":     r_map,
        "confidence": confidence_iters,
        "last_alpha": last_alpha,
        "timings":    timings,
    }
