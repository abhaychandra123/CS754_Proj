"""
CROWN-Inpaint shared utilities -- decoupled from the project's old plotting
script.

Previous versions of crown/run.py imported PATCH_SIZE / PATCH_DIM /
N_ATOMS / SPARSITY / ALS_ITERS plus two patch helpers from
`inpainting_multiscale_masked_ksvd.py`.  That module imports matplotlib at
its top, so importing crown transitively pulled in plotting dependencies.
This module re-homes the helpers and constants here so that crown is a
self-contained, plotting-free library.

Constants follow CROWN_Inpaint_specification.md Section 10.1:
    Patch size p = 8   -> n = 64
    K = 256 atoms      (4 x overcomplete; spec default)
    Sparsity s = 8     (spec default)
    ALS iterations 10  (carried over from Phases 1 & 2)
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.image import (
    extract_patches_2d,
    reconstruct_from_patches_2d,
)


# ---------------------------------------------------------------------------
# Constants (spec Section 10.1)
# ---------------------------------------------------------------------------

PATCH_SIZE: tuple = (8, 8)
PATCH_DIM:  int   = PATCH_SIZE[0] * PATCH_SIZE[1]      # 64

DEFAULT_N_ATOMS:    int = 256       # spec Section 10.1
DEFAULT_SPARSITY:   int = 8         # spec Section 10.1
DEFAULT_ALS_ITERS:  int = 10        # carried over from Phases 1-2


# ---------------------------------------------------------------------------
# Patch extraction (features-first layout)
# ---------------------------------------------------------------------------

def extract_features_first(
    image: np.ndarray,
    patch_shape: tuple = PATCH_SIZE,
) -> np.ndarray:
    """
    Extract every overlapping `patch_shape` patch and return a (p*p, N) array
    where column i contains the flattened pixels of patch i in row-major
    (C order) layout.

    Parameters
    ----------
    image : (H, W) float64    Image with no NaN.
    patch_shape : (p, p)      Patch shape; default = PATCH_SIZE = (8, 8).

    Returns
    -------
    X : (p*p, N) float64      Patch matrix.
    """
    raw = extract_patches_2d(image, patch_shape)        # (N, p, p)
    N   = raw.shape[0]
    return raw.reshape(N, -1).T                          # (p*p, N)


def extract_patches_with_mask(
    image_filled: np.ndarray,
    mask: np.ndarray,
    patch_shape: tuple = PATCH_SIZE,
) -> tuple:
    """
    Extract overlapping patches from a NaN-free image plus its binary mask,
    returning the same (X_filled, X_nan, M_flat, N) tuple shape used by the
    rest of the project.

        X_filled : (p*p, N)   patch values verbatim (no NaN injection)
        X_nan    : (p*p, N)   patch values with NaN at  mask == 0
        M_flat   : (p*p, N)   binary mask matching the patch layout
        N        : int        total number of overlapping patches

    sklearn's `extract_patches_2d` rejects NaN, so the function takes the
    zero-filled or prior-filled image as input and reintroduces NaN only
    after the shape translation.

    Used by `crown.run.train_dictionary` to feed `masked_ksvd` with NaN-
    marked patches while keeping the binary mask aligned.
    """
    if image_filled.shape != mask.shape:
        raise ValueError(
            f"image / mask shape mismatch: {image_filled.shape} vs {mask.shape}"
        )
    if np.isnan(image_filled).any():
        raise ValueError(
            "image_filled must NOT contain NaN before patch extraction"
        )

    raw_img  = extract_patches_2d(image_filled, patch_shape)   # (N, p, p)
    raw_mask = extract_patches_2d(mask,         patch_shape)   # (N, p, p)
    N        = raw_img.shape[0]

    X_flat = raw_img.reshape(N, -1).T                          # (p*p, N)
    M_flat = raw_mask.reshape(N, -1).T                         # (p*p, N)

    X_filled = X_flat.copy()
    X_nan    = X_flat.copy()
    X_nan[M_flat == 0] = np.nan

    return X_filled, X_nan, M_flat, N


# ---------------------------------------------------------------------------
# Reconstruction (overlap averaging)
# ---------------------------------------------------------------------------

def reconstruct_image_from_codes(
    D: np.ndarray,
    Alpha: np.ndarray,
    image_shape: tuple,
    patch_shape: tuple = PATCH_SIZE,
) -> np.ndarray:
    """
    Reconstruct an image from a dictionary and sparse codes via overlap
    averaging.

        patches_flat = D @ Alpha                # (n, N)
        patches_2d   = patches_flat.T.reshape(N, p, p)
        img          = reconstruct_from_patches_2d(patches_2d, image_shape)

    Parameters
    ----------
    D : (p*p, K) float64        Dictionary (unit-norm columns).
    Alpha : (K, N) float64      Sparse codes for every overlapping patch.
    image_shape : (H, W)        Target image shape.
    patch_shape : (p, p)        Patch shape; default = PATCH_SIZE = (8, 8).

    Returns
    -------
    img_recon : (H, W) float64    Image clipped to [0, 1].
    """
    if D.shape[0] != patch_shape[0] * patch_shape[1]:
        raise ValueError(
            f"D rows {D.shape[0]} != patch_shape product "
            f"{patch_shape[0] * patch_shape[1]}"
        )
    N            = Alpha.shape[1]
    patches_flat = D @ Alpha                                       # (n, N)
    patches_T    = patches_flat.T                                   # (N, n)
    patches_2d   = patches_T.reshape(N, patch_shape[0], patch_shape[1])
    img          = reconstruct_from_patches_2d(patches_2d, image_shape)
    return np.clip(img, 0.0, 1.0)
