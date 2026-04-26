"""
=============================================================================
Phase 4: Multi-Scale Masked Dictionary Learning for Image Inpainting
         Coarse-to-Fine Image Pyramid with Prior Injection
=============================================================================

MOTIVATION
----------
In Phase 3 we found that when a contiguous missing region (20×20 pixels)
is LARGER than the dictionary patch size (8×8), every patch that overlaps
the hole interior has ZERO observed pixels.  Masked OMP then returns
alpha = 0 for those patches, and reconstruction gives a black smear.

SOLUTION: Coarse-to-Fine Pyramid
---------------------------------
We exploit the fact that physical scale (pixels) is relative.
At 32×32 resolution, the same 20×20 hole becomes a 5×5 hole in pixel count.
A 5×5 hole is SMALLER than an 8×8 patch: every hole pixel is now covered
by multiple patches that contain both the hole region AND neighbouring
observed pixels.  Masked K-SVD succeeds at this coarse level.

We then propagate the coarse reconstruction upward through 3 pyramid levels:
    Level 3 (32×32)  — hole is  5×5: always solvable with 8×8 patches
    Level 2 (64×64)  — hole is 10×10: partially solvable; prior injection helps
    Level 1 (128×128)— hole is 20×20: prior injection from L2 fills the cavity

PRIOR INJECTION WITH MASK DECOUPLING — Mathematical Note
---------------------------------------------------------
If sparse coding uses the strict binary mask at Level 1, center patches
inside a 20×20 hole can have 0 observed pixels. For 8×8 patches this creates
a 4×4 dead-zone at the hole center where masked OMP returns alpha=0.

To avoid this, we explicitly decouple learning-time masking from
reconstruction-time sparse coding:

  • X_nan    : strict masked patches (missing entries set to NaN)
  • X_filled : prior-filled patches (no NaN)

Phase C (Dictionary Learning):
    masked_ksvd(X_train_nan, M_train, ...)
    Dictionary updates only use truly observed pixels.

Phase D (Sparse Coding for Reconstruction):
    Level 3:  masked_omp(X_all_filled[:, i], D, M_all[:, i], s)
              (ignore missing entries so zero-fill is not treated as truth)
    Levels 2/1: masked_omp(X_all_filled[:, i], D, all_ones, s)
                (treat injected prior as valid signal for refinement)

PYRAMID ALGORITHM
-----------------
Level 3 (coarsest, 32×32):
    • Downsample corrupted image + mask (order=0 + threshold > 0.5 for mask)
    • Extract all 8×8 patches → X_filled, X_nan, M
    • Learn D₃ (64×256) via masked K-SVD on X_train_nan + M_train
    • Sparse-code all patches from X_filled using TRUE mask M → recon_32

Level 2 (middle, 64×64):
    • Downsample 128×128 corrupted image + mask to 64×64
    • Upsample recon_32 → prior_64  (bilinear, order=1)
    • PRIOR INJECTION: img_64_filled[mask_64==0] = prior_64[mask_64==0]
    • Extract 8×8 patches from img_64_filled and mask_64 → X_filled, X_nan, M
    • Learn D₂ (64×256) via masked K-SVD on X_train_nan + M_train
    • Sparse-code all patches from X_filled using all-ones mask → recon_64

Level 1 (finest, 128×128):
    • Upsample recon_64 → prior_128
    • PRIOR INJECTION: img_128_filled[mask==0] = prior_128[mask==0]
    • Extract 8×8 patches from img_128_filled and original mask → X_filled, X_nan, M
    • Learn D₁ (64×256) via masked K-SVD on X_train_nan + M_train
    • Sparse-code all patches from X_filled using all-ones mask → final_recon_128

SHAPE TRANSLATION (applied at every level — see detailed annotation below):
    sklearn output : (N_patches, 8, 8)  — patches-first
    our API input  : (64, N_patches)    — features-first
    Forward:  patches.reshape(N, 64).T  →  (N,8,8) → (N,64) → (64,N)
    Inverse:  codes.T.reshape(N, 8, 8)  →  (64,N)  → (N,64) → (N,8,8)

Author : CS754 Project Group
License: MIT
=============================================================================
"""

import sys
import os
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

from skimage import data as skdata
from skimage.transform import resize as sk_resize
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity  as ssim_fn
from skimage.restoration import inpaint_biharmonic
from sklearn.feature_extraction.image import (
    extract_patches_2d,
    reconstruct_from_patches_2d,
)

# ---------------------------------------------------------------------------
# Import Phase 1 & 2 core math — unchanged from masked_ksvd.py
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from masked_ksvd import (
    masked_omp,
    masked_ksvd,
    baseline_ksvd,
    init_dictionary_from_data,
)
from custom_masks import generate_mask, MASK_MODES

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
# Image and corruption
IMAGE_SIZE    = (128, 128)
# Face-square hole: 20×20 block covering the cameraman's face in 128×128.
# Verified visually: face center is approximately at row=20, col=63.
# rows 10:30 (height 20), cols 55:75 (width 20) covers the face well.
HOLE_R0, HOLE_R1 = 10, 30     # row slice  [R0, R1)  → 20 rows
HOLE_C0, HOLE_C1 = 55, 75     # col slice  [C0, C1)  → 20 cols

# Dictionary learning
PATCH_SIZE   = (8, 8)
PATCH_DIM    = 64              # 8*8
N_ATOMS      = 256             # 4× overcomplete  (256 > 64)
SPARSITY     = 10              # nonzeros per patch code
ALS_ITERS    = 10              # ALS steps inside each masked_ksvd_update_atom

# Per-level hyperparameters
#   n_train  : patches sampled for dictionary learning (speed vs quality)
#   n_iter   : K-SVD outer iterations
LEVEL_CFG = {
    3: {"size": (32, 32),   "n_train": 500,  "n_iter": 25},
    2: {"size": (64, 64),   "n_train": 1500, "n_iter": 25},
    1: {"size": (128, 128), "n_train": 3000, "n_iter": 20},
}

SEED = 42

DEFAULT_MASK_MODE = "face-square"
DEFAULT_MISSING_FRAC = 0.30
DEFAULT_SQUARE_SIZE = HOLE_R1 - HOLE_R0
DEFAULT_SQUARE_CENTER = (
    (HOLE_R0 + HOLE_R1) // 2,
    (HOLE_C0 + HOLE_C1) // 2,
)
DEFAULT_SCRATCH_COUNT = 3
DEFAULT_SCRATCH_THICKNESS = 4
DEFAULT_OVERLAY_TEXT = "SAMPLE"
DEFAULT_TEXT_SCALE = 0.24
DEFAULT_TEXT_ANGLE = -90.0
DEFAULT_TEXT_STROKE = 1


# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_image() -> np.ndarray:
    """
    Load the reference grayscale image, resize to 128×128, float [0,1].
    """
    raw   = skdata.brick()                                        # (512,512) uint8
    img   = sk_resize(raw.astype(np.float64) / 255.0,
                      IMAGE_SIZE, anti_aliasing=True)               # (128,128) float64
    return img


def make_custom_mask(
    image_shape: tuple = IMAGE_SIZE,
    mask_mode: str = DEFAULT_MASK_MODE,
    missing_frac: float = DEFAULT_MISSING_FRAC,
    seed: int = SEED,
    square_size: int = DEFAULT_SQUARE_SIZE,
    square_center: tuple[int, int] | None = DEFAULT_SQUARE_CENTER,
    scratch_count: int = DEFAULT_SCRATCH_COUNT,
    scratch_thickness: int = DEFAULT_SCRATCH_THICKNESS,
    overlay_text: str = DEFAULT_OVERLAY_TEXT,
    text_scale: float = DEFAULT_TEXT_SCALE,
    text_angle: float = DEFAULT_TEXT_ANGLE,
    text_stroke: int = DEFAULT_TEXT_STROKE,
) -> np.ndarray:
    """
    Build a binary mask using the shared custom mask generator.

    Convention throughout: 1 = observed pixel, 0 = missing pixel.
    """
    return generate_mask(
        image_shape=image_shape,
        mode=mask_mode,
        missing_frac=missing_frac,
        seed=seed,
        square_size=square_size,
        square_center=square_center,
        scratch_count=scratch_count,
        scratch_thickness=scratch_thickness,
        overlay_text=overlay_text,
        text_scale=text_scale,
        text_angle=text_angle,
        text_stroke=text_stroke,
    )


def downsample_mask(mask_128: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Downsample a binary mask using nearest-neighbour interpolation, then
    threshold at 0.5 to guarantee strict binary output {0.0, 1.0}.

    WHY order=0 (nearest-neighbour)?
    ---------------------------------
    Linear or higher-order interpolation creates fractional values at hole
    boundaries (e.g., 0.25, 0.5, 0.75).  If those pass through masked K-SVD
    as a mask, entries like 0.25 are treated as "partially observed", which is
    not meaningful for our binary mask semantics.

    WHY threshold at 0.5?
    ----------------------
    After nearest-neighbour, values should already be strictly 0 or 1, but
    floating-point arithmetic can produce values like 9.99e-1 instead of 1.0.
    The threshold guarantees machine-precision binary output.

    Parameters
    ----------
    mask_128  : (128, 128) binary float64 mask
    target_size : (H, W) target resolution

    Returns
    -------
    mask_down : (H, W) binary float64 mask, values in {0.0, 1.0}
    """
    raw = sk_resize(
        mask_128,
        target_size,
        order=0,               # nearest-neighbour: no interpolation artefacts
        anti_aliasing=False,   # no smoothing (would create fractional values)
        preserve_range=True,   # keep values in [0, 1] not normalised further
    )
    return (raw > 0.5).astype(np.float64)   # strict binarisation


def mask_aware_downsample(
    img_corrupt: np.ndarray,
    mask: np.ndarray,
    target_size: tuple,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Downsample a corrupted image without leaking missing-pixel zeros into
    nearby observed pixels.  Uses normalised convolution (Shepard form).

    Why a plain sk_resize fails
    ---------------------------
    sk_resize(img_corrupt, target_size, anti_aliasing=True) applies a Gaussian
    blur before subsampling.  Because img_corrupt has zeros at missing
    positions, that blur drags observed pixels near a hole boundary toward
    zero — biasing the downsampled "observed" values darker than they should
    be.  The K-SVD pipeline at the coarse level then trains on systematically
    biased pixels (a silent dataset bias, not an algorithmic bug).

    Normalised convolution
    ----------------------
    Treat each input pixel's mask value as a confidence weight.  Smooth-and-
    subsample both the numerator (img * mask) and the denominator (mask) with
    the SAME anti-aliasing kernel, then divide.  The result at each output
    pixel is the weighted average of ONLY the observed input pixels that fall
    within the kernel's support; zeros at missing positions contribute nothing.

        img_low[y, x] = (k * (mask . img))[y, x]
                        ----------------------------
                        (k * mask)[y, x]   + eps

    where k is the anti-aliasing kernel implicit in sk_resize.

    Edge case
    ---------
    At output pixels whose entire kernel support is missing (denominator ~ 0)
    we return 0.  The caller subsequently multiplies by the strict binary
    downsampled mask (downsample_mask), so any such position lies inside a
    downsampled hole and its value is irrelevant to patch extraction.

    Sanity (strict generalisation of the original code)
    ---------------------------------------------------
    For a fully observed input (mask identically 1) the denominator is 1
    everywhere and the result reduces exactly to
        sk_resize(img, target_size, anti_aliasing=True, preserve_range=True),
    so substituting this for the original sk_resize call cannot regress the
    no-missing-pixels case.

    Parameters
    ----------
    img_corrupt : (H, W)   zero-filled corrupted image (= img * mask)
    mask        : (H, W)   binary float64 mask (1=observed, 0=missing)
    target_size : (H', W') target resolution
    eps         : float    numerical floor for division

    Returns
    -------
    img_low : (H', W') float64, mask-weighted downsampled estimate
    """
    num = sk_resize(
        img_corrupt, target_size,
        anti_aliasing=True, preserve_range=True,
    )
    den = sk_resize(
        mask, target_size,
        anti_aliasing=True, preserve_range=True,
    )
    # np.maximum(den, eps) protects the False branch from divide-by-zero
    # warnings: numpy evaluates BOTH np.where branches before selecting.
    return np.where(den > eps, num / np.maximum(den, eps), 0.0)


def corrupt_image(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply binary mask to image: missing pixels set to 0.0.

    We do NOT use NaN here because sklearn.extract_patches_2d rejects NaN.
    The mask is propagated separately; NaN is reintroduced at patch level
    inside extract_and_translate_patches().
    """
    return img * mask


# =============================================================================
# PATCH UTILITIES  (reusable across all pyramid levels)
# =============================================================================

def extract_and_translate_patches(
    image_filled: np.ndarray,
    mask: np.ndarray,
) -> tuple:
    """
    Extract overlapping 8×8 patches and perform the mandatory shape translation.

        INPUT image must contain no NaN (it may be zero-filled or prior-filled).
        OUTPUT includes both:
            • X_filled (no NaN), used for reconstruction sparse coding.
            • X_nan (strictly masked), used for dictionary learning.

    Shape Translation (annotated step by step)
    -------------------------------------------
    sklearn returns:
        raw_img_patches  shape: (N_patches, 8, 8)   ← patches-first
        raw_mask_patches shape: (N_patches, 8, 8)

    Step 1  .reshape(N, 64)
        Collapses the two spatial dims (8, 8) into one 64-d vector.
        Row-major (C order): pixel at spatial position [i, j] → flat index i*8+j.
        Shapes:  (N, 8, 8) → (N, 64)

    Step 2  .T
        Transposes to features-first layout.
        Shape: (N, 64) → (64, N)
        Now: column i = all 64 pixels of patch i.

    NaN injection:
        X_nan[j, i] = NaN  where  M_flat[j, i] == 0
        Signals masked_omp to ignore that pixel for signal i.

    Parameters
    ----------
    image_filled : (H, W) image at target resolution, with no NaN
    mask         : (H, W) binary float mask (1=observed, 0=missing)

    Returns
    -------
    X_filled : (64, N) patch matrix preserving prior-filled values (no NaN)
    X_nan    : (64, N) patch matrix with NaN at unobserved entries
    M_flat   : (64, N) binary mask matrix matching both X matrices
    N        : int     total number of overlapping patches
    """
    H, W = image_filled.shape
    assert image_filled.shape == mask.shape, \
        f"image and mask shape mismatch: {image_filled.shape} vs {mask.shape}"
    assert not np.isnan(image_filled).any(), \
        "image_filled must NOT contain NaN before patch extraction"

    # --- sklearn extraction ---------------------------------------------------
    # extract_patches_2d uses step=1 (default), producing (H-8+1)*(W-8+1) patches
    raw_img  = extract_patches_2d(image_filled, PATCH_SIZE)   # (N, 8, 8)
    raw_mask = extract_patches_2d(mask,       PATCH_SIZE)   # (N, 8, 8)

    N = raw_img.shape[0]
    # Sanity: N == (H-8+1)*(W-8+1)
    assert N == (H - 8 + 1) * (W - 8 + 1), \
        f"Expected {(H-8+1)*(W-8+1)} patches, got {N}"

    # --- Shape Translation: (N, 8, 8) → (64, N) ------------------------------
    # Step 1: flatten spatial dimensions
    X_flat = raw_img.reshape(N, PATCH_DIM)      # (N, 64)
    M_flat = raw_mask.reshape(N, PATCH_DIM)     # (N, 64)

    # Step 2: transpose to features-first
    X_flat = X_flat.T                           # (64, N)
    M_flat = M_flat.T                           # (64, N)

    # Preserve prior-filled values for reconstruction sparse coding.
    X_filled = X_flat.copy()

    # --- NaN injection: mark missing positions --------------------------------
    # M_flat[j, i] == 0  ⟺  pixel j of patch i was originally missing.
    # Setting X_nan[j, i] = NaN signals masked_omp to exclude that entry
    # from residual computation and atom correlation.
    X_nan = X_flat.copy()
    X_nan[M_flat == 0] = np.nan

    return X_filled, X_nan, M_flat, N


def subsample_for_training(
    X: np.ndarray,
    M: np.ndarray,
    n_train: int,
    seed: int,
) -> tuple:
    """
    Randomly sample n_train columns from (64, N) patch matrices.

    The SAME random index is used for X and M to preserve the
    patch ↔ mask correspondence: column i of X_train and column i of
    M_train must describe the SAME spatial patch.

    Parameters
    ----------
    X       : (64, N)   patches with NaN at missing positions
    M       : (64, N)   binary masks
    n_train : int       number of patches to draw
    seed    : int

    Returns
    -------
    X_train : (64, n_train)
    M_train : (64, n_train)
    """
    N   = X.shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=min(n_train, N), replace=False)
    # Identical index applied to both matrices → correspondence preserved
    return X[:, idx], M[:, idx]


def reconstruct_image_from_codes(
    D: np.ndarray,
    Alpha: np.ndarray,
    image_size: tuple,
) -> np.ndarray:
    """
    Reconstruct an image from sparse codes and dictionary.

    Inverse Shape Translation (annotated step by step):
    ---------------------------------------------------
    Step 1  D @ Alpha
        Matrix product: (64, K) @ (K, N) = (64, N)
        Each column is the 64-pixel reconstruction of one patch.

    Step 2  .T
        Transpose back to patches-first layout.
        Shape: (64, N) → (N, 64)

    Step 3  .reshape(N, 8, 8)
        Unfolds the 64-d flat vector back to an 8×8 spatial grid.
        Row-major (C order): flat index k → spatial position [k//8, k%8].
        Shape: (N, 64) → (N, 8, 8)

    Step 4  reconstruct_from_patches_2d
        Sums all overlapping 8×8 patches at each pixel and divides by the
        number of patches contributing to that pixel (the overlap count).
        This averaging acts as a local low-pass filter that suppresses
        patch-boundary artefacts.
        Shape: (N, 8, 8) → (H, W)

    Parameters
    ----------
    D          : (64, K)  dictionary
    Alpha      : (K, N)   sparse codes
    image_size : (H, W)   target image shape

    Returns
    -------
    img_recon : (H, W) float64, clipped to [0, 1]
    """
    N = Alpha.shape[1]

    # Step 1: synthesise patches
    patches_flat = D @ Alpha              # (64, N)

    # Step 2: transpose to patches-first
    patches_T    = patches_flat.T         # (N, 64)

    # Step 3: unfold flat 64-d vector → 8×8 spatial grid
    H, W = image_size
    patches_2d   = patches_T.reshape(N, PATCH_SIZE[0], PATCH_SIZE[1])   # (N, 8, 8)

    # Step 4: average overlapping patches
    img_recon = reconstruct_from_patches_2d(patches_2d, image_size)     # (H, W)

    return np.clip(img_recon, 0.0, 1.0)


# =============================================================================
# CORE PIPELINE: LEARN + SPARSE CODE + RECONSTRUCT (one pyramid level)
# =============================================================================

def learn_and_reconstruct(
    X_all_filled: np.ndarray,
    X_all_nan: np.ndarray,
    M_all: np.ndarray,
    use_true_mask_for_coding: bool,
    image_size: tuple,
    n_train: int,
    n_iter: int,
    level_label: str,
    seed: int = SEED,
) -> tuple:
    """
    Execute one complete pyramid level with decoupled masking:
        (1) Subsample training patches
        (2) Learn dictionary D via masked_ksvd on strict masked patches
        (3) Sparse-code ALL patches from prior-filled patches with dynamic mask
        (4) Reconstruct image

    Parameters
    ----------
    X_all_filled : (64, N) all patches with prior values, no NaN
    X_all_nan    : (64, N) all patches with NaN at missing positions
    M_all      : (64, N)   binary masks for all patches
    use_true_mask_for_coding : bool
        True  -> Phase D uses M_all[:, i] for masked OMP
        False -> Phase D uses all-ones mask for masked OMP
    image_size : (H, W)    target reconstruction shape
    n_train    : int       dictionary-learning subset size
    n_iter     : int       K-SVD outer iterations
    level_label: str       display label for progress output
    seed       : int

    Returns
    -------
    D         : (64, 256)   learned dictionary
    Alpha_all : (256, N)    sparse codes for all patches
    recon_img : (H, W)      reconstructed image
    train_err : (n_iter,)   masked RMSE curve during learning
    """
    N = X_all_filled.shape[1]
    assert X_all_nan.shape == X_all_filled.shape == M_all.shape, \
        f"Shape mismatch at {level_label}: X_filled={X_all_filled.shape}, " \
        f"X_nan={X_all_nan.shape}, M={M_all.shape}"
    assert not np.isnan(X_all_filled).any(), \
        f"[{level_label}] X_all_filled must not contain NaN"

    # ---- (1) Training subset ------------------------------------------------
    X_train_nan, M_train = subsample_for_training(X_all_nan, M_all, n_train, seed)

    # ---- (2) Dictionary learning on training subset -------------------------
    D, _, train_err = masked_ksvd(
        X_train_nan, M_train,
        K=N_ATOMS, s=SPARSITY,
        n_iter=n_iter, als_iters=ALS_ITERS,
        seed=seed, label=level_label,
    )

    # Verify unit-norm columns — required for correct OMP normalisation
    col_norms = np.linalg.norm(D, axis=0)
    assert np.allclose(col_norms, 1.0, atol=1e-5), \
        f"[{level_label}] Dictionary not unit-norm! min={col_norms.min():.5f}"

    # ---- (3) Sparse-code ALL patches with learned D -------------------------
    # We use a separate full sparse-coding pass (not limited to training subset)
    # so that every spatial position in the image is covered.
    #
    # Phase-D coding mask is dynamic by level:
    #   use_true_mask_for_coding=True  -> ignore missing entries (strict mask)
    #   use_true_mask_for_coding=False -> treat X_all_filled as fully observed
    print(f"\n  [{level_label}] Sparse-coding {N} patches ...")
    Alpha_all = np.zeros((N_ATOMS, N), dtype=np.float64)
    t0 = time.time()

    all_ones = np.ones(X_all_filled.shape[0], dtype=np.float64)
    for i in range(N):
        coding_mask = M_all[:, i] if use_true_mask_for_coding else all_ones
        Alpha_all[:, i] = masked_omp(X_all_filled[:, i], D, coding_mask, SPARSITY)

        # Progress report every 3000 patches
        if i > 0 and i % 3000 == 0:
            elapsed = time.time() - t0
            rate    = i / elapsed
            eta     = (N - i) / rate
            print(f"    patch {i}/{N}  ({rate:.0f} patches/s, ETA {eta:.1f}s)")

    sparse_time = time.time() - t0
    print(f"  [{level_label}] Sparse coding done in {sparse_time:.2f}s")

    # ---- (4) Reconstruct image ----------------------------------------------
    recon_img = reconstruct_image_from_codes(D, Alpha_all, image_size)

    return D, Alpha_all, recon_img, train_err


# =============================================================================
# MULTI-SCALE PYRAMID
# =============================================================================

def inject_prior(
    img_corrupt_zero: np.ndarray,
    mask: np.ndarray,
    prior: np.ndarray,
    level_label: str,
) -> np.ndarray:
    """
    Fill missing pixels of img_corrupt_zero with upsampled prior values.

    Operation:  img_filled[mask == 0] = prior[mask == 0]

        WHY this is done (with decoupled masks):
        ----------------------------------------
        The prior is the upsampled reconstruction from the coarser level.
        After injection:
                img_filled[j] = img_observed[j]  if mask[j]=1
                img_filled[j] = prior[j]         if mask[j]=0

        In this file, dictionary learning and sparse coding are intentionally
        decoupled:
            • Phase C learns D from strict masked patches (X_nan, M).
                        • Phase D uses a dynamic coding mask:
                            - Level 3: true mask M (ignore zero-filled missing entries)
                            - Levels 2/1: all-ones mask (use injected prior as signal)

        This lets center-hole patches (which would otherwise have 0 observed
        entries) be represented from the injected prior while still learning atoms
        only from truly observed pixels.

    Parameters
    ----------
    img_corrupt_zero : (H, W) zero-filled corrupted image at this level
    mask             : (H, W) binary mask (1=observed, 0=missing)
    prior            : (H, W) upsampled coarse reconstruction (same size)
    level_label      : str

    Returns
    -------
    img_filled : (H, W)  zero-filled where observed, prior where missing
    """
    assert img_corrupt_zero.shape == mask.shape == prior.shape, \
        f"Shape mismatch in inject_prior at {level_label}"

    img_filled = img_corrupt_zero.copy()
    n_injected = int((mask == 0).sum())
    img_filled[mask == 0] = prior[mask == 0]

    print(f"  [{level_label}] Prior injection: filled {n_injected} missing pixels "
          f"from upsampled coarse reconstruction.")
    print(f"    img_filled range: [{img_filled.min():.4f}, {img_filled.max():.4f}]")

    return img_filled


def run_pyramid(
    img_clean:    np.ndarray,
    img_corrupt:  np.ndarray,
    mask_128:     np.ndarray,
) -> dict:
    """
    Execute the full 3-level Coarse-to-Fine pyramid.

    Level 3 (32×32): Learn from scratch using original mask + image.
    Level 2 (64×64): Learn with prior injection from Level 3 result.
    Level 1 (128×128): Learn with prior injection from Level 2 result.

    At each level:
      • Image and mask are downsampled from the reference 128×128 versions.
      • The mask is downsampled with order=0 + threshold (strictly binary).
      • The image is downsampled normally (anti_aliasing=True) then multiplied
        by the binary downsampled mask to zero-fill missing pixels.

    Returns
    -------
    results : dict with keys:
        'recon_32'    : (32, 32)   Level 3 reconstruction
        'recon_64'    : (64, 64)   Level 2 reconstruction
        'recon_128'   : (128, 128) Level 1 final reconstruction
        'err_L3'      : (n_iter,)  training RMSE curve at Level 3
        'err_L2'      : (n_iter,)  training RMSE curve at Level 2
        'err_L1'      : (n_iter,)  training RMSE curve at Level 1
        'mask_32'     : (32, 32)   binary mask at 32×32
        'mask_64'     : (64, 64)   binary mask at 64×64
    """
    results = {}
    t_pyramid = time.time()

    # =========================================================================
    # LEVEL 3 — 32×32 (Coarsest)
    # =========================================================================
    # At this resolution, the original 20×20 hole (400 px at 128×128) becomes
    # a 5×5 hole (25 px at 32×32).
    #
    # CRITICAL: a 5×5 hole is SMALLER than our 8×8 patch.
    # Every pixel inside the 5×5 hole is covered by at least one 8×8 patch
    # that also contains observed pixels, so masked OMP always receives a
    # non-trivial signal and returns a meaningful sparse code.
    # This is the fundamental reason the pyramid works.
    # =========================================================================
    print()
    print("=" * 68)
    print("  LEVEL 3  (32×32)  — Coarsest scale")
    print("=" * 68)

    cfg3  = LEVEL_CFG[3]
    sz3   = cfg3["size"]   # (32, 32)

    # --- Downsample image (mask-aware: prevents zero-leakage from holes) ----
    # A plain sk_resize would Gaussian-blur the zero-filled missing pixels
    # into nearby observed pixels and bias the coarse "observed" values dark.
    # mask_aware_downsample uses normalised convolution to exclude missing
    # pixels from the contribution to each output value.
    img_32  = mask_aware_downsample(img_corrupt, mask_128, sz3)  # (32, 32)

    # --- Downsample mask (nearest-neighbour + threshold → strict binary) -----
    mask_32 = downsample_mask(mask_128, sz3)                   # (32, 32) ∈ {0,1}
    results['mask_32'] = mask_32

    # --- Zero-fill: enforce strict-mask boundary --------------------------
    # img_32 is mask-aware so observed values are unbiased.  mask_32 (nearest-
    # neighbour) defines the strict binary boundary used during patch
    # extraction; multiplying enforces that boundary in the image as well.
    img_32_zero = img_32 * mask_32                             # (32, 32)

    rows_miss32 = int((mask_32 == 0).sum())
    print(f"  Missing at 32×32: {rows_miss32} pixels  "
          f"({rows_miss32 / mask_32.size * 100:.1f}%)")

    # --- Extract patches (filled + strict-masked variants) -------------------
    X3_filled, X3_nan, M3, N3 = extract_and_translate_patches(img_32_zero, mask_32)
    print(f"  Patches: {N3} total  ({N3} = (32-8+1)²)")

    # --- Learn + reconstruct at Level 3 --------------------------------------
    _, _, recon_32, err_L3 = learn_and_reconstruct(
        X3_filled, X3_nan, M3,
        use_true_mask_for_coding=True,
        image_size=sz3,
        n_train=cfg3["n_train"],
        n_iter=cfg3["n_iter"],
        level_label="Level 3 (32×32)",
        seed=SEED,
    )
    results['recon_32'] = recon_32
    results['err_L3']   = err_L3

    print(f"  Level 3 reconstruction range: [{recon_32.min():.3f}, {recon_32.max():.3f}]")

    # =========================================================================
    # LEVEL 2 — 64×64 (Middle scale)
    # =========================================================================
    # At this resolution, the hole is 10×10.
    # A 10×10 hole in an 8×8 patch means some patches are entirely inside the
    # hole (all 64 pixels missing).  Prior injection from Level 3 partially
    # fills this cavity before dictionary learning, giving the boundary-straddling
    # patches a more informative neighbourhood.
    # =========================================================================
    print()
    print("=" * 68)
    print("  LEVEL 2  (64×64)  — Middle scale")
    print("=" * 68)

    cfg2  = LEVEL_CFG[2]
    sz2   = cfg2["size"]   # (64, 64)

    # --- Downsample 128×128 reference image (mask-aware) and mask -----------
    # See mask_aware_downsample docstring: normalised convolution prevents
    # zero-valued missing pixels from biasing the downsampled observed pixels.
    img_64     = mask_aware_downsample(img_corrupt, mask_128, sz2)  # (64, 64)
    mask_64    = downsample_mask(mask_128, sz2)                     # (64, 64) ∈ {0,1}
    img_64_zero = img_64 * mask_64
    results['mask_64'] = mask_64

    rows_miss64 = int((mask_64 == 0).sum())
    print(f"  Missing at 64×64: {rows_miss64} pixels  "
          f"({rows_miss64 / mask_64.size * 100:.1f}%)")

    # --- Upsample Level 3 reconstruction → prior for Level 2 ----------------
    # order=1 (bilinear) produces a smooth upsampled estimate.
    # This prior fills the 10×10 hole with content consistent with the coarser
    # 5×5 level reconstruction.
    prior_64 = sk_resize(
        recon_32, sz2,
        order=1,               # bilinear upsampling (smooth, no aliasing)
        anti_aliasing=False,   # no additional smoothing needed
        preserve_range=True,
    )
    prior_64 = np.clip(prior_64, 0.0, 1.0)
    print(f"  prior_64 (upsampled from recon_32): "
          f"range [{prior_64.min():.3f}, {prior_64.max():.3f}]")

    # --- PRIOR INJECTION: fill missing pixels with coarse estimate -----------
    # This produces img_64_filled where:
    #   img_64_filled[j] = img_64_zero[j]  if mask_64[j]=1  (true observation)
    #   img_64_filled[j] = prior_64[j]     if mask_64[j]=0  (missing ← prior fill)
    #
    # NOTE: Phase C will still use strict masking (X_nan + M) for learning.
    #       Phase D will code X_filled with all-ones mask so the prior is used
    #       during sparse coding/reconstruction.
    img_64_filled = inject_prior(img_64_zero, mask_64, prior_64, "Level 2 (64×64)")

    # --- Extract patches from the FILLED image and ORIGINAL mask -------------
    # Returns both X_filled (for Phase D coding) and X_nan (for Phase C learning).
    X2_filled, X2_nan, M2, N2 = extract_and_translate_patches(img_64_filled, mask_64)
    print(f"  Patches: {N2} total  ({N2} = (64-8+1)²)")

    # --- Learn + reconstruct at Level 2 --------------------------------------
    _, _, recon_64, err_L2 = learn_and_reconstruct(
        X2_filled, X2_nan, M2,
        use_true_mask_for_coding=False,
        image_size=sz2,
        n_train=cfg2["n_train"],
        n_iter=cfg2["n_iter"],
        level_label="Level 2 (64×64)",
        seed=SEED,
    )
    results['recon_64'] = recon_64
    results['err_L2']   = err_L2

    print(f"  Level 2 reconstruction range: [{recon_64.min():.3f}, {recon_64.max():.3f}]")

    # =========================================================================
    # LEVEL 1 — 128×128 (Finest scale)
    # =========================================================================
    # The original 20×20 hole: maximally challenging.
    # Prior injection from Level 2 fills the cavity before the final pass.
    # =========================================================================
    print()
    print("=" * 68)
    print("  LEVEL 1  (128×128)  — Finest scale  [final output]")
    print("=" * 68)

    cfg1  = LEVEL_CFG[1]
    sz1   = cfg1["size"]   # (128, 128)

    # --- Reference image and mask (already at 128×128) -----------------------
    img_128_zero = img_corrupt.copy()    # zero-filled, shape (128, 128)
    mask_128_    = mask_128.copy()       # binary, shape (128, 128)

    rows_miss128 = int((mask_128_ == 0).sum())
    print(f"  Missing at 128×128: {rows_miss128} pixels  "
          f"({rows_miss128 / mask_128_.size * 100:.1f}%)")

    # --- Upsample Level 2 reconstruction → prior for Level 1 ----------------
    prior_128 = sk_resize(
        recon_64, sz1,
        order=1,               # bilinear upsampling
        anti_aliasing=False,
        preserve_range=True,
    )
    prior_128 = np.clip(prior_128, 0.0, 1.0)
    print(f"  prior_128 (upsampled from recon_64): "
          f"range [{prior_128.min():.3f}, {prior_128.max():.3f}]")

    # --- PRIOR INJECTION at 128×128 ------------------------------------------
    img_128_filled = inject_prior(img_128_zero, mask_128_, prior_128, "Level 1 (128×128)")

    # --- Extract patches from the FILLED image and original mask -------------
    X1_filled, X1_nan, M1, N1 = extract_and_translate_patches(img_128_filled, mask_128_)
    print(f"  Patches: {N1} total  ({N1} = (128-8+1)²)")

    # --- Learn + reconstruct at Level 1 (final output) -----------------------
    _, _, recon_128, err_L1 = learn_and_reconstruct(
        X1_filled, X1_nan, M1,
        use_true_mask_for_coding=False,
        image_size=sz1,
        n_train=cfg1["n_train"],
        n_iter=cfg1["n_iter"],
        level_label="Level 1 (128×128)",
        seed=SEED,
    )
    results['recon_128'] = recon_128
    results['err_L1']    = err_L1

    print(f"  Level 1 reconstruction range: [{recon_128.min():.3f}, {recon_128.max():.3f}]")
    print(f"\n  Total pyramid wall-clock time: {time.time()-t_pyramid:.1f}s")

    return results


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(img_true: np.ndarray, img_pred: np.ndarray, label: str) -> dict:
    """
    PSNR and SSIM on the full image (including previously-missing pixels).
    Standard inpainting evaluation protocol.
    """
    p = psnr_fn(img_true, img_pred, data_range=1.0)
    s = ssim_fn(img_true, img_pred, data_range=1.0)
    print(f"  {label:<45s}: PSNR = {p:6.2f} dB   SSIM = {s:.4f}")
    return {"psnr": p, "ssim": s}


# =============================================================================
# VISUALISATION
# =============================================================================

def plot_results(
    img_clean:    np.ndarray,
    img_corrupt:  np.ndarray,
    mask_128:     np.ndarray,
    img_bh:       np.ndarray,
    recon_32:     np.ndarray,
    recon_64:     np.ndarray,
    recon_128:    np.ndarray,
    metrics:      dict,
    err_L3:       np.ndarray,
    err_L2:       np.ndarray,
    err_L1:       np.ndarray,
    save_path:    str,
) -> None:
    """
    Multi-panel figure:
      Row 1: Original | Corrupted | Biharmonic | L3 (32→128) | L2 (64→128) | L1 Final
      Row 2: Convergence curves (3 levels) | Error maps (3 algorithms)
    """
    BLUE   = '#2563EB'
    RED    = '#DC2626'
    GREEN  = '#16A34A'
    ORANGE = '#D97706'
    PURPLE = '#7C3AED'

    fig = plt.figure(figsize=(26, 11))
    gs  = gridspec.GridSpec(2, 6, figure=fig, hspace=0.48, wspace=0.12)

    # ---- Helper: image panel ------------------------------------------------
    def img_panel(ax_pos, img, title, subtitle="", cmap='gray'):
        ax = fig.add_subplot(ax_pos)
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(title, fontsize=10, fontweight='bold', pad=3)
        if subtitle:
            ax.set_xlabel(subtitle, fontsize=9, labelpad=2)
        ax.set_xticks([]); ax.set_yticks([])
        return ax

    # Row 1, Col 0: Original
    img_panel(gs[0, 0], img_clean,
              "Original", "128×128 ground truth")

    # Row 1, Col 1: Corrupted
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(img_corrupt, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    # Overlay hole region in red
    hole_rgba = np.zeros((*mask_128.shape, 4))
    hole_rgba[..., 0] = 1.0
    hole_rgba[..., 3] = (mask_128 == 0) * 0.7
    ax.imshow(hole_rgba, interpolation='nearest')
    ax.set_title("Corrupted", fontsize=10, fontweight='bold', pad=3)
    ax.set_xlabel(f"{(mask_128 == 0).mean()*100:.1f}% missing pixels (red)",
                  fontsize=9, labelpad=2)
    ax.set_xticks([]); ax.set_yticks([])

    # Row 1, Col 2: Biharmonic
    m = metrics['biharmonic']
    img_panel(gs[0, 2], img_bh,
              "Biharmonic Inpainting",
              f"PSNR={m['psnr']:.1f}dB  SSIM={m['ssim']:.3f}")

    # Row 1, Col 3: Level 3 result (upsampled to 128 for display)
    recon_32_up = sk_resize(recon_32, IMAGE_SIZE, order=1, anti_aliasing=False,
                             preserve_range=True)
    m = metrics['level3']
    img_panel(gs[0, 3], recon_32_up,
              "Level 3 Recon\n(32→128 upsampled)",
              f"PSNR={m['psnr']:.1f}dB  SSIM={m['ssim']:.3f}")

    # Row 1, Col 4: Level 2 result (upsampled to 128 for display)
    recon_64_up = sk_resize(recon_64, IMAGE_SIZE, order=1, anti_aliasing=False,
                             preserve_range=True)
    m = metrics['level2']
    img_panel(gs[0, 4], recon_64_up,
              "Level 2 Recon\n(64→128 upsampled)",
              f"PSNR={m['psnr']:.1f}dB  SSIM={m['ssim']:.3f}")

    # Row 1, Col 5: Final Level 1 result
    m = metrics['level1']
    img_panel(gs[0, 5], recon_128,
              "Final: Level 1\nMasked K-SVD (128×128)",
              f"PSNR={m['psnr']:.1f}dB  SSIM={m['ssim']:.3f}")

    # ---- Row 2, Col 0-2: Convergence curves (all 3 levels) -----------------
    ax_conv = fig.add_subplot(gs[1, 0:3])
    for errs, color, label in [
        (err_L3, ORANGE, f"Level 3 — 32×32  (10 iters, 500 train patches)"),
        (err_L2, PURPLE, f"Level 2 — 64×64  (12 iters, 1500 train patches)"),
        (err_L1, BLUE,   f"Level 1 — 128×128(15 iters, 3000 train patches)"),
    ]:
        iters = np.arange(1, len(errs) + 1)
        ax_conv.semilogy(iters, errs, 'o-', color=color, lw=2.0, ms=5,
                         mfc='white', mew=1.6, label=label)
        # Annotate final RMSE
        ax_conv.annotate(f"{errs[-1]:.5f}",
                         xy=(len(iters), errs[-1]),
                         xytext=(len(iters) - 4, errs[-1] * 0.6),
                         fontsize=8, color=color,
                         arrowprops=dict(arrowstyle='->', color=color, lw=1.0))

    ax_conv.set_xlabel("K-SVD Iteration", fontsize=11)
    ax_conv.set_ylabel("Masked RMSE — training patches (log scale)", fontsize=10)
    ax_conv.set_title(
        "Multi-Scale Training Convergence\n"
        "(Prior injection at Levels 2 & 3; ALS Rank-1 atom update)",
        fontsize=11, fontweight='bold'
    )
    ax_conv.grid(True, which='both', alpha=0.35)
    ax_conv.legend(fontsize=9, loc='upper right')

    # ---- Row 2, Col 3-5: Absolute error maps --------------------------------
    vmax_err = 0.25   # fixed scale: all maps on same colour axis

    def error_panel(ax_pos, img_pred, title, color):
        ax   = fig.add_subplot(ax_pos)
        err  = np.abs(img_pred - img_clean)
        ax.imshow(err, cmap='hot', vmin=0, vmax=vmax_err, interpolation='nearest')
        mean_all  = err.mean()
        ax.set_title(title, fontsize=10, fontweight='bold', color=color, pad=3)
        ax.set_xlabel(
            f"full MAE={mean_all:.4f}",
            fontsize=8, labelpad=2
        )
        ax.set_xticks([]); ax.set_yticks([])
        return ax

    error_panel(gs[1, 3], img_bh,       "Biharmonic Error",     GREEN)
    error_panel(gs[1, 4], recon_64_up,  "Level 2 Error",        PURPLE)
    ax_err = error_panel(gs[1, 5], recon_128, "Level 1 Error",  BLUE)

    # Shared colorbar (attach to last error panel)
    sm = plt.cm.ScalarMappable(cmap='hot', norm=Normalize(vmin=0, vmax=vmax_err))
    sm.set_array([])
    fig.colorbar(sm, ax=fig.axes[-3:], fraction=0.018, pad=0.04,
                 label='|reconstruction error|')

    fig.suptitle(
        "Phase 4 — Multi-Scale Masked Dictionary Learning  |  Coarse-to-Fine Pyramid\n"
        "128×128 image  |  custom mask  |  8×8 patches  |  "
        "D: 64×256  |  sparsity s=10  |  3-level pyramid",
        fontsize=12, fontweight='bold', y=1.01
    )

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Figure saved → {save_path}")


# =============================================================================
# MAIN
# =============================================================================

def parse_cli_args() -> argparse.Namespace:
    """Parse CLI flags for custom mask selection and parameters."""
    parser = argparse.ArgumentParser(
        description="Phase 4 multiscale inpainting with custom masks."
    )
    parser.add_argument(
        "--mask-mode",
        choices=MASK_MODES,
        default=DEFAULT_MASK_MODE,
        help="Mask pattern to apply at 128x128 before pyramid processing.",
    )
    parser.add_argument(
        "--missing-frac",
        type=float,
        default=DEFAULT_MISSING_FRAC,
        help="Missing fraction used when --mask-mode random.",
    )
    parser.add_argument(
        "--mask-seed",
        type=int,
        default=SEED,
        help="Random seed used by mask generators.",
    )
    parser.add_argument(
        "--square-size",
        type=int,
        default=DEFAULT_SQUARE_SIZE,
        help="Square side length for face-square mask modes.",
    )
    parser.add_argument(
        "--square-center",
        type=int,
        nargs=2,
        metavar=("ROW", "COL"),
        default=DEFAULT_SQUARE_CENTER,
        help="Square center as ROW COL for face-square mask modes.",
    )
    parser.add_argument(
        "--scratch-count",
        type=int,
        default=DEFAULT_SCRATCH_COUNT,
        help="Number of diagonal scratches for scratches mask modes.",
    )
    parser.add_argument(
        "--scratch-thickness",
        type=int,
        default=DEFAULT_SCRATCH_THICKNESS,
        help="Line thickness for scratches mask modes.",
    )
    parser.add_argument(
        "--overlay-text",
        type=str,
        default=DEFAULT_OVERLAY_TEXT,
        help="Text string for text-overlay mask mode.",
    )
    parser.add_argument(
        "--text-scale",
        type=float,
        default=DEFAULT_TEXT_SCALE,
        help="Text size fraction for text-overlay mask mode.",
    )
    parser.add_argument(
        "--text-angle",
        type=float,
        default=DEFAULT_TEXT_ANGLE,
        help="Text rotation angle (degrees) for text-overlay mode.",
    )
    parser.add_argument(
        "--text-stroke",
        type=int,
        default=DEFAULT_TEXT_STROKE,
        help="Stroke width for text-overlay mode.",
    )
    return parser.parse_args()

def main():
    args = parse_cli_args()
    square_center = tuple(args.square_center) if args.square_center is not None else None

    t_main = time.time()

    print("=" * 68)
    print("  Phase 4: Multi-Scale Masked Dictionary Learning")
    print("  Coarse-to-Fine Image Pyramid with Prior Injection")
    print("=" * 68)
    print(f"  Image size  : {IMAGE_SIZE}")
    print(f"  Mask mode   : {args.mask_mode}")
    if args.mask_mode == "random":
        print(f"  Missing frac: {args.missing_frac*100:.1f}%")
    if args.mask_mode == "face-square":
        print(f"  Square mask : size={args.square_size}, center={square_center}")
    if args.mask_mode == "scratches":
        print(f"  Scratches   : count={args.scratch_count}, thickness={args.scratch_thickness}")
    if args.mask_mode == "face-square+scratches":
        print(f"  Square mask : size={args.square_size}, center={square_center}")
        print(f"  Scratches   : count={args.scratch_count}, thickness={args.scratch_thickness}")
    if args.mask_mode == "text-overlay":
        print(f"  Text overlay: '{args.overlay_text}', scale={args.text_scale}, "
              f"angle={args.text_angle}, stroke={args.text_stroke}")
    print(f"  Dictionary  : {PATCH_DIM}×{N_ATOMS}  ({N_ATOMS//PATCH_DIM}× overcomplete)")
    print(f"  Sparsity    : s = {SPARSITY}")
    print(f"  Pyramid     : 32×32 → 64×64 → 128×128")
    print()

    # ---- Load & corrupt -------------------------------------------------------
    print("[Data Prep]  Loading image & applying selected custom mask ...")
    img_clean   = load_image()
    mask_128    = make_custom_mask(
        image_shape=IMAGE_SIZE,
        mask_mode=args.mask_mode,
        missing_frac=args.missing_frac,
        seed=args.mask_seed,
        square_size=args.square_size,
        square_center=square_center,
        scratch_count=args.scratch_count,
        scratch_thickness=args.scratch_thickness,
        overlay_text=args.overlay_text,
        text_scale=args.text_scale,
        text_angle=args.text_angle,
        text_stroke=args.text_stroke,
    )
    img_corrupt = corrupt_image(img_clean, mask_128)

    print(f"  img_clean   : {img_clean.shape}  range [{img_clean.min():.3f},{img_clean.max():.3f}]")
    print(f"  mask_128    : {mask_128.shape}   "
          f"missing = {int((mask_128==0).sum())} px "
          f"({(mask_128==0).mean()*100:.1f}%)")
    print()

    # ---- Biharmonic baseline --------------------------------------------------
    print("[Baseline]  Running biharmonic inpainting ...")
    bh_mask = (mask_128 == 0).astype(bool)   # True = pixel to inpaint
    t0      = time.time()
    img_bh  = np.clip(inpaint_biharmonic(img_corrupt, bh_mask), 0.0, 1.0)
    print(f"  Biharmonic done in {time.time()-t0:.2f}s")
    print()

    # ---- Multi-scale pyramid --------------------------------------------------
    print("[Pyramid]  Executing 3-level Coarse-to-Fine Masked K-SVD ...")
    results = run_pyramid(img_clean, img_corrupt, mask_128)

    recon_32  = results['recon_32']
    recon_64  = results['recon_64']
    recon_128 = results['recon_128']

    # ---- Upsample L3 and L2 to 128 for fair comparison -----------------------
    recon_32_at_128 = sk_resize(recon_32, IMAGE_SIZE, order=1,
                                 anti_aliasing=False, preserve_range=True)
    recon_64_at_128 = sk_resize(recon_64, IMAGE_SIZE, order=1,
                                 anti_aliasing=False, preserve_range=True)

    # ---- Evaluation -----------------------------------------------------------
    print()
    print("=" * 68)
    print("  EVALUATION  (full 128×128 image)")
    print("=" * 68)
    metrics = {
        'biharmonic': evaluate(img_clean, img_bh,          "Biharmonic inpainting"),
        'level3':     evaluate(img_clean, recon_32_at_128, "Level 3 (32→128 bilinear upsample)"),
        'level2':     evaluate(img_clean, recon_64_at_128, "Level 2 (64→128 bilinear upsample)"),
        'level1':     evaluate(img_clean, recon_128,       "Level 1 — Masked K-SVD FINAL"),
    }

    # ---- PSNR improvement summary -------------------------------------------
    psnr_bh  = metrics['biharmonic']['psnr']
    psnr_l1  = metrics['level1']['psnr']
    delta    = psnr_l1 - psnr_bh
    print()
    print(f"  Masked K-SVD (L1) vs Biharmonic : PSNR {delta:+.2f} dB  "
          f"({'better' if delta > 0 else 'worse'} than biharmonic)")
    print(f"  Level progression  L3→L2→L1 PSNR: "
          f"{metrics['level3']['psnr']:.2f} → "
          f"{metrics['level2']['psnr']:.2f} → "
          f"{metrics['level1']['psnr']:.2f} dB")

    # ---- Plot -----------------------------------------------------------------
    print()
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "outputs",
        "phase4_multiscale_results.png",
    )
    plot_results(
        img_clean, img_corrupt, mask_128,
        img_bh,
        recon_32, recon_64, recon_128,
        metrics,
        results['err_L3'], results['err_L2'], results['err_L1'],
        save_path=out_path,
    )

    print(f"\n  Total wall-clock time: {time.time()-t_main:.1f}s")
    print("=" * 68)


if __name__ == "__main__":
    main()
