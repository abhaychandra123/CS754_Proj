"""
=============================================================================
Phase 3: 2D Image Inpainting Pipeline — Masked Dictionary Learning
=============================================================================

Builds on Phase 1 & 2 by wrapping the masked K-SVD math in a full 2D image
processing pipeline using overlapping 8x8 patches.

Architecture
------------
(A) Data prep  : Load camera image → resize 128x128 → float [0,1] → corrupt 30%
(B) Extraction : sklearn extract_patches_2d → shape translation (N,8,8) → (64,N)
(C) Learning   : masked_ksvd on 3000 training patches  → D (64×256)
(D) Coding     : masked_omp on ALL 14641 patches with learned D → Alpha (256×N)
(E) Recon      : D @ Alpha → (64,N) → (N,8,8) → reconstruct_from_patches_2d
(F) Baselines  : zero-imputation K-SVD + biharmonic inpainting
(G) Evaluation : PSNR + SSIM on full 128x128 reconstructions

Shape Translation (CRITICAL — explicitly annotated throughout)
--------------
  sklearn output : (N_patches, 8, 8)   — patches indexed first
  our API input  : (64, N_patches)     — signal dimension first, patches second

  Forward:   patches.reshape(N, 64).T      # (N,8,8) -> (N,64) -> (64,N)
  Inverse:   codes.T.reshape(N, 8, 8)      # (64,N)  -> (N,64) -> (N,8,8)

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
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity  as ssim_fn
from skimage.restoration import inpaint_biharmonic
from sklearn.feature_extraction.image import (
    extract_patches_2d,
    reconstruct_from_patches_2d,
)

# ---------------------------------------------------------------------------
# Import Phase 1 & 2 core functions — keep them EXACTLY as written
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from masked_ksvd import (
    masked_omp,
    masked_ksvd,
    baseline_ksvd,
    init_dictionary_from_data,
    masked_ksvd_update_atom,
)
from custom_masks import generate_mask, MASK_MODES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_SIZE    = (128, 128)
PATCH_SIZE    = (8, 8)
PATCH_DIM     = 64          # 8*8 flattened patch dimension
N_DICT_ATOMS  = 256         # overcomplete dictionary: 256 > 64 (4x)
SPARSITY      = 10          # nonzeros per patch code; standard for 8x8 patches
MISSING_FRAC  = 0.30        # fraction of pixels randomly dropped
N_TRAIN_PATCHES = 3000      # patches used for dictionary learning (speed)
N_KSVD_ITER   = 25        # K-SVD outer iterations
N_ALS_ITER    = 10          # ALS steps per atom update
SEED          = 42


# =============================================================================
# STEP A — Data Loading & Corruption
# =============================================================================

def load_and_corrupt_image(
    seed: int = SEED,
    mask_mode: str = "random",
    missing_frac: float = MISSING_FRAC,
    square_size: int = 20,
    square_center: tuple[int, int] | None = None,
    scratch_count: int = 3,
    scratch_thickness: int = 4,
):
    """
    Load the standard 'camera' grayscale image, resize to IMAGE_SIZE,
    normalise to float [0,1], and corrupt using the selected mask mode.

    Returns
    -------
    img_clean   : (H, W)  ground-truth float image
    img_corrupt : (H, W)  zero-filled corrupted image (missing = 0)
    mask        : (H, W)  float binary mask; 1=observed, 0=missing
    """
    # Load and resize
    img_raw   = skdata.camera()                              # (512,512) uint8
    img_clean = resize(img_raw.astype(np.float64) / 255.0,
                       IMAGE_SIZE, anti_aliasing=True)       # (128,128) float64

    # Generate binary mask: 1 = observed, 0 = missing
    mask = generate_mask(
        image_shape=IMAGE_SIZE,
        mode=mask_mode,
        missing_frac=missing_frac,
        seed=seed,
        square_size=square_size,
        square_center=square_center,
        scratch_count=scratch_count,
        scratch_thickness=scratch_thickness,
    )

    # Zero-fill missing pixels.
    # NaN is NOT used here because sklearn.extract_patches_2d rejects NaN.
    # The mask is propagated separately and NaN is reintroduced at patch level.
    img_corrupt = img_clean * mask    # missing pixels become 0.0

    print(f"  Image shape     : {img_clean.shape}")
    print(f"  Mask mode       : {mask_mode}")
    print(f"  Missing pixels  : {int((mask == 0).sum())} "
          f"({(mask == 0).mean()*100:.1f}%)")
    print(f"  Observed pixels : {int(mask.sum())} "
          f"({mask.mean()*100:.1f}%)")

    return img_clean, img_corrupt, mask


# =============================================================================
# STEP B — Patch Extraction & Shape Translation
# =============================================================================

def extract_patches(image_zero: np.ndarray, mask: np.ndarray):
    """
    Extract ALL overlapping 8x8 patches and translate shapes.

    sklearn's extract_patches_2d rejects NaN, so we:
      (1) Extract patches from the zero-filled image.
      (2) Extract patches from the binary mask.
      (3) After the shape translation to (64, N), reintroduce NaN
          where mask==0 so our masked_omp receives the correct signal.

    Shape Translation (CRITICAL)
    ----------------------------
    sklearn returns : (N_patches, 8, 8)         patches-first layout
    our API expects : (64, N_patches)            features-first layout

    Two-step translation:
        Step 1: reshape(N, 64)   — collapse the 8x8 spatial grid to a 64-d vector
                shape: (N, 8, 8) -> (N, 64)
        Step 2: .T               — transpose to features-first
                shape: (N, 64)   -> (64, N)

    Inverse (for reconstruction):
        Step 1: .T               — transpose back
                shape: (64, N)   -> (N, 64)
        Step 2: reshape(N, 8, 8) — unfold the 64-d vector to 8x8 spatial grid
                shape: (N, 64)   -> (N, 8, 8)

    Parameters
    ----------
    image_zero : (H, W)  zero-filled corrupted image (no NaN)
    mask       : (H, W)  binary float mask (1=observed, 0=missing)

    Returns
    -------
    X_patches : (64, N)  patch matrix with NaN at missing positions
    M_patches : (64, N)  binary mask matrix matching X_patches
    N_patches : int      total number of overlapping patches
    """
    # ---- Extract with sklearn -----------------------------------------------
    # Both calls must use the default step=1 so reconstruct_from_patches_2d
    # can aggregate them back correctly.
    raw_patches_img  = extract_patches_2d(image_zero, PATCH_SIZE)
    # raw_patches_img shape: (N_patches, 8, 8)

    raw_patches_mask = extract_patches_2d(mask, PATCH_SIZE)
    # raw_patches_mask shape: (N_patches, 8, 8)

    N = raw_patches_img.shape[0]
    # N = (H - 8 + 1) * (W - 8 + 1) = 121 * 121 = 14641 for 128x128

    # ---- Shape Translation: (N, 8, 8) -> (64, N) ----------------------------
    #
    # Step 1: .reshape(N, 64)
    #   Collapses the two spatial dimensions (8, 8) into a single 64-d vector.
    #   row-major (C order) flattening: pixel [i,j] -> index i*8 + j
    #   Result shape: (N, 64)
    #
    # Step 2: .T
    #   Transposes to put features (pixel positions) on axis 0.
    #   Result shape: (64, N)
    #   Now column i of the result = all 64 pixels of patch i.
    X_flat = raw_patches_img.reshape(N, PATCH_DIM).T     # (64, N)
    M_flat = raw_patches_mask.reshape(N, PATCH_DIM).T    # (64, N)

    # ---- Reintroduce NaN at missing positions --------------------------------
    # After shape translation, M_flat[j, i] = 1 iff pixel j of patch i is observed.
    # We set X_flat[j, i] = NaN wherever M_flat[j, i] == 0.
    # This is the form expected by masked_omp and masked_ksvd.
    X_nanfill = X_flat.copy()
    X_nanfill[M_flat == 0] = np.nan

    print(f"  Patch shape (sklearn raw) : {raw_patches_img.shape}  (N, 8, 8)")
    print(f"  Patch shape (our API)     : {X_nanfill.shape}         (64, N)")
    print(f"  NaN entries in X          : {np.isnan(X_nanfill).sum()} "
          f"({np.isnan(X_nanfill).mean()*100:.1f}%)")

    return X_nanfill, M_flat, N


def subsample_patches(X: np.ndarray, M: np.ndarray, n_train: int, seed: int = SEED):
    """
    Randomly sample n_train columns from (64, N) matrices.

    Dictionary learning (K-SVD) is trained on a representative subset of
    patches for speed. Sparse coding for reconstruction uses all N patches.
    The same random index set is used for both X and M to maintain alignment.

    Parameters
    ----------
    X       : (64, N)  patch matrix with NaN
    M       : (64, N)  binary mask matrix
    n_train : int      number of patches to sample
    seed    : int

    Returns
    -------
    X_train : (64, n_train)  sampled patches
    M_train : (64, n_train)  sampled masks
    """
    N   = X.shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=min(n_train, N), replace=False)

    # idx is the same for both X and M, preserving (patch, mask) correspondence.
    return X[:, idx], M[:, idx]


# =============================================================================
# STEP C+D — Learn Dictionary, Then Sparse-Code All Patches
# =============================================================================

def learn_and_reconstruct(
    X_all: np.ndarray,
    M_all: np.ndarray,
    X_train: np.ndarray,
    M_train: np.ndarray,
    label: str = "Masked K-SVD",
    use_masking: bool = True,
) -> tuple:
    """
    Two-phase pipeline:
        Phase (C): Learn dictionary D on X_train (n_train patches).
        Phase (D): Sparse-code ALL patches X_all with learned D.

    Using a training subset for dictionary learning is standard practice in
    dictionary-learning literature (e.g., Mairal et al. 2009) and avoids
    O(N_all * n_iter) OMP calls during the iterative learning phase.

    Parameters
    ----------
    X_all    : (64, N_all)    all patches for reconstruction, NaN at missing
    M_all    : (64, N_all)    all binary masks
    X_train  : (64, N_train)  training subset
    M_train  : (64, N_train)  training masks
    label    : str            display label
    use_masking : bool        True = masked K-SVD; False = zero-imputation baseline

    Returns
    -------
    D            : (64, 256)   learned dictionary
    Alpha_all    : (256, N_all) sparse codes for all patches
    train_errors : (n_iter,)   masked RMSE curve during learning
    recon_time   : float       total wall-clock seconds
    """
    t_start = time.time()
    n, N_all = X_all.shape

    # ---- Phase C: Dictionary learning on training subset -------------------
    if use_masking:
        D, _, train_errors = masked_ksvd(
            X_train, M_train,
            K=N_DICT_ATOMS, s=SPARSITY,
            n_iter=N_KSVD_ITER, als_iters=N_ALS_ITER,
            seed=SEED, label=label
        )
    else:
        D, _, train_errors = baseline_ksvd(
            X_train, M_train,
            K=N_DICT_ATOMS, s=SPARSITY,
            n_iter=N_KSVD_ITER, seed=SEED
        )

    # Verify unit-norm columns (mandatory for OMP correctness)
    col_norms = np.linalg.norm(D, axis=0)
    assert np.allclose(col_norms, 1.0, atol=1e-5), \
        f"Dictionary columns are not unit-norm! min={col_norms.min():.4f}"

    # ---- Phase D: Sparse-code ALL patches with learned D -------------------
    # This is a single (non-iterative) sparse coding pass.
    # We loop over all N_all patches, applying masked_omp or unmasked omp.
    print(f"\n  [{label}] Sparse-coding {N_all} patches for reconstruction ...")
    Alpha_all  = np.zeros((N_DICT_ATOMS, N_all))
    all_ones   = np.ones(n)       # unmasked (for baseline)

    t_omp = time.time()
    for i in range(N_all):
        if i % 3000 == 0 and i > 0:
            elapsed = time.time() - t_omp
            rate    = i / elapsed
            eta     = (N_all - i) / rate
            print(f"    patch {i}/{N_all}  ({rate:.0f} patches/s, ETA {eta:.1f}s)")

        if use_masking:
            Alpha_all[:, i] = masked_omp(X_all[:, i], D, M_all[:, i], SPARSITY)
        else:
            # Baseline: zero-imputed patch, no masking
            x_zero = np.nan_to_num(X_all[:, i], nan=0.0)
            Alpha_all[:, i] = masked_omp(x_zero, D, all_ones, SPARSITY)

    recon_time = time.time() - t_start
    print(f"  [{label}] Done. Total time: {recon_time:.1f}s")

    return D, Alpha_all, train_errors, recon_time


# =============================================================================
# STEP E — Reconstruction from Patch Codes
# =============================================================================

def patches_to_image(
    D: np.ndarray,
    Alpha: np.ndarray,
    image_shape: tuple,
) -> np.ndarray:
    """
    Reconstruct a full image from a dictionary and sparse codes.

    Inverse Shape Translation:
    --------------------------
        (1) D @ Alpha : (64, 256) @ (256, N) = (64, N)  — reconstructed patches
        (2) .T        : (64, N)  -> (N, 64)              — patches-first
        (3) .reshape  : (N, 64)  -> (N, 8, 8)            — unfold to spatial grid
        (4) reconstruct_from_patches_2d: average overlapping patches -> (H, W)

    The averaging in step (4) acts as a natural low-pass filter that smooths
    boundary artefacts at patch borders.

    Parameters
    ----------
    D     : (64, 256)  learned dictionary
    Alpha : (256, N)   sparse codes for all N patches
    image_shape : (H, W)  target image shape

    Returns
    -------
    img_recon : (H, W) float image, clipped to [0, 1]
    """
    N = Alpha.shape[1]

    # Step 1: (64, 256) @ (256, N) = (64, N)
    patches_flat = D @ Alpha       # (64, N)

    # Step 2: (64, N) -> (N, 64)
    # .T transposes back to patches-first layout
    patches_T    = patches_flat.T  # (N, 64)

    # Step 3: (N, 64) -> (N, 8, 8)
    # .reshape unfolds the 64-d vector back to an 8x8 spatial grid
    patches_2d   = patches_T.reshape(N, PATCH_SIZE[0], PATCH_SIZE[1])  # (N, 8, 8)

    # Step 4: Average overlapping patches
    # reconstruct_from_patches_2d sums patches at each pixel location and
    # divides by the number of patches contributing (overlap count).
    img_recon = reconstruct_from_patches_2d(patches_2d, image_shape)   # (H, W)

    # Clip to valid image range
    img_recon = np.clip(img_recon, 0.0, 1.0)

    return img_recon


# =============================================================================
# STEP F — Biharmonic Inpainting Baseline
# =============================================================================

def run_biharmonic(img_corrupt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Run skimage's biharmonic inpainting as a classical baseline.

    skimage convention: mask=True means "this pixel needs to be inpainted."
    Our convention:     mask=1    means "this pixel IS observed."
    -> Invert: bh_mask = (mask == 0)

    Parameters
    ----------
    img_corrupt : (H, W)  zero-filled corrupted image
    mask        : (H, W)  1=observed, 0=missing

    Returns
    -------
    img_bh : (H, W)  biharmonic inpainted image
    """
    bh_mask = (mask == 0).astype(bool)   # True where pixel is MISSING
    img_bh  = inpaint_biharmonic(img_corrupt, bh_mask)
    return np.clip(img_bh, 0.0, 1.0)


# =============================================================================
# STEP G — Evaluation Metrics
# =============================================================================

def evaluate(img_true: np.ndarray, img_pred: np.ndarray, label: str) -> dict:
    """
    Compute PSNR and SSIM on the full reconstructed image.

    Both metrics are computed on the ENTIRE image, including pixels that were
    originally missing. This is the standard inpainting evaluation protocol
    because the goal is to hallucinate a plausible value at every missing pixel.

    Parameters
    ----------
    img_true : (H, W)  ground-truth float image in [0, 1]
    img_pred : (H, W)  reconstructed float image in [0, 1]
    label    : str

    Returns
    -------
    metrics : dict with keys 'psnr' and 'ssim'
    """
    psnr = psnr_fn(img_true, img_pred, data_range=1.0)
    ssim = ssim_fn(img_true, img_pred, data_range=1.0)
    print(f"  {label:<35s}  PSNR = {psnr:6.2f} dB   SSIM = {ssim:.4f}")
    return {"psnr": psnr, "ssim": ssim}


# =============================================================================
# VISUALISATION
# =============================================================================

def plot_results(
    img_clean:     np.ndarray,
    img_corrupt:   np.ndarray,
    mask:          np.ndarray,
    img_biharmonic: np.ndarray,
    img_baseline:  np.ndarray,
    img_masked:    np.ndarray,
    metrics:       dict,
    err_masked:    np.ndarray,
    err_baseline:  np.ndarray,
    save_path:     str,
) -> None:
    """
    Six-panel figure:
      Row 1: Original | Corrupted | Biharmonic | Baseline K-SVD | Masked K-SVD
      Row 2: Convergence curve | Error maps for each reconstruction
    """
    BLUE  = '#2563EB'
    RED   = '#DC2626'
    GREEN = '#16A34A'

    fig = plt.figure(figsize=(22, 10))
    gs  = gridspec.GridSpec(2, 5, figure=fig, hspace=0.45, wspace=0.12)

    # ---- Helper: image panel with metric annotations -----------------------
    def img_panel(ax, img, title, subtitle="", cmap='gray'):
        im = ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=4)
        if subtitle:
            ax.set_xlabel(subtitle, fontsize=10, labelpad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        return im

    # Row 1, Col 0 — Original
    img_panel(fig.add_subplot(gs[0, 0]),
              img_clean, "Original (Ground Truth)", "128×128 grayscale")

    # Row 1, Col 1 — Corrupted (missing pixels shown as light gray overlay)
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(img_corrupt, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    # Highlight missing pixels with a semi-transparent red overlay
    miss_rgba       = np.zeros((*mask.shape, 4))
    miss_rgba[..., 0] = 1.0          # red channel
    miss_rgba[..., 3] = (mask == 0) * 0.55   # alpha only where missing
    ax.imshow(miss_rgba, interpolation='nearest')
    ax.set_title("Corrupted Input", fontsize=11, fontweight='bold', pad=4)
    ax.set_xlabel(f"30% pixels missing (red)", fontsize=10, labelpad=3)
    ax.set_xticks([])
    ax.set_yticks([])

    # Row 1, Col 2 — Biharmonic
    m = metrics['biharmonic']
    img_panel(fig.add_subplot(gs[0, 2]),
              img_biharmonic, "Biharmonic Inpainting",
              f"PSNR={m['psnr']:.1f}dB  SSIM={m['ssim']:.3f}")

    # Row 1, Col 3 — Baseline K-SVD
    m = metrics['baseline']
    img_panel(fig.add_subplot(gs[0, 3]),
              img_baseline, "Baseline K-SVD\n(zero imputation)",
              f"PSNR={m['psnr']:.1f}dB  SSIM={m['ssim']:.3f}")

    # Row 1, Col 4 — Masked K-SVD
    m = metrics['masked']
    img_panel(fig.add_subplot(gs[0, 4]),
              img_masked, "Masked K-SVD\n(novel — ALS update)",
              f"PSNR={m['psnr']:.1f}dB  SSIM={m['ssim']:.3f}")

    # ---- Row 2, Col 0-1: Convergence curve ----------------------------------
    ax = fig.add_subplot(gs[1, 0:2])
    iters = np.arange(1, len(err_masked) + 1)
    ax.semilogy(iters, err_masked,   'o-', color=BLUE, lw=2.2, ms=6,
                mfc='white', mew=1.8, label='Masked K-SVD  (novel)')
    ax.semilogy(iters, err_baseline, 's--', color=RED, lw=2.0, ms=6,
                mfc='white', mew=1.8, label='Baseline  (zero imputation)')
    ax.set_xlabel("K-SVD Iteration", fontsize=11)
    ax.set_ylabel("Masked RMSE  (observed pixels, log scale)", fontsize=10)
    ax.set_title("Training Convergence (3000-patch subset)", fontsize=11,
                 fontweight='bold')
    ax.grid(True, which='both', alpha=0.35)
    ax.legend(fontsize=10)
    # Annotate final values
    ax.annotate(f"{err_masked[-1]:.4f}",
                xy=(len(iters), err_masked[-1]),
                xytext=(len(iters) - 5, err_masked[-1] * 0.6),
                fontsize=9, color=BLUE,
                arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.2))
    ax.annotate(f"{err_baseline[-1]:.4f}",
                xy=(len(iters), err_baseline[-1]),
                xytext=(len(iters) - 5, err_baseline[-1] * 1.5),
                fontsize=9, color=RED,
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.2))

    # ---- Row 2, Col 2-4: Absolute error maps --------------------------------
    vmax_err = 0.25    # fixed scale so maps are comparable

    def error_panel(ax, img_pred, img_true, title, color):
        err_map = np.abs(img_pred - img_true)
        im = ax.imshow(err_map, cmap='hot', vmin=0, vmax=vmax_err,
                       interpolation='nearest')
        mean_err = err_map.mean()
        ax.set_title(title, fontsize=10, fontweight='bold', color=color, pad=4)
        ax.set_xlabel(f"mean |error| = {mean_err:.4f}", fontsize=9, labelpad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        return im

    ax_bh  = fig.add_subplot(gs[1, 2])
    ax_bl  = fig.add_subplot(gs[1, 3])
    ax_mk  = fig.add_subplot(gs[1, 4])

    error_panel(ax_bh, img_biharmonic, img_clean, "Biharmonic Error",   GREEN)
    error_panel(ax_bl, img_baseline,   img_clean, "Baseline Error",     RED)
    im_err = error_panel(ax_mk, img_masked, img_clean, "Masked K-SVD Error", BLUE)

    # Shared colorbar for the three error maps
    fig.colorbar(im_err, ax=[ax_bh, ax_bl, ax_mk],
                 fraction=0.025, pad=0.04, label='|reconstruction error|')

    fig.suptitle(
        "Phase 3 — 2D Image Inpainting via Masked Dictionary Learning\n"
        "128×128 camera image  |  8×8 patches  |  30% pixels missing  |"
        "  D: 64×256  |  sparsity s=10",
        fontsize=13, fontweight='bold', y=1.01
    )

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Figure saved -> {save_path}")


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line flags for selecting corruption mask behavior."""
    parser = argparse.ArgumentParser(
        description="Phase 3 image inpainting with optional custom masks."
    )
    parser.add_argument(
        "--mask-mode",
        choices=MASK_MODES,
        default="random",
        help=(
            "Mask pattern to use. 'random' keeps current behavior; "
            "other modes use structured custom masks."
        ),
    )
    parser.add_argument(
        "--missing-frac",
        type=float,
        default=MISSING_FRAC,
        help="Missing fraction for --mask-mode random.",
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
        default=20,
        help="Square side length for face-square mask modes.",
    )
    parser.add_argument(
        "--square-center",
        type=int,
        nargs=2,
        metavar=("ROW", "COL"),
        default=None,
        help="Square center as ROW COL for face-square mask modes.",
    )
    parser.add_argument(
        "--scratch-count",
        type=int,
        default=3,
        help="Number of diagonal scratches for scratches mask modes.",
    )
    parser.add_argument(
        "--scratch-thickness",
        type=int,
        default=4,
        help="Line thickness for scratches mask modes.",
    )
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_cli_args()
    square_center = tuple(args.square_center) if args.square_center is not None else None

    t_total = time.time()

    print("=" * 68)
    print("  Phase 3: 2D Image Inpainting — Masked Dictionary Learning")
    print("=" * 68)
    print(f"  Image size     : {IMAGE_SIZE}")
    print(f"  Patch size     : {PATCH_SIZE}  ({PATCH_DIM}-d vectors)")
    print(f"  Dictionary     : {PATCH_DIM}×{N_DICT_ATOMS}  ({N_DICT_ATOMS/PATCH_DIM:.0f}x overcomplete)")
    print(f"  Sparsity       : s = {SPARSITY}")
    print(f"  Mask mode      : {args.mask_mode}")
    if args.mask_mode == "random":
        print(f"  Missing frac   : {args.missing_frac*100:.0f}%")
    print(f"  Train patches  : {N_TRAIN_PATCHES}")
    print(f"  K-SVD iters    : {N_KSVD_ITER}")
    print()

    # ---- A: Load & corrupt --------------------------------------------------
    print("[A]  Loading and corrupting image ...")
    img_clean, img_corrupt, mask = load_and_corrupt_image(
        seed=args.mask_seed,
        mask_mode=args.mask_mode,
        missing_frac=args.missing_frac,
        square_size=args.square_size,
        square_center=square_center,
        scratch_count=args.scratch_count,
        scratch_thickness=args.scratch_thickness,
    )
    print()

    # ---- B: Patch extraction ------------------------------------------------
    print("[B]  Extracting overlapping 8×8 patches ...")
    X_all, M_all, N_all = extract_patches(img_corrupt, mask)
    X_train, M_train    = subsample_patches(X_all, M_all, N_TRAIN_PATCHES, seed=SEED)
    print(f"  All patches     : {N_all}")
    print(f"  Training subset : {X_train.shape[1]}")
    print()

    # ---- C+D: Masked K-SVD (novel) ------------------------------------------
    print("[C+D]  Running Masked K-SVD (novel — ALS atom update) ...")
    D_masked, Alpha_masked, err_masked, t_masked = learn_and_reconstruct(
        X_all, M_all, X_train, M_train,
        label="Masked K-SVD (novel)",
        use_masking=True
    )

    # ---- C+D: Baseline K-SVD ------------------------------------------------
    print()
    print("[C+D]  Running Baseline K-SVD (zero imputation) ...")
    D_baseline, Alpha_baseline, err_baseline, t_baseline = learn_and_reconstruct(
        X_all, M_all, X_train, M_train,
        label="Baseline K-SVD (zero imputation)",
        use_masking=False
    )

    # ---- E: Reconstruct images from patch codes -----------------------------
    print()
    print("[E]  Reconstructing images from patch codes ...")
    img_masked   = patches_to_image(D_masked,   Alpha_masked,   IMAGE_SIZE)
    img_baseline = patches_to_image(D_baseline, Alpha_baseline, IMAGE_SIZE)
    print(f"  Masked K-SVD  recon range : [{img_masked.min():.3f}, {img_masked.max():.3f}]")
    print(f"  Baseline      recon range : [{img_baseline.min():.3f}, {img_baseline.max():.3f}]")

    # ---- F: Biharmonic baseline ---------------------------------------------
    print()
    print("[F]  Running biharmonic inpainting ...")
    t0       = time.time()
    img_bh   = run_biharmonic(img_corrupt, mask)
    print(f"  Biharmonic done in {time.time()-t0:.2f}s")

    # ---- G: Evaluation ------------------------------------------------------
    print()
    print("[G]  Evaluation (PSNR / SSIM on full 128×128 image)")
    print("  " + "-" * 58)
    metrics = {
        "biharmonic": evaluate(img_clean, img_bh,       "Biharmonic inpainting"),
        "baseline":   evaluate(img_clean, img_baseline,  "Baseline K-SVD (zero imputation)"),
        "masked":     evaluate(img_clean, img_masked,    "Masked K-SVD (novel — ALS)"),
    }

    # Summary
    print()
    print("  Summary (Masked K-SVD vs Baseline):")
    dpsnr = metrics['masked']['psnr']  - metrics['baseline']['psnr']
    dssim = metrics['masked']['ssim']  - metrics['baseline']['ssim']
    print(f"    PSNR gain : {dpsnr:+.2f} dB")
    print(f"    SSIM gain : {dssim:+.4f}")

    # ---- Plot ---------------------------------------------------------------
    print()
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "outputs",
        "phase3_inpainting_results.png",
    )
    plot_results(
        img_clean, img_corrupt, mask,
        img_bh, img_baseline, img_masked,
        metrics, err_masked, err_baseline,
        save_path=out_path
    )

    print(f"\n  Total wall-clock time: {time.time()-t_total:.1f}s")
    print("=" * 68)


if __name__ == "__main__":
    main()
