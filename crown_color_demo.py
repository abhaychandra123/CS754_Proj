"""
CROWN-Inpaint -- Color demonstration (per-channel independent).

This script extends the existing CROWN pipeline to RGB images by running the
full CROWN-Inpaint algorithm independently on each channel.  No algorithmic
modification is required: every CROWN component (regime map, confidence map,
weighted OMP, nonlocal coupling, biharmonic smoother, fusion) is scalar-valued
or per-pixel-vectorised, so the framework is color-agnostic.  This script is
intentionally separate from `crown_inpaint_experiment.py` so the main
benchmarking pipeline remains untouched.

Pipeline reuse
--------------
Image loading uses `skimage.data` (with a `--image` argument allowing
astronaut, coffee, chelsea, or a user-supplied PNG/JPG).  Mask construction
reuses `custom_masks.generate_mask` so the same hole shapes (`face-square`,
`scratches`, `text-overlay`, `random`) are available here as in the main
benchmark.  CROWN itself is invoked per channel via
`crown.run.train_dictionary` and `crown.run.run_crown_inpaint`.

Output
------
A four-panel figure: ground truth | corrupted | biharmonic baseline | CROWN.
Each panel reports per-image PSNR (dB) and SSIM, computed over the full
RGB volume with `channel_axis=-1`.  A NumPy .npz file with the raw arrays
is also saved next to the figure for reproducibility.

Usage
-----
  python3 crown_color_demo.py
  python3 crown_color_demo.py --image astronaut --mask-mode face-square
  python3 crown_color_demo.py --image coffee --mask-mode scratches --T 5
  python3 crown_color_demo.py --image astronaut --mask-mode random --missing-frac 0.3
  python3 crown_color_demo.py --image-path /path/to/img.png --mask-mode face-square
  python3 crown_color_demo.py --quick           # smaller dict / fewer iters

The `--quick` flag halves the training cost (n_train=1500, n_iter=10, T=3)
to produce a slide-ready figure in ~90 seconds on a laptop CPU.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage import data as skdata
from skimage import io as skio
from skimage.color import rgba2rgb, gray2rgb
from skimage.transform import resize as sk_resize
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity  as ssim_fn
from skimage.restoration import inpaint_biharmonic


# ---------------------------------------------------------------------------
# Project imports (mirrors crown_inpaint_experiment.py conventions)
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from crown.run    import train_dictionary, run_crown_inpaint     # noqa: E402
from crown.utils  import PATCH_SIZE                               # noqa: E402
from custom_masks import (                                        # noqa: E402
    generate_mask,
    DEFAULT_MISSING_FRAC,
    DEFAULT_SEED,
    DEFAULT_SQUARE_SIZE,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_SIZE = (128, 128)

# Built-in RGB images from skimage.data
RGB_IMAGES = {
    "astronaut":  skdata.astronaut,
    "coffee":     skdata.coffee,
    "chelsea":    skdata.chelsea,
    "rocket":     skdata.rocket,
}


# ---------------------------------------------------------------------------
# Image loader
# ---------------------------------------------------------------------------

def load_color_image(name: str | None, path: str | None,
                     image_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """
    Load an RGB image, resize to image_size, return float64 in [0, 1] of
    shape (H, W, 3).  Either `name` (skimage.data) or `path` must be set;
    if both are set, `path` takes precedence.
    """
    if path is not None:
        raw = skio.imread(path)
    else:
        if name not in RGB_IMAGES:
            raise ValueError(
                f"Unknown image '{name}'. Choose from {list(RGB_IMAGES)} "
                "or pass --image-path /to/file.png."
            )
        raw = RGB_IMAGES[name]()

    raw = np.asarray(raw)
    if raw.dtype != np.float64:
        raw = raw.astype(np.float64) / (255.0 if raw.max() > 1.5 else 1.0)

    if raw.ndim == 2:
        raw = gray2rgb(raw)
    elif raw.ndim == 3 and raw.shape[-1] == 4:
        raw = rgba2rgb(raw)
    elif raw.ndim != 3 or raw.shape[-1] != 3:
        raise ValueError(f"Unsupported image shape {raw.shape}; expected (H,W) or (H,W,3/4).")

    img = sk_resize(raw, image_size, anti_aliasing=True)
    return np.clip(img, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Per-channel biharmonic baseline (for the comparison panel)
# ---------------------------------------------------------------------------

def biharmonic_color(img_corrupted: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Per-channel biharmonic completion of the hole."""
    out = np.zeros_like(img_corrupted)
    bh_mask = (M == 0).astype(bool)
    for c in range(3):
        ch = inpaint_biharmonic(img_corrupted[..., c], bh_mask)
        ch = np.where(M == 1, img_corrupted[..., c], ch)
        out[..., c] = np.clip(ch, 0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# Per-channel CROWN
# ---------------------------------------------------------------------------

def crown_color(
    img_clean:   np.ndarray,
    M:           np.ndarray,
    *,
    T:           int  = 3,
    n_train:     int  = 3000,
    n_iter:      int  = 15,
    seed:        int  = 42,
    verbose:     bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Run CROWN-Inpaint independently on each RGB channel.

    Parameters
    ----------
    img_clean : (H, W, 3) float64    Ground-truth color image in [0, 1].
    M         : (H, W) float64       Binary mask shared across channels.
    T, n_train, n_iter, seed         Per-channel CROWN/training settings.

    Returns
    -------
    u_color   : (H, W, 3) float64    Reconstructed color image.
    info      : dict with per-channel timing and a 'history_per_channel' key.
    """
    H, W, C = img_clean.shape
    if M.shape != (H, W):
        raise ValueError(f"mask shape {M.shape} != image spatial shape {(H, W)}")
    if C != 3:
        raise ValueError(f"expected 3-channel image, got {C}")

    img_corrupted = img_clean * M[..., None]      # zero-fill hole per channel
    u_color       = np.zeros_like(img_clean)
    per_ch_times  = []
    per_ch_hist   = []

    for c in range(C):
        ch_name = ["R", "G", "B"][c]
        if verbose:
            print(f"\n  [color]  ===== channel {ch_name} =====")
        t_ch = time.time()

        y_c = img_corrupted[..., c]

        # 1. Train per-channel dictionary on observed patches only
        D_c, _ = train_dictionary(
            y_c, M,
            n_train=n_train,
            n_iter=n_iter,
            seed=seed + c,                # de-correlate per channel
            verbose=verbose,
        )

        # 2. Run full CROWN on this channel
        out = run_crown_inpaint(
            y_c, M, D_c,
            T=T,
            img_clean=img_clean[..., c],   # so per-channel PSNR is logged
            verbose=verbose,
        )
        u_color[..., c] = out["u"]

        per_ch_times.append(time.time() - t_ch)
        per_ch_hist.append(out["history"])

        if verbose:
            ch_psnr = psnr_fn(img_clean[..., c], np.clip(u_color[..., c], 0, 1),
                              data_range=1.0)
            print(f"  [color]  channel {ch_name} done in "
                  f"{per_ch_times[-1]:.1f}s   PSNR={ch_psnr:.2f}dB")

    return np.clip(u_color, 0.0, 1.0), {
        "per_channel_seconds": per_ch_times,
        "history_per_channel": per_ch_hist,
    }


# ---------------------------------------------------------------------------
# Metrics on full RGB volumes
# ---------------------------------------------------------------------------

def color_metrics(gt: np.ndarray, pred: np.ndarray, M: np.ndarray) -> dict:
    """Full-image PSNR/SSIM and hole-only PSNR for an RGB image."""
    pred_c = np.clip(pred, 0.0, 1.0)
    psnr_full = float(psnr_fn(gt, pred_c, data_range=1.0))
    ssim_full = float(ssim_fn(gt, pred_c, data_range=1.0, channel_axis=-1))

    hole = (M == 0)
    if hole.any():
        # Hole-only PSNR averaged across channels
        gt_h, pr_h = gt[hole], pred_c[hole]                  # (n_hole, 3)
        mse_hole   = float(np.mean((gt_h - pr_h) ** 2))
        psnr_hole  = 10.0 * np.log10(1.0 / max(mse_hole, 1e-12))
    else:
        psnr_hole = float("inf")

    return {"psnr": psnr_full, "ssim": ssim_full, "hole_psnr": psnr_hole}


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

def save_comparison_figure(
    img_clean:   np.ndarray,
    img_corrupt: np.ndarray,
    img_bh:      np.ndarray,
    img_crown:   np.ndarray,
    metrics_bh:  dict,
    metrics_cr:  dict,
    out_path:    str,
    image_label: str,
    mask_label:  str,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.0))
    titles = [
        "Ground truth",
        f"Corrupted ({mask_label})",
        f"Biharmonic baseline\nPSNR={metrics_bh['psnr']:.2f}dB  "
        f"SSIM={metrics_bh['ssim']:.3f}\nhole-PSNR={metrics_bh['hole_psnr']:.2f}dB",
        f"CROWN (per-channel)\nPSNR={metrics_cr['psnr']:.2f}dB  "
        f"SSIM={metrics_cr['ssim']:.3f}\nhole-PSNR={metrics_cr['hole_psnr']:.2f}dB",
    ]
    panels = [img_clean, img_corrupt, img_bh, img_crown]

    for ax, title, panel in zip(axes, titles, panels):
        ax.imshow(np.clip(panel, 0.0, 1.0))
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f"CROWN-Inpaint -- color extension (per-channel)   |   "
        f"image: {image_label}   |   mask: {mask_label}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def build_mask(args, image_shape: Tuple[int, int]) -> Tuple[np.ndarray, str]:
    """
    Construct the binary mask using the project's `generate_mask` helper.
    Returns (M, label) where label is a short string for figure titles.
    """
    M = generate_mask(
        image_shape,
        mode=args.mask_mode,
        missing_frac=args.missing_frac,
        seed=args.seed,
        square_size=args.square_size,
        square_center=tuple(args.square_center) if args.square_center else None,
        scratch_count=args.scratch_count,
        scratch_thickness=args.scratch_thickness,
        overlay_text=args.overlay_text,
    ).astype(np.float64)

    if args.mask_mode == "random":
        label = f"random {int(round(args.missing_frac * 100))}%"
    elif args.mask_mode == "face-square":
        label = f"square s={args.square_size}"
    elif args.mask_mode == "scratches":
        label = f"scratches n={args.scratch_count} t={args.scratch_thickness}"
    elif args.mask_mode == "face-square+scratches":
        label = "square + scratches"
    elif args.mask_mode == "text-overlay":
        label = f"text '{args.overlay_text}'"
    else:
        label = args.mask_mode

    return M, label


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Color CROWN-Inpaint demo (per-channel independent).",
    )
    p.add_argument("--image", default="astronaut", choices=list(RGB_IMAGES),
                   help="Built-in RGB image from skimage.data.")
    p.add_argument("--image-path", default=None,
                   help="Path to a custom image (overrides --image).")
    p.add_argument("--size", type=int, default=128,
                   help="Image side length after resize (square).")

    p.add_argument("--mask-mode", default="face-square",
                   choices=["random", "face-square", "scratches",
                            "face-square+scratches", "text-overlay"])
    p.add_argument("--missing-frac", type=float, default=DEFAULT_MISSING_FRAC)
    p.add_argument("--seed",         type=int,   default=DEFAULT_SEED)
    p.add_argument("--square-size",  type=int,   default=DEFAULT_SQUARE_SIZE)
    p.add_argument("--square-center", type=int, nargs=2, default=None,
                   help="(row, col) center of the square hole; default = image center.")
    p.add_argument("--scratch-count",     type=int, default=3)
    p.add_argument("--scratch-thickness", type=int, default=4)
    p.add_argument("--overlay-text",      default="SAMPLE")

    p.add_argument("-T", "--T",     type=int, default=3,
                   help="Outer CROWN iterations per channel.")
    p.add_argument("--n-train",     type=int, default=3000,
                   help="Patches used to train the per-channel dictionary.")
    p.add_argument("--n-iter",      type=int, default=15,
                   help="K-SVD outer iterations per channel.")

    p.add_argument("--quick", action="store_true",
                   help="Faster preset: n_train=1500, n_iter=10, T=3.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-channel CROWN progress logs.")
    p.add_argument("-o", "--out-dir", default="outputs/crown_color",
                   help="Directory for the figure and .npz dump.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.quick:
        args.n_train = 1500
        args.n_iter  = 10
        args.T       = max(args.T, 3)
        if args.T > 3: args.T = 3

    image_shape = (args.size, args.size)

    print("=" * 72)
    print(f"CROWN-Inpaint COLOR DEMO")
    print("=" * 72)
    print(f"  image       : {args.image_path or args.image}")
    print(f"  image size  : {image_shape}")
    print(f"  mask mode   : {args.mask_mode}")
    print(f"  T           : {args.T}")
    print(f"  n_train     : {args.n_train}")
    print(f"  n_iter      : {args.n_iter}")
    print(f"  seed        : {args.seed}")
    print(f"  out dir     : {args.out_dir}")

    # ---- 1. Load image ---------------------------------------------------
    img_clean = load_color_image(
        args.image if args.image_path is None else None,
        args.image_path,
        image_size=image_shape,
    )
    image_label = (os.path.basename(args.image_path)
                   if args.image_path else args.image)

    # ---- 2. Build mask ---------------------------------------------------
    M, mask_label = build_mask(args, image_shape)
    img_corrupt   = img_clean * M[..., None]

    print(f"\n  hole pixels : {int((M == 0).sum())} / {M.size} "
          f"({100 * (M == 0).mean():.1f}%)")

    # ---- 3. Biharmonic baseline -----------------------------------------
    print("\n  [color]  Biharmonic baseline (per-channel)...")
    t_bh = time.time()
    img_bh = biharmonic_color(img_corrupt, M)
    bh_seconds = time.time() - t_bh
    metrics_bh = color_metrics(img_clean, img_bh, M)
    print(f"  [color]  biharmonic done in {bh_seconds:.1f}s   "
          f"PSNR={metrics_bh['psnr']:.2f}dB  SSIM={metrics_bh['ssim']:.3f}  "
          f"hole-PSNR={metrics_bh['hole_psnr']:.2f}dB")

    # ---- 4. CROWN per-channel -------------------------------------------
    print("\n  [color]  Running CROWN per channel...")
    t_cr = time.time()
    img_crown, info = crown_color(
        img_clean, M,
        T=args.T,
        n_train=args.n_train,
        n_iter=args.n_iter,
        seed=args.seed,
        verbose=not args.quiet,
    )
    crown_seconds = time.time() - t_cr
    metrics_cr = color_metrics(img_clean, img_crown, M)
    print(f"\n  [color]  CROWN total {crown_seconds:.1f}s "
          f"({np.mean(info['per_channel_seconds']):.1f}s / channel)")
    print(f"  [color]  CROWN PSNR={metrics_cr['psnr']:.2f}dB  "
          f"SSIM={metrics_cr['ssim']:.3f}  "
          f"hole-PSNR={metrics_cr['hole_psnr']:.2f}dB")

    # ---- 5. Summary ------------------------------------------------------
    print("\n" + "-" * 72)
    print(f"  {'method':<30} {'PSNR':>8}  {'SSIM':>6}  {'hole-PSNR':>10}")
    print("-" * 72)
    print(f"  {'Biharmonic (per channel)':<30} "
          f"{metrics_bh['psnr']:>8.2f}  {metrics_bh['ssim']:>6.3f}  "
          f"{metrics_bh['hole_psnr']:>10.2f}")
    print(f"  {'CROWN (per channel)':<30} "
          f"{metrics_cr['psnr']:>8.2f}  {metrics_cr['ssim']:>6.3f}  "
          f"{metrics_cr['hole_psnr']:>10.2f}")
    print(f"  {'delta (CROWN - biharmonic)':<30} "
          f"{metrics_cr['psnr']-metrics_bh['psnr']:>+8.2f}  "
          f"{metrics_cr['ssim']-metrics_bh['ssim']:>+6.3f}  "
          f"{metrics_cr['hole_psnr']-metrics_bh['hole_psnr']:>+10.2f}")
    print("-" * 72)

    # ---- 6. Save outputs ------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{image_label}_{args.mask_mode.replace('+','-')}_T{args.T}_seed{args.seed}"
    fig_path = os.path.join(args.out_dir, f"crown_color_{tag}.png")
    npz_path = os.path.join(args.out_dir, f"crown_color_{tag}.npz")

    save_comparison_figure(
        img_clean, img_corrupt, img_bh, img_crown,
        metrics_bh, metrics_cr,
        out_path=fig_path,
        image_label=image_label,
        mask_label=mask_label,
    )
    np.savez_compressed(
        npz_path,
        img_clean=img_clean, img_corrupt=img_corrupt,
        img_bh=img_bh,       img_crown=img_crown,
        mask=M,
        metrics_bh=np.array(list(metrics_bh.values())),
        metrics_crown=np.array(list(metrics_cr.values())),
    )
    print(f"\n  figure saved to : {fig_path}")
    print(f"  arrays saved to : {npz_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
