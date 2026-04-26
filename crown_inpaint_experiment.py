"""
CROWN-Inpaint experiment driver.

Runs:
  * Biharmonic baseline
  * CROWN-Inpaint (full method)
  * Optional: ablations (no nonlocal, no fusion, no manifold)

Outputs a six-panel comparison figure plus a console summary table.

Usage examples
--------------
  python3 crown_inpaint_experiment.py
  python3 crown_inpaint_experiment.py --hole face --T 5
  python3 crown_inpaint_experiment.py --hole textured --T 5 --manifold
  python3 crown_inpaint_experiment.py --mask-mode scratches --T 5
  python3 crown_inpaint_experiment.py --mask-mode text-overlay --overlay-text SAMPLE
  python3 crown_inpaint_experiment.py --ablation no_nonlocal
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity  as ssim_fn
from skimage.restoration import inpaint_biharmonic

# Project root on sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from inpainting_multiscale_masked_ksvd import (
    load_image, IMAGE_SIZE, PATCH_SIZE,
)
from crown.run import train_dictionary, run_crown_inpaint
from custom_masks import (
    generate_mask,
    MASK_MODES,
    DEFAULT_MISSING_FRAC,
    DEFAULT_SEED,
    DEFAULT_SQUARE_SIZE,
    DEFAULT_SQUARE_CENTER,
    DEFAULT_SCRATCH_COUNT,
    DEFAULT_SCRATCH_THICKNESS,
    DEFAULT_OVERLAY_TEXT,
    DEFAULT_TEXT_SCALE,
    DEFAULT_TEXT_ANGLE,
    DEFAULT_TEXT_STROKE,
)


# ---------------------------------------------------------------------------
# Hole presets (rectangular holes used in the spec's regime analysis)
# ---------------------------------------------------------------------------

HOLE_PRESETS = {
    # Smooth-ish region of the cameraman / brick image
    "face":     {"r0": 10, "r1": 30, "c0": 55, "c1": 75},
    # Textured region (camera body / brick texture)
    "textured": {"r0": 40, "r1": 60, "c0": 64, "c1": 84},
    # Larger textured hole for stress test
    "large":    {"r0": 30, "r1": 70, "c0": 30, "c1": 70},
}


def make_block_mask(hole_spec: dict, image_shape=IMAGE_SIZE) -> np.ndarray:
    M = np.ones(image_shape, dtype=np.float64)
    M[hole_spec["r0"]:hole_spec["r1"],
      hole_spec["c0"]:hole_spec["c1"]] = 0.0
    return M


def _slug(text: str) -> str:
    """Build a compact file-name token from a user-facing mask label."""
    token = []
    for ch in str(text):
        if ch.isalnum():
            token.append(ch.lower())
        elif ch in ("-", "_"):
            token.append(ch)
        else:
            token.append("-")
    out = "".join(token).strip("-")
    while "--" in out:
        out = out.replace("--", "-")
    return out or "mask"


def _rect_from_square_mask(square_size: int,
                           square_center: tuple[int, int] | None,
                           image_shape=IMAGE_SIZE) -> dict:
    """Mirror custom_masks.square_hole_mask geometry for plot rectangles."""
    h, w = image_shape
    size = int(max(1, min(square_size, h, w)))
    if square_center is None:
        square_center = (h // 2, w // 2)
    cy, cx = int(square_center[0]), int(square_center[1])
    r0 = max(0, cy - size // 2)
    r1 = min(h, r0 + size)
    c0 = max(0, cx - size // 2)
    c1 = min(w, c0 + size)
    return {"r0": r0, "r1": r1, "c0": c0, "c1": c1}


def build_experiment_mask(args, image_shape=IMAGE_SIZE) -> tuple[np.ndarray, dict]:
    """
    Build the binary mask and plotting/output metadata for this run.

    With no --mask-mode, preserve the legacy CROWN rectangle presets.  When
    --mask-mode is supplied, delegate to custom_masks.generate_mask so that
    all mask semantics stay in one place.
    """
    if args.mask_mode is None:
        hole_spec = HOLE_PRESETS[args.hole].copy()
        hole_spec["name"] = args.hole
        M = make_block_mask(hole_spec, image_shape=image_shape)
        label = f"hole: {args.hole}"
        token = f"hole_{_slug(args.hole)}"
        return M, {
            "kind": "hole",
            "label": label,
            "token": token,
            "rect": hole_spec,
        }

    square_center = (
        tuple(args.square_center) if args.square_center is not None else None
    )
    M = generate_mask(
        image_shape=image_shape,
        mode=args.mask_mode,
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

    rect = None
    if args.mask_mode == "face-square":
        rect = _rect_from_square_mask(
            args.square_size, square_center, image_shape=image_shape
        )

    label = f"mask: {args.mask_mode}"
    token_parts = ["mask", _slug(args.mask_mode)]
    if args.mask_mode == "random":
        label += f" ({args.missing_frac * 100:.1f}% missing)"
        token_parts.append(f"p{int(round(args.missing_frac * 1000)):03d}")
    elif args.mask_mode == "face-square":
        label += f" (size={args.square_size}, center={square_center})"
        token_parts.append(f"s{args.square_size}")
    elif args.mask_mode == "scratches":
        label += (
            f" (count={args.scratch_count}, "
            f"thickness={args.scratch_thickness})"
        )
        token_parts.extend([f"n{args.scratch_count}", f"t{args.scratch_thickness}"])
    elif args.mask_mode == "face-square+scratches":
        label += (
            f" (square={args.square_size}, center={square_center}, "
            f"scratches={args.scratch_count}x{args.scratch_thickness})"
        )
        token_parts.extend([
            f"s{args.square_size}",
            f"n{args.scratch_count}",
            f"t{args.scratch_thickness}",
        ])
    elif args.mask_mode == "text-overlay":
        label += (
            f" ('{args.overlay_text}', scale={args.text_scale}, "
            f"angle={args.text_angle}, stroke={args.text_stroke})"
        )
        token_parts.append(_slug(args.overlay_text))

    return M.astype(np.float64), {
        "kind": "custom",
        "mode": args.mask_mode,
        "label": label,
        "token": "_".join(token_parts),
        "rect": rect,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(img_true, img_pred, M):
    img_pred = np.clip(img_pred, 0.0, 1.0)
    p = float(psnr_fn(img_true, img_pred, data_range=1.0))
    s = float(ssim_fn(img_true, img_pred, data_range=1.0))
    hole = (M == 0)
    if hole.any():
        mse = float(np.mean((img_true[hole] - img_pred[hole]) ** 2))
        hp  = 10.0 * np.log10(1.0 / max(mse, 1e-12))
        # Boundary band: 4-pixel ring around the hole perimeter
        from scipy.ndimage import binary_dilation, binary_erosion
        ring = binary_dilation(hole, iterations=4) & ~binary_erosion(hole, iterations=4)
        bm  = float(np.mean(np.abs(img_true[ring] - img_pred[ring])))
    else:
        hp = float("inf")
        bm = 0.0
    return {"psnr": p, "ssim": s, "hole_psnr": hp, "boundary_mae": bm}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(img_clean, y_corr, M, img_bh, crown_out,
                     mask_info, metrics, save_path,
                     ablation_label=None):
    BLUE  = "#2563EB"
    GREEN = "#16A34A"
    RED   = "#DC2626"
    fig = plt.figure(figsize=(22, 11))
    gs  = gridspec.GridSpec(2, 5, figure=fig, hspace=0.42, wspace=0.10)
    rect = mask_info.get("rect")
    hole = (M == 0)

    def add_mask_overlay(ax, edge_color=None, tint=False, tint_color="red"):
        if tint:
            overlay = np.zeros((*M.shape, 4))
            if tint_color == "red":
                overlay[..., 0] = 1.0
            elif tint_color == "cyan":
                overlay[..., 1] = 1.0
                overlay[..., 2] = 1.0
            overlay[..., 3] = hole * 0.35
            ax.imshow(overlay)

        if edge_color is None:
            return

        if rect is not None:
            r0, r1 = rect["r0"], rect["r1"]
            c0, c1 = rect["c0"], rect["c1"]
            ax.add_patch(Rectangle(
                (c0 - 0.5, r0 - 0.5),
                c1 - c0, r1 - r0,
                lw=1.2, ec=edge_color, fc="none", ls="--",
            ))
        elif hole.any():
            ax.contour(
                hole.astype(float),
                levels=[0.5],
                colors=edge_color,
                linewidths=0.9,
            )

    def img_panel(sp, img, title, sub=""):
        ax = fig.add_subplot(sp)
        ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=3)
        ax.set_xlabel(sub, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    img_panel(gs[0, 0], img_clean, "Original")
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(y_corr, cmap="gray", vmin=0, vmax=1)
    add_mask_overlay(ax, tint=True, tint_color="red")
    ax.set_title("Corrupted", fontsize=11, fontweight="bold", pad=3)
    ax.set_xticks([]); ax.set_yticks([])

    m = metrics["biharmonic"]
    img_panel(
        gs[0, 2], img_bh, "Biharmonic",
        f"PSNR={m['psnr']:.2f} dB   Hole-PSNR={m['hole_psnr']:.2f} dB"
    )
    m = metrics["crown"]
    img_panel(
        gs[0, 3], crown_out["u"],
        f"CROWN-Inpaint{' ('+ablation_label+')' if ablation_label else ''}",
        f"PSNR={m['psnr']:.2f} dB   Hole-PSNR={m['hole_psnr']:.2f} dB"
    )

    # Regime map panel
    ax = fig.add_subplot(gs[0, 4])
    im = ax.imshow(crown_out["regime"], cmap="viridis", vmin=0, vmax=1)
    ax.set_title("Regime map  r", fontsize=11, fontweight="bold", pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.045)

    # Convergence curve
    history = crown_out["history"]
    if any("psnr" in h for h in history):
        ax = fig.add_subplot(gs[1, 0:2])
        iters = [h["iter"] for h in history if "psnr" in h]
        ps    = [h["psnr"] for h in history if "psnr" in h]
        hps   = [h.get("hole_psnr", float("nan")) for h in history if "psnr" in h]
        ax.plot(iters, ps,  "o-", color=BLUE,  lw=2, ms=6, mfc="white",
                label="full-image PSNR")
        ax.plot(iters, hps, "s--", color=RED,   lw=2, ms=6, mfc="white",
                label="hole-only PSNR")
        ax.axhline(metrics["biharmonic"]["psnr"], color=GREEN, ls=":", lw=2,
                   label="biharmonic full PSNR")
        ax.axhline(metrics["biharmonic"]["hole_psnr"], color="#7C3AED",
                   ls=":", lw=2, label="biharmonic hole PSNR")
        for it, p in zip(iters, ps):
            ax.annotate(f"{p:.2f}", (it, p),
                        xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set_xlabel("Outer iteration"); ax.set_ylabel("PSNR (dB)")
        ax.set_title("CROWN-Inpaint convergence trajectory",
                     fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    # Confidence at last iteration
    if crown_out["confidence"]:
        c_last = crown_out["confidence"][-1]
        ax = fig.add_subplot(gs[1, 2])
        im = ax.imshow(c_last, cmap="magma", vmin=0, vmax=1)
        add_mask_overlay(ax, edge_color="cyan")
        ax.set_title(f"Confidence c (final iter)", fontsize=10, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.045)

    # Error maps (hot)
    vmax = 0.25

    def err_panel(sp, pred, title, color):
        ax = fig.add_subplot(sp)
        err = np.abs(np.clip(pred, 0, 1) - img_clean)
        ax.imshow(err, cmap="hot", vmin=0, vmax=vmax)
        add_mask_overlay(ax, edge_color="cyan")
        ax.set_title(title, fontsize=10, fontweight="bold", color=color, pad=3)
        ax.set_xlabel(f"hole MAE={err[M==0].mean():.4f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    err_panel(gs[1, 3], img_bh,        "Biharmonic |error|",  GREEN)
    err_panel(gs[1, 4], crown_out["u"], "CROWN |error|",       BLUE)

    title_label = mask_info["label"]
    if rect is not None:
        title_label = (
            f"{title_label} "
            f"({rect['r1'] - rect['r0']}x{rect['c1'] - rect['c0']})"
        )
    fig.suptitle(
        f"CROWN-Inpaint  vs  Biharmonic   [{title_label}]",
        fontsize=13, fontweight="bold", y=1.01,
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved -> {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="CROWN-Inpaint experiment runner")
    ap.add_argument("--hole", choices=list(HOLE_PRESETS.keys()), default="face",
                    help="legacy rectangular hole preset name")
    ap.add_argument("--mask-mode", choices=MASK_MODES, default=None,
                    help="custom mask mode from custom_masks.py; overrides --hole")
    ap.add_argument("--missing-frac", type=float, default=DEFAULT_MISSING_FRAC,
                    help="missing fraction used when --mask-mode random")
    ap.add_argument("--mask-seed", type=int, default=DEFAULT_SEED,
                    help="random seed used by custom mask generators")
    ap.add_argument("--square-size", type=int, default=DEFAULT_SQUARE_SIZE,
                    help="square side length for face-square mask modes")
    ap.add_argument("--square-center", type=int, nargs=2,
                    metavar=("ROW", "COL"), default=DEFAULT_SQUARE_CENTER,
                    help="square center as ROW COL for face-square mask modes")
    ap.add_argument("--scratch-count", type=int, default=DEFAULT_SCRATCH_COUNT,
                    help="number of diagonal scratches for scratches mask modes")
    ap.add_argument("--scratch-thickness", type=int,
                    default=DEFAULT_SCRATCH_THICKNESS,
                    help="line thickness for scratches mask modes")
    ap.add_argument("--overlay-text", type=str, default=DEFAULT_OVERLAY_TEXT,
                    help="text string for text-overlay mask mode")
    ap.add_argument("--text-scale", type=float, default=DEFAULT_TEXT_SCALE,
                    help="text size fraction for text-overlay mask mode")
    ap.add_argument("--text-angle", type=float, default=DEFAULT_TEXT_ANGLE,
                    help="text rotation angle in degrees for text-overlay mode")
    ap.add_argument("--text-stroke", type=int, default=DEFAULT_TEXT_STROKE,
                    help="stroke width for text-overlay mode")
    ap.add_argument("--T", type=int, default=5, help="outer iterations")
    ap.add_argument("--K_s", type=int, default=10,
                    help="inner harmonic Jacobi steps")
    ap.add_argument("--n_train", type=int, default=3000,
                    help="patches used for dictionary training")
    ap.add_argument("--n_iter_dict", type=int, default=25,
                    help="K-SVD iterations for dictionary training")
    ap.add_argument("--manifold", action="store_true",
                    help="enable optional stochastic manifold correction")
    ap.add_argument("--ablation",
                    choices=["none", "no_nonlocal", "no_fusion", "no_manifold"],
                    default="none",
                    help="run a specific ablation in addition to the comparison")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = os.path.join(_THIS_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("  CROWN-Inpaint experiment")
    print("=" * 70)
    print(f"  hole preset      : {args.hole}")
    print(f"  custom mask mode : {args.mask_mode or 'none'}")
    print(f"  outer iters T    : {args.T}")
    print(f"  manifold enabled : {args.manifold}")
    print(f"  ablation         : {args.ablation}")

    # ---- Data ---------------------------------------------------------------
    img_clean = load_image()
    M, mask_info = build_experiment_mask(args)
    y_corr = img_clean * M

    print(f"  selected mask    : {mask_info['label']}")
    if args.mask_mode == "random":
        print(f"  mask seed        : {args.mask_seed}")
    elif args.mask_mode in ("scratches", "face-square+scratches"):
        print(f"  mask seed        : {args.mask_seed}")

    # Quick gradient diagnostic
    gx, gy   = np.gradient(img_clean)
    grad_mag = np.sqrt(gx**2 + gy**2)
    hole_g   = float(grad_mag[M == 0].mean())
    print(f"  hole gradient |∇I| (mean) : {hole_g:.4f}  "
          f"({'SMOOTH' if hole_g < 0.06 else 'TEXTURED'})")
    print(f"  missing pixels            : {(M==0).sum()} "
          f"({(M==0).mean()*100:.1f}%)")

    # ---- Biharmonic baseline -----------------------------------------------
    print("\n  [Baseline] biharmonic ...")
    t0 = time.time()
    img_bh = np.clip(
        inpaint_biharmonic(y_corr, (M == 0).astype(bool)),
        0.0, 1.0,
    )
    print(f"  done in {time.time() - t0:.2f}s")

    # ---- Dictionary training -----------------------------------------------
    print()
    D, train_err = train_dictionary(
        y_corr, M,
        n_train=args.n_train,
        n_iter=args.n_iter_dict,
        seed=args.seed,
        verbose=True,
    )

    # ---- CROWN-Inpaint -----------------------------------------------------
    print("\n  [CROWN-Inpaint] running outer loop ...")

    nonlocal_enabled = True
    manifold_enabled = args.manifold
    fusion_disabled  = False

    label_extra = ""
    if args.ablation == "no_nonlocal":
        nonlocal_enabled = False
        label_extra = "no NL"
    elif args.ablation == "no_fusion":
        fusion_disabled  = True
        label_extra = "smooth=0 (sparse only)"
    elif args.ablation == "no_manifold":
        manifold_enabled = False
        label_extra = "no manifold"

    if fusion_disabled:
        # Force regime map to all-ones so the fusion picks the sparse branch
        # everywhere.  We achieve this without changing the API by patching
        # `compute_regime_map` for this run; cleaner is to expose a flag in
        # run_crown_inpaint, but a one-shot monkey-patch keeps changes
        # localised to the experiment driver.
        import crown.run as _crown_run
        _orig_regime = _crown_run.compute_regime_map
        def _all_ones_regime(*a, **kw):
            ref = _orig_regime(*a, **kw)
            return np.ones_like(ref)
        _crown_run.compute_regime_map = _all_ones_regime

    crown_out = run_crown_inpaint(
        y=y_corr, M=M, D=D,
        T=args.T, K_s=args.K_s,
        nonlocal_enabled=nonlocal_enabled,
        manifold_enabled=manifold_enabled,
        manifold_seed=args.seed,
        img_clean=img_clean,
        verbose=True,
    )

    if fusion_disabled:
        _crown_run.compute_regime_map = _orig_regime

    # ---- Metrics + plot ----------------------------------------------------
    metrics = {
        "biharmonic": compute_metrics(img_clean, img_bh,         M),
        "crown":      compute_metrics(img_clean, crown_out["u"], M),
    }
    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  {'method':<14s}  {'PSNR':>8s}  {'SSIM':>8s}  "
          f"{'hole-PSNR':>10s}  {'bnd-MAE':>8s}")
    print("  " + "-" * 56)
    for name, m in [("biharmonic",   metrics["biharmonic"]),
                    ("CROWN-Inpaint", metrics["crown"])]:
        print(f"  {name:<14s}  {m['psnr']:>8.2f}  {m['ssim']:>8.4f}  "
              f"{m['hole_psnr']:>10.2f}  {m['boundary_mae']:>8.4f}")
    dp = metrics["crown"]["psnr"]      - metrics["biharmonic"]["psnr"]
    dh = metrics["crown"]["hole_psnr"] - metrics["biharmonic"]["hole_psnr"]
    ds = metrics["crown"]["ssim"]      - metrics["biharmonic"]["ssim"]
    print(f"  {'Δ (CROWN-BH)':<14s}  {dp:>+8.2f}  {ds:>+8.4f}  {dh:>+10.2f}")

    plot_comparison(
        img_clean, y_corr, M, img_bh, crown_out,
        mask_info=mask_info, metrics=metrics,
        save_path=os.path.join(out_dir, f"crown_{mask_info['token']}.png"),
        ablation_label=label_extra or None,
    )


if __name__ == "__main__":
    main()
