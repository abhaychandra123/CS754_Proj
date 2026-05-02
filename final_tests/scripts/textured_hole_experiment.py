#!/usr/bin/env python3
"""Final textured-hole inpainting experiment.

This script evaluates 24x24 square holes deliberately placed on textured
regions of 128x128 crops. It writes:

* results.csv: one row per crop and method
* summary.json: aggregate headline metrics
* visual_grid.png: input | biharmonic | CROWN | GT
* visual_grid_with_ksvd.png: input | biharmonic | masked K-SVD | CROWN | GT
* per_crop/*.png and per_crop/*.npy

Run from the repository root:

    MPLCONFIGDIR=/private/tmp/mpl-cache \
    nohup venv/bin/python -u final_tests/scripts/textured_hole_experiment.py \
      > final_tests/results/textured_hole.log 2>&1 &
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cs754_proj_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/cs754_proj_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from skimage import data as skdata
from skimage.color import rgb2gray
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from skimage.transform import resize as sk_resize
from skimage.util import img_as_float

from crown.run import run_crown_inpaint, train_dictionary
from crown.smooth import biharmonic_init
from crown.utils import PATCH_SIZE, extract_patches_with_mask, reconstruct_image_from_codes
from masked_ksvd import masked_omp


SCRIPT_VERSION = "1.0"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "final_tests" / "results" / "textured_hole"
METHODS = ("biharmonic", "masked_ksvd", "crown_full")


@dataclass(frozen=True)
class CropSpec:
    name: str
    source: str
    texture: str
    crop_y: int
    crop_x: int
    hole_y: int
    hole_x: int


# Fixed textured regions. The 24x24 hole positions are intentionally placed
# within named texture areas; they are not center squares and not random draws.
CROP_SPECS: tuple[CropSpec, ...] = (
    CropSpec("brick_wall_01", "brick", "brick_wall", 0, 28, 44, 72),
    CropSpec("brick_wall_02", "brick", "brick_wall", 150, 10, 76, 34),
    CropSpec("brick_wall_03", "brick", "brick_wall", 300, 250, 36, 82),
    CropSpec("brick_wall_04", "brick", "brick_wall", 60, 320, 88, 42),
    CropSpec("fabric_suit_01", "astronaut", "fabric_suit", 278, 62, 74, 40),
    CropSpec("fabric_suit_02", "astronaut", "fabric_suit", 310, 215, 46, 76),
    CropSpec("fabric_suit_03", "astronaut", "fabric_suit", 320, 20, 34, 56),
    CropSpec("hair_01", "astronaut", "hair", 24, 130, 40, 70),
    CropSpec("hair_02", "astronaut", "hair", 40, 176, 24, 28),
    CropSpec("hair_fur_03", "chelsea", "hair_fur", 0, 92, 54, 78),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run final textured 24x24-hole inpainting experiment."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-crops", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--hole-size", type=int, default=24)
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--sparsity", type=int, default=8)
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--dict-iters", type=int, default=25)
    parser.add_argument("--als-iters", type=int, default=10)
    parser.add_argument("--crown-T", type=int, default=5)
    parser.add_argument("--K-s", type=int, default=10)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use tiny parameters for a fast smoke test; not paper results.",
    )
    parser.add_argument(
        "--verbose-algo",
        action="store_true",
        help="Show inner Masked K-SVD and CROWN logs.",
    )
    return parser.parse_args()


def source_loaders() -> dict[str, Callable[[], np.ndarray]]:
    return {
        "brick": lambda: to_gray_float(skdata.brick()),
        "astronaut": lambda: to_gray_float(skdata.astronaut()),
        "chelsea": lambda: to_gray_float(skdata.chelsea()),
    }


def to_gray_float(img: np.ndarray) -> np.ndarray:
    arr = img_as_float(img)
    if arr.ndim == 3:
        arr = rgb2gray(arr[..., :3])
    return np.asarray(arr, dtype=np.float64)


def make_run_dir(output_dir: Path, run_name: str | None) -> Path:
    if run_name is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_name = f"run_{stamp}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "per_crop").mkdir(parents=True, exist_ok=True)
    return run_dir


def crop_image(img: np.ndarray, spec: CropSpec, crop_size: int) -> np.ndarray:
    y0, x0 = int(spec.crop_y), int(spec.crop_x)
    y1, x1 = y0 + crop_size, x0 + crop_size
    if y0 < 0 or x0 < 0 or y1 > img.shape[0] or x1 > img.shape[1]:
        raise ValueError(f"crop {spec.name} is out of bounds for {img.shape}")
    return img[y0:y1, x0:x1].copy()


def make_square_mask(shape: tuple[int, int], y0: int, x0: int, hole_size: int) -> np.ndarray:
    h, w = shape
    if y0 < 0 or x0 < 0 or y0 + hole_size > h or x0 + hole_size > w:
        raise ValueError(
            f"hole {(y0, x0, hole_size)} is out of bounds for image shape {shape}"
        )
    mask = np.ones(shape, dtype=np.float64)
    mask[y0 : y0 + hole_size, x0 : x0 + hole_size] = 0.0
    return mask


def hard_project(pred: np.ndarray, clean: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.clip(np.where(mask == 1, clean, pred), 0.0, 1.0)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def psnr_from_mse(value: float) -> float:
    return float(10.0 * math.log10(1.0 / max(value, 1e-12)))


def boundary_band(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    hole = mask == 0
    outer = binary_dilation(hole, iterations=radius)
    inner = binary_erosion(hole, iterations=radius, border_value=0)
    band = outer & ~inner
    return band


def compute_metrics(clean: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    hole = mask == 0
    band = boundary_band(mask)
    full_mse = mse(clean, pred)
    hole_mse = mse(clean[hole], pred[hole])
    return {
        "full_psnr": float(psnr_fn(clean, pred, data_range=1.0)),
        "full_ssim": float(ssim_fn(clean, pred, data_range=1.0)),
        "full_mse": full_mse,
        "full_mae": float(np.mean(np.abs(clean - pred))),
        "hole_psnr": psnr_from_mse(hole_mse),
        "hole_mse": hole_mse,
        "hole_mae": float(np.mean(np.abs(clean[hole] - pred[hole]))),
        "boundary_mae": float(np.mean(np.abs(clean[band] - pred[band]))),
        "observed_max_abs_err": float(np.max(np.abs(clean[mask == 1] - pred[mask == 1]))),
    }


@contextlib.contextmanager
def maybe_silence(enabled: bool):
    if enabled:
        yield
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            yield


def sparse_code_masked_patches(
    X_nan: np.ndarray,
    M_all: np.ndarray,
    D: np.ndarray,
    sparsity: int,
) -> np.ndarray:
    n_features, n_patches = X_nan.shape
    K = D.shape[1]
    alpha = np.zeros((K, n_patches), dtype=np.float64)
    s_eff = min(int(sparsity), K)
    for i in range(n_patches):
        alpha[:, i] = masked_omp(X_nan[:, i], D, M_all[:, i], s_eff)
    return alpha


def run_biharmonic(clean: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    y = clean * mask
    pred = biharmonic_init(y, mask)
    return hard_project(pred, clean, mask), {}


def train_crop_dictionary(
    y: np.ndarray,
    mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    with maybe_silence(args.verbose_algo):
        D, train_err = train_dictionary(
            y,
            mask,
            n_train=args.n_train,
            n_iter=args.dict_iters,
            K=args.K,
            sparsity=min(args.sparsity, args.K),
            als_iters=args.als_iters,
            seed=args.seed,
            verbose=args.verbose_algo,
        )
    return D, train_err


def run_masked_ksvd_with_dictionary(
    clean: np.ndarray,
    mask: np.ndarray,
    D: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    y = clean * mask
    _, X_nan, M_all, _ = extract_patches_with_mask(y, mask, patch_shape=PATCH_SIZE)
    alpha = sparse_code_masked_patches(X_nan, M_all, D, args.sparsity)
    pred = reconstruct_image_from_codes(D, alpha, clean.shape, patch_shape=PATCH_SIZE)
    return hard_project(pred, clean, mask), {
        "avg_nnz": float((alpha != 0).sum() / max(alpha.shape[1], 1)),
    }


def run_crown_with_dictionary(
    clean: np.ndarray,
    mask: np.ndarray,
    D: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    y = clean * mask
    with maybe_silence(args.verbose_algo):
        out = run_crown_inpaint(
            y,
            mask,
            D,
            T=args.crown_T,
            sparsity=min(args.sparsity, args.K),
            K_s=args.K_s,
            img_clean=clean,
            verbose=args.verbose_algo,
        )
    pred = hard_project(out["u"], clean, mask)
    return pred, {
        "crown_iters_completed": int(len(out["history"]) - 1),
        "avg_outer_time_sec": float(np.mean(out["timings"])) if out["timings"] else 0.0,
    }


def save_crop_grid(
    path: Path,
    title: str,
    images: list[tuple[str, np.ndarray]],
    mask: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, len(images), figsize=(3.0 * len(images), 3.1))
    if len(images) == 1:
        axes = [axes]
    for ax, (label, img) in zip(axes, images):
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.contour(mask == 0, levels=[0.5], colors="tab:red", linewidths=0.7)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_visual_grid(
    path: Path,
    records: list[dict[str, Any]],
    columns: list[tuple[str, str]],
) -> None:
    n_rows = len(records)
    n_cols = len(columns)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.25 * n_cols, 2.05 * n_rows),
        squeeze=False,
    )
    for row, rec in enumerate(records):
        for col, (label, key) in enumerate(columns):
            ax = axes[row][col]
            ax.imshow(rec[key], cmap="gray", vmin=0.0, vmax=1.0)
            ax.contour(rec["mask"] == 0, levels=[0.5], colors="tab:red", linewidths=0.45)
            if row == 0:
                ax.set_title(label, fontsize=10)
            if col == 0:
                ax.set_ylabel(rec["name"], fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "crop",
        "source",
        "texture",
        "method",
        "hole_y",
        "hole_x",
        "hole_size",
        "K",
        "sparsity",
        "n_train",
        "dict_iters",
        "als_iters",
        "crown_T",
        "K_s",
        "runtime_sec",
        "full_psnr",
        "full_ssim",
        "full_mse",
        "full_mae",
        "hole_psnr",
        "hole_mse",
        "hole_mae",
        "boundary_mae",
        "observed_max_abs_err",
        "train_rmse_initial",
        "train_rmse_final",
        "avg_nnz",
        "crown_iters_completed",
        "avg_outer_time_sec",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for row in rows:
        by_method[row["method"]].append(row)

    method_summary = {}
    for method, method_rows in by_method.items():
        if not method_rows:
            continue
        method_summary[method] = {
            "n": len(method_rows),
            "hole_psnr_mean": float(np.mean([r["hole_psnr"] for r in method_rows])),
            "hole_psnr_std": float(np.std([r["hole_psnr"] for r in method_rows])),
            "full_psnr_mean": float(np.mean([r["full_psnr"] for r in method_rows])),
            "full_ssim_mean": float(np.mean([r["full_ssim"] for r in method_rows])),
            "runtime_sec_mean": float(np.mean([r["runtime_sec"] for r in method_rows])),
        }

    deltas = []
    for crop in sorted({r["crop"] for r in rows}):
        crop_rows = {r["method"]: r for r in rows if r["crop"] == crop}
        if "biharmonic" in crop_rows and "crown_full" in crop_rows:
            deltas.append(
                crop_rows["crown_full"]["hole_psnr"]
                - crop_rows["biharmonic"]["hole_psnr"]
            )

    return {
        "script_version": SCRIPT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "parameters": {
            "seed": args.seed,
            "crop_size": args.crop_size,
            "hole_size": args.hole_size,
            "K": args.K,
            "sparsity": args.sparsity,
            "n_train": args.n_train,
            "dict_iters": args.dict_iters,
            "als_iters": args.als_iters,
            "crown_T": args.crown_T,
            "K_s": args.K_s,
            "quick": args.quick,
        },
        "headline": {
            "metric": "mean(crown_full hole_psnr - biharmonic hole_psnr)",
            "crown_minus_biharmonic_hole_psnr_db": (
                float(np.mean(deltas)) if deltas else None
            ),
            "per_crop_deltas_db": deltas,
        },
        "method_summary": method_summary,
    }


def apply_quick_overrides(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.K = min(args.K, 16)
    args.sparsity = min(args.sparsity, 3)
    args.n_train = min(args.n_train, 100)
    args.dict_iters = min(args.dict_iters, 2)
    args.als_iters = min(args.als_iters, 2)
    args.crown_T = min(args.crown_T, 1)
    if args.max_crops is None:
        args.max_crops = 1


def main() -> None:
    args = parse_args()
    apply_quick_overrides(args)
    run_dir = make_run_dir(args.output_dir, args.run_name)
    loaders = source_loaders()
    crop_specs = CROP_SPECS[: args.max_crops] if args.max_crops else CROP_SPECS

    print(f"[textured-hole] writing results to {run_dir}")
    print(
        "[textured-hole] parameters: "
        f"K={args.K}, s={args.sparsity}, n_train={args.n_train}, "
        f"dict_iters={args.dict_iters}, crown_T={args.crown_T}, "
        f"crops={len(crop_specs)}"
    )

    rows: list[dict[str, Any]] = []
    visual_records: list[dict[str, Any]] = []

    for crop_idx, spec in enumerate(crop_specs, start=1):
        print(f"[textured-hole] crop {crop_idx}/{len(crop_specs)}: {spec.name}")
        clean = crop_image(loaders[spec.source](), spec, args.crop_size)
        if clean.shape != (args.crop_size, args.crop_size):
            clean = sk_resize(
                clean,
                (args.crop_size, args.crop_size),
                anti_aliasing=True,
                preserve_range=True,
            )
        clean = np.clip(clean.astype(np.float64), 0.0, 1.0)
        mask = make_square_mask(clean.shape, spec.hole_y, spec.hole_x, args.hole_size)
        y = clean * mask

        t0 = time.time()
        u_bh, bh_extra = run_biharmonic(clean, mask)
        bh_time = time.time() - t0

        t0 = time.time()
        D, train_err = train_crop_dictionary(y, mask, args)
        train_time = time.time() - t0
        train_extra = {
            "train_rmse_initial": float(train_err[0]) if len(train_err) else "",
            "train_rmse_final": float(train_err[-1]) if len(train_err) else "",
        }

        t0 = time.time()
        u_ksvd, ksvd_extra = run_masked_ksvd_with_dictionary(clean, mask, D, args)
        ksvd_time = train_time + (time.time() - t0)

        t0 = time.time()
        u_crown, crown_extra = run_crown_with_dictionary(clean, mask, D, args)
        crown_time = train_time + (time.time() - t0)

        method_outputs = {
            "biharmonic": (u_bh, bh_time, bh_extra),
            "masked_ksvd": (u_ksvd, ksvd_time, {**train_extra, **ksvd_extra}),
            "crown_full": (u_crown, crown_time, {**train_extra, **crown_extra}),
        }

        for method, (pred, runtime_sec, extra) in method_outputs.items():
            row = {
                "crop": spec.name,
                "source": spec.source,
                "texture": spec.texture,
                "method": method,
                "hole_y": spec.hole_y,
                "hole_x": spec.hole_x,
                "hole_size": args.hole_size,
                "K": args.K,
                "sparsity": args.sparsity,
                "n_train": args.n_train,
                "dict_iters": args.dict_iters,
                "als_iters": args.als_iters,
                "crown_T": args.crown_T,
                "K_s": args.K_s,
                "runtime_sec": float(runtime_sec),
            }
            row.update(compute_metrics(clean, pred, mask))
            row.update(extra)
            rows.append(row)
            print(
                f"  {method:12s} hole_psnr={row['hole_psnr']:.2f} dB "
                f"full_psnr={row['full_psnr']:.2f} dB runtime={runtime_sec:.1f}s"
            )

        np.save(run_dir / "per_crop" / f"{spec.name}_gt.npy", clean)
        np.save(run_dir / "per_crop" / f"{spec.name}_input.npy", y)
        np.save(run_dir / "per_crop" / f"{spec.name}_mask.npy", mask)
        np.save(run_dir / "per_crop" / f"{spec.name}_biharmonic.npy", u_bh)
        np.save(run_dir / "per_crop" / f"{spec.name}_masked_ksvd.npy", u_ksvd)
        np.save(run_dir / "per_crop" / f"{spec.name}_crown_full.npy", u_crown)

        save_crop_grid(
            run_dir / "per_crop" / f"{spec.name}_comparison.png",
            f"{spec.name} ({spec.texture})",
            [
                ("Input", y),
                ("Biharmonic", u_bh),
                ("Masked K-SVD", u_ksvd),
                ("CROWN", u_crown),
                ("GT", clean),
            ],
            mask,
        )
        visual_records.append(
            {
                **asdict(spec),
                "input": y,
                "biharmonic": u_bh,
                "masked_ksvd": u_ksvd,
                "crown_full": u_crown,
                "gt": clean,
                "mask": mask,
            }
        )

    write_csv(run_dir / "results.csv", rows)
    summary = summarize(rows, args)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    save_visual_grid(
        run_dir / "visual_grid.png",
        visual_records,
        [
            ("Input", "input"),
            ("Biharmonic", "biharmonic"),
            ("CROWN", "crown_full"),
            ("GT", "gt"),
        ],
    )
    save_visual_grid(
        run_dir / "visual_grid_with_ksvd.png",
        visual_records,
        [
            ("Input", "input"),
            ("Biharmonic", "biharmonic"),
            ("Masked K-SVD", "masked_ksvd"),
            ("CROWN", "crown_full"),
            ("GT", "gt"),
        ],
    )

    headline = summary["headline"]["crown_minus_biharmonic_hole_psnr_db"]
    print(f"[textured-hole] wrote {run_dir / 'results.csv'}")
    print(f"[textured-hole] wrote {run_dir / 'visual_grid.png'}")
    print(
        "[textured-hole] headline CROWN - biharmonic hole PSNR: "
        f"{headline:.2f} dB" if headline is not None else "[textured-hole] no headline"
    )


if __name__ == "__main__":
    main()
