#!/usr/bin/env python3
"""Final random-dropout missing-ratio sweep.

This script runs Biharmonic, Masked K-SVD, and CROWN-full on five 128x128
images with random pixel dropout at 10%, 30%, 50%, and 70%.

Primary plot: missing-pixel PSNR vs. missing ratio. Full-image PSNR is also
saved as a secondary plot, but the headline/crossover logic uses hole PSNR
because observed pixels are hard-projected back to ground truth.

Outputs per run:

* results.csv: one row per image, missing ratio, and method
* aggregate.csv: mean/std by missing ratio and method
* summary.json: crossover and CROWN dominance diagnostics
* psnr_vs_missing_ratio.png: aggregate hole PSNR plot
* full_psnr_vs_missing_ratio.png: aggregate full PSNR plot
* per_image_psnr_grid.png: per-image hole PSNR lines
* examples_missing_70.png: input | biharmonic | K-SVD | CROWN | GT examples

Run from the repository root:

    MPLCONFIGDIR=/private/tmp/mpl-cache \
    nohup venv/bin/python -u final_tests/scripts/missing_ratio_sweep.py \
      > final_tests/results/missing_ratio_sweep.log 2>&1 &
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
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "final_tests" / "results" / "missing_ratio_sweep"
DEFAULT_IMAGES = ("brick", "camera", "coins", "moon", "astronaut")
DEFAULT_MISSING_RATIOS = (0.10, 0.30, 0.50, 0.70)
METHODS = ("biharmonic", "ksvd", "crown_full")
METHOD_LABELS = {
    "biharmonic": "Biharmonic",
    "ksvd": "K-SVD",
    "crown_full": "CROWN",
}
METHOD_COLORS = {
    "biharmonic": "tab:blue",
    "ksvd": "tab:orange",
    "crown_full": "tab:green",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run final random-dropout missing-ratio sweep."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--images", default=",".join(DEFAULT_IMAGES))
    parser.add_argument(
        "--missing-ratios",
        default=",".join(str(r) for r in DEFAULT_MISSING_RATIOS),
        help="Comma-separated fractions, e.g. 0.1,0.3,0.5,0.7.",
    )
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--sparsity", type=int, default=8)
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--dict-iters", type=int, default=25)
    parser.add_argument("--als-iters", type=int, default=10)
    parser.add_argument("--crown-T", type=int, default=5)
    parser.add_argument("--K-s", type=int, default=10)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-ratios", type=int, default=None)
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


def parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_ratio_list(value: str) -> list[float]:
    ratios = [float(part.strip()) for part in value.split(",") if part.strip()]
    ratios = [ratio / 100.0 if ratio > 1.0 else ratio for ratio in ratios]
    bad = [ratio for ratio in ratios if not 0.0 < ratio < 1.0]
    if bad:
        raise ValueError(
            "missing ratios must be fractions in (0, 1) or percentages in (1, 100), "
            f"got {bad}"
        )
    return ratios


def source_loaders() -> dict[str, Callable[[], np.ndarray]]:
    return {
        "brick": lambda: to_gray_float(skdata.brick()),
        "camera": lambda: to_gray_float(skdata.camera()),
        "coins": lambda: to_gray_float(skdata.coins()),
        "moon": lambda: to_gray_float(skdata.moon()),
        "astronaut": lambda: to_gray_float(skdata.astronaut()),
        "chelsea": lambda: to_gray_float(skdata.chelsea()),
        "grass": lambda: to_gray_float(skdata.grass()),
        "gravel": lambda: to_gray_float(skdata.gravel()),
        "coffee": lambda: to_gray_float(skdata.coffee()),
    }


def to_gray_float(img: np.ndarray) -> np.ndarray:
    arr = img_as_float(img)
    if arr.ndim == 3:
        arr = rgb2gray(arr[..., :3])
    return np.asarray(arr, dtype=np.float64)


def prepare_image(name: str, image_size: int, loaders: dict[str, Callable[[], np.ndarray]]) -> np.ndarray:
    if name not in loaders:
        raise ValueError(f"unknown image '{name}'. Available: {sorted(loaders)}")
    img = loaders[name]()
    img = sk_resize(
        img,
        (image_size, image_size),
        anti_aliasing=True,
        preserve_range=True,
    )
    return np.clip(img.astype(np.float64), 0.0, 1.0)


def make_run_dir(output_dir: Path, run_name: str | None) -> Path:
    if run_name is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_name = f"run_{stamp}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "examples").mkdir(parents=True, exist_ok=True)
    return run_dir


def make_random_dropout_mask(shape: tuple[int, int], missing_ratio: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = (rng.uniform(size=shape) >= missing_ratio).astype(np.float64)
    if mask.sum() == 0:
        y = int(rng.integers(0, shape[0]))
        x = int(rng.integers(0, shape[1]))
        mask[y, x] = 1.0
    return mask


def hard_project(pred: np.ndarray, clean: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.clip(np.where(mask == 1, clean, pred), 0.0, 1.0)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def psnr_from_mse(value: float) -> float:
    return float(10.0 * math.log10(1.0 / max(value, 1e-12)))


def boundary_band(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    missing = mask == 0
    outer = binary_dilation(missing, iterations=radius)
    inner = binary_erosion(missing, iterations=radius, border_value=0)
    return outer & ~inner


def compute_metrics(clean: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    missing = mask == 0
    band = boundary_band(mask)
    full_mse = mse(clean, pred)
    missing_mse = mse(clean[missing], pred[missing])
    return {
        "full_psnr": float(psnr_fn(clean, pred, data_range=1.0)),
        "full_ssim": float(ssim_fn(clean, pred, data_range=1.0)),
        "full_mse": full_mse,
        "full_mae": float(np.mean(np.abs(clean - pred))),
        "hole_psnr": psnr_from_mse(missing_mse),
        "hole_mse": missing_mse,
        "hole_mae": float(np.mean(np.abs(clean[missing] - pred[missing]))),
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
    _n_features, n_patches = X_nan.shape
    K = D.shape[1]
    alpha = np.zeros((K, n_patches), dtype=np.float64)
    s_eff = min(int(sparsity), K)
    for i in range(n_patches):
        alpha[:, i] = masked_omp(X_nan[:, i], D, M_all[:, i], s_eff)
    return alpha


def run_biharmonic(clean: np.ndarray, mask: np.ndarray) -> np.ndarray:
    y = clean * mask
    pred = biharmonic_init(y, mask)
    return hard_project(pred, clean, mask)


def train_masked_dictionary(
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


def run_ksvd_with_dictionary(
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
    return hard_project(out["u"], clean, mask), {
        "crown_iters_completed": int(len(out["history"]) - 1),
        "avg_outer_time_sec": float(np.mean(out["timings"])) if out["timings"] else 0.0,
    }


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "image",
        "missing_ratio",
        "missing_frac_actual",
        "method",
        "seed",
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


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = sorted({(r["missing_ratio"], r["method"]) for r in rows})
    for missing_ratio, method in keys:
        group = [r for r in rows if r["missing_ratio"] == missing_ratio and r["method"] == method]
        out.append(
            {
                "missing_ratio": missing_ratio,
                "method": method,
                "num_images": len(group),
                "hole_psnr_mean": float(np.mean([r["hole_psnr"] for r in group])),
                "hole_psnr_std": float(np.std([r["hole_psnr"] for r in group])),
                "full_psnr_mean": float(np.mean([r["full_psnr"] for r in group])),
                "full_psnr_std": float(np.std([r["full_psnr"] for r in group])),
                "full_ssim_mean": float(np.mean([r["full_ssim"] for r in group])),
                "hole_mae_mean": float(np.mean([r["hole_mae"] for r in group])),
                "runtime_sec_mean": float(np.mean([r["runtime_sec"] for r in group])),
                "missing_frac_actual_mean": float(
                    np.mean([r["missing_frac_actual"] for r in group])
                ),
            }
        )
    return out


def write_aggregate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "missing_ratio",
        "method",
        "num_images",
        "hole_psnr_mean",
        "hole_psnr_std",
        "full_psnr_mean",
        "full_psnr_std",
        "full_ssim_mean",
        "hole_mae_mean",
        "runtime_sec_mean",
        "missing_frac_actual_mean",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean_lookup(aggregate: list[dict[str, Any]], metric: str) -> dict[tuple[float, str], float]:
    return {(row["missing_ratio"], row["method"]): row[metric] for row in aggregate}


def linear_crossover(
    ratios: list[float],
    values_a_minus_b: list[float],
) -> float | None:
    for idx in range(1, len(ratios)):
        x0, x1 = ratios[idx - 1], ratios[idx]
        y0, y1 = values_a_minus_b[idx - 1], values_a_minus_b[idx]
        if y0 == 0:
            return float(x0)
        if (y0 < 0 <= y1) or (y0 > 0 >= y1):
            denom = y1 - y0
            if abs(denom) < 1e-12:
                return float(x1)
            return float(x0 + (0.0 - y0) * (x1 - x0) / denom)
    if values_a_minus_b and values_a_minus_b[0] >= 0:
        return float(ratios[0])
    return None


def summarize(
    rows: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    images: list[str],
    ratios: list[float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    hole = mean_lookup(aggregate, "hole_psnr_mean")
    ksvd_minus_bh = [
        hole[(ratio, "ksvd")] - hole[(ratio, "biharmonic")] for ratio in ratios
    ]
    crown_minus_bh = [
        hole[(ratio, "crown_full")] - hole[(ratio, "biharmonic")] for ratio in ratios
    ]
    crown_minus_ksvd = [
        hole[(ratio, "crown_full")] - hole[(ratio, "ksvd")] for ratio in ratios
    ]
    crown_margin_vs_best = [
        hole[(ratio, "crown_full")]
        - max(hole[(ratio, "biharmonic")], hole[(ratio, "ksvd")])
        for ratio in ratios
    ]
    discrete = next(
        (float(ratio) for ratio, delta in zip(ratios, ksvd_minus_bh) if delta > 0),
        None,
    )

    per_image_crossover = {}
    for image in images:
        deltas = []
        for ratio in ratios:
            by_method = {
                row["method"]: row
                for row in rows
                if row["image"] == image and row["missing_ratio"] == ratio
            }
            deltas.append(by_method["ksvd"]["hole_psnr"] - by_method["biharmonic"]["hole_psnr"])
        per_image_crossover[image] = {
            "ksvd_minus_biharmonic_by_ratio": {
                str(ratio): float(delta) for ratio, delta in zip(ratios, deltas)
            },
            "linear_crossover_missing_ratio": linear_crossover(ratios, deltas),
            "first_discrete_ratio_where_ksvd_beats_biharmonic": next(
                (float(ratio) for ratio, delta in zip(ratios, deltas) if delta > 0),
                None,
            ),
        }

    return {
        "script_version": SCRIPT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "parameters": {
            "seed": args.seed,
            "image_size": args.image_size,
            "images": images,
            "missing_ratios": ratios,
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
            "metric": "mean hole_psnr across images",
            "ksvd_minus_biharmonic_by_ratio": {
                str(ratio): float(delta) for ratio, delta in zip(ratios, ksvd_minus_bh)
            },
            "ksvd_biharmonic_linear_crossover_missing_ratio": linear_crossover(
                ratios, ksvd_minus_bh
            ),
            "first_discrete_ratio_where_ksvd_beats_biharmonic": discrete,
            "crown_minus_biharmonic_by_ratio": {
                str(ratio): float(delta) for ratio, delta in zip(ratios, crown_minus_bh)
            },
            "crown_minus_ksvd_by_ratio": {
                str(ratio): float(delta) for ratio, delta in zip(ratios, crown_minus_ksvd)
            },
            "crown_margin_vs_best_baseline_by_ratio": {
                str(ratio): float(delta)
                for ratio, delta in zip(ratios, crown_margin_vs_best)
            },
            "crown_dominates_all_ratios": all(delta > 0 for delta in crown_margin_vs_best),
        },
        "per_image_crossover": per_image_crossover,
    }


def plot_metric(
    path: Path,
    aggregate: list[dict[str, Any]],
    ratios: list[float],
    metric_mean: str,
    metric_std: str,
    ylabel: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.array(ratios, dtype=np.float64) * 100.0
    for method in METHODS:
        group = [row for row in aggregate if row["method"] == method]
        by_ratio = {row["missing_ratio"]: row for row in group}
        y = np.array([by_ratio[ratio][metric_mean] for ratio in ratios])
        err = np.array([by_ratio[ratio][metric_std] for ratio in ratios])
        ax.errorbar(
            x,
            y,
            yerr=err,
            marker="o",
            linewidth=2.0,
            capsize=4,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    ax.set_xlabel("Missing pixels (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_per_image_grid(path: Path, rows: list[dict[str, Any]], images: list[str], ratios: list[float]) -> None:
    n_images = len(images)
    n_cols = min(3, n_images)
    n_rows = int(math.ceil(n_images / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.1 * n_rows), squeeze=False)
    x = np.array(ratios, dtype=np.float64) * 100.0
    for idx, image in enumerate(images):
        ax = axes[idx // n_cols][idx % n_cols]
        image_rows = [row for row in rows if row["image"] == image]
        for method in METHODS:
            by_ratio = {
                row["missing_ratio"]: row
                for row in image_rows
                if row["method"] == method
            }
            y = [by_ratio[ratio]["hole_psnr"] for ratio in ratios]
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.8,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        ax.set_title(image)
        ax.set_xlabel("Missing (%)")
        ax.set_ylabel("Hole PSNR (dB)")
        ax.grid(True, alpha=0.25)
    for idx in range(n_images, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(METHODS), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_examples_grid(
    path: Path,
    examples: list[dict[str, Any]],
    example_ratio: float,
) -> None:
    selected = [ex for ex in examples if ex["missing_ratio"] == example_ratio]
    if not selected:
        return
    columns = [
        ("Input", "input"),
        ("Biharmonic", "biharmonic"),
        ("K-SVD", "ksvd"),
        ("CROWN", "crown_full"),
        ("GT", "gt"),
    ]
    fig, axes = plt.subplots(
        len(selected),
        len(columns),
        figsize=(2.25 * len(columns), 2.0 * len(selected)),
        squeeze=False,
    )
    for row_idx, rec in enumerate(selected):
        for col_idx, (label, key) in enumerate(columns):
            ax = axes[row_idx][col_idx]
            ax.imshow(rec[key], cmap="gray", vmin=0.0, vmax=1.0)
            if row_idx == 0:
                ax.set_title(label, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(rec["image"], fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"Examples at {int(example_ratio * 100)}% random dropout", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def apply_quick_overrides(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.K = min(args.K, 16)
    args.sparsity = min(args.sparsity, 3)
    args.n_train = min(args.n_train, 100)
    args.dict_iters = min(args.dict_iters, 2)
    args.als_iters = min(args.als_iters, 2)
    args.crown_T = min(args.crown_T, 1)
    if args.max_images is None:
        args.max_images = 1
    if args.max_ratios is None:
        args.max_ratios = 2


def run_one_case(
    image_name: str,
    clean: np.ndarray,
    missing_ratio: float,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mask_seed = int(args.seed + 1009 * DEFAULT_IMAGES.index(image_name)) if image_name in DEFAULT_IMAGES else args.seed
    ratio_seed = int(mask_seed + round(missing_ratio * 1000))
    mask = make_random_dropout_mask(clean.shape, missing_ratio, ratio_seed)
    y = clean * mask
    missing_frac_actual = float(1.0 - mask.mean())

    t0 = time.time()
    u_bh = run_biharmonic(clean, mask)
    bh_time = time.time() - t0

    t0 = time.time()
    D, train_err = train_masked_dictionary(y, mask, args)
    train_time = time.time() - t0
    train_extra = {
        "train_rmse_initial": float(train_err[0]) if len(train_err) else "",
        "train_rmse_final": float(train_err[-1]) if len(train_err) else "",
    }

    t0 = time.time()
    u_ksvd, ksvd_extra = run_ksvd_with_dictionary(clean, mask, D, args)
    ksvd_time = train_time + (time.time() - t0)

    t0 = time.time()
    u_crown, crown_extra = run_crown_with_dictionary(clean, mask, D, args)
    crown_time = train_time + (time.time() - t0)

    outputs = {
        "biharmonic": (u_bh, bh_time, {}),
        "ksvd": (u_ksvd, ksvd_time, {**train_extra, **ksvd_extra}),
        "crown_full": (u_crown, crown_time, {**train_extra, **crown_extra}),
    }
    rows = []
    for method, (pred, runtime_sec, extra) in outputs.items():
        row = {
            "image": image_name,
            "missing_ratio": float(missing_ratio),
            "missing_frac_actual": missing_frac_actual,
            "method": method,
            "seed": args.seed,
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

    example = {
        "image": image_name,
        "missing_ratio": float(missing_ratio),
        "input": y,
        "biharmonic": u_bh,
        "ksvd": u_ksvd,
        "crown_full": u_crown,
        "gt": clean,
        "mask": mask,
    }
    return rows, example


def main() -> None:
    args = parse_args()
    apply_quick_overrides(args)
    loaders = source_loaders()
    images = parse_csv_list(args.images)
    ratios = parse_ratio_list(args.missing_ratios)
    if args.max_images is not None:
        images = images[: args.max_images]
    if args.max_ratios is not None:
        ratios = ratios[: args.max_ratios]
    run_dir = make_run_dir(args.output_dir, args.run_name)

    print(f"[missing-ratio] writing results to {run_dir}")
    print(
        "[missing-ratio] parameters: "
        f"K={args.K}, s={args.sparsity}, n_train={args.n_train}, "
        f"dict_iters={args.dict_iters}, crown_T={args.crown_T}, "
        f"images={images}, ratios={ratios}"
    )

    prepared = {
        name: prepare_image(name, args.image_size, loaders)
        for name in images
    }
    rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    total = len(images) * len(ratios)
    case_idx = 0
    for image_name in images:
        for ratio in ratios:
            case_idx += 1
            print(
                f"[missing-ratio] case {case_idx}/{total}: "
                f"{image_name}, missing={int(ratio * 100)}%"
            )
            case_rows, example = run_one_case(image_name, prepared[image_name], ratio, args)
            rows.extend(case_rows)
            examples.append(example)
            for row in case_rows:
                print(
                    f"  {METHOD_LABELS[row['method']]:10s} "
                    f"hole_psnr={row['hole_psnr']:.2f} dB "
                    f"full_psnr={row['full_psnr']:.2f} dB "
                    f"runtime={row['runtime_sec']:.1f}s"
                )

    aggregate = aggregate_rows(rows)
    write_rows_csv(run_dir / "results.csv", rows)
    write_aggregate_csv(run_dir / "aggregate.csv", aggregate)
    summary = summarize(rows, aggregate, images, ratios, args)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_metric(
        run_dir / "psnr_vs_missing_ratio.png",
        aggregate,
        ratios,
        "hole_psnr_mean",
        "hole_psnr_std",
        "Missing-pixel PSNR (dB)",
        "Random Dropout Sweep",
    )
    plot_metric(
        run_dir / "full_psnr_vs_missing_ratio.png",
        aggregate,
        ratios,
        "full_psnr_mean",
        "full_psnr_std",
        "Full-image PSNR (dB)",
        "Random Dropout Sweep",
    )
    plot_per_image_grid(run_dir / "per_image_psnr_grid.png", rows, images, ratios)
    save_examples_grid(
        run_dir / f"examples_missing_{int(max(ratios) * 100)}.png",
        examples,
        max(ratios),
    )

    headline = summary["headline"]
    print(f"[missing-ratio] wrote {run_dir / 'results.csv'}")
    print(f"[missing-ratio] wrote {run_dir / 'aggregate.csv'}")
    print(f"[missing-ratio] wrote {run_dir / 'psnr_vs_missing_ratio.png'}")
    print(
        "[missing-ratio] K-SVD/Biharmonic crossover: "
        f"{headline['ksvd_biharmonic_linear_crossover_missing_ratio']}"
    )
    print(
        "[missing-ratio] CROWN dominates all ratios: "
        f"{headline['crown_dominates_all_ratios']}"
    )


if __name__ == "__main__":
    main()
