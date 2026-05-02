#!/usr/bin/env python3
"""Dataset sweep for CROWN evaluation.

This script runs Biharmonic, Masked K-SVD, and CROWN-full on a downloaded
dataset (ImageNet validation subset, etc.) with random pixel dropout.

Outputs per run:
* results.csv: one row per image, missing ratio, and method
* aggregate.csv: mean/std by missing ratio and method
* summary.json: crossover and CROWN dominance diagnostics
* psnr_vs_missing_ratio.png: aggregate hole PSNR plot

Run from the repository root:

    nohup venv/bin/python -u dataset_sweep.py \
      > final_tests/results/dataset_sweep.log 2>&1 &
"""

from __future__ import annotations

import argparse
import glob
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

PROJECT_ROOT = Path(__file__).resolve().parent
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
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
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
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "final_tests" / "results" / "dataset_sweep"
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
        description="Run dataset evaluation sweep."
    )
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "dataset" / "images", help="Directory containing images")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=128)
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

def parse_ratio_list(value: str) -> list[float]:
    ratios = [float(part.strip()) for part in value.split(",") if part.strip()]
    ratios = [ratio / 100.0 if ratio > 1.0 else ratio for ratio in ratios]
    bad = [ratio for ratio in ratios if not 0.0 < ratio < 1.0]
    if bad:
        raise ValueError(f"missing ratios must be fractions in (0, 1), got {bad}")
    return ratios

def to_gray_float(img: np.ndarray) -> np.ndarray:
    arr = img_as_float(img)
    if arr.ndim == 3:
        arr = rgb2gray(arr[..., :3])
    return np.asarray(arr, dtype=np.float64)

def load_image(filepath: str, image_size: int) -> np.ndarray:
    img = np.array(Image.open(filepath))
    img = to_gray_float(img)
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
    run_dir.mkdir(parents=True, exist_ok=True)
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

def run_one_case(
    image_name: str,
    clean: np.ndarray,
    missing_ratio: float,
    args: argparse.Namespace,
    idx: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mask_seed = int(args.seed + 1009 * idx)
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


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: return
    fieldnames = list(rows[0].keys())
    # Try to order some important ones first
    priority = ["image", "missing_ratio", "method", "hole_psnr", "full_psnr", "full_ssim", "runtime_sec"]
    fieldnames = priority + [k for k in fieldnames if k not in priority]
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
                "runtime_sec_mean": float(np.mean([r["runtime_sec"] for r in group])),
            }
        )
    return out

def write_aggregate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

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
        if not group: continue
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
            color=METHOD_COLORS.get(method, "black"),
            label=METHOD_LABELS.get(method, method),
        )
    ax.set_xlabel("Missing pixels (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    args = parse_args()
    if args.quick:
        args.K = min(args.K, 16)
        args.sparsity = min(args.sparsity, 3)
        args.n_train = min(args.n_train, 100)
        args.dict_iters = min(args.dict_iters, 2)
        args.als_iters = min(args.als_iters, 2)
        args.crown_T = min(args.crown_T, 1)

    ratios = parse_ratio_list(args.missing_ratios)
    
    # Load dataset
    image_files = sorted(glob.glob(os.path.join(args.dataset_dir, "*.png")))
    if not image_files:
        print(f"No PNG images found in {args.dataset_dir}!")
        print("Please run dataset_download.py first.")
        sys.exit(1)
        
    if args.max_images is not None:
        image_files = image_files[:args.max_images]

    run_dir = make_run_dir(args.output_dir, args.run_name)
    print(f"[dataset-sweep] writing results to {run_dir}")
    print(f"[dataset-sweep] processing {len(image_files)} images, ratios={ratios}")

    rows: list[dict[str, Any]] = []
    total = len(image_files) * len(ratios)
    case_idx = 0
    
    for i, file_path in enumerate(image_files):
        image_name = os.path.basename(file_path).split('.')[0]
        try:
            clean = load_image(file_path, args.image_size)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

        for ratio in ratios:
            case_idx += 1
            print(f"[dataset-sweep] case {case_idx}/{total}: {image_name}, missing={int(ratio * 100)}%")
            case_rows, _ = run_one_case(image_name, clean, ratio, args, i)
            rows.extend(case_rows)

    if not rows:
        print("No cases ran successfully.")
        return

    aggregate = aggregate_rows(rows)
    write_rows_csv(run_dir / "results.csv", rows)
    write_aggregate_csv(run_dir / "aggregate.csv", aggregate)

    plot_metric(
        run_dir / "psnr_vs_missing_ratio.png",
        aggregate,
        ratios,
        "hole_psnr_mean",
        "hole_psnr_std",
        "Missing-pixel PSNR (dB)",
        "Dataset Sweep (ImageNet subsets)",
    )
    plot_metric(
        run_dir / "full_psnr_vs_missing_ratio.png",
        aggregate,
        ratios,
        "full_psnr_mean",
        "full_psnr_std",
        "Full-image PSNR (dB)",
        "Dataset Sweep (ImageNet subsets)",
    )

    print(f"[dataset-sweep] evaluation complete. Results in {run_dir}")


if __name__ == "__main__":
    main()
