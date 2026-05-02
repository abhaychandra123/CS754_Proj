"""Small overnight experiment on larger images.

This script compares biharmonic initialization, masked K-SVD, and CROWN-
Inpaint on resized versions of the same reference image.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from skimage import data as skdata
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from skimage.transform import resize as sk_resize

from crown.run import run_crown_inpaint, train_dictionary
from crown.smooth import biharmonic_init
from crown.utils import extract_patches_with_mask, reconstruct_image_from_codes
from custom_masks import square_hole_mask
from masked_ksvd import masked_omp


OUTPUT_DIR = PROJECT_ROOT / "final_tests" / "results" / "big_image_try"
SEED = 42
K = 256
SPARSITY = 8
DICT_ITERS = 15
ALS_ITERS = 10
TRAIN_PATCHES = 3000
CROWN_T = 5
SIZES = [256, 512]


brick_512 = skdata.brick().astype(np.float64) / 255.0

METHOD_LABELS = {
    "biharmonic": "Biharmonic",
    "masked_ksvd": "Masked K-SVD",
    "crown": "CROWN",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run larger brick inpainting comparisons.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--sizes", default=",".join(str(size) for size in SIZES))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--K", type=int, default=K)
    parser.add_argument("--sparsity", type=int, default=SPARSITY)
    parser.add_argument("--dict-iters", type=int, default=DICT_ITERS)
    parser.add_argument("--als-iters", type=int, default=ALS_ITERS)
    parser.add_argument("--n-train", type=int, default=TRAIN_PATCHES)
    parser.add_argument("--crown-T", type=int, default=CROWN_T)
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Use cheaper settings: K=128, s=6, dict_iters=5, n_train=1000, "
            "crown_T=3, and only 256 unless --sizes is explicitly set."
        ),
    )
    return parser.parse_args()


def parse_sizes(value: str) -> list[int]:
    sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not sizes:
        raise ValueError("at least one size is required")
    return sizes


def apply_args(args: argparse.Namespace) -> list[int]:
    global OUTPUT_DIR, SEED, K, SPARSITY, DICT_ITERS, ALS_ITERS, TRAIN_PATCHES, CROWN_T

    sizes_was_default = args.sizes == ",".join(str(size) for size in SIZES)
    if args.fast:
        args.K = min(args.K, 128)
        args.sparsity = min(args.sparsity, 6)
        args.dict_iters = min(args.dict_iters, 5)
        args.n_train = min(args.n_train, 1000)
        args.crown_T = min(args.crown_T, 3)
        if sizes_was_default:
            args.sizes = "256"

    OUTPUT_DIR = args.output_dir
    SEED = int(args.seed)
    K = int(args.K)
    SPARSITY = int(args.sparsity)
    DICT_ITERS = int(args.dict_iters)
    ALS_ITERS = int(args.als_iters)
    TRAIN_PATCHES = int(args.n_train)
    CROWN_T = int(args.crown_T)
    return parse_sizes(args.sizes)


def make_square_hole(size: int, hole_size: int | None = None) -> np.ndarray:
    if hole_size is None:
        hole_size = size // 4
    return square_hole_mask((size, size), square_size=hole_size, center=None)


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
    return outer & ~inner


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
        "boundary_mae": float(np.mean(np.abs(clean[band] - pred[band]))) if band.any() else 0.0,
        "observed_max_abs_err": float(np.max(np.abs(clean[mask == 1] - pred[mask == 1]))),
    }


def run_masked_ksvd_image(img: np.ndarray, mask: np.ndarray, D: np.ndarray) -> np.ndarray:
    y = img * mask
    _, X_nan, M_all, _ = extract_patches_with_mask(y, mask)

    alpha = np.zeros((D.shape[1], X_nan.shape[1]), dtype=np.float64)
    s_eff = min(SPARSITY, D.shape[1])
    for i in range(X_nan.shape[1]):
        alpha[:, i] = masked_omp(X_nan[:, i], D, M_all[:, i], s_eff)

    pred = reconstruct_image_from_codes(D, alpha, img.shape)
    return hard_project(pred, img, mask)


def write_results_csv(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "size",
        "method",
        "K",
        "sparsity",
        "n_train",
        "dict_iters",
        "als_iters",
        "crown_T",
        "dict_train_sec",
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
    ]
    with (OUTPUT_DIR / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_summary(rows: list[dict[str, object]]) -> None:
    summary = {
        "parameters": {
            "seed": SEED,
            "K": K,
            "sparsity": SPARSITY,
            "dict_iters": DICT_ITERS,
            "als_iters": ALS_ITERS,
            "n_train": TRAIN_PATCHES,
            "crown_T": CROWN_T,
        },
        "by_size": {},
    }
    for size in sorted({int(row["size"]) for row in rows}):
        size_rows = [row for row in rows if int(row["size"]) == size]
        by_method = {str(row["method"]): row for row in size_rows}
        summary["by_size"][str(size)] = {
            method: {
                "hole_psnr": float(row["hole_psnr"]),
                "full_psnr": float(row["full_psnr"]),
                "full_ssim": float(row["full_ssim"]),
                "runtime_sec": row.get("runtime_sec", ""),
            }
            for method, row in by_method.items()
        }
        if "crown" in by_method and "biharmonic" in by_method:
            summary["by_size"][str(size)]["crown_minus_biharmonic_hole_psnr"] = (
                float(by_method["crown"]["hole_psnr"])
                - float(by_method["biharmonic"]["hole_psnr"])
            )
        if "masked_ksvd" in by_method and "biharmonic" in by_method:
            summary["by_size"][str(size)]["masked_ksvd_minus_biharmonic_hole_psnr"] = (
                float(by_method["masked_ksvd"]["hole_psnr"])
                - float(by_method["biharmonic"]["hole_psnr"])
            )
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_results(
    size: int,
    clean: np.ndarray,
    mask: np.ndarray,
    observed: np.ndarray,
    outputs: dict[str, np.ndarray],
    runtimes: dict[str, float],
) -> list[dict[str, object]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / f"gt_{size}.npy", clean)
    np.save(OUTPUT_DIR / f"input_{size}.npy", observed)
    np.save(OUTPUT_DIR / f"mask_{size}.npy", mask)
    for method, arr in outputs.items():
        np.save(OUTPUT_DIR / f"{method}_{size}.npy", arr)

    rows = []
    for method, pred in outputs.items():
        row = {
            "size": size,
            "method": method,
            "K": K,
            "sparsity": SPARSITY,
            "n_train": TRAIN_PATCHES,
            "dict_iters": DICT_ITERS,
            "als_iters": ALS_ITERS,
            "crown_T": CROWN_T if method == "crown" else "",
            "dict_train_sec": runtimes.get("dict_train", ""),
            "runtime_sec": runtimes.get(method, ""),
        }
        row.update(compute_metrics(clean, pred, mask))
        rows.append(row)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    panels = [
        ("Input", observed),
        (
            f"Biharmonic\n{rows[0]['hole_psnr']:.2f} dB hole",
            outputs["biharmonic"],
        ),
        (
            f"Masked K-SVD\n{rows[1]['hole_psnr']:.2f} dB hole",
            outputs["masked_ksvd"],
        ),
        (
            f"CROWN\n{rows[2]['hole_psnr']:.2f} dB hole",
            outputs["crown"],
        ),
        ("GT", clean),
    ]
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.contour(mask == 0, levels=[0.5], colors="tab:red", linewidths=0.5)
        ax.set_title(f"{title}\n{size}x{size}" if title in {"Input", "GT"} else title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"comparison_{size}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return rows


def main() -> None:
    sizes = apply_args(parse_args())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"[big-image] sizes={sizes}, K={K}, s={SPARSITY}, n_train={TRAIN_PATCHES}, "
        f"dict_iters={DICT_ITERS}, crown_T={CROWN_T}"
    )
    print(f"[big-image] writing to {OUTPUT_DIR}")

    all_rows = []
    for size in sizes:
        print(f"[big-image] size {size}x{size}")
        img = sk_resize(brick_512, (size, size), anti_aliasing=True, preserve_range=False)
        mask = make_square_hole(size, hole_size=size // 4)
        y = img * mask

        t0 = time.time()
        u_bh = biharmonic_init(y, mask)
        bh_time = time.time() - t0

        t0 = time.time()
        D, _ = train_dictionary(
            y,
            mask,
            n_train=TRAIN_PATCHES,
            n_iter=DICT_ITERS,
            K=K,
            sparsity=min(SPARSITY, K),
            als_iters=ALS_ITERS,
            seed=SEED,
            verbose=False,
        )
        dict_time = time.time() - t0

        t0 = time.time()
        u_ksvd = run_masked_ksvd_image(img, mask, D)
        ksvd_sparse_time = time.time() - t0

        t0 = time.time()
        out = run_crown_inpaint(
            y,
            mask,
            D,
            T=CROWN_T,
            sparsity=min(SPARSITY, K),
            verbose=False,
        )
        u_crown = hard_project(out["u"], img, mask)
        crown_loop_time = time.time() - t0

        rows = save_results(
            size,
            img,
            mask,
            y,
            {
                "biharmonic": u_bh,
                "masked_ksvd": u_ksvd,
                "crown": u_crown,
            },
            {
                "biharmonic": bh_time,
                "dict_train": dict_time,
                "masked_ksvd": dict_time + ksvd_sparse_time,
                "crown": dict_time + crown_loop_time,
            },
        )
        all_rows.extend(rows)
        print(
            f"  trained shared D in {dict_time:.1f}s; "
            f"K-SVD sparse pass {ksvd_sparse_time:.1f}s; "
            f"CROWN loop {crown_loop_time:.1f}s"
        )

    write_results_csv(all_rows)
    write_summary(all_rows)


if __name__ == "__main__":
    main()
