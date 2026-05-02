"""Small overnight experiment on larger images.

This script compares biharmonic initialization, masked K-SVD, and CROWN-
Inpaint on resized versions of the same reference image.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage import data as skdata
from skimage.transform import resize as sk_resize

from crown.run import run_crown_inpaint, train_dictionary
from crown.smooth import biharmonic_init
from crown.utils import extract_patches_with_mask, reconstruct_image_from_codes
from custom_masks import square_hole_mask
from masked_ksvd import masked_ksvd, masked_omp


OUTPUT_DIR = PROJECT_ROOT / "final_tests" / "results" / "big_image_try"
SEED = 42
K = 256
SPARSITY = 8
DICT_ITERS = 15
ALS_ITERS = 10
TRAIN_PATCHES = 3000


brick_512 = skdata.brick().astype(np.float64) / 255.0


def make_square_hole(size: int, hole_size: int | None = None) -> np.ndarray:
    if hole_size is None:
        hole_size = size // 4
    return square_hole_mask((size, size), square_size=hole_size)


def hard_project(pred: np.ndarray, clean: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.clip(np.where(mask == 1, clean, pred), 0.0, 1.0)


def run_masked_ksvd_image(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    y = img * mask
    _, X_nan, M_all, _ = extract_patches_with_mask(y, mask)

    rng = np.random.default_rng(SEED)
    idx = rng.choice(X_nan.shape[1], size=min(TRAIN_PATCHES, X_nan.shape[1]), replace=False)
    X_train = X_nan[:, idx]
    M_train = M_all[:, idx]

    D, _, _ = masked_ksvd(
        X_train,
        M_train,
        K=K,
        s=min(SPARSITY, K),
        n_iter=DICT_ITERS,
        als_iters=ALS_ITERS,
        seed=SEED,
        label="Big image masked K-SVD",
    )

    alpha = np.zeros((D.shape[1], X_nan.shape[1]), dtype=np.float64)
    s_eff = min(SPARSITY, D.shape[1])
    for i in range(X_nan.shape[1]):
        alpha[:, i] = masked_omp(X_nan[:, i], D, M_all[:, i], s_eff)

    pred = reconstruct_image_from_codes(D, alpha, img.shape)
    return hard_project(pred, img, mask)


def save_results(size: int, u_bh: np.ndarray, u_ksvd: np.ndarray, u_crown: np.ndarray) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / f"biharmonic_{size}.npy", u_bh)
    np.save(OUTPUT_DIR / f"masked_ksvd_{size}.npy", u_ksvd)
    np.save(OUTPUT_DIR / f"crown_{size}.npy", u_crown)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(
        axes,
        [u_bh, u_ksvd, u_crown],
        ["Biharmonic", "Masked K-SVD", "CROWN"],
    ):
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"{title} ({size}x{size})")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"comparison_{size}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for size in [256, 512]:
        img = sk_resize(brick_512, (size, size), anti_aliasing=True, preserve_range=False)
        mask = make_square_hole(size, hole_size=size // 4)
        y = img * mask

        u_bh = biharmonic_init(y, mask)
        u_ksvd = run_masked_ksvd_image(img, mask)

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
        out = run_crown_inpaint(y, mask, D, T=5, sparsity=min(SPARSITY, K), verbose=False)
        u_crown = hard_project(out["u"], img, mask)

        save_results(size, u_bh, u_ksvd, u_crown)


if __name__ == "__main__":
    main()
