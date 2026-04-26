#!/usr/bin/env python3
"""
Experiment sweep driver for CS754 inpainting comparisons.

This script is intentionally standalone: it does not call the existing
experiment scripts' main() functions, and it does not modify algorithm files.
It reuses the repository's algorithm primitives and writes reproducible CSV,
JSON, and plot outputs for paper-ready comparisons.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import itertools
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


# Matplotlib is imported by some existing project modules at import time.
# Point it at a writable cache before any such import happens.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/cs754_proj_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/cs754_proj_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from skimage import data as skdata
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim_fn
from skimage.restoration import inpaint_biharmonic
from skimage.transform import resize as sk_resize
from skimage.util import img_as_float

from custom_masks import generate_mask
from masked_ksvd import baseline_ksvd, masked_ksvd, masked_omp
from crown.utils import (
    PATCH_SIZE,
    extract_patches_with_mask,
    reconstruct_image_from_codes,
)
import crown.run as crown_run
import inpainting_multiscale_masked_ksvd as multiscale


SCRIPT_VERSION = "1.0"
IMAGE_SIZE = (128, 128)
PATCH_DIM = PATCH_SIZE[0] * PATCH_SIZE[1]

ALL_METHODS = (
    "biharmonic",
    "zero_ksvd",
    "masked_ksvd",
    "multiscale_ksvd",
    "crown_full",
    "crown_no_nonlocal",
    "crown_sparse_only",
)

BUILTIN_IMAGES = ("brick", "camera", "coins", "moon")


@dataclass(frozen=True)
class MaskSpec:
    name: str
    mode: str
    missing_frac: float = 0.30
    square_size: int = 20
    square_center: tuple[int, int] | None = (64, 64)
    scratch_count: int = 3
    scratch_thickness: int = 4
    overlay_text: str = "SAMPLE"
    text_scale: float = 0.24
    text_angle: float = -90.0
    text_stroke: int = 1

    def kwargs(self, seed: int) -> dict[str, Any]:
        return {
            "image_shape": IMAGE_SIZE,
            "mode": self.mode,
            "missing_frac": self.missing_frac,
            "seed": seed,
            "square_size": self.square_size,
            "square_center": self.square_center,
            "scratch_count": self.scratch_count,
            "scratch_thickness": self.scratch_thickness,
            "overlay_text": self.overlay_text,
            "text_scale": self.text_scale,
            "text_angle": self.text_angle,
            "text_stroke": self.text_stroke,
        }


@dataclass(frozen=True)
class RunSpec:
    suite: str
    method: str
    image: str
    mask: str
    seed: int
    K: int
    sparsity: int
    n_train: int
    dict_iters: int
    als_iters: int
    crown_T: int
    K_s: int

    @property
    def run_id(self) -> str:
        parts = [
            self.suite,
            self.method,
            self.image,
            self.mask,
            f"seed{self.seed}",
            f"K{self.K}",
            f"s{self.sparsity}",
            f"nt{self.n_train}",
            f"di{self.dict_iters}",
            f"als{self.als_iters}",
            f"T{self.crown_T}",
            f"Ks{self.K_s}",
        ]
        readable = "__".join(parts)
        if len(readable) <= 180:
            return readable
        digest = hashlib.sha1(readable.encode("utf-8")).hexdigest()[:12]
        return "__".join(parts[:5] + [digest])


MASK_SPECS: dict[str, MaskSpec] = {
    "random_p10": MaskSpec("random_p10", "random", missing_frac=0.10),
    "random_p30": MaskSpec("random_p30", "random", missing_frac=0.30),
    "random_p50": MaskSpec("random_p50", "random", missing_frac=0.50),
    "square_s16": MaskSpec("square_s16", "face-square", square_size=16),
    "square_s24": MaskSpec("square_s24", "face-square", square_size=24),
    "square_s32": MaskSpec("square_s32", "face-square", square_size=32),
    "scratches_n3_t4": MaskSpec(
        "scratches_n3_t4", "scratches", scratch_count=3, scratch_thickness=4
    ),
    "scratches_n4_t8": MaskSpec(
        "scratches_n4_t8", "scratches", scratch_count=4, scratch_thickness=8
    ),
    "square_s24_scratches_n3_t4": MaskSpec(
        "square_s24_scratches_n3_t4",
        "face-square+scratches",
        square_size=24,
        scratch_count=3,
        scratch_thickness=4,
    ),
    "text_sample": MaskSpec("text_sample", "text-overlay", overlay_text="SAMPLE"),
}

LARGE_MASKS = tuple(MASK_SPECS.keys())
SENSITIVITY_IMAGES = ("brick", "camera")
SENSITIVITY_MASKS = ("random_p30", "square_s24", "scratches_n4_t8")

METRIC_KEYS = (
    "full_psnr",
    "full_ssim",
    "full_mse",
    "full_mae",
    "hole_psnr",
    "hole_mse",
    "hole_mae",
    "boundary_mae",
    "missing_frac_actual",
    "runtime_sec",
    "train_rmse_final",
    "observed_max_abs_err",
)

RUN_FIELDNAMES = (
    "run_id",
    "suite",
    "method",
    "image",
    "mask",
    "seed",
    "K",
    "sparsity",
    "n_train",
    "dict_iters",
    "als_iters",
    "crown_T",
    "K_s",
    "mask_mode",
    "missing_frac_target",
    "square_size",
    "scratch_count",
    "scratch_thickness",
    "status",
    *METRIC_KEYS,
    "train_rmse_initial",
    "train_rmse_L3_final",
    "train_rmse_L2_final",
    "train_rmse_L1_final",
    "crown_iters_completed",
    "avg_outer_time_sec",
    "error",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def parse_int_list(value: str | None) -> list[int] | None:
    items = parse_csv_list(value)
    if items is None:
        return None
    return [int(item) for item in items]


def select_known(
    requested: list[str] | None,
    default: Iterable[str],
    allowed: Iterable[str],
    label: str,
) -> list[str]:
    allowed_set = set(allowed)
    if requested is None:
        return list(default)
    if len(requested) == 1 and requested[0] == "all":
        return list(allowed)
    unknown = [item for item in requested if item not in allowed_set]
    if unknown:
        raise ValueError(
            f"unknown {label}: {unknown}. Allowed values: {sorted(allowed_set)}"
        )
    return requested


def parse_methods(value: str) -> list[str]:
    items = parse_csv_list(value) or ["all"]
    return select_known(items, ALL_METHODS, ALL_METHODS, "methods")


def load_builtin_image(name: str) -> np.ndarray:
    if name == "brick":
        raw = skdata.brick()
    elif name == "camera":
        raw = skdata.camera()
    elif name == "coins":
        raw = skdata.coins()
    elif name == "moon":
        raw = skdata.moon()
    else:
        raise ValueError(f"unsupported image '{name}'")

    img = img_as_float(raw)
    if img.ndim == 3:
        img = rgb2gray(img)
    img = sk_resize(
        img.astype(np.float64),
        IMAGE_SIZE,
        anti_aliasing=True,
        preserve_range=True,
    )
    return np.clip(img.astype(np.float64), 0.0, 1.0)


def build_mask(mask_name: str, seed: int) -> np.ndarray:
    spec = MASK_SPECS[mask_name]
    mask = generate_mask(**spec.kwargs(seed))
    if mask.shape != IMAGE_SIZE:
        raise ValueError(f"mask {mask_name} shape {mask.shape} != {IMAGE_SIZE}")
    return (mask > 0.5).astype(np.float64)


def psnr_from_mse(mse: float) -> float:
    return float(10.0 * np.log10(1.0 / max(float(mse), 1e-12)))


def compute_metrics(img_true: np.ndarray, img_pred: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    pred = np.clip(img_pred, 0.0, 1.0)
    err = img_true - pred
    full_mse = float(np.mean(err * err))
    full_mae = float(np.mean(np.abs(err)))
    hole = mask == 0
    obs = mask == 1

    if hole.any():
        hole_err = err[hole]
        hole_mse = float(np.mean(hole_err * hole_err))
        hole_mae = float(np.mean(np.abs(hole_err)))
        ring = binary_dilation(hole, iterations=4) & ~binary_erosion(hole, iterations=4)
        boundary_mae = float(np.mean(np.abs(err[ring]))) if ring.any() else 0.0
    else:
        hole_mse = 0.0
        hole_mae = 0.0
        boundary_mae = 0.0

    observed_max_abs_err = float(np.max(np.abs(err[obs]))) if obs.any() else 0.0
    return {
        "full_psnr": psnr_from_mse(full_mse),
        "full_ssim": float(ssim_fn(img_true, pred, data_range=1.0)),
        "full_mse": full_mse,
        "full_mae": full_mae,
        "hole_psnr": psnr_from_mse(hole_mse),
        "hole_mse": hole_mse,
        "hole_mae": hole_mae,
        "boundary_mae": boundary_mae,
        "missing_frac_actual": float(hole.mean()),
        "observed_max_abs_err": observed_max_abs_err,
    }


def hard_project(pred: np.ndarray, clean: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.clip(np.where(mask == 1, clean, pred), 0.0, 1.0)


def validate_prediction(pred: np.ndarray, clean: np.ndarray, mask: np.ndarray, row: dict[str, Any]) -> None:
    if pred.shape != IMAGE_SIZE:
        raise RuntimeError(f"prediction shape {pred.shape} != {IMAGE_SIZE}")
    if not np.isfinite(pred).all():
        raise RuntimeError("prediction contains NaN or inf")
    if float(pred.min()) < -1e-8 or float(pred.max()) > 1.0 + 1e-8:
        raise RuntimeError(
            f"prediction outside [0,1]: min={pred.min():.6g}, max={pred.max():.6g}"
        )
    obs_err = float(np.max(np.abs(pred[mask == 1] - clean[mask == 1]))) if (mask == 1).any() else 0.0
    if obs_err > 1e-8:
        raise RuntimeError(f"observed pixels not preserved: max error {obs_err:.3e}")
    for key in METRIC_KEYS:
        if key in row and row[key] not in ("", None):
            value = float(row[key])
            if not math.isfinite(value):
                raise RuntimeError(f"metric {key} is not finite: {value}")


@contextlib.contextmanager
def maybe_silence(verbose: bool):
    if verbose:
        yield
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            yield


def sample_training_columns(
    X: np.ndarray, M: np.ndarray, n_train: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_cols = X.shape[1]
    idx = rng.choice(n_cols, size=min(int(n_train), n_cols), replace=False)
    return X[:, idx], M[:, idx]


def sparse_code_all(
    X: np.ndarray,
    M: np.ndarray,
    D: np.ndarray,
    sparsity: int,
    use_mask: bool,
) -> np.ndarray:
    n, n_patches = X.shape
    K = D.shape[1]
    alpha = np.zeros((K, n_patches), dtype=np.float64)
    all_ones = np.ones(n, dtype=np.float64)
    s_eff = min(int(sparsity), K)
    for i in range(n_patches):
        if use_mask:
            alpha[:, i] = masked_omp(X[:, i], D, M[:, i], s_eff)
        else:
            x_zero = np.nan_to_num(X[:, i], nan=0.0)
            alpha[:, i] = masked_omp(x_zero, D, all_ones, s_eff)
    return alpha


def run_biharmonic(clean: np.ndarray, mask: np.ndarray, spec: RunSpec, verbose: bool) -> tuple[np.ndarray, dict[str, Any]]:
    y = clean * mask
    pred = inpaint_biharmonic(y, (mask == 0).astype(bool))
    pred = hard_project(pred, clean, mask)
    return pred, {}


def run_zero_ksvd_image(clean: np.ndarray, mask: np.ndarray, spec: RunSpec, verbose: bool) -> tuple[np.ndarray, dict[str, Any]]:
    y = clean * mask
    _, X_nan, M_all, _ = extract_patches_with_mask(y, mask)
    X_train, M_train = sample_training_columns(X_nan, M_all, spec.n_train, spec.seed)
    with maybe_silence(verbose):
        D, _, train_err = baseline_ksvd(
            X_train,
            M_train,
            K=spec.K,
            s=min(spec.sparsity, spec.K),
            n_iter=spec.dict_iters,
            als_iters=spec.als_iters,
            seed=spec.seed,
        )
    alpha = sparse_code_all(X_nan, M_all, D, spec.sparsity, use_mask=False)
    pred = reconstruct_image_from_codes(D, alpha, IMAGE_SIZE)
    pred = hard_project(pred, clean, mask)
    return pred, {
        "train_rmse_initial": float(train_err[0]) if len(train_err) else "",
        "train_rmse_final": float(train_err[-1]) if len(train_err) else "",
    }


def run_masked_ksvd_image(clean: np.ndarray, mask: np.ndarray, spec: RunSpec, verbose: bool) -> tuple[np.ndarray, dict[str, Any]]:
    y = clean * mask
    _, X_nan, M_all, _ = extract_patches_with_mask(y, mask)
    X_train, M_train = sample_training_columns(X_nan, M_all, spec.n_train, spec.seed)
    with maybe_silence(verbose):
        D, _, train_err = masked_ksvd(
            X_train,
            M_train,
            K=spec.K,
            s=min(spec.sparsity, spec.K),
            n_iter=spec.dict_iters,
            als_iters=spec.als_iters,
            seed=spec.seed,
            label=f"masked_ksvd:{spec.run_id}",
        )
    alpha = sparse_code_all(X_nan, M_all, D, spec.sparsity, use_mask=True)
    pred = reconstruct_image_from_codes(D, alpha, IMAGE_SIZE)
    pred = hard_project(pred, clean, mask)
    return pred, {
        "train_rmse_initial": float(train_err[0]) if len(train_err) else "",
        "train_rmse_final": float(train_err[-1]) if len(train_err) else "",
    }


@contextlib.contextmanager
def patched_multiscale_globals(spec: RunSpec):
    old = {
        "N_ATOMS": multiscale.N_ATOMS,
        "SPARSITY": multiscale.SPARSITY,
        "ALS_ITERS": multiscale.ALS_ITERS,
        "SEED": multiscale.SEED,
        "LEVEL_CFG": multiscale.LEVEL_CFG,
    }
    n_train = int(spec.n_train)
    multiscale.N_ATOMS = int(spec.K)
    multiscale.SPARSITY = min(int(spec.sparsity), int(spec.K))
    multiscale.ALS_ITERS = int(spec.als_iters)
    multiscale.SEED = int(spec.seed)
    multiscale.LEVEL_CFG = {
        3: {"size": (32, 32), "n_train": max(25, n_train // 6), "n_iter": int(spec.dict_iters)},
        2: {"size": (64, 64), "n_train": max(50, n_train // 2), "n_iter": int(spec.dict_iters)},
        1: {"size": (128, 128), "n_train": n_train, "n_iter": int(spec.dict_iters)},
    }
    try:
        yield
    finally:
        multiscale.N_ATOMS = old["N_ATOMS"]
        multiscale.SPARSITY = old["SPARSITY"]
        multiscale.ALS_ITERS = old["ALS_ITERS"]
        multiscale.SEED = old["SEED"]
        multiscale.LEVEL_CFG = old["LEVEL_CFG"]


def run_multiscale_ksvd_image(clean: np.ndarray, mask: np.ndarray, spec: RunSpec, verbose: bool) -> tuple[np.ndarray, dict[str, Any]]:
    y = clean * mask
    with patched_multiscale_globals(spec):
        with maybe_silence(verbose):
            results = multiscale.run_pyramid(clean, y, mask)
    pred = hard_project(results["recon_128"], clean, mask)
    err_l3 = results.get("err_L3", np.array([]))
    err_l2 = results.get("err_L2", np.array([]))
    err_l1 = results.get("err_L1", np.array([]))
    return pred, {
        "train_rmse_L3_final": float(err_l3[-1]) if len(err_l3) else "",
        "train_rmse_L2_final": float(err_l2[-1]) if len(err_l2) else "",
        "train_rmse_L1_final": float(err_l1[-1]) if len(err_l1) else "",
        "train_rmse_final": float(err_l1[-1]) if len(err_l1) else "",
    }


@contextlib.contextmanager
def force_crown_sparse_only(enabled: bool):
    if not enabled:
        yield
        return
    original = crown_run.compute_regime_map

    def _all_sparse_regime(y: np.ndarray, M: np.ndarray, *args, **kwargs) -> np.ndarray:
        return np.ones_like(y, dtype=np.float64)

    crown_run.compute_regime_map = _all_sparse_regime
    try:
        yield
    finally:
        crown_run.compute_regime_map = original


def run_crown_variant(clean: np.ndarray, mask: np.ndarray, spec: RunSpec, verbose: bool) -> tuple[np.ndarray, dict[str, Any]]:
    y = clean * mask
    nonlocal_enabled = spec.method != "crown_no_nonlocal"
    sparse_only = spec.method == "crown_sparse_only"
    with maybe_silence(verbose):
        D, train_err = crown_run.train_dictionary(
            y,
            mask,
            n_train=spec.n_train,
            n_iter=spec.dict_iters,
            K=spec.K,
            sparsity=min(spec.sparsity, spec.K),
            als_iters=spec.als_iters,
            seed=spec.seed,
            verbose=verbose,
        )
        with force_crown_sparse_only(sparse_only):
            out = crown_run.run_crown_inpaint(
                y=y,
                M=mask,
                D=D,
                T=spec.crown_T,
                K_s=spec.K_s,
                sparsity=min(spec.sparsity, spec.K),
                nonlocal_enabled=nonlocal_enabled,
                manifold_enabled=False,
                manifold_seed=spec.seed,
                img_clean=clean,
                verbose=verbose,
            )
    pred = hard_project(out["u"], clean, mask)
    history = out.get("history", [])
    timings = out.get("timings", [])
    return pred, {
        "train_rmse_initial": float(train_err[0]) if len(train_err) else "",
        "train_rmse_final": float(train_err[-1]) if len(train_err) else "",
        "crown_iters_completed": max(0, len(history) - 1),
        "avg_outer_time_sec": float(np.mean(timings)) if timings else "",
    }


RUNNERS = {
    "biharmonic": run_biharmonic,
    "zero_ksvd": run_zero_ksvd_image,
    "masked_ksvd": run_masked_ksvd_image,
    "multiscale_ksvd": run_multiscale_ksvd_image,
    "crown_full": run_crown_variant,
    "crown_no_nonlocal": run_crown_variant,
    "crown_sparse_only": run_crown_variant,
}


def run_one(spec: RunSpec, image_cache: dict[str, np.ndarray], mask_cache: dict[tuple[str, int], np.ndarray], verbose: bool) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    clean = image_cache.setdefault(spec.image, load_builtin_image(spec.image))
    mask_key = (spec.mask, spec.seed)
    if mask_key not in mask_cache:
        mask_cache[mask_key] = build_mask(spec.mask, spec.seed)
    mask = mask_cache[mask_key]
    mask_spec = MASK_SPECS[spec.mask]

    row: dict[str, Any] = {
        "run_id": spec.run_id,
        **asdict(spec),
        "mask_mode": mask_spec.mode,
        "missing_frac_target": mask_spec.missing_frac if mask_spec.mode == "random" else "",
        "square_size": mask_spec.square_size,
        "scratch_count": mask_spec.scratch_count,
        "scratch_thickness": mask_spec.scratch_thickness,
        "status": "completed",
        "error": "",
    }

    t0 = time.time()
    pred, extra = RUNNERS[spec.method](clean, mask, spec, verbose=verbose)
    row["runtime_sec"] = float(time.time() - t0)
    row.update(extra)
    row.update(compute_metrics(clean, pred, mask))
    validate_prediction(pred, clean, mask, row)
    return row, clean, mask, pred


def default_profile_values(profile: str) -> dict[str, Any]:
    if profile == "smoke":
        return {
            "images": ["brick"],
            "masks": ["random_p30"],
            "seeds": [42],
            "K": [32],
            "sparsity": [3],
            "dict_iters": [1],
            "n_train": [100],
            "als_iters": 2,
            "crown_T": [1],
            "K_s": 1,
        }
    if profile == "large":
        return {
            "images": list(BUILTIN_IMAGES),
            "masks": list(LARGE_MASKS),
            "seeds": [42, 69, 123],
            "K": [256],
            "sparsity": [8],
            "dict_iters": [25],
            "n_train": [3000],
            "als_iters": 10,
            "crown_T": [5],
            "K_s": 10,
        }
    raise ValueError(f"unsupported profile {profile}")


def enumerate_runs(args: argparse.Namespace) -> tuple[list[RunSpec], dict[str, Any]]:
    defaults = default_profile_values(args.profile)

    methods = parse_methods(args.methods)
    images = select_known(
        parse_csv_list(args.images),
        defaults["images"],
        BUILTIN_IMAGES,
        "images",
    )
    masks = select_known(
        parse_csv_list(args.masks),
        defaults["masks"],
        MASK_SPECS.keys(),
        "masks",
    )

    seeds = parse_int_list(args.seeds) or defaults["seeds"]
    K_values = parse_int_list(args.K_values) or defaults["K"]
    sparsity_values = parse_int_list(args.sparsity) or defaults["sparsity"]
    dict_iters_values = parse_int_list(args.dict_iters) or defaults["dict_iters"]
    n_train_values = parse_int_list(args.n_train) or defaults["n_train"]
    crown_T_values = parse_int_list(args.crown_T) or defaults["crown_T"]

    for name, values in [
        ("seeds", seeds),
        ("K", K_values),
        ("sparsity", sparsity_values),
        ("dict-iters", dict_iters_values),
        ("n-train", n_train_values),
        ("crown-T", crown_T_values),
    ]:
        if not values:
            raise ValueError(f"{name} cannot be empty")
        if any(int(v) < 1 for v in values):
            raise ValueError(f"{name} values must be >= 1: {values}")

    base_grid = list(
        itertools.product(
            images,
            masks,
            seeds,
            K_values,
            sparsity_values,
            n_train_values,
            dict_iters_values,
            crown_T_values,
            methods,
        )
    )
    runs: list[RunSpec] = [
        RunSpec(
            suite="paper",
            method=method,
            image=image,
            mask=mask,
            seed=int(seed),
            K=int(K),
            sparsity=int(sparsity),
            n_train=int(n_train),
            dict_iters=int(dict_iters),
            als_iters=int(defaults["als_iters"]),
            crown_T=int(crown_T),
            K_s=int(defaults["K_s"]),
        )
        for image, mask, seed, K, sparsity, n_train, dict_iters, crown_T, method in base_grid
    ]

    if args.profile == "large" and not args.no_sensitivity:
        sens_images = [item for item in SENSITIVITY_IMAGES if item in images]
        sens_masks = [item for item in SENSITIVITY_MASKS if item in masks]
        sens_methods = [m for m in methods if m != "biharmonic"]
        sens_K = parse_int_list(args.K_values) or [128, 256, 384]
        sens_s = parse_int_list(args.sparsity) or [4, 8, 12]
        sens_dict = parse_int_list(args.dict_iters) or [10, 25, 40]
        sens_n_train = parse_int_list(args.n_train) or [1000, 3000]
        sens_T = parse_int_list(args.crown_T) or [1, 3, 5, 8]

        for image, mask, seed, K, sparsity, n_train, dict_iters, method in itertools.product(
            sens_images,
            sens_masks,
            seeds,
            sens_K,
            sens_s,
            sens_n_train,
            sens_dict,
            sens_methods,
        ):
            T_values = sens_T if method.startswith("crown_") else [defaults["crown_T"][0]]
            for crown_T in T_values:
                runs.append(
                    RunSpec(
                        suite="sensitivity",
                        method=method,
                        image=image,
                        mask=mask,
                        seed=int(seed),
                        K=int(K),
                        sparsity=int(sparsity),
                        n_train=int(n_train),
                        dict_iters=int(dict_iters),
                        als_iters=int(defaults["als_iters"]),
                        crown_T=int(crown_T),
                        K_s=int(defaults["K_s"]),
                    )
                )

    seen: set[str] = set()
    deduped: list[RunSpec] = []
    for run in runs:
        if run.run_id in seen:
            continue
        seen.add(run.run_id)
        deduped.append(run)

    config = {
        "profile": args.profile,
        "methods": methods,
        "images": images,
        "masks": masks,
        "seeds": seeds,
        "K": K_values,
        "sparsity": sparsity_values,
        "dict_iters": dict_iters_values,
        "n_train": n_train_values,
        "als_iters": defaults["als_iters"],
        "crown_T": crown_T_values,
        "K_s": defaults["K_s"],
        "include_sensitivity": args.profile == "large" and not args.no_sensitivity,
    }
    return deduped, config


def read_completed_run_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {
            row["run_id"]
            for row in reader
            if row.get("run_id") and row.get("status") == "completed"
        }


def append_run_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in RUN_FIELDNAMES})


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    group_keys = [
        "suite",
        "method",
        "image",
        "mask",
        "K",
        "sparsity",
        "n_train",
        "dict_iters",
        "crown_T",
    ]
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        if row.get("status") != "completed":
            continue
        key = tuple(row.get(k, "") for k in group_keys)
        grouped.setdefault(key, []).append(row)

    agg: list[dict[str, Any]] = []
    for key, group in grouped.items():
        out: dict[str, Any] = {k: v for k, v in zip(group_keys, key)}
        out["num_runs"] = len(group)
        seeds = sorted({row.get("seed", "") for row in group})
        out["seeds"] = ",".join(seeds)
        for metric in METRIC_KEYS:
            values = []
            for row in group:
                try:
                    value = float(row.get(metric, ""))
                except ValueError:
                    continue
                if math.isfinite(value):
                    values.append(value)
            if values:
                arr = np.array(values, dtype=float)
                out[f"{metric}_mean"] = float(arr.mean())
                out[f"{metric}_std"] = float(arr.std(ddof=0))
        agg.append(out)

    agg.sort(
        key=lambda row: (
            row.get("suite", ""),
            row.get("method", ""),
            row.get("image", ""),
            row.get("mask", ""),
            str(row.get("K", "")),
        )
    )
    return agg


def mean_by_key(rows: list[dict[str, str]], key: str, metric: str, suite: str | None = None) -> tuple[list[str], np.ndarray, np.ndarray]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        if row.get("status") != "completed":
            continue
        if suite is not None and row.get("suite") != suite:
            continue
        try:
            value = float(row.get(metric, ""))
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        buckets.setdefault(row.get(key, ""), []).append(value)
    labels = sorted(label for label in buckets if label)
    means = np.array([np.mean(buckets[label]) for label in labels], dtype=float)
    stds = np.array([np.std(buckets[label]) for label in labels], dtype=float)
    return labels, means, stds


def plot_bar_summary(rows: list[dict[str, str]], plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    for key, filename, title in [
        ("method", "method_summary_hole_psnr.png", "Hole PSNR by Method"),
        ("mask", "mask_summary_hole_psnr.png", "Hole PSNR by Mask"),
    ]:
        labels, means, stds = mean_by_key(rows, key=key, metric="hole_psnr", suite="paper")
        if not labels:
            continue
        fig_w = max(9.0, 0.6 * len(labels) + 3.0)
        fig, ax = plt.subplots(figsize=(fig_w, 5.5))
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=3, color="#2563EB", alpha=0.88)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Hole PSNR (dB)")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / filename, dpi=150)
        plt.close(fig)


def plot_sensitivity(rows: list[dict[str, str]], plots_dir: Path) -> None:
    sens = [row for row in rows if row.get("suite") == "sensitivity" and row.get("status") == "completed"]
    if not sens:
        return
    scored = []
    for row in sens:
        try:
            score = float(row["hole_psnr"])
        except (KeyError, ValueError):
            continue
        if math.isfinite(score):
            label = (
                f"{row['method']} {row['image']} {row['mask']} "
                f"K={row['K']} s={row['sparsity']} it={row['dict_iters']} "
                f"nt={row['n_train']} T={row['crown_T']}"
            )
            scored.append((score, label))
    if not scored:
        return
    scored.sort(reverse=True, key=lambda item: item[0])
    top = scored[: min(15, len(scored))]
    labels = [item[1] for item in top]
    values = np.array([item[0] for item in top], dtype=float)
    y = np.arange(len(top))
    fig_h = max(6.0, 0.48 * len(top) + 2.0)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.barh(y, values, color="#16A34A", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Hole PSNR (dB)")
    ax.set_title("Top Sensitivity Configurations")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "hyperparameter_sensitivity_top.png", dpi=150)
    plt.close(fig)


def plot_visual_comparison(
    plots_dir: Path,
    image_name: str,
    mask_name: str,
    seed: int,
    clean: np.ndarray,
    mask: np.ndarray,
    preds: dict[str, np.ndarray],
) -> None:
    if not preds:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    items = [("original", clean), ("corrupted", clean * mask)] + sorted(preds.items())
    n = len(items)
    cols = min(4, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.0 * rows))
    axes_arr = np.array(axes, dtype=object).reshape(-1)
    for ax, (label, img) in zip(axes_arr, items):
        ax.imshow(np.clip(img, 0.0, 1.0), cmap="gray", vmin=0, vmax=1)
        if label == "corrupted":
            overlay = np.zeros((*mask.shape, 4), dtype=float)
            overlay[..., 0] = 1.0
            overlay[..., 3] = (mask == 0) * 0.45
            ax.imshow(overlay)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes_arr[len(items) :]:
        ax.axis("off")
    fig.suptitle(f"Selected Comparison: {image_name}, {mask_name}, seed={seed}")
    fig.tight_layout()
    fig.savefig(plots_dir / "selected_visual_comparison.png", dpi=150)
    plt.close(fig)


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def print_dry_run(specs: list[RunSpec], config: dict[str, Any], total_specs: int) -> None:
    print("Dry run: no algorithms executed.")
    print(f"Profile          : {config['profile']}")
    print(f"Methods          : {', '.join(config['methods'])}")
    print(f"Images           : {', '.join(config['images'])}")
    print(f"Masks            : {', '.join(config['masks'])}")
    print(f"Total run specs  : {total_specs}")
    if len(specs) != total_specs:
        print(f"Listed specs     : {len(specs)}")
    print("First run IDs:")
    for spec in specs[: min(12, len(specs))]:
        print(f"  {spec.run_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run reproducible inpainting experiment sweeps."
    )
    parser.add_argument("--profile", choices=("smoke", "large"), default="large")
    parser.add_argument("--out-dir", default="outputs/experiment_sweeps")
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma list or 'all'. Allowed: " + ",".join(ALL_METHODS),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--images", default=None, help="Comma list or 'all'.")
    parser.add_argument("--masks", default=None, help="Comma list or 'all'.")
    parser.add_argument("--seeds", default=None, help="Comma-separated integer seeds.")
    parser.add_argument("--K", dest="K_values", default=None, help="Comma-separated atom counts.")
    parser.add_argument("--sparsity", default=None, help="Comma-separated sparsity values.")
    parser.add_argument("--dict-iters", default=None, help="Comma-separated dictionary iteration counts.")
    parser.add_argument("--n-train", default=None, help="Comma-separated training patch counts.")
    parser.add_argument("--crown-T", default=None, help="Comma-separated CROWN outer iteration counts.")
    parser.add_argument(
        "--no-sensitivity",
        action="store_true",
        help="For large profile, skip the representative hyperparameter grid.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show inner algorithm logs from K-SVD and CROWN.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_runs is not None and args.max_runs < 1:
        parser.error("--max-runs must be >= 1")

    out_dir = Path(args.out_dir)
    runs_csv = out_dir / "runs.csv"
    aggregate_csv = out_dir / "aggregate.csv"
    manifest_json = out_dir / "manifest.json"
    plots_dir = out_dir / "plots"

    all_specs, config = enumerate_runs(args)

    if args.dry_run:
        limited_specs = all_specs[: args.max_runs] if args.max_runs else all_specs
        print_dry_run(limited_specs, config, total_specs=len(all_specs))
        write_manifest(
            manifest_json,
            {
                "script_version": SCRIPT_VERSION,
                "created_at": utc_now(),
                "dry_run": True,
                "config": config,
                "total_specs": len(all_specs),
                "listed_specs": len(limited_specs),
                "args": vars(args),
                "python": sys.version,
                "platform": platform.platform(),
            },
        )
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume and runs_csv.exists():
        runs_csv.unlink()

    completed = read_completed_run_ids(runs_csv) if args.resume else set()
    pending = [spec for spec in all_specs if spec.run_id not in completed]
    total_pending_before_cap = len(pending)
    if args.max_runs is not None:
        pending = pending[: args.max_runs]

    print("=" * 78)
    print("Experiment sweep")
    print("=" * 78)
    print(f"Profile          : {args.profile}")
    print(f"Output dir       : {out_dir}")
    print(f"Total specs      : {len(all_specs)}")
    print(f"Already complete : {len(completed)}")
    print(f"Pending selected : {len(pending)} / {total_pending_before_cap}")
    print(f"Methods          : {', '.join(config['methods'])}")
    print()

    image_cache: dict[str, np.ndarray] = {}
    mask_cache: dict[tuple[str, int], np.ndarray] = {}
    visual_key: tuple[Any, ...] | None = None
    visual_payload: dict[str, Any] | None = None

    started_at = utc_now()
    n_ok = 0
    n_failed = 0

    for idx, spec in enumerate(pending, start=1):
        print(f"[{idx:04d}/{len(pending):04d}] {spec.run_id}")
        try:
            row, clean, mask, pred = run_one(spec, image_cache, mask_cache, verbose=args.verbose)
            append_run_row(runs_csv, row)
            n_ok += 1

            key = (
                spec.suite,
                spec.image,
                spec.mask,
                spec.seed,
                spec.K,
                spec.sparsity,
                spec.n_train,
                spec.dict_iters,
                spec.crown_T,
            )
            if visual_key is None and spec.suite == "paper":
                visual_key = key
                visual_payload = {
                    "image": spec.image,
                    "mask": spec.mask,
                    "seed": spec.seed,
                    "clean": clean,
                    "mask_array": mask,
                    "preds": {},
                }
            if visual_key == key and visual_payload is not None:
                visual_payload["preds"][spec.method] = pred

            print(
                f"  done in {row['runtime_sec']:.2f}s | "
                f"PSNR={row['full_psnr']:.2f} | "
                f"hole-PSNR={row['hole_psnr']:.2f}"
            )
        except Exception as exc:
            n_failed += 1
            err_row = {
                "run_id": spec.run_id,
                **asdict(spec),
                "status": "failed",
                "error": repr(exc),
            }
            append_run_row(runs_csv, err_row)
            print(f"  FAILED: {exc!r}")
            if args.verbose:
                raise

    rows = read_rows(runs_csv)
    agg_rows = aggregate_rows(rows)
    write_csv(aggregate_csv, agg_rows)
    plot_bar_summary(rows, plots_dir)
    plot_sensitivity(rows, plots_dir)
    if visual_payload is not None:
        plot_visual_comparison(
            plots_dir,
            image_name=visual_payload["image"],
            mask_name=visual_payload["mask"],
            seed=int(visual_payload["seed"]),
            clean=visual_payload["clean"],
            mask=visual_payload["mask_array"],
            preds=visual_payload["preds"],
        )

    write_manifest(
        manifest_json,
        {
            "script_version": SCRIPT_VERSION,
            "created_at": started_at,
            "finished_at": utc_now(),
            "dry_run": False,
            "resume": bool(args.resume),
            "config": config,
            "total_specs": len(all_specs),
            "completed_before_run": len(completed),
            "pending_before_cap": total_pending_before_cap,
            "pending_executed": len(pending),
            "succeeded_this_invocation": n_ok,
            "failed_this_invocation": n_failed,
            "runs_csv": str(runs_csv),
            "aggregate_csv": str(aggregate_csv),
            "plots_dir": str(plots_dir),
            "args": vars(args),
            "python": sys.version,
            "platform": platform.platform(),
        },
    )

    print()
    print("=" * 78)
    print("Sweep complete")
    print("=" * 78)
    print(f"Succeeded this run : {n_ok}")
    print(f"Failed this run    : {n_failed}")
    print(f"Runs CSV           : {runs_csv}")
    print(f"Aggregate CSV      : {aggregate_csv}")
    print(f"Manifest           : {manifest_json}")
    print(f"Plots              : {plots_dir}")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
