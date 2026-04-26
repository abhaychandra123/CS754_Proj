#!/usr/bin/env python3
"""
Parallel experiment sweep driver for CS754 inpainting comparisons.

This wrapper reuses experiment_sweep.py internals and parallelizes independent
run specs across multiple worker processes.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import multiprocessing as mp
import os
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


PARALLEL_SCRIPT_VERSION = "1.0"

# Keep per-worker BLAS thread counts low to avoid CPU oversubscription.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import experiment_sweep as core


_WORKER_IMAGE_CACHE: dict[str, Any] = {}
_WORKER_MASK_CACHE: dict[tuple[str, int], Any] = {}


def _set_worker_thread_env(threads: int) -> None:
    value = str(max(1, int(threads)))
    os.environ["OPENBLAS_NUM_THREADS"] = value
    os.environ["OMP_NUM_THREADS"] = value
    os.environ["MKL_NUM_THREADS"] = value
    os.environ["NUMEXPR_NUM_THREADS"] = value


def _spec_visual_key(spec: core.RunSpec) -> tuple[Any, ...]:
    return (
        spec.suite,
        spec.image,
        spec.mask,
        spec.seed,
        spec.K,
        spec.sparsity,
        spec.n_train,
        spec.dict_iters,
        spec.crown_T,
        spec.als_iters,
        spec.K_s,
    )


def _run_worker(
    spec: core.RunSpec,
    verbose: bool,
    visual_key: tuple[Any, ...] | None,
) -> dict[str, Any]:
    row, clean, mask, pred = core.run_one(
        spec,
        image_cache=_WORKER_IMAGE_CACHE,
        mask_cache=_WORKER_MASK_CACHE,
        verbose=verbose,
    )
    payload: dict[str, Any] = {"row": row}
    if visual_key is not None and _spec_visual_key(spec) == visual_key:
        payload["visual"] = {
            "method": spec.method,
            "image": spec.image,
            "mask": spec.mask,
            "seed": spec.seed,
            "clean": clean,
            "mask_array": mask,
            "pred": pred,
        }
    return payload


def build_parallel_parser() -> argparse.ArgumentParser:
    parser = core.build_parser()
    parser.description = "Run reproducible inpainting experiment sweeps in parallel."
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of worker processes (default: min(8, cpu_count)).",
    )
    parser.add_argument(
        "--worker-threads",
        type=int,
        default=1,
        help="BLAS/OpenMP threads per worker to avoid oversubscription.",
    )
    parser.add_argument(
        "--queue-factor",
        type=int,
        default=2,
        help="How many tasks to queue per worker (default: 2).",
    )
    return parser


def _submit_next(
    executor: cf.ProcessPoolExecutor,
    inflight: dict[cf.Future[dict[str, Any]], core.RunSpec],
    specs_iter,
    args: argparse.Namespace,
    visual_key: tuple[Any, ...] | None,
) -> bool:
    try:
        spec = next(specs_iter)
    except StopIteration:
        return False
    future = executor.submit(_run_worker, spec, args.verbose, visual_key)
    inflight[future] = spec
    return True


def _run_parallel(
    pending: list[core.RunSpec],
    args: argparse.Namespace,
    runs_csv: Path,
    visual_key: tuple[Any, ...] | None,
) -> tuple[int, int, dict[str, Any] | None]:
    n_ok = 0
    n_failed = 0
    visual_payload: dict[str, Any] | None = None

    max_workers = int(args.jobs)
    max_prefetch = max_workers * max(1, int(args.queue_factor))

    specs_iter = iter(pending)
    ctx = mp.get_context("spawn")
    inflight: dict[cf.Future[dict[str, Any]], core.RunSpec] = {}

    with cf.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        while len(inflight) < min(max_prefetch, len(pending)):
            if not _submit_next(executor, inflight, specs_iter, args, visual_key):
                break

        finished = 0
        while inflight:
            done, _ = cf.wait(inflight.keys(), return_when=cf.FIRST_COMPLETED)
            for future in done:
                spec = inflight.pop(future)
                finished += 1
                print(f"[{finished:04d}/{len(pending):04d}] {spec.run_id}")
                try:
                    payload = future.result()
                    row = payload["row"]
                    core.append_run_row(runs_csv, row)
                    n_ok += 1
                    if visual_payload is None and payload.get("visual"):
                        item = payload["visual"]
                        visual_payload = {
                            "image": item["image"],
                            "mask": item["mask"],
                            "seed": item["seed"],
                            "clean": item["clean"],
                            "mask_array": item["mask_array"],
                            "preds": {item["method"]: item["pred"]},
                        }
                    elif visual_payload is not None and payload.get("visual"):
                        item = payload["visual"]
                        visual_payload["preds"][item["method"]] = item["pred"]

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
                    core.append_run_row(runs_csv, err_row)
                    print(f"  FAILED: {exc!r}")
                    if args.verbose:
                        raise

                _submit_next(executor, inflight, specs_iter, args, visual_key)

    return n_ok, n_failed, visual_payload


def _run_serial(
    pending: list[core.RunSpec],
    args: argparse.Namespace,
    runs_csv: Path,
    visual_key: tuple[Any, ...] | None,
) -> tuple[int, int, dict[str, Any] | None]:
    image_cache: dict[str, Any] = {}
    mask_cache: dict[tuple[str, int], Any] = {}
    visual_payload: dict[str, Any] | None = None
    n_ok = 0
    n_failed = 0

    for idx, spec in enumerate(pending, start=1):
        print(f"[{idx:04d}/{len(pending):04d}] {spec.run_id}")
        try:
            row, clean, mask, pred = core.run_one(
                spec,
                image_cache=image_cache,
                mask_cache=mask_cache,
                verbose=args.verbose,
            )
            core.append_run_row(runs_csv, row)
            n_ok += 1

            if visual_key is not None and _spec_visual_key(spec) == visual_key:
                if visual_payload is None:
                    visual_payload = {
                        "image": spec.image,
                        "mask": spec.mask,
                        "seed": spec.seed,
                        "clean": clean,
                        "mask_array": mask,
                        "preds": {},
                    }
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
            core.append_run_row(runs_csv, err_row)
            print(f"  FAILED: {exc!r}")
            if args.verbose:
                raise

    return n_ok, n_failed, visual_payload


def main(argv: list[str] | None = None) -> int:
    parser = build_parallel_parser()
    args = parser.parse_args(argv)

    if args.max_runs is not None and args.max_runs < 1:
        parser.error("--max-runs must be >= 1")
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.worker_threads < 1:
        parser.error("--worker-threads must be >= 1")
    if args.queue_factor < 1:
        parser.error("--queue-factor must be >= 1")

    _set_worker_thread_env(args.worker_threads)

    out_dir = Path(args.out_dir)
    runs_csv = out_dir / "runs.csv"
    aggregate_csv = out_dir / "aggregate.csv"
    manifest_json = out_dir / "manifest.json"
    plots_dir = out_dir / "plots"

    all_specs, config = core.enumerate_runs(args)

    if args.dry_run:
        limited_specs = all_specs[: args.max_runs] if args.max_runs else all_specs
        core.print_dry_run(limited_specs, config, total_specs=len(all_specs))
        core.write_manifest(
            manifest_json,
            {
                "script_version": core.SCRIPT_VERSION,
                "parallel_script_version": PARALLEL_SCRIPT_VERSION,
                "created_at": core.utc_now(),
                "dry_run": True,
                "parallel": {
                    "jobs": int(args.jobs),
                    "worker_threads": int(args.worker_threads),
                    "queue_factor": int(args.queue_factor),
                },
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

    completed = core.read_completed_run_ids(runs_csv) if args.resume else set()
    pending = [spec for spec in all_specs if spec.run_id not in completed]
    total_pending_before_cap = len(pending)
    if args.max_runs is not None:
        pending = pending[: args.max_runs]

    visual_key: tuple[Any, ...] | None = None
    for spec in pending:
        if spec.suite == "paper":
            visual_key = _spec_visual_key(spec)
            break

    print("=" * 78)
    print("Parallel experiment sweep")
    print("=" * 78)
    print(f"Profile          : {args.profile}")
    print(f"Output dir       : {out_dir}")
    print(f"Total specs      : {len(all_specs)}")
    print(f"Already complete : {len(completed)}")
    print(f"Pending selected : {len(pending)} / {total_pending_before_cap}")
    print(f"Methods          : {', '.join(config['methods'])}")
    print(f"Jobs             : {args.jobs}")
    print(f"Worker threads   : {args.worker_threads}")
    print()

    started_at = core.utc_now()
    if not pending:
        n_ok = 0
        n_failed = 0
        visual_payload = None
    elif args.jobs == 1:
        n_ok, n_failed, visual_payload = _run_serial(
            pending,
            args,
            runs_csv,
            visual_key,
        )
    else:
        n_ok, n_failed, visual_payload = _run_parallel(
            pending,
            args,
            runs_csv,
            visual_key,
        )

    rows = core.read_rows(runs_csv)
    agg_rows = core.aggregate_rows(rows)
    core.write_csv(aggregate_csv, agg_rows)
    core.plot_bar_summary(rows, plots_dir)
    core.plot_sensitivity(rows, plots_dir)

    if visual_payload is not None:
        core.plot_visual_comparison(
            plots_dir,
            image_name=visual_payload["image"],
            mask_name=visual_payload["mask"],
            seed=int(visual_payload["seed"]),
            clean=visual_payload["clean"],
            mask=visual_payload["mask_array"],
            preds=visual_payload["preds"],
        )

    core.write_manifest(
        manifest_json,
        {
            "script_version": core.SCRIPT_VERSION,
            "parallel_script_version": PARALLEL_SCRIPT_VERSION,
            "created_at": started_at,
            "finished_at": core.utc_now(),
            "dry_run": False,
            "resume": bool(args.resume),
            "parallel": {
                "jobs": int(args.jobs),
                "worker_threads": int(args.worker_threads),
                "queue_factor": int(args.queue_factor),
            },
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
    print("Parallel sweep complete")
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
