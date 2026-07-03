#!/usr/bin/env python3
"""
LANTERN — Latent Relationship Mining & Atomicity-Aware Dual-View Routing
=======================================================================

Reproducible reference implementation.

Quick start:
    python run.py                # run on gc, mc, mf sequentially
    python run.py --dataset mc   # run on a single dataset

Datasets:
    gc — Google Core (Chromium issue tracker)
    mc — Mozilla Core (Bugzilla Core product)
    mf — Mozilla Firefox (Bugzilla Firefox product)
"""
import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lantern.config import DATASETS


def run_all():
    """Train and evaluate LANTERN on all three datasets."""
    from train import train

    all_results = {}
    total_start = time.time()

    for ds in DATASETS:
        t0 = time.time()
        print(f"\n{'#' * 60}")
        print(f"#  Dataset: {ds}")
        print(f"{'#' * 60}")
        metrics = train(ds)
        elapsed = time.time() - t0
        all_results[ds] = metrics
        print(f"  [{ds}] elapsed: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    # ── Summary ──
    print(f"\n{'=' * 90}")
    print("  Evaluation Summary")
    print(f"{'=' * 90}")
    header = (
        f"{'Dataset':<8} {'MRR':>8} {'HR@1':>8} {'HR@3':>8} {'HR@5':>8} "
        f"{'HR@10':>8} {'NDCG@1':>8} {'NDCG@3':>8} {'NDCG@5':>8} {'NDCG@10':>8}"
    )
    print(header)
    print("-" * 90)
    for ds in DATASETS:
        m = all_results[ds]
        print(
            f"{ds:<8} {m['MRR']:8.4f} {m['HR@1']:8.4f} {m['HR@3']:8.4f} "
            f"{m['HR@5']:8.4f} {m['HR@10']:8.4f} {m['NDCG@1']:8.4f} "
            f"{m['NDCG@3']:8.4f} {m['NDCG@5']:8.4f} {m['NDCG@10']:8.4f}"
        )

    total_elapsed = time.time() - total_start
    print(f"\n  Total wall time: {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LANTERN: Latent Relationship Mining & Dual-View Routing"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=DATASETS + [None],
        help="Single dataset to run (default: all three)",
    )
    args = parser.parse_args()

    if args.dataset:
        from train import train

        train(args.dataset)
    else:
        run_all()
