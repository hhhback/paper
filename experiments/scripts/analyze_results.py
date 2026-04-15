#!/usr/bin/env python3
"""Post-hoc statistical analysis CLI.

Reads results CSVs from multiple variants and performs statistical comparison
against a baseline variant using paired bootstrap CIs, Cohen's d, and
Holm-Bonferroni correction.

Usage:
    python analyze_results.py --results-dir /path/to/results --baseline Ours

Expected layout:
    results_dir/
        Ours/results.csv
        No-Memory/results.csv
        Full-Context/results.csv
        ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.evaluation.stats import (
    cohens_d,
    holm_bonferroni,
    paired_bootstrap_ci,
)

METRIC_COLUMNS = [
    "precision@1", "precision@3", "precision@5",
    "recall@5", "ndcg@10", "hr@10", "mrr",
]


def _load_variant(results_dir: Path, variant_name: str) -> pd.DataFrame:
    path = results_dir / variant_name / "results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Results CSV not found: {path}")
    df = pd.read_csv(path)
    df = df.set_index("user_id")
    return df


def compare_variants(
    results_dir: Path,
    baseline_name: str,
    n_comparisons: int = 9,
) -> pd.DataFrame:
    """Compare all non-baseline variants against the baseline.

    Returns a summary DataFrame with one row per (variant, metric) pair.
    """
    baseline_df = _load_variant(results_dir, baseline_name)

    # Discover other variant directories
    variant_dirs = sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and d.name != baseline_name and (d / "results.csv").exists()
    )

    rows: list[dict] = []
    all_p_values: list[float] = []
    row_indices: list[int] = []

    for variant_dir in variant_dirs:
        variant_name = variant_dir.name
        variant_df = _load_variant(results_dir, variant_name)

        # Align on shared users
        shared_users = baseline_df.index.intersection(variant_df.index)
        if len(shared_users) == 0:
            continue

        bl = baseline_df.loc[shared_users]
        vr = variant_df.loc[shared_users]

        for metric in METRIC_COLUMNS:
            if metric not in bl.columns or metric not in vr.columns:
                continue

            a = np.array(bl[metric].values, dtype=float)
            b = np.array(vr[metric].values, dtype=float)

            boot = paired_bootstrap_ci(a, b)
            d = cohens_d(a, b)

            row = {
                "variant": variant_name,
                "metric": metric,
                "baseline_mean": float(np.mean(a)),
                "variant_mean": float(np.mean(b)),
                "mean_diff": boot.mean_diff,
                "ci_lower": boot.ci_lower,
                "ci_upper": boot.ci_upper,
                "p_value": boot.p_value,
                "cohens_d": d,
            }
            row_indices.append(len(rows))
            all_p_values.append(boot.p_value)
            rows.append(row)

    # Holm-Bonferroni correction per metric (m=9 comparisons each)
    if rows:
        by_metric: dict[str, list[int]] = {}
        for i, row in enumerate(rows):
            by_metric.setdefault(row["metric"], []).append(i)

        for metric, indices in by_metric.items():
            p_vals = [rows[i]["p_value"] for i in indices]
            rejections = holm_bonferroni(p_vals, alpha=0.05)
            for idx, reject in zip(indices, rejections):
                rows[idx]["significant"] = reject

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Statistical comparison of experiment variants.",
    )
    parser.add_argument(
        "--results-dir", required=True, type=Path,
        help="Directory containing variant subdirectories with results.csv",
    )
    parser.add_argument(
        "--baseline", default="Ours",
        help="Name of the baseline variant directory (default: Ours)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Path to save summary CSV (default: results_dir/comparison.csv)",
    )
    parser.add_argument(
        "--n-comparisons", type=int, default=9,
        help="Total number of comparisons for Holm-Bonferroni (default: 9)",
    )
    args = parser.parse_args()

    summary = compare_variants(
        args.results_dir, args.baseline, args.n_comparisons,
    )

    if summary.empty:
        print("No comparisons found. Check that variant directories exist.")
        return

    # Print summary table
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(summary.to_string(index=False))

    # Save CSV
    output_path = args.output or (args.results_dir / "comparison.csv")
    summary.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
