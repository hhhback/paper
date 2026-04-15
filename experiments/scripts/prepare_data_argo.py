"""CLI entry point: prepare experiment data for Argo K8s pod context.

Usage::

    spark-submit prepare_data_argo.py --end-date 20250315 --output-dir /path/to/output

Orchestrates:
1. Compute train/test time windows
2. Select experiment users with sufficient activity
3. Build ground truth (test-period category visits)
4. Write user_ids parquet to HDFS (for production Spark jobs)
5. Save user_ids.json and ground_truth.json locally

LLM extraction (Stage 3-4 in prepare_data.py) is handled by separate Argo K8s pods.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure experiment package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.dataset import (
    build_ground_truth,
    compute_time_windows,
    select_experiment_users,
)

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare experiment data for Argo: select users, build ground truth, write HDFS parquet.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date of the experiment window in YYYYMMDD format.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Root directory for all experiment outputs.",
    )
    parser.add_argument(
        "--profile-path",
        type=str,
        default="hdfs://camino/user/airspace-fs-kr/service/new_place/profile/",
        help="Parquet path for user demographic data.",
    )
    parser.add_argument(
        "--n-users",
        type=int,
        default=2000,
        help="Number of experiment users to select (default: 2000).",
    )
    parser.add_argument(
        "--min-clicks",
        type=int,
        default=15,
        help="Minimum click count per period for user selection (default: 15).",
    )
    parser.add_argument(
        "--hdfs-output-base",
        type=str,
        default="hdfs://camino/user/airspace-fs-kr/service/memory_agent/experiment",
        help="HDFS base path for writing user_ids_parquet (default: hdfs://camino/user/airspace-fs-kr/service/memory_agent/experiment).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Spark (no custom config — Argo's common-c3s template handles it) --
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .appName("experiment-prepare-data")
        .enableHiveSupport()
        .getOrCreate()
    )

    # 1. Compute time windows
    windows = compute_time_windows(args.end_date)
    logger.info(
        "Time windows — train: %s~%s, test: %s~%s",
        windows.train_start, windows.train_end,
        windows.test_start, windows.test_end,
    )

    # 2. Select experiment users
    user_ids = select_experiment_users(
        spark, windows,
        n_users=args.n_users,
        min_clicks=args.min_clicks,
    )
    logger.info("Selected %d users", len(user_ids))

    # 3. Build ground truth for test period
    ground_truth = build_ground_truth(
        spark, user_ids,
        test_start=windows.test_start,
        test_end=windows.test_end,
    )
    logger.info("Ground truth built for %d users", len(ground_truth))

    # 4. Write user_ids parquet to HDFS
    #    Required by downstream production Spark jobs:
    #    - data.py --sampled_ids_path reads parquet from HDFS
    #    - explicit_preprocessing.py --user_input only triggers parquet read for hdfs:// prefix
    user_ids_df = spark.createDataFrame([(uid,) for uid in user_ids], ["idhash"])
    hdfs_parquet_path = f"{args.hdfs_output_base}/{args.end_date}/user_ids_parquet"
    user_ids_df.coalesce(1).write.mode("overwrite").parquet(hdfs_parquet_path)
    logger.info("Wrote user_ids parquet to %s", hdfs_parquet_path)

    # 5. Save user_ids and ground_truth locally
    user_ids_path = output_dir / "user_ids.json"
    with open(user_ids_path, "w", encoding="utf-8") as f:
        json.dump(user_ids, f, ensure_ascii=False)
    logger.info("Saved user_ids to %s", user_ids_path)

    gt_path = output_dir / "ground_truth.json"
    gt_serializable = {uid: sorted(cats) for uid, cats in ground_truth.items()}
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_serializable, f, ensure_ascii=False, indent=2)
    logger.info("Saved ground_truth to %s", gt_path)

    logger.info("Data preparation complete. Output: %s", output_dir)
    spark.stop()


if __name__ == "__main__":
    main()
