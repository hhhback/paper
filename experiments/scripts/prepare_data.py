"""CLI entry point: prepare experiment data (user selection, ground truth, LLM extraction).

Usage::

    spark-submit prepare_data.py --end-date 20250315 --output-dir /path/to/output

Orchestrates:
1. Compute train/test time windows
2. Select experiment users with sufficient activity
3. Build ground truth (test-period category visits)
4. Run per-day LLM extraction over the train window
5. Save user_ids and ground_truth for Stage B
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure experiment package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.dataset import (
    build_ground_truth,
    compute_time_windows,
    select_experiment_users,
)
from experiments.re_extract import extract_for_experiment

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare experiment data: select users, build ground truth, run LLM extraction.",
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
    return parser.parse_args()


def _to_dash(datestr: str) -> str:
    """YYYYMMDD -> YYYY-MM-DD."""
    return f"{datestr[:4]}-{datestr[4:6]}-{datestr[6:8]}"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Spark --
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .appName("experiment-prepare-data")
        .config("spark.sql.shuffle.partitions", "500")
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

    # 4. LLM extraction over train period
    from openai import AsyncOpenAI

    llm_client = AsyncOpenAI(
        base_url="https://bruno.maas.navercorp.com/v1",
        api_key=os.environ["API_KEY"],
    )

    train_start_dash = _to_dash(windows.train_start)
    train_end_dash = _to_dash(windows.train_end)

    extract_for_experiment(
        spark=spark,
        user_ids=user_ids,
        train_start=train_start_dash,
        train_end=train_end_dash,
        output_dir=output_dir / "output",
        llm_client=llm_client,
        profile_path=args.profile_path,
        demo_path=args.profile_path,
    )

    # 5. Save user_ids and ground_truth for Stage B
    user_ids_path = output_dir / "user_ids.json"
    with open(user_ids_path, "w", encoding="utf-8") as f:
        json.dump(user_ids, f, ensure_ascii=False)
    logger.info("Saved user_ids to %s", user_ids_path)

    gt_path = output_dir / "ground_truth.json"
    # Convert sets to lists for JSON serialization
    gt_serializable = {uid: sorted(cats) for uid, cats in ground_truth.items()}
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_serializable, f, ensure_ascii=False, indent=2)
    logger.info("Saved ground_truth to %s", gt_path)

    logger.info("Data preparation complete. Output: %s", output_dir)
    spark.stop()


if __name__ == "__main__":
    main()
