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

NOTE: This script is self-contained (no imports from experiments package) because
spark-submit on YARN only ships the application file, not the full repo.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inlined from experiments/dataset.py (YARN can't import experiments package)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimeWindows:
    train_start: str  # YYYYMMDD
    train_end: str
    test_start: str
    test_end: str


def _to_dash(datestr):
    """YYYYMMDD -> YYYY-MM-DD."""
    return "%s-%s-%s" % (datestr[:4], datestr[4:6], datestr[6:8])


def compute_time_windows(end_date, train_days=21, test_days=14):
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    test_end_dt = end_dt
    test_start_dt = end_dt - timedelta(days=test_days - 1)
    train_end_dt = test_start_dt - timedelta(days=1)
    train_start_dt = train_end_dt - timedelta(days=train_days - 1)
    return TimeWindows(
        train_start=train_start_dt.strftime("%Y%m%d"),
        train_end=train_end_dt.strftime("%Y%m%d"),
        test_start=test_start_dt.strftime("%Y%m%d"),
        test_end=test_end_dt.strftime("%Y%m%d"),
    )


def select_experiment_users(spark, windows, n_users=2000, min_clicks=15):
    from pyspark.sql import functions as F

    train_start = _to_dash(windows.train_start)
    train_end = _to_dash(windows.train_end)
    test_start = _to_dash(windows.test_start)
    test_end = _to_dash(windows.test_end)

    train_users = spark.sql(
        "SELECT idhash, COUNT(*) AS train_clicks "
        "FROM airspace_recsys__db_real.data_poi_click_log "
        "WHERE datestr BETWEEN '%s' AND '%s' "
        "GROUP BY idhash HAVING train_clicks >= %d" % (train_start, train_end, min_clicks)
    )
    test_users = spark.sql(
        "SELECT idhash, COUNT(*) AS test_clicks "
        "FROM airspace_recsys__db_real.data_poi_click_log "
        "WHERE datestr BETWEEN '%s' AND '%s' "
        "GROUP BY idhash HAVING test_clicks >= %d" % (test_start, test_end, min_clicks)
    )
    both_periods = train_users.join(test_users, on="idhash", how="inner")

    test_clicks_with_cat = spark.sql(
        "SELECT c.idhash, p.category_path_kr "
        "FROM airspace_recsys__db_real.data_poi_click_log AS c "
        "JOIN airspace_recsys__db_real.data_poi_base AS p ON c.sid = p.sid "
        "WHERE c.datestr BETWEEN '%s' AND '%s'" % (test_start, test_end)
    ).withColumn(
        "top_category",
        F.trim(F.split(F.col("category_path_kr"), ">").getItem(1)),
    ).filter(F.col("top_category").isNotNull())

    cat_diversity = (
        test_clicks_with_cat
        .groupBy("idhash")
        .agg(F.countDistinct("top_category").alias("n_categories"))
        .filter(F.col("n_categories") >= 2)
    )
    candidates = both_periods.join(cat_diversity, on="idhash", how="inner")

    data_type_flags = spark.sql(
        "SELECT idhash, "
        "MAX(CASE WHEN query IS NOT NULL AND TRIM(query) != '' THEN 1 ELSE 0 END) AS has_implicit, "
        "MAX(CASE WHEN query IS NULL OR TRIM(query) = '' THEN 1 ELSE 0 END) AS has_explicit "
        "FROM airspace_recsys__db_real.data_poi_click_log "
        "WHERE datestr BETWEEN '%s' AND '%s' "
        "GROUP BY idhash" % (train_start, test_end)
    )

    ranked = (
        candidates
        .join(data_type_flags, on="idhash", how="left")
        .fillna(0, subset=["has_implicit", "has_explicit"])
        .withColumn("has_both", F.col("has_implicit") + F.col("has_explicit"))
        .withColumn("total_clicks", F.col("train_clicks") + F.col("test_clicks"))
        .orderBy(F.desc("has_both"), F.desc("total_clicks"))
        .limit(n_users)
    )
    rows = ranked.select("idhash").collect()
    selected = [row["idhash"] for row in rows]
    logger.info("Selected %d experiment users (requested %d, min_clicks=%d)", len(selected), n_users, min_clicks)
    return selected


def build_ground_truth(spark, user_ids, test_start, test_end):
    from pyspark.sql import functions as F

    rows_df = spark.createDataFrame([(uid,) for uid in user_ids], ["idhash"])
    rows_df.createOrReplaceTempView("target_ids")

    start_dash = _to_dash(test_start)
    end_dash = _to_dash(test_end)

    click_df = spark.sql(
        "SELECT c.idhash, c.sid "
        "FROM airspace_recsys__db_real.data_poi_click_log AS c "
        "LEFT SEMI JOIN target_ids AS tid ON c.idhash = tid.idhash "
        "WHERE datestr BETWEEN '%s' AND '%s'" % (start_dash, end_dash)
    )
    poi_base_df = spark.sql("SELECT sid, category_path_kr FROM airspace_recsys__db_real.data_poi_base")

    joined = (
        click_df
        .join(poi_base_df, on="sid", how="left")
        .withColumn("top_category", F.trim(F.split(F.col("category_path_kr"), ">").getItem(1)))
        .filter(F.col("top_category").isNotNull())
        .select("idhash", "top_category")
        .distinct()
    )
    rows = joined.collect()
    ground_truth = {}
    for row in rows:
        ground_truth.setdefault(row["idhash"], set()).add(row["top_category"])
    return ground_truth


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare experiment data for Argo: select users, build ground truth, write HDFS parquet.",
    )
    parser.add_argument("--end-date", type=str, required=True, help="End date in YYYYMMDD format.")
    parser.add_argument("--output-dir", type=str, required=True, help="Root directory for experiment outputs.")
    parser.add_argument("--profile-path", type=str,
                        default="hdfs://camino/user/airspace-fs-kr/service/new_place/profile/",
                        help="Parquet path for user demographic data.")
    parser.add_argument("--n-users", type=int, default=2000, help="Number of users (default: 2000).")
    parser.add_argument("--min-clicks", type=int, default=15, help="Min clicks per period (default: 15).")
    parser.add_argument("--hdfs-output-base", type=str,
                        default="hdfs://camino/user/airspace-fs-kr/service/memory_agent/experiment",
                        help="HDFS base path for user_ids_parquet.")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("experiment-prepare-data").enableHiveSupport().getOrCreate()

    # 1. Compute time windows
    windows = compute_time_windows(args.end_date)
    logger.info("Time windows — train: %s~%s, test: %s~%s",
                windows.train_start, windows.train_end, windows.test_start, windows.test_end)

    # 2. Select experiment users
    user_ids = select_experiment_users(spark, windows, n_users=args.n_users, min_clicks=args.min_clicks)
    logger.info("Selected %d users", len(user_ids))

    # 3. Build ground truth
    ground_truth = build_ground_truth(spark, user_ids, test_start=windows.test_start, test_end=windows.test_end)
    logger.info("Ground truth built for %d users", len(ground_truth))

    # 4. Write user_ids parquet to HDFS (required by data.py and explicit_preprocessing.py)
    user_ids_df = spark.createDataFrame([(uid,) for uid in user_ids], ["idhash"])
    hdfs_parquet_path = "%s/%s/user_ids_parquet" % (args.hdfs_output_base, args.end_date)
    user_ids_df.coalesce(1).write.mode("overwrite").parquet(hdfs_parquet_path)
    logger.info("Wrote user_ids parquet to %s", hdfs_parquet_path)

    # 5. Save locally
    with open(str(output_dir / "user_ids.json"), "w") as f:
        json.dump(user_ids, f, ensure_ascii=False)
    logger.info("Saved user_ids to %s", output_dir / "user_ids.json")

    gt_serializable = {uid: sorted(cats) for uid, cats in ground_truth.items()}
    with open(str(output_dir / "ground_truth.json"), "w") as f:
        json.dump(gt_serializable, f, ensure_ascii=False, indent=2)
    logger.info("Saved ground_truth to %s", output_dir / "ground_truth.json")

    logger.info("Data preparation complete. Output: %s", output_dir)
    spark.stop()


if __name__ == "__main__":
    main()
