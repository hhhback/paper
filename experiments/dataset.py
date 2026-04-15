"""Temporal split and data loading for the experiment pipeline.

Provides:
- Time window computation with non-overlapping train/test periods
- Spark-based interaction loading and ground truth derivation
- Local JSONL loading for per-user daily extraction texts
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Time windows
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimeWindows:
    train_start: str  # YYYYMMDD
    train_end: str
    test_start: str
    test_end: str


def compute_time_windows(
    end_date: str,
    train_days: int = 21,
    test_days: int = 14,
) -> TimeWindows:
    """Compute non-overlapping train/test windows ending at *end_date*.

    Layout::

        |<--- train_days --->|<--- test_days --->|
        train_start      train_end  test_start  test_end (= end_date)
    """
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    test_end_dt = end_dt
    test_start_dt = end_dt - timedelta(days=test_days - 1)
    train_end_dt = test_start_dt - timedelta(days=1)
    train_start_dt = train_end_dt - timedelta(days=train_days - 1)

    windows = TimeWindows(
        train_start=train_start_dt.strftime("%Y%m%d"),
        train_end=train_end_dt.strftime("%Y%m%d"),
        test_start=test_start_dt.strftime("%Y%m%d"),
        test_end=test_end_dt.strftime("%Y%m%d"),
    )

    # Invariant: train and test must not overlap
    assert train_end_dt < test_start_dt, (
        f"Train/test overlap: train_end={windows.train_end}, "
        f"test_start={windows.test_start}"
    )
    return windows


# ---------------------------------------------------------------------------
# Spark helpers — date format conversion
# ---------------------------------------------------------------------------

def _to_dash(datestr: str) -> str:
    """Convert YYYYMMDD to YYYY-MM-DD for Spark SQL BETWEEN clauses."""
    return f"{datestr[:4]}-{datestr[4:6]}-{datestr[6:8]}"


def _register_target_ids(spark: "SparkSession", user_ids: list[str]) -> None:
    """Register *user_ids* as a temporary view ``target_ids`` with column ``idhash``."""
    rows = [(uid,) for uid in user_ids]
    spark.createDataFrame(rows, ["idhash"]).createOrReplaceTempView("target_ids")


# ---------------------------------------------------------------------------
# Raw interaction loading
# ---------------------------------------------------------------------------

def load_raw_interactions(
    spark: "SparkSession",
    user_ids: list[str],
    start_date: str,
    end_date: str,
) -> "DataFrame":
    """Load click-log interactions for *user_ids* within [start_date, end_date].

    Dates are in YYYYMMDD format.  Returns the raw DataFrame from
    ``data_poi_click_log`` filtered by the user set and date range.
    """
    _register_target_ids(spark, user_ids)

    start_dash = _to_dash(start_date)
    end_dash = _to_dash(end_date)

    return spark.sql(f"""
        SELECT etimestamp, idhash, age, gender, sid, query
        FROM airspace_recsys__db_real.data_poi_click_log AS c
        LEFT SEMI JOIN target_ids AS tid
            ON c.idhash = tid.idhash
        WHERE datestr BETWEEN '{start_dash}' AND '{end_dash}'
    """)


# ---------------------------------------------------------------------------
# Ground truth: per-user visited POI category set in test window
# ---------------------------------------------------------------------------

def build_ground_truth(
    spark: "SparkSession",
    user_ids: list[str],
    test_start: str,
    test_end: str,
) -> dict[str, set[str]]:
    """Build ground truth: ``{user_id: {top_category, ...}}`` for the test window.

    Matches production derivation (data.py:133):
    ``F.trim(F.split(F.col('category_path_kr'), '>').getItem(1))``
    """
    from pyspark.sql import functions as F

    _register_target_ids(spark, user_ids)

    start_dash = _to_dash(test_start)
    end_dash = _to_dash(test_end)

    click_df = spark.sql(f"""
        SELECT c.idhash, c.sid
        FROM airspace_recsys__db_real.data_poi_click_log AS c
        LEFT SEMI JOIN target_ids AS tid
            ON c.idhash = tid.idhash
        WHERE datestr BETWEEN '{start_dash}' AND '{end_dash}'
    """)

    poi_base_df = spark.sql("""
        SELECT sid, category_path_kr
        FROM airspace_recsys__db_real.data_poi_base
    """)

    joined = (
        click_df
        .join(poi_base_df, on="sid", how="left")
        .withColumn(
            "top_category",
            F.trim(F.split(F.col("category_path_kr"), ">").getItem(1)),
        )
        .filter(F.col("top_category").isNotNull())
        .select("idhash", "top_category")
        .distinct()
    )

    rows = joined.collect()

    ground_truth: dict[str, set[str]] = {}
    for row in rows:
        ground_truth.setdefault(row["idhash"], set()).add(row["top_category"])

    return ground_truth


# ---------------------------------------------------------------------------
# User selection (AC2)
# ---------------------------------------------------------------------------

def select_experiment_users(
    spark: "SparkSession",
    windows: TimeWindows,
    n_users: int = 2000,
    min_clicks: int = 15,
) -> list[str]:
    """Select users with sufficient activity in both train and test periods.

    Selection criteria:
    1. ≥ *min_clicks* interactions in the train period
    2. ≥ *min_clicks* interactions in the test period
    3. ≥ 2 distinct top-level categories visited in the test period
    4. Users with both implicit and explicit data are preferred (sorted first)
    5. Ties broken by total click count (descending)

    Returns up to *n_users* user ID hashes.
    """
    from pyspark.sql import functions as F

    train_start = _to_dash(windows.train_start)
    train_end = _to_dash(windows.train_end)
    test_start = _to_dash(windows.test_start)
    test_end = _to_dash(windows.test_end)

    # Users with enough clicks in the train period
    train_users = spark.sql(f"""
        SELECT idhash, COUNT(*) AS train_clicks
        FROM airspace_recsys__db_real.data_poi_click_log
        WHERE datestr BETWEEN '{train_start}' AND '{train_end}'
        GROUP BY idhash
        HAVING train_clicks >= {min_clicks}
    """)

    # Users with enough clicks in the test period
    test_users = spark.sql(f"""
        SELECT idhash, COUNT(*) AS test_clicks
        FROM airspace_recsys__db_real.data_poi_click_log
        WHERE datestr BETWEEN '{test_start}' AND '{test_end}'
        GROUP BY idhash
        HAVING test_clicks >= {min_clicks}
    """)

    # Inner join: users active in both periods
    both_periods = train_users.join(test_users, on="idhash", how="inner")

    # Category diversity filter: ≥ 2 distinct top categories in test period
    test_clicks_with_cat = spark.sql(f"""
        SELECT c.idhash, p.category_path_kr
        FROM airspace_recsys__db_real.data_poi_click_log AS c
        JOIN airspace_recsys__db_real.data_poi_base AS p ON c.sid = p.sid
        WHERE c.datestr BETWEEN '{test_start}' AND '{test_end}'
    """).withColumn(
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

    # Check for implicit/explicit data availability per user
    # Implicit: clicks with a query (search-driven)
    # Explicit: clicks without a query (direct browsing)
    data_type_flags = spark.sql(f"""
        SELECT
            idhash,
            MAX(CASE WHEN query IS NOT NULL AND TRIM(query) != '' THEN 1 ELSE 0 END) AS has_implicit,
            MAX(CASE WHEN query IS NULL OR TRIM(query) = '' THEN 1 ELSE 0 END) AS has_explicit
        FROM airspace_recsys__db_real.data_poi_click_log
        WHERE datestr BETWEEN '{train_start}' AND '{test_end}'
        GROUP BY idhash
    """)

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

    logger.info(
        "Selected %d experiment users (requested %d, min_clicks=%d)",
        len(selected), n_users, min_clicks,
    )
    return selected


# ---------------------------------------------------------------------------
# Local JSONL loaders (Stage B — per-user daily extraction texts)
# ---------------------------------------------------------------------------

def _load_jsonl_records(path: Path, text_field: str) -> dict[str, str]:
    """Load a JSONL file, returning ``{idhash: text}``."""
    records: dict[str, str] = {}
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            idhash = (obj.get("idhash") or "").strip()
            text = (obj.get(text_field) or "").strip()
            if idhash and text:
                records[idhash] = text
    return records


def load_user_daily_texts(
    data_dir: Path,
    user_id: str,
    implicit_only: bool = False,
    explicit_only: bool = False,
) -> dict[str, str]:
    """Load per-day extraction JSONLs for a single user.

    Returns ``{datestr: merged_text}`` where each date's text is assembled
    from implicit and/or explicit persona results following the production
    merge format (build_memories.py:87-88).

    Directory layout expected under *data_dir*::

        output/implicit_data/{datestr}/persona_results_{datestr}.jsonl
        output/explicit_data/{datestr}/persona_results_{datestr}.jsonl
    """
    implicit_root = data_dir / "output" / "implicit_data"
    explicit_root = data_dir / "output" / "explicit_data"

    # Collect all available datestrs from both directories
    datestrs: set[str] = set()
    if not explicit_only and implicit_root.is_dir():
        datestrs.update(d.name for d in implicit_root.iterdir() if d.is_dir())
    if not implicit_only and explicit_root.is_dir():
        datestrs.update(d.name for d in explicit_root.iterdir() if d.is_dir())

    result: dict[str, str] = {}
    for datestr in sorted(datestrs):
        imp_text: str | None = None
        exp_text: str | None = None

        if not explicit_only:
            imp_path = implicit_root / datestr / f"persona_results_{datestr}.jsonl"
            imp_records = _load_jsonl_records(imp_path, "llm_response")
            imp_text = imp_records.get(user_id)

        if not implicit_only:
            exp_path = explicit_root / datestr / f"persona_results_{datestr}.jsonl"
            exp_records = _load_jsonl_records(exp_path, "persona_text")
            exp_text = exp_records.get(user_id)

        if imp_text and exp_text:
            result[datestr] = f"[implicit data]\n{imp_text}\n\n[explicit data]\n{exp_text}"
        elif imp_text:
            result[datestr] = imp_text
        elif exp_text:
            result[datestr] = exp_text

    return result


def load_all_users(
    data_dir: Path,
    user_ids: list[str],
    implicit_only: bool = False,
    explicit_only: bool = False,
) -> dict[str, dict[str, str]]:
    """Load daily texts for all users. Returns ``{user_id: {datestr: text}}``."""
    return {
        uid: load_user_daily_texts(
            data_dir, uid,
            implicit_only=implicit_only,
            explicit_only=explicit_only,
        )
        for uid in user_ids
    }
