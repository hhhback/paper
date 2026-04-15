"""Re-extraction DAG — run LLM extraction on raw Spark data for the train window.

Produces two independent extraction streams:
- **Implicit**: click/save log data → prompt_click_gen() → LLM (temperature=0.0) → llm_response
- **Explicit**: review data from iceberg__data_visit_review → explicit prompts → LLM (temperature=0.2) → persona_text

These are fundamentally different data sources with different prompts, matching
the production pipeline (async_run.py for implicit, explicit_extract.py for explicit).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

# Production pipeline imports
sys.path.insert(0, os.environ.get("PROD_SRC", "/home1/irteam/work/fs-feature/src/user/memory"))
import data
import prompt_extract

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)

# Path to the explicit extraction prompt template
_EXPLICIT_PROMPT_PATH = Path("/home1/irteam/work/fs-feature/src/user/memory/explicit_prompts/prompt_format.yaml")


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _date_range(start: str, end: str) -> list[str]:
    """Yield YYYY-MM-DD strings from *start* to *end* inclusive."""
    fmt = "%Y-%m-%d"
    cur = datetime.strptime(start, fmt)
    end_dt = datetime.strptime(end, fmt)
    dates: list[str] = []
    while cur <= end_dt:
        dates.append(cur.strftime(fmt))
        cur += timedelta(days=1)
    return dates


# ---------------------------------------------------------------------------
# Click preprocessing (mirrors data.py:107-146 exactly)
# ---------------------------------------------------------------------------

def _preprocess_click(click_df, poi_base_df, demo_df):
    """Apply the same preprocessing as production data.py main()."""
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    # Day-of-week
    days_mapping = data.days_mapping
    days_expr = F.create_map([F.lit(x) for x in chain(*days_mapping.items())])
    click_df = (
        click_df
        .withColumn("temp_timestamp", F.to_timestamp("etimestamp"))
        .withColumn("dat_name", days_expr.getItem(F.dayofweek("temp_timestamp")))
        .drop("temp_timestamp")
    )

    # Join POI base (adds category_name_kr, road_addr, title, place_review_count)
    click_df = click_df.join(poi_base_df, on="sid", how="left")

    # Join demo
    df = click_df.join(demo_df, on="idhash", how="left")

    # Derived columns
    df = (
        df
        .withColumn("region", F.split(F.col("road_addr"), " ").getItem(0))
        .withColumn("top_category", F.trim(F.split(F.col("category_path_kr"), ">").getItem(1)))
    )

    # Review percentile rank within (region, top_category)
    window_spec = Window.partitionBy("region", "top_category").orderBy(F.col("place_review_count").desc())
    df = df.withColumn("review_pct_rank", F.round(F.percent_rank().over(window_spec) * 100, 1))

    # Bin into S/A/B/C/D
    custom_bins = [10, 30, 60, 85, 100]
    labels = ["S", "A", "B", "C", "D"]
    expr = F.when(F.col("review_pct_rank") <= custom_bins[0], labels[0])
    for b, l in zip(custom_bins[1:], labels[1:]):
        expr = expr.when(F.col("review_pct_rank") <= b, l)
    df = df.withColumn("review_top_display", expr.otherwise("정보없음"))

    return df


# ---------------------------------------------------------------------------
# Save-data preprocessing (mirrors data.py:120-123)
# ---------------------------------------------------------------------------

def _preprocess_save(save_df, poi_base_df):
    from pyspark.sql import functions as F

    return (
        save_df
        .join(poi_base_df, on="sid", how="left")
        .withColumn("category", F.trim(F.element_at(F.split(F.col("category_path_kr"), ">"), -1)))
    )


# ---------------------------------------------------------------------------
# Explicit data: review preprocessing (mirrors explicit_preprocessing.py)
# ---------------------------------------------------------------------------

def _load_explicit_prompts() -> dict:
    """Load explicit extraction prompts from the production YAML."""
    with open(_EXPLICIT_PROMPT_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _query_review_data(spark: "SparkSession", datestr: str):
    """Query review data for a single day from iceberg__data_visit_review.

    Mirrors explicit_preprocessing.py: query → filter deleted/empty → dedup by review_group_id.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    review_raw = spark.sql(f"""
        SELECT
            idno AS idhash, review_group_id, place_id, created_date_time,
            visit_date_time, text_review,
            flatten(votedkeyword.codes) AS keyword_list, is_deleted
        FROM hadoop_cat.airspace_recsys__db_real.iceberg__data_visit_review
        WHERE DATE(iceberg_updated_time) = '{datestr}'
    """)

    # Filter to target users
    review_data = review_raw.join(
        spark.table("target_ids"),
        review_raw["idhash"] == spark.table("target_ids")["idhash"],
        "left_semi",
    )

    # Text cleanup and filter
    processed = (
        review_data
        .withColumn("text_review", F.trim(F.regexp_replace(F.col("text_review"), r"[\n\t\r]+", " ")))
        .filter(
            (~F.col("is_deleted"))
            & F.col("text_review").isNotNull()
            & (F.col("text_review") != "")
        )
    )

    # Dedup by review_group_id
    window_spec = Window.partitionBy("review_group_id").orderBy(F.col("created_date_time").desc())
    result = (
        processed
        .withColumn("_rn", F.row_number().over(window_spec))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )

    return result


def _preprocess_explicit(review_df, poi_df, demo_df):
    """Join review data with POI and demo info, matching explicit_preprocessing.py output."""
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
    from explicit_preprocessing import get_demo_text_spark

    merged = (
        review_df
        .join(poi_df, on="place_id", how="inner")
        .join(demo_df, on="idhash", how="left")
    )

    window_spec = Window.partitionBy("idhash").orderBy(F.col("created_date_time").desc())

    result = (
        merged
        .withColumn("demo_text", F.coalesce(get_demo_text_spark(), F.lit("")))
        .withColumn("row", F.row_number().over(window_spec))
        .withColumn(
            "keyword_list",
            F.concat(F.lit("["), F.array_join(F.col("keyword_list"), ", "), F.lit("]")),
        )
        .select(
            "idhash", "row",
            F.col("created_date_time").alias("write_date_time"),
            "visit_date_time",
            F.date_format(F.col("visit_date_time"), "E").alias("visit_day"),
            "place_name", "address", "category_path_kr",
            "text_review", "keyword_list", "demo_text",
        )
    )

    return result


def _build_explicit_prompts(
    review_pd,
    user_prompt_template: str,
    few_shot_rich: str,
    few_shot_sparse: str,
) -> list[tuple[str, str]]:
    """Build per-user explicit prompts from preprocessed review DataFrame.

    Each user's reviews are chunked (max 50) and formatted using the explicit prompt template.
    Returns list of (idhash, prompt_text) pairs.
    """
    prompts: list[tuple[str, str]] = []

    for idhash, user_reviews in review_pd.groupby("idhash"):
        user_reviews = user_reviews.sort_values(
            ["write_date_time", "place_name"], ascending=False,
        )
        user_demo = user_reviews["demo_text"].iloc[0]
        num_reviews = len(user_reviews)
        few_shot = few_shot_sparse if num_reviews <= 4 else few_shot_rich

        # Chunk by 50 reviews (matching production explicit_extract.py)
        chunk_size = 50
        for start_idx in range(0, num_reviews, chunk_size):
            chunk = user_reviews.iloc[start_idx : start_idx + chunk_size]
            clean_chunk = chunk.drop(
                columns=["idhash", "demo_text", "created_date_time"],
                errors="ignore",
            )

            prompt_text = user_prompt_template.format(
                few_shot=few_shot,
                demographic_prompt=user_demo,
                user_prompt=clean_chunk.to_string(index=False),
            )
            prompts.append((idhash, prompt_text))

    return prompts


# ---------------------------------------------------------------------------
# LLM call (simplified from async_run.py)
# ---------------------------------------------------------------------------

async def _call_llm(
    client, prompt: str, semaphore: asyncio.Semaphore,
    system_message: str, temperature: float = 0.0,
) -> dict:
    max_retries = 5
    async with semaphore:
        for _ in range(max_retries):
            try:
                completion = await client.chat.completions.create(
                    model=os.getenv("LLM_MODEL_NAME", "qwen/qwen3-235b-a22b-instruct-2507"),
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt},
                    ],
                    stream=False,
                    temperature=temperature,
                )
                content = completion.choices[0].message.content
                if content is None:
                    await asyncio.sleep(10)
                    continue
                usage = completion.usage
                return {
                    "content": content,
                    "usage": {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                    },
                }
            except Exception as e:
                logger.warning("LLM call error: %s", e)
                await asyncio.sleep(10)
    return {"content": None, "usage": None}


async def _extract_batch(
    prompts: list[tuple[str, str]],
    system_prompt: str,
    client: "AsyncOpenAI",
    temperature: float = 0.0,
    concurrency: int = 200,
) -> list[dict]:
    """Run LLM extraction for a list of (idhash, prompt) pairs."""
    semaphore = asyncio.Semaphore(concurrency)

    async def _one(idhash: str, prompt: str):
        res = await _call_llm(client, prompt, semaphore, system_prompt, temperature)
        return {
            "idhash": idhash,
            "prompts": prompt,
            "llm_response": res["content"],
            "usage": res["usage"],
        }

    tasks = [_one(uid, p) for uid, p in prompts]
    return await asyncio.gather(*tasks)


def _format_explicit_persona(raw_content: str, datestr: str) -> str | None:
    """Parse explicit LLM response and format as persona_text.

    Mirrors production explicit_extract.py:76-80: extracts User_Info from JSON array.
    """
    import json as _json
    import re

    if not raw_content:
        return None

    # Try parsing as JSON
    try:
        persona_data = _json.loads(raw_content)
    except (_json.JSONDecodeError, TypeError):
        # Try regex fallback for JSON array
        arr_match = re.search(r"\[.*\]", raw_content, re.DOTALL)
        if arr_match:
            try:
                persona_data = _json.loads(arr_match.group(0))
            except (_json.JSONDecodeError, TypeError):
                return raw_content  # Fall back to raw content
        else:
            return raw_content

    if isinstance(persona_data, list):
        text_lines = []
        for p in persona_data:
            if isinstance(p, dict) and "User_Info" in p:
                text_lines.append(f"- {p['User_Info']}")
        if text_lines:
            return (
                f"사용자의 리뷰 기반 페르소나 분석 결과 ({datestr}):\n"
                + "\n".join(text_lines)
            )

    return raw_content


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_implicit_results(results: list[dict], output_dir: Path, datestr: str) -> None:
    """Write implicit JSONL for one day."""
    implicit_dir = output_dir / "implicit_data" / datestr
    implicit_dir.mkdir(parents=True, exist_ok=True)
    implicit_path = implicit_dir / f"persona_results_{datestr}.jsonl"

    with open(implicit_path, "w", encoding="utf-8") as f:
        for rec in results:
            line = {
                "idhash": rec["idhash"],
                "llm_response": rec["llm_response"],
                "prompts": rec["prompts"],
                "usage": rec["usage"],
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    logger.info("Saved %d implicit records to %s", len(results), implicit_path)


def _save_explicit_results(results: list[dict], output_dir: Path, datestr: str) -> None:
    """Write explicit JSONL for one day."""
    explicit_dir = output_dir / "explicit_data" / datestr
    explicit_dir.mkdir(parents=True, exist_ok=True)
    explicit_path = explicit_dir / f"persona_results_{datestr}.jsonl"

    with open(explicit_path, "w", encoding="utf-8") as f:
        for rec in results:
            persona_text = _format_explicit_persona(rec["llm_response"], datestr)
            if not persona_text:
                continue
            line = {
                "idhash": rec["idhash"],
                "persona_text": persona_text,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    logger.info("Saved explicit records to %s", explicit_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_for_experiment(
    spark: "SparkSession",
    user_ids: list[str],
    train_start: str,
    train_end: str,
    output_dir: Path,
    llm_client: "AsyncOpenAI",
    profile_path: str,
    demo_path: str | None = None,
) -> None:
    """Run per-day LLM extraction over the train window.

    Produces two independent extraction streams:
    - Implicit: click/save logs → prompt_click_gen → LLM (temp=0.0)
    - Explicit: review data → explicit prompts → LLM (temp=0.2)

    Args:
        spark: Active SparkSession with Hive support.
        user_ids: Target user ID hashes.
        train_start: First date inclusive (YYYY-MM-DD).
        train_end: Last date inclusive (YYYY-MM-DD).
        output_dir: Root directory for output files.
        llm_client: AsyncOpenAI-compatible client for LLM calls.
        profile_path: Parquet path for ``data.get_user_demo_data()``.
        demo_path: Parquet path for explicit extraction demo data. Defaults to profile_path.
    """
    import pandas as pd

    if demo_path is None:
        demo_path = profile_path

    # 1. Register target_ids — MUST happen before any data.py call
    user_ids_df = spark.createDataFrame([(uid,) for uid in user_ids], ["idhash"])
    user_ids_df.createOrReplaceTempView("target_ids")

    # 2. Load date-independent base data once
    # For implicit extraction
    poi_base_df = data.get_poi_base_data(spark)
    demo_df = data.get_user_demo_data(spark, profile_path)
    implicit_system_prompt = prompt_extract.generate_system_prompt()

    # For explicit extraction
    from explicit_preprocessing import get_poi_data
    explicit_poi_df = get_poi_data(spark)
    explicit_demo_raw = spark.read.parquet(demo_path)
    demo_id_col = "idno" if "idno" in explicit_demo_raw.columns else "idhash"
    explicit_demo_df = explicit_demo_raw.dropDuplicates([demo_id_col])
    if demo_id_col != "idhash":
        explicit_demo_df = explicit_demo_df.withColumnRenamed(demo_id_col, "idhash")

    explicit_prompts_cfg = _load_explicit_prompts()
    explicit_system_prompt = explicit_prompts_cfg["system_prompt"]
    explicit_user_template = explicit_prompts_cfg["user_prompt_template"]
    explicit_few_shot_rich = explicit_prompts_cfg["few_shot_examples"]["rich"]
    explicit_few_shot_sparse = explicit_prompts_cfg["few_shot_examples"]["sparse"]

    dates = _date_range(train_start, train_end)
    logger.info("Extracting %d days: %s -> %s", len(dates), train_start, train_end)

    # 3. Per-day extraction loop
    for datestr in dates:
        logger.info("Processing date %s", datestr)

        # ===== IMPLICIT EXTRACTION (click/save logs) =====
        click_df = data.get_poi_click_data(spark, datestr, datestr)
        save_df = data.get_poi_save_data(spark, datestr, datestr)

        processed_click = _preprocess_click(click_df, poi_base_df, demo_df)
        processed_save = _preprocess_save(save_df, poi_base_df)

        click_pd = processed_click.toPandas()
        save_pd = processed_save.toPandas()

        if not click_pd.empty:
            click_pd["etimestamp"] = pd.to_datetime(click_pd["etimestamp"])
            if not save_pd.empty:
                save_pd["etimestamp"] = pd.to_datetime(save_pd["etimestamp"])

            click_groups = click_pd.sort_values("etimestamp").groupby("idhash")
            save_groups = (
                save_pd.sort_values("etimestamp").groupby("idhash")
                if not save_pd.empty else None
            )

            implicit_prompts: list[tuple[str, str]] = []
            for idhash, click_group in click_groups:
                try:
                    save_group = save_groups.get_group(idhash) if save_groups is not None else None
                except KeyError:
                    save_group = None
                prompt_text = prompt_extract.prompt_click_gen(click_group, save_group)
                implicit_prompts.append((idhash, prompt_text))

            if implicit_prompts:
                logger.info("Date %s: %d users with clicks (implicit)", datestr, len(implicit_prompts))
                implicit_results = asyncio.run(
                    _extract_batch(implicit_prompts, implicit_system_prompt, llm_client, temperature=0.0)
                )
                _save_implicit_results(implicit_results, output_dir, datestr)
        else:
            logger.info("No clicks on %s, skipping implicit", datestr)

        # ===== EXPLICIT EXTRACTION (review data) =====
        review_df = _query_review_data(spark, datestr)
        review_with_poi = (
            review_df
            .join(explicit_poi_df, on="place_id", how="inner")
            .join(explicit_demo_df, on="idhash", how="left")
        )

        review_pd = review_with_poi.toPandas()

        if not review_pd.empty:
            # Add demo_text column using pandas equivalent
            # (get_demo_text_spark is a Spark expression, so we compute it in Spark first)
            from explicit_preprocessing import get_demo_text_spark
            from pyspark.sql import functions as F

            review_with_demo = review_with_poi.withColumn(
                "demo_text", F.coalesce(get_demo_text_spark(), F.lit("")),
            )
            review_pd = review_with_demo.toPandas()

            explicit_user_prompts = _build_explicit_prompts(
                review_pd, explicit_user_template,
                explicit_few_shot_rich, explicit_few_shot_sparse,
            )

            if explicit_user_prompts:
                logger.info(
                    "Date %s: %d user-chunks with reviews (explicit)",
                    datestr, len(explicit_user_prompts),
                )
                explicit_results = asyncio.run(
                    _extract_batch(
                        explicit_user_prompts, explicit_system_prompt, llm_client,
                        temperature=0.2,  # Matches production explicit_extract.py
                    )
                )
                _save_explicit_results(explicit_results, output_dir, datestr)
        else:
            logger.info("No reviews on %s, skipping explicit", datestr)
