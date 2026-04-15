"""Wrapper that runs explicit_extract.py per-day over a preprocessed parquet."""

import argparse
import logging
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

EXTRACT_SCRIPT = "/code/src/user/memory/explicit_extract.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run explicit_extract.py for each day in [train_start, train_end]."
    )
    parser.add_argument("--preprocess-output", required=True, help="Path to preprocessed parquet file")
    parser.add_argument("--output-dir", required=True, help="Base output directory")
    parser.add_argument("--train-start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--train-end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--prompt-path", required=True, help="Path to prompt YAML")
    parser.add_argument("--model-name", required=True, help="LLM model name")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for extraction")
    return parser.parse_args()


def date_range(start: str, end: str):
    current = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date()
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def main():
    args = parse_args()

    logger.info("Reading preprocessed parquet: %s", args.preprocess_output)
    df = pd.read_parquet(args.preprocess_output)
    df["_date"] = pd.to_datetime(df["write_date_time"]).dt.date
    logger.info("Loaded %d rows spanning %s to %s", len(df), df["_date"].min(), df["_date"].max())

    for day in date_range(args.train_start, args.train_end):
        date_str = day.strftime("%Y-%m-%d")
        day_df = df[df["_date"] == day]

        if day_df.empty:
            logger.warning("No rows for %s, skipping", date_str)
            continue

        logger.info("Processing %s: %d rows", date_str, len(day_df))
        temp_dir = tempfile.mkdtemp()
        try:
            temp_path = os.path.join(temp_dir, date_str)
            os.makedirs(temp_path, exist_ok=True)
            day_df.drop(columns=["_date"]).to_parquet(
                os.path.join(temp_path, "part-00000.parquet"), index=False
            )

            output_path = os.path.join(args.output_dir, "explicit_data", date_str)
            os.makedirs(output_path, exist_ok=True)

            subprocess.check_call([
                "python3", EXTRACT_SCRIPT,
                "--output_base", temp_path,
                "--output_dir", output_path,
                "--datestr", date_str,
                "--prompt-path", args.prompt_path,
                "--batch_size", str(args.batch_size),
                "--model_name", args.model_name,
            ])
            logger.info("Completed %s", date_str)
        finally:
            shutil.rmtree(temp_dir)

    logger.info("All dates processed")


if __name__ == "__main__":
    main()
