#!/usr/bin/env python3
"""CLI entry point for running a single experiment variant."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.config import get_preset
from experiments.runner import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an experiment preset on user data.",
    )
    parser.add_argument(
        "--preset", required=True,
        help='Preset name (e.g., "ours", "no_memory", "ablation_dynamic")',
    )
    parser.add_argument(
        "--data-dir", required=True, type=Path,
        help="Path to Stage A output (contains output/, user_ids.json, ground_truth.json)",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Path to write results",
    )
    parser.add_argument(
        "--resume", action="store_true", default=True, dest="resume",
        help="Resume from checkpoint (default)",
    )
    parser.add_argument(
        "--no-resume", action="store_false", dest="resume",
        help="Start fresh, ignoring any existing checkpoint",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = get_preset(args.preset)

    # Load user IDs
    user_ids_path = args.data_dir / "user_ids.json"
    with open(user_ids_path, encoding="utf-8") as f:
        user_ids: list[str] = json.load(f)

    # Load ground truth: {user_id: [category, ...]} -> {user_id: set}
    gt_path = args.data_dir / "ground_truth.json"
    with open(gt_path, encoding="utf-8") as f:
        raw_gt: dict[str, list[str]] = json.load(f)
    ground_truths = {uid: set(cats) for uid, cats in raw_gt.items()}

    # The runner loads daily_texts per user internally via load_user_daily_texts,
    # but run_all also accepts a pre-loaded users dict. We pass an empty dict
    # since the runner loads texts on-the-fly in its loop.
    # However, run_all iterates over users.keys() to determine the user set,
    # so we provide a dict with user_ids as keys.
    users = {uid: {} for uid in user_ids}

    runner = ExperimentRunner(config, args.data_dir, args.output_dir)
    df = runner.run_all(users, ground_truths, resume=args.resume)

    logging.getLogger(__name__).info(
        "Finished preset=%s, %d users processed.", args.preset, len(df),
    )


if __name__ == "__main__":
    main()
