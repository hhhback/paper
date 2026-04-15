#!/usr/bin/env python3
"""Test pipeline: run all (or selected) experiment variants end-to-end.

Creates synthetic data or uses existing Stage A output, runs each variant
through memory construction → prediction → evaluation, and produces a
comparison table with statistical tests.

Usage examples::

    # Quick smoke test: 5 synthetic users, all 10 variants
    python run_test_pipeline.py --mode synthetic --n-users 5

    # Run specific variants on real data
    python run_test_pipeline.py --mode real --data-dir /path/to/stage_a_output \\
        --variants ours no_memory ablation_dynamic

    # Run only baselines
    python run_test_pipeline.py --mode real --data-dir /path/to/stage_a_output \\
        --group baselines

    # Run only ablations
    python run_test_pipeline.py --mode real --data-dir /path/to/stage_a_output \\
        --group ablations
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Mock memory_agent if OPENAI_API_KEY is not set (production dependency).
# This allows running the pipeline in test/synthetic mode without the
# production LLM environment.
if not os.environ.get("OPENAI_API_KEY"):
    for _mod in [
        "memory_agent", "memory_agent.agent", "memory_agent.config",
        "memory_agent.llm", "memory_agent.storage", "memory_agent.models",
    ]:
        if _mod not in sys.modules:
            sys.modules[_mod] = MagicMock()

from experiments.config import (
    ABLATION_NAMES,
    BASELINE_NAMES,
    POI_CATEGORIES,
    PRESETS,
    get_preset,
)
from experiments.evaluation.metrics import compute_all_metrics
from experiments.runner import ExperimentRunner

logger = logging.getLogger(__name__)

ALL_VARIANTS = BASELINE_NAMES + ABLATION_NAMES


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def _create_synthetic_data(data_dir: Path, n_users: int, n_days: int = 5) -> tuple[list[str], dict]:
    """Generate synthetic JSONL files, user_ids.json, ground_truth.json.

    Produces plausible Korean text for both implicit and explicit streams
    so that all constructor variants have data to work with.
    """
    from datetime import datetime, timedelta

    user_ids = [f"test_user_{i:04d}" for i in range(n_users)]
    base_date = datetime(2025, 3, 1)
    dates = [(base_date + timedelta(days=d)).strftime("%Y-%m-%d") for d in range(n_days)]

    # Implicit data (click-log style extraction)
    implicit_templates = [
        "이 사용자는 {cat} 카테고리의 장소를 주로 방문합니다. 주중에는 강남 지역에서 활동하고 주말에는 분당 일대에서 여가를 즐깁니다.",
        "사용자의 클릭 로그를 분석한 결과, {cat} 관련 장소에 대한 관심이 높으며, 맛집 탐방과 카페 방문을 즐기는 패턴이 관찰됩니다.",
        "최근 방문 기록에서 {cat} 카테고리 비중이 높고, 리뷰 평점이 높은 장소를 선호하는 경향이 있습니다.",
    ]

    # Explicit data (review-based persona)
    explicit_templates = [
        "사용자의 리뷰 기반 페르소나 분석 결과 (2025-03):\n- {cat} 관련 장소 방문 시 분위기와 서비스를 중시함\n- 주차 편의성을 필수적으로 고려하는 차량 이동 중심 생활\n- 가족 동반 외식 빈도가 높음",
        "사용자의 리뷰 기반 페르소나 분석 결과 (2025-03):\n- {cat} 카테고리에서 맛과 가성비를 중시하는 식성\n- 주말 오후 시간대 활동 비중이 높음\n- 건강 관리에 관심이 있어 운동 관련 장소도 방문",
        "사용자의 리뷰 기반 페르소나 분석 결과 (2025-03):\n- {cat} 분야에 높은 관심을 보이며 신규 장소 탐방을 즐김\n- 인스타그램 감성의 공간을 선호\n- 소규모 모임 장소로 적합한 곳을 자주 검색",
    ]

    output_dir = data_dir / "output"

    for date_idx, datestr in enumerate(dates):
        imp_dir = output_dir / "implicit_data" / datestr
        exp_dir = output_dir / "explicit_data" / datestr
        imp_dir.mkdir(parents=True, exist_ok=True)
        exp_dir.mkdir(parents=True, exist_ok=True)

        imp_path = imp_dir / f"persona_results_{datestr}.jsonl"
        exp_path = exp_dir / f"persona_results_{datestr}.jsonl"

        with (
            open(imp_path, "w", encoding="utf-8") as imp_f,
            open(exp_path, "w", encoding="utf-8") as exp_f,
        ):
            for i, uid in enumerate(user_ids):
                cat = POI_CATEGORIES[(i + date_idx) % len(POI_CATEGORIES)]
                template_idx = (i + date_idx) % len(implicit_templates)

                imp_line = {
                    "idhash": uid,
                    "llm_response": implicit_templates[template_idx].format(cat=cat),
                }
                imp_f.write(json.dumps(imp_line, ensure_ascii=False) + "\n")

                exp_line = {
                    "idhash": uid,
                    "persona_text": explicit_templates[template_idx].format(cat=cat),
                }
                exp_f.write(json.dumps(exp_line, ensure_ascii=False) + "\n")

    # Ground truth: each user visited 3-4 categories
    ground_truth: dict[str, list[str]] = {}
    for i, uid in enumerate(user_ids):
        start = i % len(POI_CATEGORIES)
        cats = [POI_CATEGORIES[(start + j) % len(POI_CATEGORIES)] for j in range(3)]
        ground_truth[uid] = cats

    # Save metadata
    with open(data_dir / "user_ids.json", "w", encoding="utf-8") as f:
        json.dump(user_ids, f, ensure_ascii=False)

    with open(data_dir / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)

    logger.info("Created synthetic data: %d users × %d days", n_users, n_days)
    return user_ids, ground_truth


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_variant(
    preset_name: str,
    data_dir: Path,
    output_dir: Path,
    user_ids: list[str],
    ground_truths: dict[str, set[str]],
    mock_llm: bool = False,
    resume: bool = True,
) -> pd.DataFrame | None:
    """Run a single experiment variant and return its results DataFrame."""
    config = get_preset(preset_name)
    logger.info("=" * 60)
    logger.info("Running variant: %s (%s)", config.name, preset_name)
    logger.info("=" * 60)

    users = {uid: {} for uid in user_ids}
    start_time = time.time()

    try:
        if mock_llm:
            # Mock predict_categories for testing without LLM
            def _mock_predict(_llm_client, _memory_context: str) -> list[str]:
                return list(POI_CATEGORIES[:10])

            with patch("experiments.runner.predict_categories", side_effect=_mock_predict):
                runner = ExperimentRunner(config, data_dir, output_dir)
                df = runner.run_all(users, ground_truths, resume=resume)
        else:
            runner = ExperimentRunner(config, data_dir, output_dir)
            df = runner.run_all(users, ground_truths, resume=resume)

        elapsed = time.time() - start_time
        logger.info(
            "Variant %s completed: %d users in %.1fs",
            config.name, len(df), elapsed,
        )
        return df

    except Exception:
        logger.exception("Variant %s FAILED", config.name)
        return None


def run_pipeline(
    variant_names: list[str],
    data_dir: Path,
    output_dir: Path,
    mock_llm: bool = False,
    resume: bool = True,
) -> pd.DataFrame:
    """Run multiple variants and produce a comparison summary."""
    # Load user IDs and ground truth
    with open(data_dir / "user_ids.json", encoding="utf-8") as f:
        user_ids: list[str] = json.load(f)

    with open(data_dir / "ground_truth.json", encoding="utf-8") as f:
        raw_gt: dict[str, list[str]] = json.load(f)
    ground_truths = {uid: set(cats) for uid, cats in raw_gt.items()}

    logger.info("Pipeline: %d users, %d variants", len(user_ids), len(variant_names))

    results_summary: list[dict] = []

    for preset_name in variant_names:
        df = run_variant(
            preset_name, data_dir, output_dir, user_ids, ground_truths,
            mock_llm=mock_llm, resume=resume,
        )
        if df is None:
            continue

        config = get_preset(preset_name)
        metric_cols = [c for c in df.columns if c != "user_id"]
        row = {"variant": config.name, "preset": preset_name, "n_users": len(df)}
        for col in metric_cols:
            row[f"{col}_mean"] = df[col].mean()
            row[f"{col}_std"] = df[col].std()
        results_summary.append(row)

    summary_df = pd.DataFrame(results_summary)

    # Print comparison table
    if not summary_df.empty:
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        pd.set_option("display.float_format", "{:.4f}".format)

        # Show mean metrics per variant
        display_cols = ["variant", "n_users"]
        for metric in ["precision@1", "precision@3", "precision@5", "recall@5", "ndcg@10", "hr@10", "mrr"]:
            mean_col = f"{metric}_mean"
            if mean_col in summary_df.columns:
                display_cols.append(mean_col)

        print(summary_df[display_cols].to_string(index=False))

        # Save summary
        summary_path = output_dir / "pipeline_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")

    # Run statistical analysis if "ours" was included
    ours_dir = output_dir / "Ours"
    if ours_dir.exists() and (ours_dir / "results.csv").exists() and len(variant_names) > 1:
        print("\n" + "-" * 80)
        print("STATISTICAL ANALYSIS (baseline=Ours)")
        print("-" * 80)
        try:
            from experiments.scripts.analyze_results import compare_variants
            stat_df = compare_variants(output_dir, baseline_name="Ours")
            if not stat_df.empty:
                print(stat_df.to_string(index=False))
                stat_path = output_dir / "comparison.csv"
                stat_df.to_csv(stat_path, index=False)
                print(f"\nStatistical comparison saved to {stat_path}")
        except Exception:
            logger.exception("Statistical analysis failed")

    return summary_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test pipeline: run experiment variants and compare results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Variant groups:
  baselines   — no_memory, full_context, flat_memory, fixed_hierarchy, ours
  ablations   — ablation_dynamic, ablation_elevation, ablation_backward,
                ablation_no_implicit, ablation_no_explicit
  all         — all 10 variants

Individual variants:
  %(all_variants)s
        """ % {"all_variants": ", ".join(ALL_VARIANTS)},
    )

    parser.add_argument(
        "--mode", choices=["synthetic", "real"], default="synthetic",
        help="Data mode: 'synthetic' generates fake data, 'real' uses Stage A output (default: synthetic)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Path to Stage A output (required for --mode real)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Path to write results (default: auto-generated in /tmp)",
    )
    parser.add_argument(
        "--n-users", type=int, default=5,
        help="Number of synthetic users (only for --mode synthetic, default: 5)",
    )
    parser.add_argument(
        "--n-days", type=int, default=5,
        help="Number of synthetic days (only for --mode synthetic, default: 5)",
    )

    # Variant selection (mutually exclusive group + individual)
    variant_group = parser.add_mutually_exclusive_group()
    variant_group.add_argument(
        "--group", choices=["baselines", "ablations", "all"], default=None,
        help="Run a predefined group of variants",
    )
    variant_group.add_argument(
        "--variants", nargs="+", choices=ALL_VARIANTS, default=None,
        help="Run specific variants by preset name",
    )

    parser.add_argument(
        "--mock-llm", action="store_true",
        help="Mock LLM calls (always predicts first 10 categories). Useful for testing pipeline mechanics.",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh, ignoring any existing checkpoints",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Determine variants to run
    if args.variants:
        variant_names = args.variants
    elif args.group == "baselines":
        variant_names = BASELINE_NAMES
    elif args.group == "ablations":
        variant_names = ABLATION_NAMES
    elif args.group == "all":
        variant_names = ALL_VARIANTS
    else:
        variant_names = ALL_VARIANTS  # default: run all

    # Determine data directory
    if args.mode == "synthetic":
        import tempfile
        base_tmp = Path(tempfile.mkdtemp(prefix="exp_pipeline_"))
        data_dir = base_tmp / "data"
        data_dir.mkdir()
        output_dir = args.output_dir or (base_tmp / "results")
        output_dir.mkdir(parents=True, exist_ok=True)

        _create_synthetic_data(data_dir, args.n_users, args.n_days)
        logger.info("Synthetic data created at %s", data_dir)

        # For synthetic mode, always mock LLM unless user explicitly has API key
        if not args.mock_llm and "API_KEY" not in __import__("os").environ:
            logger.info("No API_KEY found, enabling --mock-llm for synthetic mode")
            args.mock_llm = True

    elif args.mode == "real":
        if args.data_dir is None:
            parser.error("--data-dir is required for --mode real")
        data_dir = args.data_dir
        output_dir = args.output_dir or (data_dir.parent / "results")
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Data dir: %s", data_dir)
    logger.info("Output dir: %s", output_dir)
    logger.info("Variants: %s", variant_names)
    logger.info("Mock LLM: %s", args.mock_llm)

    summary = run_pipeline(
        variant_names=variant_names,
        data_dir=data_dir,
        output_dir=output_dir,
        mock_llm=args.mock_llm,
        resume=not args.no_resume,
    )

    if summary.empty:
        logger.warning("No variants completed successfully")
        sys.exit(1)

    print(f"\nPipeline complete. Results in: {output_dir}")


if __name__ == "__main__":
    main()
