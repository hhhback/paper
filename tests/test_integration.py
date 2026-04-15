"""E2E integration test for the experiment pipeline.

Verifies the full pipeline (data loading -> memory construction -> prediction ->
evaluation -> results CSV) using synthetic data and mocked LLM calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.config import POI_CATEGORIES
from experiments.evaluation.metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATES = ["2025-01-01", "2025-01-02", "2025-01-03"]
NUM_USERS = 10
FAKE_PREDICTIONS = POI_CATEGORIES[:10]  # deterministic prediction


def _user_id(i: int) -> str:
    return f"user_{i:04d}"


def _setup_synthetic_data(data_dir: Path, user_ids: list[str]) -> dict[str, list[str]]:
    """Create synthetic JSONL files, user_ids.json, and ground_truth.json.

    Returns the ground_truth dict (user_id -> list of categories) for later
    verification.
    """
    # Create per-day JSONL files for implicit and explicit data
    for datestr in DATES:
        for data_type, text_field in [
            ("implicit_data", "llm_response"),
            ("explicit_data", "persona_text"),
        ]:
            day_dir = data_dir / "output" / data_type / datestr
            day_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = day_dir / f"persona_results_{datestr}.jsonl"
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for uid in user_ids:
                    record = {
                        "idhash": uid,
                        text_field: f"Sample {data_type} text for {uid} on {datestr}.",
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # user_ids.json
    with open(data_dir / "user_ids.json", "w", encoding="utf-8") as f:
        json.dump(user_ids, f)

    # ground_truth.json: each user gets 3 categories from POI_CATEGORIES
    ground_truth: dict[str, list[str]] = {}
    for i, uid in enumerate(user_ids):
        start = i % len(POI_CATEGORIES)
        cats = []
        for j in range(3):
            cats.append(POI_CATEGORIES[(start + j) % len(POI_CATEGORIES)])
        ground_truth[uid] = cats

    with open(data_dir / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, ensure_ascii=False)

    return ground_truth


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _mock_predict_categories(_llm_client, _memory_context: str) -> list[str]:
    """Deterministic prediction: always returns first 10 POI categories."""
    return list(FAKE_PREDICTIONS)


# memory_agent mocking is handled by conftest.py — import runner after that.
from experiments.config import get_preset  # noqa: E402
from experiments.runner import ExperimentRunner  # noqa: E402


# ---------------------------------------------------------------------------
# Unit test: compute_all_metrics
# ---------------------------------------------------------------------------

class TestComputeAllMetrics:
    def test_perfect_prediction(self):
        gt = {"음식점", "카페/디저트", "쇼핑"}
        predicted = ["음식점", "카페/디저트", "쇼핑"] + POI_CATEGORIES[3:10]
        metrics = compute_all_metrics(predicted, gt)

        assert metrics["precision@1"] == 1.0
        assert metrics["precision@3"] == 1.0
        assert metrics["recall@5"] == 1.0
        assert metrics["hr@10"] == 1.0
        assert metrics["mrr"] == 1.0

    def test_no_hits(self):
        gt = {"음식점", "카페/디저트"}
        predicted = ["교통/모빌리티", "여행/숙박", "레저/스포츠",
                      "문화/예술", "자기계발", "반려동물",
                      "쇼핑", "의료", "미용", "미용"]
        metrics = compute_all_metrics(predicted, gt)

        assert metrics["precision@1"] == 0.0
        assert metrics["precision@3"] == 0.0
        assert metrics["recall@5"] == 0.0
        assert metrics["mrr"] == 0.0

    def test_partial_hits(self):
        gt = {"음식점", "카페/디저트", "쇼핑"}
        predicted = ["음식점", "교통/모빌리티", "쇼핑",
                      "여행/숙박", "카페/디저트",
                      "레저/스포츠", "문화/예술", "자기계발", "반려동물", "의료"]
        metrics = compute_all_metrics(predicted, gt)

        assert metrics["precision@1"] == 1.0       # 음식점 hit
        assert metrics["precision@3"] == pytest.approx(2 / 3)  # 2 of 3
        assert metrics["recall@5"] == 1.0           # all 3 in top 5
        assert metrics["hr@10"] == 1.0
        assert metrics["mrr"] == 1.0                # first hit at rank 1

    def test_all_metric_keys_present(self):
        metrics = compute_all_metrics(["음식점"], {"음식점"})
        expected_keys = {
            "precision@1", "precision@3", "precision@5",
            "recall@5", "ndcg@10", "hr@10", "mrr",
        }
        assert set(metrics.keys()) == expected_keys


# ---------------------------------------------------------------------------
# E2E integration test
# ---------------------------------------------------------------------------

METRIC_COLUMNS = [
    "user_id", "precision@1", "precision@3", "precision@5",
    "recall@5", "ndcg@10", "hr@10", "mrr",
]


class TestE2EIntegration:
    """Full pipeline test: data loading -> no_memory constructor -> prediction -> CSV."""

    def _run_pipeline(
        self, data_dir: Path, output_dir: Path, user_ids: list[str],
        ground_truth_raw: dict[str, list[str]], resume: bool = True,
    ) -> pd.DataFrame:
        """Run the no_memory variant through ExperimentRunner.run_all().

        memory_agent is mocked by conftest.py; predict_categories is patched
        here to avoid real LLM calls.
        """
        with patch(
            "experiments.runner.predict_categories",
            side_effect=_mock_predict_categories,
        ):
            config = get_preset("no_memory")
            runner = ExperimentRunner(config, data_dir, output_dir)
            users = {uid: {} for uid in user_ids}
            ground_truths = {uid: set(cats) for uid, cats in ground_truth_raw.items()}
            return runner.run_all(users, ground_truths, resume=resume)

    def test_full_pipeline_10_users(self, tmp_path: Path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        output_dir = tmp_path / "results"
        output_dir.mkdir()

        user_ids = [_user_id(i) for i in range(NUM_USERS)]
        gt_raw = _setup_synthetic_data(data_dir, user_ids)

        df = self._run_pipeline(data_dir, output_dir, user_ids, gt_raw)

        # Verify results.csv exists and has correct shape
        results_path = output_dir / "No-Memory" / "results.csv"
        assert results_path.exists(), f"results.csv not found at {results_path}"

        saved_df = pd.read_csv(results_path)
        assert len(saved_df) == NUM_USERS, (
            f"Expected {NUM_USERS} rows, got {len(saved_df)}"
        )

        # All metric columns present
        for col in METRIC_COLUMNS:
            assert col in saved_df.columns, f"Missing column: {col}"

        # All users present
        assert set(saved_df["user_id"]) == set(user_ids)

        # Metric values are valid floats in [0, 1]
        for col in METRIC_COLUMNS[1:]:  # skip user_id
            assert saved_df[col].notna().all(), f"NaN in column {col}"
            assert (saved_df[col] >= 0.0).all(), f"Negative value in {col}"
            assert (saved_df[col] <= 1.0).all(), f"Value > 1 in {col}"

        # Checkpoint file exists
        checkpoint_path = output_dir / "No-Memory" / "checkpoint.jsonl"
        assert checkpoint_path.exists(), "Checkpoint file not found"

    def test_checkpoint_resume(self, tmp_path: Path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        output_dir = tmp_path / "results"
        output_dir.mkdir()

        # Phase 1: run with 10 users
        original_ids = [_user_id(i) for i in range(NUM_USERS)]
        gt_raw = _setup_synthetic_data(data_dir, original_ids)

        self._run_pipeline(data_dir, output_dir, original_ids, gt_raw)

        # Verify phase 1 completed
        checkpoint_path = output_dir / "No-Memory" / "checkpoint.jsonl"
        results_path = output_dir / "No-Memory" / "results.csv"
        assert checkpoint_path.exists()
        assert results_path.exists()

        # Delete results.csv but keep checkpoint
        results_path.unlink()

        # Phase 2: add 5 more users, re-run with resume=True
        extra_ids = [_user_id(NUM_USERS + i) for i in range(5)]
        all_ids = original_ids + extra_ids

        # Regenerate data files to include new users
        gt_raw = _setup_synthetic_data(data_dir, all_ids)

        # Track how many times predict_categories is called
        called_for: list[str] = []

        def _tracking_predict(llm_client, memory_context: str) -> list[str]:
            called_for.append("call")
            return _mock_predict_categories(llm_client, memory_context)

        with patch(
            "experiments.runner.predict_categories",
            side_effect=_tracking_predict,
        ):
            config = get_preset("no_memory")
            runner = ExperimentRunner(config, data_dir, output_dir)
            users = {uid: {} for uid in all_ids}
            ground_truths = {uid: set(cats) for uid, cats in gt_raw.items()}
            df = runner.run_all(users, ground_truths, resume=True)

        # Only 5 new users should have been processed
        assert len(called_for) == 5, (
            f"Expected 5 predict calls (new users only), got {len(called_for)}"
        )

        # Final results should contain all 15 users
        assert len(df) == NUM_USERS + 5

        # Re-saved CSV should also have 15
        saved_df = pd.read_csv(output_dir / "No-Memory" / "results.csv")
        assert len(saved_df) == NUM_USERS + 5
        assert set(saved_df["user_id"]) == set(all_ids)
