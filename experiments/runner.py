"""ExperimentRunner — runs experiment variants across users with checkpoint/resume."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

_PROD_SRC = os.environ.get("PROD_SRC", "/home1/irteam/work/fs-feature/src/user/memory")
if _PROD_SRC not in sys.path:
    sys.path.insert(0, _PROD_SRC)

from memory_agent.llm import LLMClient

from experiments.config import ExperimentConfig
from experiments.constructors.base import AbstractConstructor
from experiments.constructors.flat_memory import FlatMemoryConstructor
from experiments.constructors.fixed_hierarchy import FixedHierarchyConstructor
from experiments.constructors.full_context import FullContextConstructor
from experiments.constructors.no_memory import NoMemoryConstructor
from experiments.constructors.ours import OursConstructor
from experiments.dataset import load_user_daily_texts
from experiments.evaluation.metrics import compute_all_metrics
from experiments.evaluation.predictor import predict_categories

logger = logging.getLogger(__name__)


def _create_constructor(
    config: ExperimentConfig, llm_client: LLMClient,
) -> AbstractConstructor:
    """Create the appropriate constructor based on config.constructor_type."""
    factories = {
        "no_memory": lambda: NoMemoryConstructor(config),
        "full_context": lambda: FullContextConstructor(config),
        "flat_memory": lambda: FlatMemoryConstructor(config, llm_client),
        "fixed_hierarchy": lambda: FixedHierarchyConstructor(config),
        "ours": lambda: OursConstructor(config),
    }
    factory = factories.get(config.constructor_type)
    if factory is None:
        raise ValueError(
            f"Unknown constructor_type: {config.constructor_type!r}. "
            f"Available: {list(factories.keys())}"
        )
    return factory()


class ExperimentRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        data_dir: Path,
        output_dir: Path,
    ):
        self.config = config
        self.data_dir = data_dir
        self.output_dir = output_dir

        self._llm_client = LLMClient()
        self._constructor = _create_constructor(config, self._llm_client)

        self._variant_dir = output_dir / config.name
        self._checkpoint_path = self._variant_dir / "checkpoint.jsonl"
        self._results_path = self._variant_dir / "results.csv"

    # ------------------------------------------------------------------
    # Single user
    # ------------------------------------------------------------------

    def run_single_user(
        self,
        user_id: str,
        daily_texts: dict[str, str],
        ground_truth: set[str],
    ) -> dict:
        memory_dir = self._variant_dir / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)

        self._constructor.build_memory(user_id, daily_texts, memory_dir)
        memory_context = self._constructor.get_memory_context(user_id, memory_dir)
        predicted = predict_categories(self._llm_client, memory_context)
        metrics = compute_all_metrics(predicted, ground_truth)
        metrics["user_id"] = user_id
        return metrics

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _load_checkpoint(self) -> list[dict]:
        """Load completed records from checkpoint JSONL."""
        records: list[dict] = []
        if not self._checkpoint_path.exists():
            return records
        with open(self._checkpoint_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _append_checkpoint(self, record: dict) -> None:
        """Append a single record to the checkpoint JSONL file."""
        self._variant_dir.mkdir(parents=True, exist_ok=True)
        with open(self._checkpoint_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Run all users
    # ------------------------------------------------------------------

    def run_all(
        self,
        users: dict[str, dict[str, str]],
        ground_truths: dict[str, set[str]],
        resume: bool = True,
    ) -> pd.DataFrame:
        self._variant_dir.mkdir(parents=True, exist_ok=True)

        # Load checkpoint
        completed_records = self._load_checkpoint() if resume else []
        completed_ids = {r["user_id"] for r in completed_records}

        if completed_ids:
            logger.info(
                "[%s] Resuming: %d users already completed, skipping.",
                self.config.name, len(completed_ids),
            )

        pending_ids = [uid for uid in users if uid not in completed_ids]
        total = len(users)
        done = len(completed_ids)

        for uid in pending_ids:
            done += 1
            logger.info(
                "[%s] Running user %s (%d/%d)",
                self.config.name, uid, done, total,
            )
            daily_texts = load_user_daily_texts(
                self.data_dir, uid,
                implicit_only=self.config.implicit_only,
                explicit_only=self.config.explicit_only,
            )
            gt = ground_truths.get(uid, set())
            metrics = self.run_single_user(uid, daily_texts, gt)
            self._append_checkpoint(metrics)
            completed_records.append(metrics)

        # Save final results CSV
        df = pd.DataFrame(completed_records)
        column_order = [
            "user_id", "precision@1", "precision@3", "precision@5",
            "recall@5", "ndcg@10", "hr@10", "mrr",
        ]
        df = df[[c for c in column_order if c in df.columns]]
        df.to_csv(self._results_path, index=False)
        logger.info(
            "[%s] Results saved to %s (%d users)",
            self.config.name, self._results_path, len(df),
        )
        return df
