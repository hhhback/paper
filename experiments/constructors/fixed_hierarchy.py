from pathlib import Path

from experiments.config import ExperimentConfig
from experiments.constructors.base import AbstractConstructor
from experiments.constructors.ours import OursConstructor


class FixedHierarchyConstructor(AbstractConstructor):
    def __init__(self, config: ExperimentConfig, llm_client=None):
        self._delegate = OursConstructor(config)

    def build_memory(self, user_id: str, daily_texts: dict[str, str], memory_dir: Path) -> None:
        self._delegate.build_memory(user_id, daily_texts, memory_dir)

    def get_memory_context(self, user_id: str, memory_dir: Path) -> str:
        return self._delegate.get_memory_context(user_id, memory_dir)
