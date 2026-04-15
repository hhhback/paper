from pathlib import Path

from experiments.config import ExperimentConfig
from experiments.constructors.base import AbstractConstructor


class NoMemoryConstructor(AbstractConstructor):
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def build_memory(self, user_id: str, daily_texts: dict[str, str], memory_dir: Path) -> None:
        pass

    def get_memory_context(self, user_id: str, memory_dir: Path) -> str:
        return ""
