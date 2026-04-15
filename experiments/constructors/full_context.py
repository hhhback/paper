from pathlib import Path

import tiktoken

from experiments.config import ExperimentConfig
from experiments.constructors.base import AbstractConstructor


class FullContextConstructor(AbstractConstructor):
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._enc = tiktoken.get_encoding("cl100k_base")

    def build_memory(self, user_id: str, daily_texts: dict[str, str], memory_dir: Path) -> None:
        merged = "\n".join(daily_texts[d] for d in sorted(daily_texts))

        tokens = self._enc.encode(merged)
        if len(tokens) > self.config.context_budget:
            merged = self._enc.decode(tokens[: self.config.context_budget])

        user_dir = memory_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        (user_dir / "context.txt").write_text(merged, encoding="utf-8")

    def get_memory_context(self, user_id: str, memory_dir: Path) -> str:
        path = memory_dir / user_id / "context.txt"
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")
