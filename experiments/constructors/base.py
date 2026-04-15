from abc import ABC, abstractmethod
from pathlib import Path


class AbstractConstructor(ABC):
    @abstractmethod
    def build_memory(self, user_id: str, daily_texts: dict[str, str], memory_dir: Path) -> None:
        """Build memory from per-day extraction texts.

        Args:
            daily_texts: {datestr: extraction_text} ordered by date.
                         Baselines may merge internally; "ours" iterates day-by-day.
        """
        ...

    @abstractmethod
    def get_memory_context(self, user_id: str, memory_dir: Path) -> str:
        """Return formatted memory context string for prediction."""
        ...
