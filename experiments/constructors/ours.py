"""OursConstructor — wraps ExperimentMemoryAgent with ablation flags.

ExperimentMemoryAgent subclasses the production MemoryAgent, overriding
methods to enable/disable specific code paths based on ExperimentConfig flags.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import tiktoken

# Add production codebase to sys.path so we can import memory_agent
_PROD_SRC = os.environ.get("PROD_SRC", "/home1/irteam/work/fs-feature/src/user/memory")
if _PROD_SRC not in sys.path:
    sys.path.insert(0, _PROD_SRC)

from memory_agent.agent import MemoryAgent
from memory_agent.storage import (
    ensure_user_dirs,
    list_paths,
    load_all_notes,
    load_hierarchy,
    normalize_path,
    save_all_notes,
    save_hierarchy,
)

from experiments.config import ExperimentConfig
from experiments.constructors.base import AbstractConstructor

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  ExperimentMemoryAgent — production MemoryAgent + ablation overrides
# --------------------------------------------------------------------------- #

class ExperimentMemoryAgent(MemoryAgent):
    """MemoryAgent subclass that disables code paths based on ablation flags."""

    def __init__(
        self,
        config: ExperimentConfig,
        memory_dir: Path | None = None,
        llm=None,
    ):
        super().__init__(memory_dir=memory_dir, llm=llm)
        self.config = config

    # -- preference filtering (implicit_only / explicit_only) --------------- #

    def _filter_preference_profile(self, preference_profile: dict) -> dict:
        """Zero out preference types based on implicit_only / explicit_only."""
        if self.config.implicit_only:
            preference_profile = {
                **preference_profile,
                "explicit_preferences": [],
            }
        if self.config.explicit_only:
            preference_profile = {
                **preference_profile,
                "implicit_preferences": [],
            }
        return preference_profile

    # -- dynamic routing filter --------------------------------------------- #

    def _filter_assignments_to_base(self, assignments: list[dict]) -> list[dict]:
        """Keep only assignments whose target_path is in _base_paths()."""
        allowed = set(self._base_paths())
        filtered = []
        for a in assignments:
            raw_path = a.get("target_path")
            if not raw_path:
                continue
            if normalize_path(raw_path) in allowed:
                filtered.append(a)
        return filtered

    # -- forward pass overrides --------------------------------------------- #

    def _forward_pass(self, user_id: str, interaction_text: str, label: str) -> None:
        """Shared logic for initialize_user_memory and update_interaction.

        Mirrors the production flow exactly, inserting ablation filters at the
        appropriate points:
          - preference profile filtering (implicit_only / explicit_only)
          - assignment filtering (disable_dynamic_routing)
        """
        logger.info("%s: start user_id=%s", label, user_id)
        user_dir = ensure_user_dirs(self.memory_dir, user_id)
        hierarchy = load_hierarchy(user_dir)
        notes = load_all_notes(user_dir, user_id)

        # Step 1: extract preferences
        preference_profile = self.llm.extract_preferences(interaction_text)

        # --- ablation: implicit_only / explicit_only ---
        preference_profile = self._filter_preference_profile(preference_profile)

        if not self._has_preference_profile(preference_profile):
            save_hierarchy(user_dir, hierarchy)
            return

        base_paths = self._base_paths()
        existing_paths = list_paths(hierarchy, include_root=True)

        # Step 2: match paths
        assignments = self.llm.match_paths(preference_profile, base_paths, existing_paths)

        # --- ablation: disable_dynamic_routing ---
        if self.config.disable_dynamic_routing:
            assignments = self._filter_assignments_to_base(assignments)

        # Step 3: apply assignments
        self._apply_assignments(
            assignments=assignments,
            hierarchy=hierarchy,
            notes=notes,
        )

        # Step 4: save
        save_all_notes(user_dir, user_id, notes)
        save_hierarchy(user_dir, hierarchy)
        logger.info("%s: done user_id=%s", label, user_id)

    def initialize_user_memory(self, user_id: str, interaction_text: str) -> None:
        self._forward_pass(user_id, interaction_text, "init")

    def update_interaction(self, user_id: str, interaction_text: str) -> None:
        self._forward_pass(user_id, interaction_text, "update")

    # -- backward pass overrides -------------------------------------------- #

    def backward_cleanup(self, user_id: str) -> None:
        if self.config.disable_backward:
            return
        super().backward_cleanup(user_id)

    def _lift_common_traits(self, hierarchy, notes) -> None:
        if self.config.disable_elevation:
            return
        super()._lift_common_traits(hierarchy, notes)


# --------------------------------------------------------------------------- #
#  Memory context formatting
# --------------------------------------------------------------------------- #

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _depth_first_paths(notes: dict[str, object]) -> list[str]:
    """Return note paths in hierarchy depth-first order.

    Builds a tree from path segments and traverses depth-first,
    sorting children alphabetically at each level.
    """
    tree: dict = {}
    for path in notes:
        segments = [s.strip() for s in path.split(" > ") if s.strip()]
        node = tree
        for seg in segments:
            node = node.setdefault(seg, {})

    result: list[str] = []

    def _walk(node: dict, prefix: list[str]) -> None:
        path_str = " > ".join(prefix)
        if path_str in notes:
            result.append(path_str)
        for child_name in sorted(node.keys()):
            _walk(node[child_name], prefix + [child_name])

    for root_name in sorted(tree.keys()):
        _walk(tree[root_name], [root_name])

    return result


def _format_note(note) -> str:
    """Format a single Note into the context string."""
    keywords = ", ".join(note.keywords) if note.keywords else ""
    user_info = ", ".join(note.user_info) if note.user_info else ""
    return (
        f"[{note.path}]\n"
        f"요약: {note.summary}\n"
        f"사용자 정보: {user_info}\n"
        f"키워드: {keywords}\n"
    )


def _truncate_to_budget(ordered_notes: list, budget: int) -> str:
    """Concatenate notes depth-first, truncating deepest leaves first.

    Strategy:
    1. Format all notes and compute token counts.
    2. If total fits within budget, return as-is.
    3. Otherwise, remove notes from deepest to shallowest until the
       remainder fits. Root-level notes (depth 1) are never removed.
    """
    if not ordered_notes:
        return ""

    entries = []
    for note in ordered_notes:
        text = _format_note(note)
        depth = note.path.count(" > ") + 1
        tokens = len(_TOKENIZER.encode(text))
        entries.append({"note": note, "text": text, "depth": depth, "tokens": tokens})

    total_tokens = sum(e["tokens"] for e in entries)
    if total_tokens <= budget:
        return "\n".join(e["text"] for e in entries)

    # Sort candidates for removal: deepest first, then reverse DFS order
    # (later items in the DFS order are removed first among same-depth nodes)
    removable = [
        (i, e) for i, e in enumerate(entries) if e["depth"] > 1
    ]
    # Stable sort: deepest first, then highest index first (reverse DFS)
    removable.sort(key=lambda x: (-x[1]["depth"], -x[0]))

    removed = set()
    for idx, entry in removable:
        if total_tokens <= budget:
            break
        removed.add(idx)
        total_tokens -= entry["tokens"]

    return "\n".join(
        e["text"] for i, e in enumerate(entries) if i not in removed
    )


# --------------------------------------------------------------------------- #
#  OursConstructor
# --------------------------------------------------------------------------- #

class OursConstructor(AbstractConstructor):
    """Constructor that uses ExperimentMemoryAgent with ablation flags."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def build_memory(self, user_id: str, daily_texts: dict[str, str], memory_dir: Path) -> None:
        import shutil
        user_path = memory_dir / "users" / user_id
        if user_path.exists() and any(user_path.iterdir()):
            logger.warning("Cleaning stale memory dir: %s", user_path)
            shutil.rmtree(user_path)

        agent = ExperimentMemoryAgent(
            config=self.config,
            memory_dir=memory_dir,
        )

        sorted_dates = sorted(daily_texts.keys())
        for i, date in enumerate(sorted_dates):
            text = daily_texts[date]
            if i == 0:
                agent.initialize_user_memory(user_id, text)
            else:
                agent.update_interaction(user_id, text)
                if not self.config.disable_backward:
                    agent.backward_cleanup(user_id)

    def get_memory_context(self, user_id: str, memory_dir: Path) -> str:
        user_dir = memory_dir / "users" / user_id
        notes = load_all_notes(user_dir, user_id)
        if not notes:
            return ""

        ordered_paths = _depth_first_paths(notes)
        ordered_notes = [notes[p] for p in ordered_paths if p in notes]

        return _truncate_to_budget(ordered_notes, self.config.context_budget)
