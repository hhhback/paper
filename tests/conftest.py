"""Pytest configuration: mock production dependencies before any test imports."""

import sys
from unittest.mock import MagicMock

# memory_agent is a production package that requires OPENAI_API_KEY at import
# time and lives outside this repo. Mock the entire package tree so that
# experiments.runner (and its transitive imports) can be loaded without the
# production environment.

_MEMORY_AGENT_MODULES = [
    "memory_agent",
    "memory_agent.agent",
    "memory_agent.config",
    "memory_agent.llm",
    "memory_agent.storage",
]

for mod_name in _MEMORY_AGENT_MODULES:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()
