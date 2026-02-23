"""
AI Session Tools - Modern Python library for analyzing Claude Code session data.

A composable, well-designed library with thin CLI layer for discovering, extracting,
and analyzing Claude Code session files.

Copyright (c) 2026 Andrew Hundt
Licensed under the Apache License, Version 2.0

Example usage as library:
    from ai_session_tools import SessionRecoveryEngine, FilterSpec

    engine = SessionRecoveryEngine(projects_dir, recovery_dir)
    filters = FilterSpec(min_edits=5)
    results = engine.search("*.py", filters)
"""

try:
    from importlib.metadata import version
    __version__ = version("ai_session_tools")
except Exception:
    __version__ = "1.0.0"

__author__ = "Andrew Hundt"

from .engine import SessionRecoveryEngine
from .filters import ChainedFilter, LocationMatcher, MessageFilter, SearchFilter
from .formatters import CsvFormatter, JsonFormatter, PlainFormatter, ResultFormatter, TableFormatter
from .models import (
    FileVersion,
    FilterSpec,
    MessageType,
    RecoveredFile,
    RecoveryStatistics,
    SessionMessage,
    SessionMetadata,
)
from .types import (
    ComposableFilter,
    ComposableSearch,
    Extractable,
    Filterable,
    Formatter,
    Reporter,
    Searchable,
    Storage,
)

__all__ = [
    "ChainedFilter",
    "ComposableFilter",
    "ComposableSearch",
    "CsvFormatter",
    "Extractable",
    "FileVersion",
    "Filterable",
    "FilterSpec",
    "Formatter",
    "JsonFormatter",
    "LocationMatcher",
    "MessageFilter",
    "MessageType",
    "PlainFormatter",
    "RecoveredFile",
    "RecoveryStatistics",
    "Reporter",
    "ResultFormatter",
    "Searchable",
    "SearchFilter",
    "SessionMessage",
    "SessionMetadata",
    "SessionRecoveryEngine",
    "Storage",
    "TableFormatter",
]
