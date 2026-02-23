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

from .engine import SessionRecoveryEngine
from .extractors import FileExtractor, MessageExtractor
from .filters import LocationMatcher, MessageFilter, SearchFilter
from .formatters import CsvFormatter, JsonFormatter, ResultFormatter, TableFormatter
from .models import (
    FileLocation,
    FileType,
    FileVersion,
    FilterSpec,
    RecoveredFile,
    SearchOptions,
    SessionMessage,
)

__version__ = "2.0.0"
__author__ = "Claude Code Recovery"

__all__ = [
    "CsvFormatter",
    "FileExtractor",
    "FileLocation",
    "FileType",
    "FileVersion",
    "FilterSpec",
    "JsonFormatter",
    "LocationMatcher",
    "MessageExtractor",
    "MessageFilter",
    "RecoveredFile",
    "ResultFormatter",
    "SearchFilter",
    "SearchOptions",
    "SessionMessage",
    "SessionRecoveryEngine",
    "TableFormatter",
]
