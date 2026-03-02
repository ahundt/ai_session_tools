"""
AI Session Tools — Python library + CLI for Claude Code, AI Studio, and Gemini CLI sessions.

Copyright (c) 2026 Andrew Hundt
Licensed under the Apache License, Version 2.0

Quickstart (Claude Code single-source):
    from ai_session_tools import SessionRecoveryEngine, FilterSpec

    engine = SessionRecoveryEngine(projects_dir, recovery_dir)
    results = engine.search("*.py", FilterSpec(min_edits=5))

Quickstart (multi-source, auto-configured):
    from ai_session_tools import get_session_backend

    backend = get_session_backend()        # auto-detects Claude, AI Studio, Gemini CLI
    sessions = backend.get_sessions(since="7d")
    messages = backend.search_messages("authentication")

Configuration:
    from ai_session_tools import load_config, write_config

    cfg = load_config()
    cfg.setdefault("source_dirs", {})["aistudio"] = ["/path/to/ai-studio"]
    write_config(cfg)
"""

try:
    from importlib.metadata import version
    __version__ = version("ai_session_tools")
except Exception:
    __version__ = "0.3.0"

__author__ = "Andrew Hundt"

# Core engine + multi-source entry points
from .engine import (
    SessionRecoveryEngine,
    SessionBackend,
    MultiSourceEngine,
    get_session_backend,
    get_multi_engine,
    parse_date_input,
)

# Source backends (also importable directly from ai_session_tools.sources)
from .sources import AiStudioSource, GeminiCliSource

# Filters
from .filters import ChainedFilter, LocationMatcher, MessageFilter, SearchFilter

# Formatters
from .formatters import (
    CsvFormatter,
    JsonFormatter,
    MessageFormatter,
    PlainFormatter,
    ResultFormatter,
    TableFormatter,
    get_formatter,
)

# Models
from .models import (
    ContextMatch,
    CorrectionMatch,
    FileVersion,
    FilterSpec,
    MessageType,
    PlanningCommandCount,
    RecoveredFile,
    RecoveryStatistics,
    SessionAnalysis,
    SessionInfo,
    SessionMessage,
    SessionMetadata,
)

# Type protocols
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

# Config API
from .config import (
    load_config,
    get_config_path,
    write_config,
    get_config_section,
)

__all__ = [
    # Core engine
    "SessionRecoveryEngine",
    "SessionBackend",
    "MultiSourceEngine",
    "get_session_backend",
    "get_multi_engine",
    "parse_date_input",
    # Source backends
    "AiStudioSource",
    "GeminiCliSource",
    # Filters
    "ChainedFilter",
    "LocationMatcher",
    "MessageFilter",
    "SearchFilter",
    # Formatters
    "CsvFormatter",
    "JsonFormatter",
    "MessageFormatter",
    "PlainFormatter",
    "ResultFormatter",
    "TableFormatter",
    "get_formatter",
    # Models
    "ContextMatch",
    "CorrectionMatch",
    "FileVersion",
    "FilterSpec",
    "MessageType",
    "PlanningCommandCount",
    "RecoveredFile",
    "RecoveryStatistics",
    "SessionAnalysis",
    "SessionInfo",
    "SessionMessage",
    "SessionMetadata",
    # Type protocols
    "ComposableFilter",
    "ComposableSearch",
    "Extractable",
    "Filterable",
    "Formatter",
    "Reporter",
    "Searchable",
    "Storage",
    # Config
    "load_config",
    "get_config_path",
    "write_config",
    "get_config_section",
]
