# AI Session Tools

Modern Python library for analyzing and extracting data from Claude Code session files.

## Features

- **Library-first design** - Use as both CLI tool and importable Python library
- **Composable architecture** - Protocol-based interfaces for extensibility
- **Modern Python** - Full type hints, frozen dataclasses, enums, and protocols
- **Multiple output formats** - Table, JSON, CSV, and plain text
- **Advanced filtering** - Search, filter, and analyze sessions with composable predicates

## Installation

```bash
pip install ai_session_tools
```

Or with development dependencies:

```bash
pip install "ai_session_tools[dev]"
```

## Quick Start

### As CLI

```bash
# Search for files by pattern
ai_session_tools search --pattern "*.py"

# Extract file history
ai_session_tools history --file myfile.py

# View session statistics
ai_session_tools stats

# Extract messages from a session
ai_session_tools messages --session <session-id>
```

### As Library

```python
from ai_session_tools import SessionRecoveryEngine, SearchFilter, FilterSpec
from pathlib import Path

# Initialize
engine = SessionRecoveryEngine(projects_dir, recovery_dir)

# Simple search
results = engine.search("*.py")

# Advanced filtering
filters = FilterSpec(min_edits=5)
results = engine.search("*.py", filters)

# Composable filters
file_filter = SearchFilter().by_edits(min_edits=5).by_type("python")
filtered = file_filter(results)

# Message access
messages = engine.get_messages("session_id")

# Statistics
stats = engine.get_statistics()
print(f"Total files: {stats.total_files}")
```

## Architecture

```
ai_session_tools/
├── __init__.py         # Clean public API
├── models.py           # Data models (frozen dataclasses, enums)
├── types.py            # Protocol interfaces for extensibility
├── filters.py          # Composable filter implementations
├── engine.py           # Core business logic (clean 361-line API)
├── formatters.py       # Output formatting (Table, JSON, CSV, Plain)
├── extractors.py       # Extraction strategies
└── cli.py              # Thin CLI orchestration layer
```

## Design Principles

1. **Library-First** - All business logic in library, CLI is thin orchestration
2. **Composable** - Mix and match components with fluent APIs
3. **Modern Python** - Type hints, protocols, frozen dataclasses, enums
4. **Clean Separation** - Models, engine, filters, formatters, CLI all separate
5. **Extensible** - Protocol-based interfaces for custom implementations

## Development

```bash
# Install with dev dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Lint and format
uv run ruff check ai_session_tools/
uv run ruff format ai_session_tools/

# Run CLI
uv run ai_session_tools --help
```

## License

Apache License 2.0 - See LICENSE file for details

Copyright (c) 2026 Andrew Hundt

## Contributing

Contributions welcome! Please ensure:
- All tests pass (`pytest`)
- Code is linted (`ruff check`)
- Code is formatted (`ruff format`)
- Type hints are complete
