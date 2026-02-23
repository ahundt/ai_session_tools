# AI Session Tools

Find and recover source files and conversations from your Claude Code sessions.

## The problem

Claude Code saves every conversation and every file it writes into session folders under `~/.claude/projects/`. But these are raw JSONL files with UUIDs for names — not something you can easily browse. If you want to find "that Python file Claude wrote last week" or "the session where we discussed the auth bug," you'd have to manually dig through dozens of folders and parse JSON.

## What this tool does

`aise` indexes all your Claude Code sessions and gives you four capabilities:

| Capability | What it does | Example |
|-----------|-------------|---------|
| **Find source files** | List every `.py`, `.md`, `.rs`, etc. that Claude wrote or edited across all sessions | `aise search files --pattern "*.py"` |
| **Recover source files** | Save the latest version — or every version — of a file to disk | `aise extract --name cli.py` |
| **Search conversations** | Find which sessions discussed a topic by searching message text | `aise search messages --query "auth bug"` |
| **Read conversations** | Print all user/assistant messages from a specific session | `aise get --session ab841016-...` |

Works as a CLI tool (`aise`) and as an importable Python library (`from ai_session_tools import ...`).

## Install

```bash
# Install as a global CLI tool
uv tool install ai-session-tools

# Or install from a local clone
git clone https://github.com/anthropics/ai_session_tools.git
uv tool install ./ai_session_tools
```

This gives you two equivalent commands: `aise` (short) and `ai_session_tools` (long).

To use as a library instead:

```bash
uv add ai-session-tools
```

## Quick start

```bash
# 1. See what files exist across all your Claude Code sessions
aise search

# 2. Find a specific file you remember working on
aise search files --pattern "*session*"

# 3. Recover it to disk
aise extract --name session_manager.py
# → Saved to ./recovered/session_manager.py
```

## CLI reference

### Find source files

List and filter source files that Claude wrote or edited during sessions.

```bash
# List all files across all sessions
aise search

# Only Python files
aise search files --pattern "*.py"

# Files edited 5+ times (likely important project files, not one-off scripts)
aise search files --min-edits 5

# Only Python and Markdown, modified after a specific date
aise search files -i py,md --after-date 2026-01-15

# Exclude compiled/temporary files
aise search files -x pyc,tmp,o

# Only files from specific sessions
aise search files --include-sessions ab841016,cd923f57

# Output as JSON instead of a table
aise search files --format json
```

**What "edits" means:** The number of times a file was written or modified across all sessions. A file with 20 edits is probably a core project file. A file with 1 edit is probably a one-off script.

### Recover source files to disk

```bash
# Save the most recent version of a file
aise extract --name cli.py
# → writes ./recovered/cli.py

# Save to a specific directory
aise extract --name cli.py --output-dir ./backup

# Save EVERY version (shows the file's edit history)
aise history --name cli.py
# → writes ./recovered/cli_v1.py, cli_v2.py, cli_v3.py, ...
```

Use `aise search` first to find the exact filename.

### Search conversations

Find sessions where specific topics were discussed.

```bash
# Find messages containing "authentication" across all sessions
aise search messages --query "authentication"

# Only search your messages (not Claude's responses)
aise search messages --query "error" --type user

# Show more results
aise search messages --query "TODO" --limit 50

# Truncate long messages to 200 chars for a compact overview
aise search messages --query "refactor" --max-chars 200
```

### Read messages from a specific session

Once you know a session ID (from `aise search` output, or from `ls ~/.claude/projects/*/`):

```bash
# Read all messages from a session
aise get --session ab841016-f07b-444c-bb18-22f6b373be52

# Read only your messages
aise get --session ab841016 --type user --limit 50

# Compact JSON output with truncated messages
aise get --session ab841016 --format json --max-chars 200
```

### Search files AND messages at once

```bash
# Find Python files and messages mentioning "error" in one command
aise search --pattern "*.py" --query "error"
```

When you pass both `--pattern` and `--query`, it shows two result sections.

### Show summary statistics

```bash
aise stats
# → Sessions: 42, Files: 318, Versions: 1205, Largest: engine.py (47 edits)
```

### Command ordering

Both orderings are equivalent — use whichever reads more naturally to you:

```bash
aise search files --pattern "*.py"     # "search" then narrow to "files"
aise files search --pattern "*.py"     # start in "files" then "search"

aise search messages --query "error"   # "search" then narrow to "messages"
aise messages search --query "error"   # start in "messages" then "search"
```

### Environment variables

Override where the tool looks for session data:

| Variable | Default | What it controls |
|----------|---------|------------------|
| `AI_SESSION_TOOLS_PROJECTS` | `~/.claude/projects` | Where Claude Code stores session folders |
| `AI_SESSION_TOOLS_RECOVERY` | `~/.claude/recovery` | Where `extract` and `history` write output |

### All flags reference

Run `aise --help`, `aise search --help`, `aise files search --help`, etc. for full flag documentation.

## Python library usage

```python
from pathlib import Path
from ai_session_tools import SessionRecoveryEngine, FilterSpec, SearchFilter

# Point the engine at your Claude Code data
engine = SessionRecoveryEngine(
    Path.home() / ".claude" / "projects",
    Path.home() / ".claude" / "recovery",
)

# List all files Claude ever wrote or edited
all_files = engine.search("*")
for f in all_files:
    print(f"{f.name}  ({f.total_edits} edits, last modified {f.last_modified})")

# Filter to heavily-edited Python/Markdown files from 2026
filters = FilterSpec(min_edits=5, after_date="2026-01-01")
filters.with_extensions(include={"py", "md"})
results = engine.search("*", filters)

# Chain filters after search results
search_filter = SearchFilter().by_edits(min_edits=3).by_extension("py")
filtered = search_filter(all_files)

# Search conversation text across all sessions
matches = engine.search_messages("authentication")
for msg in matches:
    print(f"[{msg.type.value}] {msg.session_id}: {msg.content[:100]}")

# Read messages from one session
messages = engine.get_messages("ab841016-f07b-444c-bb18-22f6b373be52")
user_only = engine.get_messages("ab841016", message_type="user")

# Recover files to disk
engine.extract_final("cli.py", Path("./recovered"))       # latest version
engine.extract_all("cli.py", Path("./versions"))           # all versions

# Summary statistics
stats = engine.get_statistics()
print(f"{stats.total_sessions} sessions, {stats.total_files} files, {stats.total_versions} versions")
```

### Key classes

| Class | What it does |
|-------|-------------|
| `SessionRecoveryEngine` | Main entry point — search files, read messages, extract, get stats |
| `FilterSpec` | Build filters for file search: edits, date range, extensions, sessions |
| `SearchFilter` | Chain filters on search results: `.by_edits(5).by_extension("py")` |
| `RecoveredFile` | One source file found in session data — has `.name`, `.total_edits`, `.last_modified` |
| `SessionMessage` | One conversation message — has `.type` (user/assistant), `.content`, `.timestamp` |
| `RecoveryStatistics` | Counts: `.total_sessions`, `.total_files`, `.total_versions` |

## Project structure

```
ai_session_tools/
├── __init__.py      # Public API — all exports listed here
├── engine.py        # SessionRecoveryEngine: search, extract, messages, stats
├── models.py        # Data classes: RecoveredFile, SessionMessage, FilterSpec, etc.
├── filters.py       # SearchFilter chain and filter predicates
├── formatters.py    # Output as table, JSON, CSV, or plain text
├── types.py         # Protocol interfaces (Searchable, Extractable, etc.)
└── cli.py           # CLI commands (thin wrappers over the engine)
```

## Development

```bash
git clone https://github.com/anthropics/ai_session_tools.git
cd ai_session_tools

# Install with dev dependencies
uv sync --all-extras

# Run tests (104 tests)
uv run pytest

# Lint and format
uv run ruff check ai_session_tools/
uv run ruff format ai_session_tools/

# Type check
uv run mypy ai_session_tools/
```

## License

Apache License 2.0 — Copyright (c) 2026 Andrew Hundt
