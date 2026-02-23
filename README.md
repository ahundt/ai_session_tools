# AI Session Tools

Find and recover source files and conversations from your Claude Code sessions.

## The problem

Claude Code saves every conversation and every file it writes into session folders under `~/.claude/projects/`. But these are raw JSONL files with UUIDs for names — not something you can easily browse. If you want to find "that Python file Claude wrote last week" or "the session where we discussed the auth bug," you'd have to manually dig through dozens of folders and parse JSON.

## What this tool does

`aise` reads directly from your Claude Code session files and gives you five capabilities:

| Capability | What it does | Example |
|-----------|-------------|---------|
| **Find source files** | List every `.py`, `.md`, `.rs`, etc. that Claude wrote or edited across all sessions | `aise files search --pattern "*.py"` |
| **Inspect file history** | Show all recorded versions of a file with line counts, timestamps, session IDs | `aise files history cli.py` |
| **Extract file content** | Print a file to stdout or write it to disk | `aise files extract cli.py` |
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

This gives two equivalent commands: `aise` (short) and `ai_session_tools` (long).

To use as a library instead:

```bash
uv add ai-session-tools
```

## Quick start

```bash
# 1. See all files across all your Claude Code sessions
aise files search

# 2. Find a specific file (the table includes a sessions column)
aise files search --pattern "*session*"

# 3. Check its version history (read-only table — no disk writes)
aise files history session_manager.py

# 4. Print the latest version to stdout
aise files extract session_manager.py

# 5. Redirect to a file
aise files extract session_manager.py > session_manager.py
```

All `aise files <cmd>` commands have root aliases: `aise search`, `aise history`, `aise extract`. Use whichever reads more naturally — see [Command ordering](#command-ordering).

## CLI reference

### Disambiguation workflow

When you're looking for a specific file version:

```bash
# Step 1 — find the file; table shows session IDs for each file
aise files search --pattern "cli.py"

# Step 2 — see all versions for one session
aise files history cli.py --session ab841016

# Step 3 — get the exact version you want
aise files extract cli.py --session ab841016 --version 2
```

### Find source files

List and filter source files that Claude wrote or edited during sessions.

```bash
# List all files across all sessions (table includes sessions column)
aise files search

# Only Python files
aise files search --pattern "*.py"

# Files edited 5+ times (likely important project files, not one-off scripts)
aise files search --min-edits 5

# Only Python and Markdown, modified after a specific date or datetime
aise files search -i py,md --after 2026-01-15
aise files search -i py,md --after 2026-01-15T14:30:00

# Exclude compiled/temporary files
aise files search -x pyc,tmp,o

# Only files from specific sessions
aise files search --include-sessions ab841016,cd923f57

# Output as JSON instead of a table
aise files search --format json
```

**What "edits" means:** The number of times a file was written or modified across all sessions. A file with 20 edits is probably a core project file. A file with 1 edit is probably a one-off script.

### Inspect file version history

```bash
# Show a read-only version table (version#, lines, Δlines, timestamp, session)
aise files history cli.py

# Narrow to one session
aise files history cli.py --session ab841016

# Export all versions to disk as cli_v1.py, cli_v2.py, ...
aise files history cli.py --export

# Preview what --export would write without touching disk
aise files history cli.py --export --dry-run

# Export to a specific directory
aise files history cli.py --export --export-dir ./versions

# Dump all versions to stdout with === v1 === headers (for piping to AI tools)
aise files history cli.py --stdout
```

`files history` is **read-only by default** — no disk writes unless you pass `--export`.

### Extract file content

```bash
# Print latest version to stdout (pipe-friendly)
aise files extract cli.py

# Specific version
aise files extract cli.py --version 2

# Redirect to a file
aise files extract cli.py > cli.py

# Pipe to clipboard
aise files extract cli.py | pbcopy

# Limit to one session
aise files extract cli.py --session ab841016

# Write to the path Claude originally created the file (from session data)
aise files extract cli.py --restore

# Write to a specific directory
aise files extract cli.py --output-dir ./backup

# Preview what would happen without any I/O
aise files extract cli.py --restore --dry-run
aise files extract cli.py --output-dir ./backup --dry-run
```

**stdout vs stderr:** File content goes to stdout; status messages ("Extracting: cli.py  (v3, 85 lines)") go to stderr. This keeps piping and redirection clean.

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

```
$ aise stats

Recovery Statistics
  Sessions:      42
  Files:         318
  Versions:      1205
  Largest File:  engine.py (47 edits)
```

### Command ordering

Both orderings are equivalent — use whichever reads more naturally:

```bash
aise search files --pattern "*.py"     # "search" then narrow to "files"
aise files search --pattern "*.py"     # start in "files" then "search"

aise extract cli.py                    # root alias
aise files extract cli.py             # domain group form

aise history cli.py                    # root alias
aise files history cli.py             # domain group form

aise search messages --query "error"   # "search" then narrow to "messages"
aise messages search --query "error"   # start in "messages" then "search"
```

### Environment variables and --claude-dir

Override where the tool looks for session data:

| Variable / Flag | Default | What it controls |
|----------------|---------|-----------------|
| `--claude-dir PATH` | `~/.claude` | Base Claude config directory (projects and recovery are resolved relative to this) |
| `CLAUDE_CONFIG_DIR` | `~/.claude` | Same as `--claude-dir`, read from environment |
| `AI_SESSION_TOOLS_PROJECTS` | `~/.claude/projects` | Path to Claude Code session folders (overrides base dir) |
| `AI_SESSION_TOOLS_RECOVERY` | `~/.claude/recovery` | Path for recovery output when writing files (overrides base dir) |

Priority: `--claude-dir` > `CLAUDE_CONFIG_DIR` > `~/.claude`

```bash
# Use a different Claude config directory (e.g. external drive)
aise --claude-dir /Volumes/External/.claude files search
```

### All flags reference

Run `aise --help`, `aise files extract --help`, `aise files history --help`, etc. for full flag documentation.

## Python library usage

```python
from pathlib import Path
from ai_session_tools import SessionRecoveryEngine, FilterSpec

# Point the engine at your Claude Code data
engine = SessionRecoveryEngine(
    Path.home() / ".claude" / "projects",
    Path.home() / ".claude" / "recovery",
)

# List all files Claude ever wrote or edited
all_files = engine.search("*")
for f in all_files:
    print(f"{f.name}  ({f.edits} edits, last modified {f.last_modified})")

# Filter to heavily-edited Python/Markdown files from 2026
filters = FilterSpec(min_edits=5, after="2026-01-01T00:00:00")
filters.with_extensions(include={"py", "md"})
results = engine.search("*", filters)

# Search conversation text across all sessions
matches = engine.search_messages("authentication")
for msg in matches:
    print(f"[{msg.type.value}] {msg.session_id}: {msg.content[:100]}")

# Read messages from one session
messages = engine.get_messages("ab841016-f07b-444c-bb18-22f6b373be52")
user_only = engine.get_messages("ab841016", message_type="user")

# Get all recorded versions of a file
versions = engine.get_versions("cli.py")
for v in versions:
    print(f"v{v.version_num}: {v.line_count} lines  {v.timestamp}  session {v.session_id[:16]}")

# Write files to disk (library-level helpers)
engine.extract_final("cli.py", Path("./recovered"))   # latest version
engine.extract_all("cli.py", Path("./versions"))       # all versions

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
| `RecoveredFile` | One source file found in session data — `.name`, `.edits`, `.last_modified`, `.sessions` |
| `FileVersion` | One recorded version — `.version_num`, `.line_count`, `.timestamp`, `.session_id` |
| `SessionMessage` | One conversation message — `.type` (user/assistant), `.content`, `.timestamp` |
| `RecoveryStatistics` | Counts: `.total_sessions`, `.total_files`, `.total_versions` |

## Project structure

```
ai_session_tools/
├── __init__.py      # Public API — all exports listed here
├── engine.py        # SessionRecoveryEngine: search, extract, messages, stats
├── models.py        # Data classes: RecoveredFile, FileVersion, FilterSpec, etc.
├── filters.py       # SearchFilter chain and filter predicates
├── formatters.py    # Output as table (with sessions column), JSON, CSV, or plain text
├── types.py         # Protocol interfaces (Searchable, Extractable, etc.)
└── cli.py           # CLI commands (thin wrappers over the engine)
```

## Development

```bash
git clone https://github.com/anthropics/ai_session_tools.git
cd ai_session_tools

# Install with dev dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Lint and format
uv run ruff check ai_session_tools/
uv run ruff format ai_session_tools/

# Type check
uv run mypy ai_session_tools/
```

## License

Apache License 2.0 — Copyright (c) 2026 Andrew Hundt
