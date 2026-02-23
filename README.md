# AI Session Tools

Find and recover source files and conversations from your Claude Code sessions.

## The problem

Claude Code saves every conversation and every file it writes into session folders under `~/.claude/projects/`. But these are raw JSONL files with UUIDs for names — not something you can easily browse. If you want to find "that Python file Claude wrote last week" or "the session where we discussed the auth bug," you'd have to manually dig through dozens of folders and parse JSON.

## What this tool does

`aise` reads directly from your Claude Code session files and gives you these capabilities:

| Capability | Command |
|-----------|---------|
| List sessions with metadata (project, branch, date, message count) | `aise list` |
| Find source files Claude wrote or edited | `aise files search --pattern "*.py"` |
| Inspect file version history | `aise files history cli.py` |
| Extract a file version to stdout or disk | `aise files extract cli.py` |
| Cross-reference session edits against a current file | `aise files cross-ref ./cli.py` |
| Search conversation messages full-text | `aise messages search "auth bug"` |
| Search tool invocations (Bash, Edit, Write, Read, …) | `aise tools search Write "login"` |
| Read all messages from one session | `aise get ab841016` |
| Find user correction messages | `aise messages corrections` |
| Count slash command usage | `aise messages planning` |
| Export one session to markdown | `aise export session ab841016` |
| Export recent sessions to markdown | `aise export recent 7 --output week.md` |
| View / create the config file | `aise config show` / `aise config init` |
| Summary statistics | `aise stats` |

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
# 1. List your Claude Code sessions (newest first)
aise list

# 2. See all files across all sessions
aise files search

# 3. Find a specific file
aise files search --pattern "*session*"

# 4. Check its version history
aise files history session_manager.py

# 5. Print the latest version to stdout
aise files extract session_manager.py

# 6. Redirect to a file
aise files extract session_manager.py > session_manager.py
```

## CLI reference

### List sessions

```bash
# All sessions, newest first (project, branch, date, message count)
aise list

# Filter by project directory substring
aise list --project myproject

# Sessions since a date
aise list --after 2026-01-15

# JSON output
aise list --format json

# Limit to 10
aise list --limit 10
```

### Find source files

```bash
# All files (shows edits, sessions, last modified)
aise files search

# Only Python files
aise files search --pattern "*.py"

# Files edited 5+ times
aise files search --min-edits 5

# Python and Markdown, after a date
aise files search -i py,md --after 2026-01-15

# Exclude compiled files
aise files search -x pyc,tmp,o

# JSON output
aise files search --format json

# 'find' is an alias for 'search'
aise files find --pattern "*.py"
```

### Inspect file version history

```bash
# Show a read-only version table (version#, lines, Δlines, timestamp, session)
aise files history cli.py

# Narrow to one session
aise files history cli.py --session ab841016

# Export all versions to disk as cli_v1.py, cli_v2.py, …
aise files history cli.py --export

# Preview without touching disk
aise files history cli.py --export --dry-run

# Dump all versions to stdout (for piping to AI tools)
aise files history cli.py --stdout
```

`files history` is **read-only by default** — no disk writes unless you pass `--export`.

### Extract file content

```bash
# Print latest version to stdout
aise files extract cli.py

# Specific version
aise files extract cli.py --version 2

# Pipe to clipboard
aise files extract cli.py | pbcopy

# Limit to one session
aise files extract cli.py --session ab841016

# Write to a directory
aise files extract cli.py --output-dir ./backup

# Preview without I/O
aise files extract cli.py --output-dir ./backup --dry-run
```

File content goes to stdout; status messages go to stderr — keeps piping clean.

### Cross-reference session edits

```bash
# Show which edits Claude made to cli.py are present in the current file
aise files cross-ref ./cli.py

# Limit to one session
aise files cross-ref ./cli.py --session ab841016

# JSON output
aise files cross-ref ./cli.py --format json
```

### Search conversation messages

```bash
# Full-text search across all sessions
aise messages search "authentication"

# Only your messages (not Claude's)
aise messages search "error" --type user

# More results with truncation
aise messages search "TODO" --limit 50 --max-chars 200

# JSON output
aise messages search "refactor" --format json

# 'find' is an alias for 'search'
aise messages find "authentication"
```

### Search tool invocations

Find the actual tool calls Claude made — what files it wrote, what commands it ran:

```bash
# All Write tool calls
aise tools search Write

# Bash calls containing "git commit"
aise tools search Bash "git commit"

# Edit calls to cli.py
aise tools search Edit "cli.py"

# JSON output
aise tools search Write --format json

# 'find' is an alias for 'search'
aise tools find Write

# Also works via messages search with --tool flag
aise messages search "*" --tool Write
```

### Read messages from a session

```bash
# Read all messages (positional session ID)
aise get ab841016

# Read only your messages
aise get ab841016 --type user --limit 50

# JSON output
aise get ab841016 --format json

# Same via messages subcommand
aise messages get ab841016
```

Find session IDs with `aise list`.

### Find user corrections

Detects messages where you corrected Claude's behavior:

```bash
# Show all correction messages (categorized as regression/skip_step/misunderstanding/incomplete)
aise messages corrections

# Filter by project
aise messages corrections --project myproject

# More results
aise messages corrections --limit 50

# JSON output
aise messages corrections --format json
```

### Slash command usage

Count slash commands you've invoked across all sessions:

```bash
# Auto-discover ALL slash commands (no config needed)
aise messages planning

# Filter by project or date range
aise messages planning --project myproject --after 2026-01-01

# JSON output
aise messages planning --format json

# Use specific regex patterns instead of auto-discovery
aise messages planning --commands "/ar:plannew,/ar:pn"
```

The default **discovery mode** finds every message that starts with `/command` — `/ar:plannew`, `/commit`, `/help`, whatever you've used. No configuration required.

### Export to markdown

```bash
# Export one session to stdout
aise export session ab841016

# Redirect to a file
aise export session ab841016 > session.md

# Write to a file explicitly
aise export session ab841016 --output session.md

# Preview without writing
aise export session ab841016 --dry-run

# Export all sessions from the last 7 days to a single file
aise export recent --output week.md

# Last 14 days, filtered by project
aise export recent 14 --project myproject --output week.md
```

System messages (`[Request interrupted`, `<task-notification>`, `<system-reminder>`) are filtered out automatically.

### Search across files and messages at once

```bash
# Find Python files AND messages mentioning "error" in one command
aise search --pattern "*.py" --query "error"

# Tool calls with a query (auto-routes to messages)
aise search --tool Write --query "login"
aise search tools --tool Bash --query "git"

# 'find' is an alias for 'search'
aise find files --pattern "*.py"
aise find --tool Write --query "login"
```

### Statistics

```bash
aise stats
```

```
Recovery Statistics
  Sessions:      42
  Files:         318
  Versions:      1205
  Largest File:  engine.py (47 edits)
```

---

## Configuration

`aise` works out of the box — no config file required. The optional config lets you customize correction detection patterns and specify a fixed set of slash commands to track.

### Config file location

| Priority | Source |
|----------|--------|
| 1 | `--config /path/to/config.json` CLI flag |
| 2 | `AI_SESSION_TOOLS_CONFIG` environment variable |
| 3 | macOS default: `~/Library/Application Support/ai_session_tools/config.json` |
| 3 | Linux default: `~/.config/ai_session_tools/config.json` |

### Config commands

```bash
# Show the config file path (whether or not it exists)
aise config path

# Show current configuration (file contents + source)
aise config show
aise config show --format json

# Create a starter config.json with documented defaults
aise config init

# Overwrite an existing config file
aise config init --force
```

### Config file format

```json
{
  "correction_patterns": [
    "regression:you deleted",
    "regression:you removed",
    "skip_step:you forgot",
    "skip_step:you missed",
    "misunderstanding:that's wrong",
    "incomplete:also need"
  ],
  "planning_commands": [
    "/ar:plannew",
    "/ar:pn",
    "/ar:planrefine",
    "/ar:pr"
  ]
}
```

**`correction_patterns`** — list of `"CATEGORY:KEYWORD"` strings. Overrides built-in correction detection patterns when present. Category becomes the label in `aise messages corrections` output.

**`planning_commands`** — list of slash command strings (not regex). When set, `aise messages planning` counts only these commands instead of auto-discovering all slash commands.

### Priority chain

Every setting follows: **CLI flag > environment variable > config file > built-in default**

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAUDE_CONFIG_DIR` | `~/.claude` | Base Claude config directory |
| `AI_SESSION_TOOLS_PROJECTS` | `~/.claude/projects` | Path to session folders |
| `AI_SESSION_TOOLS_RECOVERY` | `~/.claude/recovery` | Recovery output path |
| `AI_SESSION_TOOLS_CONFIG` | OS config dir | Config file path |

```bash
# Use an external drive
aise --claude-dir /Volumes/External/.claude files search

# Point at a non-default projects directory
AI_SESSION_TOOLS_PROJECTS=/data/claude-sessions aise list
```

---

## Command ordering

Both orderings are equivalent — use whichever reads more naturally:

```bash
aise search files --pattern "*.py"     # "search" then narrow to "files"
aise files search --pattern "*.py"     # start in "files" then "search"

aise search messages --query "error"   # "search" then "messages"
aise messages search "error"           # "messages" then "search" (positional query)

aise find files --pattern "*.py"       # "find" is a root alias for "search"
aise files find --pattern "*.py"       # "find" in files subapp

aise tools search Write                # tools subapp with positional tool name
aise search --tool Write               # root search with --tool flag (equivalent)
```

---

## Python library usage

```python
from pathlib import Path
from ai_session_tools import SessionRecoveryEngine, FilterSpec

engine = SessionRecoveryEngine(
    Path.home() / ".claude" / "projects",
    Path.home() / ".claude" / "recovery",
)

# List sessions
sessions = engine.get_sessions(project_filter="myproject", after="2026-01-01")
for s in sessions:
    print(f"{s.session_id[:16]}  {s.git_branch}  {s.message_count} messages")

# List all files Claude ever wrote or edited
all_files = engine.search("*")
for f in all_files:
    print(f"{f.name}  ({f.edits} edits, last modified {f.last_modified})")

# Filter to heavily-edited Python files
filters = FilterSpec(min_edits=5, after="2026-01-01T00:00:00")
filters.with_extensions(include={"py"})
results = engine.search("*", filters)

# Search conversation text
matches = engine.search_messages("authentication")
for msg in matches:
    print(f"[{msg.type.value}] {msg.session_id}: {msg.content[:100]}")

# Search tool invocations
write_calls = engine.search_messages("", tool="Write")
bash_git = engine.search_messages("git commit", tool="Bash")

# Find user corrections
corrections = engine.find_corrections(project_filter="myproject", limit=20)
for c in corrections:
    print(f"{c.category}: {c.matched_pattern} — {c.content[:80]}")

# Count slash command usage (auto-discovery)
planning = engine.analyze_planning_usage()  # discovers all /command patterns
for p in planning:
    print(f"{p.command}: {p.count} uses across {len(p.session_ids)} sessions")

# Cross-reference session edits against current file
current = Path("cli.py").read_text()
refs = engine.cross_reference_session("cli.py", current)
for r in refs:
    mark = "✓" if r["found_in_current"] else "✗"
    print(f"{mark} {r['tool']} {r['timestamp'][:10]}: {r['content_snippet'][:60]}")

# Export a session to markdown
md = engine.export_session_markdown("ab841016")
Path("session.md").write_text(md)

# Get all recorded versions of a file
versions = engine.get_versions("cli.py")
for v in versions:
    print(f"v{v.version_num}: {v.line_count} lines  {v.timestamp}  session {v.session_id[:16]}")

# Summary statistics
stats = engine.get_statistics()
print(f"{stats.total_sessions} sessions, {stats.total_files} files, {stats.total_versions} versions")
```

### Key classes

| Class | Description |
|-------|-------------|
| `SessionRecoveryEngine` | Main entry point — all search, extract, and analysis methods |
| `FilterSpec` | Build filters for file search: edits, date range, extensions, sessions |
| `SessionInfo` | One session: `session_id`, `project_dir`, `git_branch`, `cwd`, `message_count` |
| `RecoveredFile` | One source file: `name`, `edits`, `last_modified`, `sessions` |
| `FileVersion` | One file version: `version_num`, `line_count`, `timestamp`, `session_id` |
| `SessionMessage` | One conversation message: `type` (user/assistant), `content`, `timestamp` |
| `CorrectionMatch` | One correction: `category`, `matched_pattern`, `content`, `session_id` |
| `PlanningCommandCount` | One slash command: `command`, `count`, `session_ids`, `project_dirs` |
| `RecoveryStatistics` | Aggregate counts: `total_sessions`, `total_files`, `total_versions` |

---

## Project structure

```
ai_session_tools/
├── __init__.py      # Public API — all exports listed here
├── engine.py        # SessionRecoveryEngine: search, extract, messages, analysis
├── models.py        # Data classes: RecoveredFile, FileVersion, SessionInfo, etc.
├── filters.py       # SearchFilter chain and filter predicates
├── formatters.py    # Output as table, JSON, CSV, or plain text
├── types.py         # Protocol interfaces (Searchable, Extractable, etc.)
└── cli.py           # CLI commands (thin wrappers over the engine)
```

---

## Development

```bash
git clone https://github.com/anthropics/ai_session_tools.git
cd ai_session_tools

# Install with dev dependencies
uv sync --all-extras

# Run tests (integration tests require ~/.claude/projects/)
uv run pytest                    # unit tests only (fast)
uv run pytest -m ""             # all tests including integration

# Lint and format
uv run ruff check ai_session_tools/
uv run ruff format ai_session_tools/

# Type check
uv run mypy ai_session_tools/
```

---

## License

Apache License 2.0 — Copyright (c) 2026 Andrew Hundt
