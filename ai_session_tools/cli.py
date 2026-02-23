"""
Thin CLI layer - orchestrates library components without business logic.

Supports dual-ordering: both `aise search files` and `aise files search` work.
Root commands default to broad behavior; domain groups narrow the scope.

Copyright (c) 2026 Andrew Hundt
Licensed under the Apache License, Version 2.0
"""

import os
import re as _re
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Set

import typer
from rich.console import Console

from .engine import SessionRecoveryEngine
from .formatters import MessageFormatter, get_formatter
from .models import FileVersion, FilterSpec

app = typer.Typer(
    help=(
        "Analyze and extract data from Claude Code sessions stored in ~/.claude/projects/.\n\n"
        "Claude Code stores each conversation as a session — a folder of JSONL files containing "
        "user/assistant messages and source code snapshots. This tool searches those sessions to "
        "find source files that were written or edited, and to search conversation messages.\n\n"
        "Override default paths with environment variables:\n\n"
        "  CLAUDE_CONFIG_DIR          Path to Claude config dir (default: ~/.claude)\n\n"
        "  AI_SESSION_TOOLS_PROJECTS  Path to Claude projects dir (default: ~/.claude/projects)\n\n"
        "  AI_SESSION_TOOLS_RECOVERY  Path to recovery output dir (default: ~/.claude/recovery)"
    ),
)
files_app = typer.Typer(
    help=(
        "Search, extract, and inspect source files (*.py, *.md, *.rs, etc.) found in Claude Code session data.\n\n"
        "These are files that Claude wrote or edited during sessions — not the JSONL session files themselves."
    ),
)
messages_app = typer.Typer(
    help=(
        "Search and read user/assistant conversation messages stored in Claude Code session JSONL files.\n\n"
        "Each session contains timestamped messages with a type (user or assistant) and text content."
    ),
)

app.add_typer(files_app, name="files", rich_help_panel="Domain Groups")
app.add_typer(messages_app, name="messages", rich_help_panel="Domain Groups")

console = Console()
err_console = Console(stderr=True)

# Module-level override set by --claude-dir global option
_g_claude_dir: Optional[str] = None


# ── Root app callback (global --claude-dir option) ────────────────────────────

@app.callback(invoke_without_command=True)
def app_callback(
    ctx: typer.Context,
    claude_dir: Optional[str] = typer.Option(
        None, "--claude-dir",
        help=(
            "Path to the Claude configuration directory. "
            "Default: $CLAUDE_CONFIG_DIR if set, otherwise ~/.claude. "
            "Example: --claude-dir /Volumes/External/.claude"
        ),
        envvar="CLAUDE_CONFIG_DIR",
    ),
) -> None:
    global _g_claude_dir
    _g_claude_dir = claude_dir
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(0)


# ── Engine factory ────────────────────────────────────────────────────────────

def get_engine(projects_dir: Optional[str] = None, recovery_dir: Optional[str] = None) -> SessionRecoveryEngine:
    """
    Get or create recovery engine.

    Priority for Claude config dir: --claude-dir CLI flag > CLAUDE_CONFIG_DIR env var > ~/.claude

    Also supports:
        AI_SESSION_TOOLS_PROJECTS: Path to Claude projects directory (overrides base dir)
        AI_SESSION_TOOLS_RECOVERY: Path to recovery directory (overrides base dir)
    """
    # Priority: --claude-dir (CLI) > CLAUDE_CONFIG_DIR (env) > ~/.claude (default)
    # _g_claude_dir is set by --claude-dir; if not set, read env var directly (for non-CLI callers)
    claude_dir = _g_claude_dir or os.getenv("CLAUDE_CONFIG_DIR")

    if projects_dir is None:
        projects_dir = os.getenv("AI_SESSION_TOOLS_PROJECTS")
        if projects_dir is None:
            base = Path(claude_dir).expanduser() if claude_dir else Path.home() / ".claude"
            projects_dir = str(base / "projects")

    if recovery_dir is None:
        recovery_dir = os.getenv("AI_SESSION_TOOLS_RECOVERY")
        if recovery_dir is None:
            base = Path(claude_dir).expanduser() if claude_dir else Path.home() / ".claude"
            recovery_dir = str(base / "recovery")

    # expanduser() so that env var values like "~/.claude/projects" work correctly
    return SessionRecoveryEngine(Path(projects_dir).expanduser(), Path(recovery_dir).expanduser())


# ── Shared helper functions ───────────────────────────────────────────────────

def _parse_session_set(s: Optional[str]) -> Set[str]:
    """Parse comma-separated session IDs string to set. Empty string or None → empty set."""
    if not s:
        return set()
    return {x.strip() for x in s.split(",") if x.strip()}


def _project_dir_name(path: str) -> str:
    """Convert a working directory path to the ~/.claude/projects/ directory name.

    Matches Claude Code's own encoding: replace /[^a-zA-Z0-9-]/g with '-'.
    Hyphens are preserved; slashes, dots, underscores, spaces all become '-'.

    Source: ~/source/happy/packages/happy-cli/src/claude/utils/path.ts getProjectPath()

    Examples:
        /Users/athundt/.claude        → -Users-athundt--claude
        /Users/me/my_project          → -Users-me-my-project
        /var/www/my.site.com/public   → -var-www-my-site-com-public
    """
    abs_path = os.path.abspath(path)
    return _re.sub(r'[^a-zA-Z0-9-]', '-', abs_path)


def _sessions_for_project(engine: SessionRecoveryEngine, project_path: str) -> Set[str]:
    """Return all session IDs whose JSONL files live under the given project directory."""
    dir_name = _project_dir_name(project_path)
    project_dir = engine.projects_dir / dir_name
    if not project_dir.is_dir():
        return set()
    return {f.stem for f in project_dir.glob("*.jsonl")}


def _filter_versions(
    versions: List[FileVersion],
    project_sessions: Set[str] = frozenset(),
    session: Optional[str] = None,
    include_sessions: Set[str] = frozenset(),
    exclude_sessions: Set[str] = frozenset(),
    after: Optional[str] = None,
    before: Optional[str] = None,
) -> List[FileVersion]:
    """Filter FileVersion list by composable criteria. All filters are ANDed."""
    result = versions
    # --project: restrict to project's session IDs
    if project_sessions:
        result = [v for v in result if v.session_id in project_sessions]
    # --session: prefix match (convenience, single value)
    if session:
        result = [v for v in result if v.session_id.startswith(session)]
    # --include-sessions: keep only matching sessions (prefix match each)
    if include_sessions:
        result = [v for v in result if any(v.session_id.startswith(i) for i in include_sessions)]
    # --exclude-sessions: remove matching sessions
    if exclude_sessions:
        result = [v for v in result if not any(v.session_id.startswith(i) for i in exclude_sessions)]
    # --after / --before: filter by version timestamp
    if after:
        result = [v for v in result if v.timestamp and v.timestamp >= after[:len(v.timestamp)]]
    if before:
        result = [v for v in result if v.timestamp and v.timestamp <= before[:len(v.timestamp)]]
    return result


def _version_src_path(engine: SessionRecoveryEngine, v: FileVersion) -> Path:
    """Return the on-disk path of a specific version file."""
    return (
        engine.recovery_dir
        / f"session_all_versions_{v.session_id}"
        / f"{v.filename}_v{v.version_num:06d}_line_{v.line_count}.txt"
    )


def _resolve_output_path(target: Path) -> Path:
    """Return target path; if it already exists, append .recovered[_N] before the extension.

    Examples: cli.py → cli.recovered.py → cli.recovered_1.py → cli.recovered_2.py → …
    """
    if not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    candidate = target.parent / f"{stem}.recovered{suffix}"
    counter = 1
    while candidate.exists():
        candidate = target.parent / f"{stem}.recovered_{counter}{suffix}"
        counter += 1
    return candidate


def _do_extract(
    engine: SessionRecoveryEngine,
    name: str,
    version: Optional[int] = None,
    session: Optional[str] = None,
    output_dir: Optional[str] = None,
    restore: bool = False,
    dry_run: bool = False,
) -> None:
    """Extract a file version: stdout by default, or write with --restore/--output-dir."""
    versions = engine.get_versions(name)

    if session:
        versions = [v for v in versions if v.session_id.startswith(session)]
        if not versions:
            err_console.print(f"[red]No versions of[/red] {name} [red]found in session[/red] {session}")
            raise typer.Exit(code=1)

    if not versions:
        err_console.print(f"[red]File not found in session data:[/red] {name}")
        raise typer.Exit(code=1)

    if version is None:
        v = max(versions, key=lambda x: x.version_num)
    else:
        v = next((x for x in versions if x.version_num == version), None)
        if v is None:
            max_v = max(x.version_num for x in versions)
            err_console.print(f"[red]Version {version} not found.[/red] Available: 1\u2013{max_v}")
            raise typer.Exit(code=1)

    label = f"v{v.version_num}, {v.line_count} lines"
    src = _version_src_path(engine, v)

    if restore:
        original = engine.get_original_path(name)
        if original:
            target = _resolve_output_path(Path(original))
        else:
            err_console.print(f"[red]No original path recorded in session data for:[/red] {name}")
            raise typer.Exit(code=1)
        err_console.print(f"Writing to: {target}  ({label})")
        if dry_run:
            err_console.print("[yellow][dry run] no files written[/yellow]")
            return
        if not src.exists():
            err_console.print(f"[red]Version file not found on disk:[/red] {src}")
            raise typer.Exit(code=1)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(src.read_text(errors="replace"))

    elif output_dir:
        target = Path(output_dir).expanduser() / name
        err_console.print(f"Writing to: {target}  ({label})")
        if dry_run:
            err_console.print("[yellow][dry run] no files written[/yellow]")
            return
        if not src.exists():
            err_console.print(f"[red]Version file not found on disk:[/red] {src}")
            raise typer.Exit(code=1)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(src.read_text(errors="replace"))

    else:
        # Default: stdout
        if dry_run:
            err_console.print(f"[yellow][dry run] would print {v.line_count} lines to stdout[/yellow]")
            return
        err_console.print(f"Extracting: {name}  ({label})")
        if src.exists():
            sys.stdout.write(src.read_text(errors="replace"))
        else:
            err_console.print(f"[red]Version file not found on disk:[/red] {src}")
            raise typer.Exit(code=1)


def _do_history_display(engine: SessionRecoveryEngine, name: str, versions: Optional[List[FileVersion]] = None) -> None:
    """Show version history table (read-only, no disk writes)."""
    if versions is None:
        versions = engine.get_versions(name)
    if not versions:
        console.print(f"[yellow]No versions found for:[/yellow] {name}")
        return
    from rich.table import Table
    table = Table(title=f"Version history: {name}  ({len(versions)} versions)")
    table.add_column("Version", style="cyan", justify="right")
    table.add_column("Lines", justify="right", style="magenta")
    table.add_column("\u0394Lines", justify="right")
    table.add_column("Timestamp", style="dim")
    table.add_column("Session", style="blue")
    prev_lines = None
    for v in versions:
        if prev_lines is None:
            delta = "\u2014"
        elif v.line_count >= prev_lines:
            delta = f"+{v.line_count - prev_lines}"
        else:
            delta = str(v.line_count - prev_lines)
        short_session = v.session_id[:16] + "\u2026" if len(v.session_id) > 16 else v.session_id
        table.add_row(f"v{v.version_num}", str(v.line_count), delta, v.timestamp or "\u2014", short_session)
        prev_lines = v.line_count
    console.print(table)


def _do_history_export(
    engine: SessionRecoveryEngine,
    name: str,
    export_dir: Optional[str] = None,
    dry_run: bool = False,
    versions: Optional[List[FileVersion]] = None,
) -> None:
    """Export all versions as named files: cli_v1.py, cli_v2.py, …"""
    if versions is None:
        versions = engine.get_versions(name)
    if not versions:
        console.print(f"[yellow]No versions found:[/yellow] {name}")
        return

    stem = Path(name).stem
    suffix = Path(name).suffix

    # Resolve export dir
    env_dir = os.getenv("AI_SESSION_TOOLS_OUTPUT")
    if export_dir:
        out_dir = Path(export_dir).expanduser()
    elif env_dir:
        out_dir = Path(env_dir).expanduser()
    else:
        original = engine.get_original_path(name)
        out_dir = Path(original).parent / "versions" if original else Path("./recovered/versions")

    # Always show what will be written before writing
    console.print(f"[bold]Exporting {len(versions)} versions to:[/bold] {out_dir}")
    for v in versions:
        console.print(f"  {stem}_v{v.version_num}{suffix}  ({v.line_count} lines)")

    if dry_run:
        console.print("[yellow][dry run] no files written[/yellow]")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for v in versions:
        src = _version_src_path(engine, v)
        if src.exists():
            shutil.copy2(src, out_dir / f"{stem}_v{v.version_num}{suffix}")
            written += 1

    if written:
        console.print(f"\n[green]Exported {written} files to:[/green] {out_dir}")
    else:
        console.print("[red]No version files found to export[/red]")
        raise typer.Exit(code=1)


def _do_history_stdout(engine: SessionRecoveryEngine, name: str, versions: Optional[List[FileVersion]] = None) -> None:
    """Print all versions to stdout with === vN === headers."""
    if versions is None:
        versions = engine.get_versions(name)
    if not versions:
        err_console.print(f"[yellow]No versions found:[/yellow] {name}")
        return
    for v in versions:
        src = _version_src_path(engine, v)
        sys.stdout.write(f"=== {name} v{v.version_num} ({v.line_count} lines, session {v.session_id[:16]}) ===\n")
        if src.exists():
            sys.stdout.write(src.read_text(errors="ignore"))
        sys.stdout.write("\n")


def _do_files_search(
    engine: SessionRecoveryEngine,
    pattern: str = "*",
    min_edits: int = 0,
    max_edits: Optional[int] = None,
    include_extensions: Optional[str] = None,
    exclude_extensions: Optional[str] = None,
    include_sessions: Optional[str] = None,
    exclude_sessions: Optional[str] = None,
    after: Optional[str] = None,
    before: Optional[str] = None,
    limit: Optional[int] = None,
    fmt: str = "table",
) -> None:
    """Search recovered files with filtering."""
    inc_ext = set(e.strip() for e in include_extensions.split(",") if e.strip()) if include_extensions else set()
    exc_ext = set(e.strip() for e in exclude_extensions.split(",") if e.strip()) if exclude_extensions else set()
    inc_sessions = set(s.strip() for s in include_sessions.split(",") if s.strip()) if include_sessions else set()
    exc_sessions = set(s.strip() for s in exclude_sessions.split(",") if s.strip()) if exclude_sessions else set()

    filters = FilterSpec(min_edits=min_edits, max_edits=max_edits, after=after, before=before)
    if inc_ext or exc_ext:
        filters.with_extensions(include=inc_ext or None, exclude=exc_ext or None)
    if inc_sessions or exc_sessions:
        filters.with_sessions(include=inc_sessions or None, exclude=exc_sessions or None)

    results = engine.search(pattern, filters)
    if limit:
        results = results[:limit]

    if not results:
        console.print("[yellow]No files found[/yellow]")
        return

    formatter = get_formatter(fmt, f"Recovered Files: {pattern}")
    console.print(formatter.format_many(results))
    console.print(f"\n[bold]Found {len(results)} files[/bold]")


def _do_messages_search(
    engine: SessionRecoveryEngine,
    query: str,
    message_type: Optional[str] = None,
    limit: int = 10,
    max_chars: int = 0,
    fmt: str = "table",
) -> None:
    """Search messages across all sessions."""
    messages_list = engine.search_messages(query, message_type)[:limit]

    if not messages_list:
        console.print("[yellow]No messages match query[/yellow]")
        return

    formatter = MessageFormatter(max_chars=max_chars)
    console.print(formatter.format_many(messages_list))
    console.print(f"\n[bold]Found {len(messages_list)} messages[/bold]")


def _do_get(
    engine: SessionRecoveryEngine,
    session_id: str,
    message_type: Optional[str] = None,
    limit: int = 10,
    max_chars: int = 0,
    fmt: str = "table",
) -> None:
    """Get messages from a specific session."""
    messages_list = engine.get_messages(session_id, message_type)[:limit]

    if not messages_list:
        console.print("[yellow]No messages found[/yellow]")
        return

    formatter = MessageFormatter(max_chars=max_chars)
    console.print(formatter.format_many(messages_list))
    console.print(f"\n[bold]Found {len(messages_list)} messages[/bold]")


def _do_stats(engine: SessionRecoveryEngine) -> None:
    """Show recovery statistics."""
    stats_data = engine.get_statistics()
    console.print(
        f"""
[bold cyan]Recovery Statistics[/bold cyan]
  Sessions:      {stats_data.total_sessions}
  Files:         {stats_data.total_files}
  Versions:      {stats_data.total_versions}
  Largest File:  {stats_data.largest_file} ({stats_data.largest_file_edits} edits)
"""
    )


# ── files_app commands ────────────────────────────────────────────────────────

@files_app.command("search")
def files_search(
    pattern: str = typer.Option("*", "--pattern", "-p", help="Filename glob/regex to match (e.g. '*.py', 'cli*'). Default: * (all files)"),
    min_edits: int = typer.Option(0, "--min-edits", help="Only show files edited at least this many times across all sessions. Default: 0"),
    max_edits: Optional[int] = typer.Option(None, "--max-edits", help="Only show files edited at most this many times. Default: unlimited"),
    include_extensions: Optional[str] = typer.Option(None, "--include-extensions", "-i", help="Only these file extensions, comma-separated (e.g. py,md,json)"),
    exclude_extensions: Optional[str] = typer.Option(None, "--exclude-extensions", "-x", help="Skip these file extensions, comma-separated (e.g. pyc,tmp)"),
    include_sessions: Optional[str] = typer.Option(None, "--include-sessions", help="Only search these session UUIDs, comma-separated"),
    exclude_sessions: Optional[str] = typer.Option(None, "--exclude-sessions", help="Skip these session UUIDs, comma-separated"),
    after: Optional[str] = typer.Option(None, "--after", help="Only files modified after this datetime (e.g. 2026-01-15 or 2026-01-15T14:30:00)"),
    before: Optional[str] = typer.Option(None, "--before", help="Only files modified before this datetime (e.g. 2026-12-31 or 2026-12-31T23:59:59)"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Max results to return. Default: unlimited"),
    fmt: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv, plain. Default: table"),
) -> None:
    """Search source files found in Claude Code session data.

    Finds files (*.py, *.md, etc.) that Claude wrote or edited during sessions.
    Filter by extension, edit count, datetime range, or session ID.

    Examples:
        aise files search                                  # all files
        aise files search --pattern "*.py"                 # Python files only
        aise files search --min-edits 3                    # files edited 3+ times across sessions
        aise files search -i py,md --after 2026-01-15      # Python/Markdown since Jan 15
        aise files search --after 2026-01-15T14:30:00      # files modified after specific time
    """
    _do_files_search(get_engine(), pattern, min_edits, max_edits, include_extensions, exclude_extensions, include_sessions, exclude_sessions, after, before, limit, fmt)


@files_app.command("extract")
def files_extract(
    name: str = typer.Argument(..., help="Filename (e.g. cli.py). Use 'aise files search' to find names."),
    version: Optional[int] = typer.Option(
        None, "--version", "-v",
        help="Version number to extract (default: latest). Run 'files history FILENAME' to see available versions.",
    ),
    session: Optional[str] = typer.Option(
        None, "--session", "-s",
        help=(
            "Limit to versions from this session ID (prefix match). "
            "Use 'files search FILENAME' or 'files history FILENAME' to find session IDs."
        ),
    ),
    restore: bool = typer.Option(
        False, "--restore",
        help=(
            "Write to the path Claude originally created/edited this file, "
            "as recorded in session data. Adds .recovered suffix if the file already exists."
        ),
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", "-o",
        help=(
            "Write to this directory instead of stdout. "
            "Example: --output-dir ./backup  writes ./backup/cli.py"
        ),
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Show what would be extracted/written without producing any output or writing any files.",
    ),
) -> None:
    """Extract a source file (cli.py, engine.py, etc.) from Claude Code session data.

    This works on SOURCE FILES that Claude wrote or edited — NOT the session JSONL files.
    By default prints the latest version to stdout (pipe-friendly).

    Run 'files history FILENAME' to see all available versions, then --version N to get one.

    Version:
      (none)            Latest version [default]
      --version N       Specific version number

    Destination (mutually exclusive):
      (none)            Print content to stdout [default]
      --restore         Write to the path Claude originally created this file
      --output-dir DIR  Write to DIR/filename

    Use --dry-run with any destination to preview without writing.

    Examples:
        aise files history cli.py                        # see all versions first
        aise files extract cli.py                        # latest → stdout
        aise files extract cli.py --version 2            # v2 → stdout
        aise files extract cli.py > cli.py               # redirect to file
        aise files extract cli.py | pbcopy               # pipe to clipboard
        aise files extract cli.py --restore              # restore to original path
        aise files extract cli.py --restore --dry-run    # preview restore
        aise files extract cli.py --output-dir ./backup  # write to ./backup/
    """
    _do_extract(get_engine(), name, version=version, session=session, output_dir=output_dir, restore=restore, dry_run=dry_run)


@files_app.command("history")
def files_history(
    name: str = typer.Argument(..., help="Filename (e.g. cli.py). Use 'aise files search' to find names."),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Limit to versions from this session ID (prefix match)."),
    export: bool = typer.Option(False, "--export", help="Write all versions to disk as cli_v1.py, cli_v2.py, etc."),
    export_dir: Optional[str] = typer.Option(None, "--export-dir", help="Where to write exported files. Default: versions/ alongside original path."),
    stdout_mode: bool = typer.Option(False, "--stdout", help="Print all versions to stdout with === v1 === headers (for scripting/AI)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="With --export: show what would be written without writing."),
) -> None:
    """Show version history of a source file from Claude Code session data. Read-only by default.

    Displays a table of all recorded versions (version number, line count, Δlines, timestamp,
    session ID). READ-ONLY — no files are written unless you use --export or --stdout.

    SOURCE FILES ONLY: shows history of files Claude wrote/edited, not the session JSONL files.

    Examples:
        aise files history cli.py                           # show version table
        aise files history cli.py --export                  # write cli_v1.py, cli_v2.py, ...
        aise files history cli.py --export --dry-run        # preview export
        aise files history cli.py --export --export-dir ./versions
        aise files history cli.py --stdout                  # all versions to stdout
    """
    engine = get_engine()
    versions = engine.get_versions(name)

    if session:
        versions = [v for v in versions if v.session_id.startswith(session)]

    if not versions:
        err_console.print(f"[red]No versions found for:[/red] {name}  (check filters)")
        raise typer.Exit(code=1)

    if stdout_mode:
        _do_history_stdout(engine, name, versions=versions)
    else:
        _do_history_display(engine, name, versions=versions)
        if export:
            _do_history_export(engine, name, export_dir=export_dir, dry_run=dry_run, versions=versions)
        elif dry_run:
            console.print("[yellow]--dry-run has no effect without --export[/yellow]")


# ── messages_app commands ─────────────────────────────────────────────────────

@messages_app.command("search")
def messages_search(
    query: str = typer.Option(..., "--query", "-q", help="Text to search for in user/assistant message content"),
    message_type: Optional[str] = typer.Option(None, "--type", "-t", help="Show only 'user' or 'assistant' messages. Default: both"),
    limit: int = typer.Option(10, "--limit", help="Max messages to return. Default: 10"),
    max_chars: int = typer.Option(0, "--max-chars", help="Truncate each message to this many characters. 0 = show full message. Default: 0"),
    fmt: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv, plain. Default: table"),
) -> None:
    """Search user/assistant conversation messages across all sessions.

    Searches the text content of messages stored in Claude Code session JSONL files.

    Examples:
        aise messages search --query "authentication"
        aise messages search --query "error" --type user --limit 20
        aise messages search --query "TODO" --max-chars 200
    """
    _do_messages_search(get_engine(), query, message_type, limit, max_chars, fmt)


@messages_app.command("get")
def messages_get(
    session_id: str = typer.Option(..., "--session", "-s", help="Session UUID (e.g. ab841016-f07b-444c-bb18-22f6b373be52). Find IDs via 'aise search' or in ~/.claude/projects/"),
    message_type: Optional[str] = typer.Option(None, "--type", "-t", help="Show only 'user' or 'assistant' messages. Default: both"),
    limit: int = typer.Option(10, "--limit", help="Max messages to return. Default: 10"),
    max_chars: int = typer.Option(0, "--max-chars", help="Truncate each message to this many characters. 0 = show full message. Default: 0"),
    fmt: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv, plain. Default: table"),
) -> None:
    """Read messages from one specific Claude Code session.

    Each session has a UUID like ab841016-f07b-444c-bb18-22f6b373be52.
    Find session IDs by running 'aise search' or listing ~/.claude/projects/*/.

    Examples:
        aise messages get --session ab841016-f07b-444c-bb18-22f6b373be52
        aise messages get -s ab841016 --type user --limit 50
    """
    _do_get(get_engine(), session_id, message_type, limit, max_chars, fmt)


# ── Root commands ─────────────────────────────────────────────────────────────

@app.command()
def search(
    domain: Optional[str] = typer.Argument(None, help="Optional: 'files' (source files) or 'messages' (conversation text). Omit to auto-detect from flags.", metavar="[files|messages]"),
    pattern: Optional[str] = typer.Option(None, "--pattern", "-p", help="[files] Filename glob/regex to match (e.g. '*.py', 'cli*'). Default: * (all files)"),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="[messages] Text to search for in user/assistant message content"),
    min_edits: int = typer.Option(0, "--min-edits", help="[files] Only show files edited at least this many times across all sessions. Default: 0"),
    max_edits: Optional[int] = typer.Option(None, "--max-edits", help="[files] Only show files edited at most this many times. Default: unlimited"),
    include_extensions: Optional[str] = typer.Option(None, "--include-extensions", "-i", help="[files] Only these file extensions, comma-separated (e.g. py,md,json)"),
    exclude_extensions: Optional[str] = typer.Option(None, "--exclude-extensions", "-x", help="[files] Skip these file extensions, comma-separated (e.g. pyc,tmp)"),
    include_sessions: Optional[str] = typer.Option(None, "--include-sessions", help="[files] Only search these session UUIDs, comma-separated"),
    exclude_sessions: Optional[str] = typer.Option(None, "--exclude-sessions", help="[files] Skip these session UUIDs, comma-separated"),
    after: Optional[str] = typer.Option(None, "--after", help="[files] Only files modified after this datetime (e.g. 2026-01-15 or 2026-01-15T14:30:00)"),
    before: Optional[str] = typer.Option(None, "--before", help="[files] Only files modified before this datetime (e.g. 2026-12-31 or 2026-12-31T23:59:59)"),
    message_type: Optional[str] = typer.Option(None, "--type", "-t", help="[messages] Show only 'user' or 'assistant' messages. Default: both"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Max results to return. Default: unlimited for files, 10 for messages"),
    max_chars: int = typer.Option(0, "--max-chars", help="[messages] Truncate each message to this many characters. 0 = full message. Default: 0"),
    fmt: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv, plain. Default: table"),
) -> None:
    """
    Search source files and/or conversation messages from Claude Code sessions.

    Searches two kinds of data:
      files    — source code files (*.py, *.md, etc.) that Claude wrote or edited
      messages — user/assistant conversation text from session JSONL files

    Auto-detects what to search based on flags:
      --pattern only → searches files
      --query only   → searches messages
      both flags     → searches both, shows two sections
      neither flag   → lists all source files (sorted by edit count)

    Flags marked [files] only apply to file search; [messages] only apply to message search.

    Examples:
        aise search                                    # list all source files
        aise search files --pattern "*.py"             # Python files only
        aise search messages --query "error"           # messages containing "error"
        aise search --pattern "*.py" --query "error"   # both: Python files + messages with "error"
        aise search files --min-edits 5 -i py,md       # heavily-edited Python/Markdown files
    """
    # Validate domain
    if domain is not None and domain not in ("files", "messages"):
        typer.echo(f"Error: domain must be 'files' or 'messages', got '{domain}'", err=True)
        raise typer.Exit(1)

    # Validate flag/domain conflicts
    if domain == "messages" and pattern is not None:
        typer.echo("Error: --pattern applies to files, not messages. Use --query.", err=True)
        raise typer.Exit(1)
    if domain == "files" and query is not None:
        typer.echo("Error: --query applies to messages, not files. Use --pattern.", err=True)
        raise typer.Exit(1)

    # Auto-detect domain from flags when not explicitly specified
    if domain is None:
        if query is not None and pattern is None:
            domain = "messages"
        elif pattern is not None and query is None:
            domain = "files"
        elif pattern is not None and query is not None:
            domain = "both"
        else:
            domain = "files"  # default: show all files

    engine = get_engine()

    if domain in ("files", "both"):
        _do_files_search(
            engine, pattern or "*", min_edits, max_edits,
            include_extensions, exclude_extensions,
            include_sessions, exclude_sessions,
            after, before, limit, fmt,
        )

    if domain in ("messages", "both"):
        msg_limit = limit if limit is not None else 10
        _do_messages_search(engine, query or "", message_type, msg_limit, max_chars, fmt)


@app.command()
def extract(
    name: str = typer.Argument(..., help="Filename to extract (e.g. cli.py). Use 'aise search' to find available names."),
    version: Optional[int] = typer.Option(
        None, "--version", "-v",
        help="Version number to extract (default: latest). Run 'history FILENAME' to see available versions.",
    ),
    session: Optional[str] = typer.Option(
        None, "--session", "-s",
        help="Limit to versions from this session ID (prefix match).",
    ),
    restore: bool = typer.Option(
        False, "--restore",
        help="Write to the path Claude originally created/edited this file.",
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", "-o",
        help="Write to this directory instead of stdout. Example: --output-dir ./backup",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Show what would be extracted/written without producing any output.",
    ),
) -> None:
    """Extract a source file from Claude Code session data (stdout by default).

    Equivalent to 'aise files extract'. Use 'aise search' to find available filenames.
    By default prints the latest version to stdout (pipe-friendly).
    """
    _do_extract(get_engine(), name, version=version, session=session, output_dir=output_dir, restore=restore, dry_run=dry_run)


@app.command()
def history(
    name: str = typer.Argument(..., help="Filename to show history for (e.g. cli.py)."),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Limit to versions from this session ID (prefix match)."),
    export: bool = typer.Option(False, "--export", help="Write all versions to disk as cli_v1.py, cli_v2.py, etc."),
    export_dir: Optional[str] = typer.Option(None, "--export-dir", help="Where to write exported files."),
    stdout_mode: bool = typer.Option(False, "--stdout", help="Print all versions to stdout with === v1 === headers."),
    dry_run: bool = typer.Option(False, "--dry-run", help="With --export: show what would be written without writing."),
) -> None:
    """Show version history of a source file (read-only by default).

    Equivalent to 'aise files history'. Creates a table of all recorded versions.
    READ-ONLY — no files are written unless you use --export or --stdout.
    """
    engine = get_engine()
    versions = engine.get_versions(name)

    if session:
        versions = [v for v in versions if v.session_id.startswith(session)]

    if not versions:
        err_console.print(f"[red]No versions found for:[/red] {name}  (check filters)")
        raise typer.Exit(code=1)

    if stdout_mode:
        _do_history_stdout(engine, name, versions=versions)
    else:
        _do_history_display(engine, name, versions=versions)
        if export:
            _do_history_export(engine, name, export_dir=export_dir, dry_run=dry_run, versions=versions)
        elif dry_run:
            console.print("[yellow]--dry-run has no effect without --export[/yellow]")


@app.command()
def get(
    session_id: str = typer.Option(..., "--session", "-s", help="Session UUID (e.g. ab841016-f07b-444c-bb18-22f6b373be52)"),
    message_type: Optional[str] = typer.Option(None, "--type", "-t", help="Show only 'user' or 'assistant' messages. Default: both"),
    limit: int = typer.Option(10, "--limit", help="Max messages to return. Default: 10"),
    max_chars: int = typer.Option(0, "--max-chars", help="Truncate each message to this many characters. 0 = show full message. Default: 0"),
    fmt: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv, plain. Default: table"),
) -> None:
    """Read messages from one specific Claude Code session.

    Equivalent to 'aise messages get'. Session IDs are UUIDs found in ~/.claude/projects/
    or in output from 'aise search'.
    """
    _do_get(get_engine(), session_id, message_type, limit, max_chars, fmt)


@app.command()
def stats() -> None:
    """Show counts of sessions, files, versions, and the most-edited file."""
    _do_stats(get_engine())


# ── Entry point ───────────────────────────────────────────────────────────────

def cli_main():
    """CLI entry point."""
    app()
