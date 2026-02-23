"""
Thin CLI layer - orchestrates library components without business logic.

Supports dual-ordering: both `aise search files` and `aise files search` work.
Root commands default to broad behavior; domain groups narrow the scope.

Copyright (c) 2026 Andrew Hundt
Licensed under the Apache License, Version 2.0
"""

import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .engine import SessionRecoveryEngine
from .formatters import MessageFormatter, get_formatter
from .models import FilterSpec

app = typer.Typer(
    help=(
        "Analyze and extract data from Claude Code sessions stored in ~/.claude/projects/.\n\n"
        "Claude Code stores each conversation as a session — a folder of JSONL files containing "
        "user/assistant messages and source code snapshots. This tool searches those sessions to "
        "find source files that were written or edited, and to search conversation messages.\n\n"
        "Override default paths with environment variables:\n\n"
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


# ── Engine factory ────────────────────────────────────────────────────────────

def get_engine(projects_dir: Optional[str] = None, recovery_dir: Optional[str] = None) -> SessionRecoveryEngine:
    """
    Get or create recovery engine.

    Supports environment variables:
        AI_SESSION_TOOLS_PROJECTS: Path to Claude projects directory
        AI_SESSION_TOOLS_RECOVERY: Path to recovery directory
    """
    if projects_dir is None:
        projects_dir = os.getenv("AI_SESSION_TOOLS_PROJECTS", str(Path.home() / ".claude" / "projects"))
    if recovery_dir is None:
        recovery_dir = os.getenv("AI_SESSION_TOOLS_RECOVERY", str(Path.home() / ".claude" / "recovery"))
    # expanduser() so that env var values like "~/.claude/projects" work correctly
    return SessionRecoveryEngine(Path(projects_dir).expanduser(), Path(recovery_dir).expanduser())


# ── Shared helper functions (business logic lives here, not in CLI funcs) ─────

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


def _do_extract(engine: SessionRecoveryEngine, name: str, output_dir: str) -> None:
    """Extract final version of a file."""
    path = engine.extract_final(name, Path(output_dir).expanduser())
    if path:
        console.print(f"[green]Extracted:[/green] {path}")
    else:
        console.print(f"[red]File not found:[/red] {name}")
        raise typer.Exit(code=1)


def _do_history(engine: SessionRecoveryEngine, name: str, output_dir: str) -> None:
    """Extract all versions of a file."""
    versions = engine.get_versions(name)
    if not versions:
        console.print(f"[yellow]No versions found for: {name}[/yellow]")
        return

    paths = engine.extract_all(name, Path(output_dir).expanduser())
    if paths:
        console.print(f"[green]Extracted {len(paths)} versions to:[/green] {output_dir}")
    else:
        console.print(f"[red]Failed to extract:[/red] {name}")
        raise typer.Exit(code=1)


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
    name: str = typer.Option(..., "--name", "-n", help="Filename to save (e.g. cli.py). Use 'aise files search' to find available names."),
    output_dir: str = typer.Option("./recovered", "--output-dir", "-o", help="Directory to write the file to. Default: ./recovered"),
) -> None:
    """Save the most recent version of a source file found in session data.

    Writes the file to --output-dir. Use 'aise files search' to find available filenames.

    Examples:
        aise files extract --name cli.py
        aise files extract --name cli.py --output-dir ./backup
    """
    _do_extract(get_engine(), name, output_dir)


@files_app.command("history")
def files_history(
    name: str = typer.Option(..., "--name", "-n", help="Filename to extract all versions of (e.g. cli.py)"),
    output_dir: str = typer.Option("./recovered", "--output-dir", "-o", help="Output directory. Default: ./recovered"),
) -> None:
    """Save every recorded version of a source file, showing its edit history.

    Creates numbered files (e.g. cli_v1.py, cli_v2.py, ...) in --output-dir.
    Each version corresponds to one edit found across all sessions.

    Examples:
        aise files history --name cli.py
        aise files history --name cli.py --output-dir ./versions
    """
    _do_history(get_engine(), name, output_dir)


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
    name: str = typer.Option(..., "--name", "-n", help="Filename to save (e.g. cli.py). Use 'aise search' to find available names."),
    output_dir: str = typer.Option("./recovered", "--output-dir", "-o", help="Directory to write the file to. Default: ./recovered"),
) -> None:
    """Save the most recent version of a source file found in session data.

    Equivalent to 'aise files extract'. Use 'aise search' to find available filenames.
    """
    _do_extract(get_engine(), name, output_dir)


@app.command()
def history(
    name: str = typer.Option(..., "--name", "-n", help="Filename to extract all versions of (e.g. cli.py)"),
    output_dir: str = typer.Option("./recovered", "--output-dir", "-o", help="Directory to write version files to. Default: ./recovered"),
) -> None:
    """Save every recorded version of a source file, showing its edit history.

    Equivalent to 'aise files history'. Creates numbered files (e.g. cli_v1.py, cli_v2.py).
    """
    _do_history(get_engine(), name, output_dir)


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
