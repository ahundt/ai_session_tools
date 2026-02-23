"""
Thin CLI layer - orchestrates library components without business logic.

Supports dual-ordering: both `ais search files` and `ais files search` work.
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

app = typer.Typer(help="AI Session Tools — analyze and extract Claude Code session data")
files_app = typer.Typer(help="Commands scoped to recovered files (*.py, *.md, etc.)")
messages_app = typer.Typer(help="Commands scoped to session messages (JSONL data)")

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
    return SessionRecoveryEngine(Path(projects_dir), Path(recovery_dir))


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
    after_date: Optional[str] = None,
    before_date: Optional[str] = None,
    limit: Optional[int] = None,
    fmt: str = "table",
) -> None:
    """Search recovered files with filtering."""
    inc_ext = set(e.strip() for e in include_extensions.split(",") if e.strip()) if include_extensions else set()
    exc_ext = set(e.strip() for e in exclude_extensions.split(",") if e.strip()) if exclude_extensions else set()
    inc_sessions = set(s.strip() for s in include_sessions.split(",") if s.strip()) if include_sessions else set()
    exc_sessions = set(s.strip() for s in exclude_sessions.split(",") if s.strip()) if exclude_sessions else set()

    filters = FilterSpec(min_edits=min_edits, max_edits=max_edits, after_date=after_date, before_date=before_date)
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
    path = engine.extract_final(name, Path(output_dir))
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

    paths = engine.extract_all(name, Path(output_dir))
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
    pattern: str = typer.Option("*", "--pattern", "-p", help="Filename glob/regex. Default: *"),
    min_edits: int = typer.Option(0, "--min-edits", help="Minimum times this file was edited across all sessions. Default: 0"),
    max_edits: Optional[int] = typer.Option(None, "--max-edits", help="Maximum edits. Default: unlimited"),
    include_extensions: Optional[str] = typer.Option(None, "--include-extensions", "-i", help="Include only these extensions (comma-separated, e.g. py,md,json)"),
    exclude_extensions: Optional[str] = typer.Option(None, "--exclude-extensions", "-x", help="Exclude these extensions (comma-separated, e.g. pyc,tmp)"),
    include_sessions: Optional[str] = typer.Option(None, "--include-sessions", help="Include only these session IDs (comma-separated)"),
    exclude_sessions: Optional[str] = typer.Option(None, "--exclude-sessions", help="Exclude these session IDs (comma-separated)"),
    after_date: Optional[str] = typer.Option(None, "--after-date", help="Only files modified after this date (ISO: 2026-01-01)"),
    before_date: Optional[str] = typer.Option(None, "--before-date", help="Only files modified before this date (ISO: 2026-12-31)"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Max results. Default: unlimited"),
    fmt: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv, plain. Default: table"),
) -> None:
    """Search recovered session files with filtering."""
    _do_files_search(get_engine(), pattern, min_edits, max_edits, include_extensions, exclude_extensions, include_sessions, exclude_sessions, after_date, before_date, limit, fmt)


@files_app.command("extract")
def files_extract(
    name: str = typer.Option(..., "--name", "-n", help="Filename to extract (e.g. session_manager.py)"),
    output_dir: str = typer.Option("./recovered", "--output-dir", "-o", help="Output directory. Default: ./recovered"),
) -> None:
    """Extract the final (most recent) version of a recovered file."""
    _do_extract(get_engine(), name, output_dir)


@files_app.command("history")
def files_history(
    name: str = typer.Option(..., "--name", "-n", help="Filename (e.g. session_manager.py)"),
    output_dir: str = typer.Option("./recovered", "--output-dir", "-o", help="Output directory. Default: ./recovered"),
) -> None:
    """Extract ALL versions (edit history) of a recovered file."""
    _do_history(get_engine(), name, output_dir)


# ── messages_app commands ─────────────────────────────────────────────────────

@messages_app.command("search")
def messages_search(
    query: str = typer.Option(..., "--query", "-q", help="Text to search for in message content"),
    message_type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter: user | assistant. Default: all"),
    limit: int = typer.Option(10, "--limit", help="Max results. Default: 10"),
    max_chars: int = typer.Option(0, "--max-chars", help="Max chars per message. 0 = full content. Default: 0"),
    fmt: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv, plain. Default: table"),
) -> None:
    """Search for text within session messages across all sessions."""
    _do_messages_search(get_engine(), query, message_type, limit, max_chars, fmt)


@messages_app.command("get")
def messages_get(
    session_id: str = typer.Option(..., "--session", "-s", help="Session ID"),
    message_type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter: user | assistant. Default: all"),
    limit: int = typer.Option(10, "--limit", help="Max results. Default: 10"),
    max_chars: int = typer.Option(0, "--max-chars", help="Max chars per message. 0 = full content. Default: 0"),
    fmt: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv, plain. Default: table"),
) -> None:
    """Get messages from a specific session."""
    _do_get(get_engine(), session_id, message_type, limit, max_chars, fmt)


# ── Root commands ─────────────────────────────────────────────────────────────

@app.command()
def search(
    domain: Optional[str] = typer.Argument(None, help="Narrow to domain: 'files' or 'messages'. Omit to auto-detect from flags.", metavar="[files|messages]"),
    pattern: Optional[str] = typer.Option(None, "--pattern", "-p", help="Filename glob/regex (files only). Default: *"),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Text to search in messages (messages only)"),
    min_edits: int = typer.Option(0, "--min-edits", help="Minimum file edits across all sessions. Default: 0"),
    max_edits: Optional[int] = typer.Option(None, "--max-edits", help="Maximum file edits. Default: unlimited"),
    include_extensions: Optional[str] = typer.Option(None, "--include-extensions", "-i", help="Include only these extensions (comma-separated)"),
    exclude_extensions: Optional[str] = typer.Option(None, "--exclude-extensions", "-x", help="Exclude these extensions (comma-separated)"),
    include_sessions: Optional[str] = typer.Option(None, "--include-sessions", help="Include only these session IDs (comma-separated)"),
    exclude_sessions: Optional[str] = typer.Option(None, "--exclude-sessions", help="Exclude these session IDs (comma-separated)"),
    after_date: Optional[str] = typer.Option(None, "--after-date", help="Only files modified after this date (ISO: 2026-01-01)"),
    before_date: Optional[str] = typer.Option(None, "--before-date", help="Only files modified before this date"),
    message_type: Optional[str] = typer.Option(None, "--type", "-t", help="Message type filter: user | assistant"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Max results. Default: unlimited for files, 10 for messages"),
    max_chars: int = typer.Option(0, "--max-chars", help="Max chars per message. 0 = full content. Default: 0"),
    fmt: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv, plain. Default: table"),
) -> None:
    """
    Search recovered files and/or session messages.

    Without a domain argument, auto-detects from flags:
      --pattern only → files search
      --query only → messages search
      both → shows both sections
      neither → files search (all files, sorted by edits)

    Examples:
        ais search                                    # all files
        ais search files --pattern "*.py"             # Python files only
        ais search messages --query "error"           # messages containing "error"
        ais search --pattern "*.py" --query "error"   # both domains
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
            after_date, before_date, limit, fmt,
        )

    if domain in ("messages", "both"):
        msg_limit = limit if limit is not None else 10
        _do_messages_search(engine, query or "", message_type, msg_limit, max_chars, fmt)


@app.command()
def extract(
    name: str = typer.Option(..., "--name", "-n", help="Filename to extract (e.g. session_manager.py)"),
    output_dir: str = typer.Option("./recovered", "--output-dir", "-o", help="Output directory. Default: ./recovered"),
) -> None:
    """Extract the final (most recent) version of a recovered file."""
    _do_extract(get_engine(), name, output_dir)


@app.command()
def history(
    name: str = typer.Option(..., "--name", "-n", help="Filename (e.g. session_manager.py)"),
    output_dir: str = typer.Option("./recovered", "--output-dir", "-o", help="Output directory. Default: ./recovered"),
) -> None:
    """Extract ALL versions (edit history) of a recovered file."""
    _do_history(get_engine(), name, output_dir)


@app.command()
def get(
    session_id: str = typer.Option(..., "--session", "-s", help="Session ID to query"),
    message_type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter: user | assistant. Default: all"),
    limit: int = typer.Option(10, "--limit", help="Max results. Default: 10"),
    max_chars: int = typer.Option(0, "--max-chars", help="Max chars per message. 0 = full content. Default: 0"),
    fmt: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv, plain. Default: table"),
) -> None:
    """Get messages from a specific Claude Code session."""
    _do_get(get_engine(), session_id, message_type, limit, max_chars, fmt)


@app.command()
def stats() -> None:
    """Show recovery statistics and summary."""
    _do_stats(get_engine())


# ── Entry point ───────────────────────────────────────────────────────────────

def cli_main():
    """CLI entry point."""
    app()
