"""
Thin CLI layer - orchestrates library components without business logic.

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

app = typer.Typer(help="AI Session Tools - Analyze and extract Claude Code session data")
console = Console()


def get_engine(projects_dir: Optional[str] = None, recovery_dir: Optional[str] = None) -> SessionRecoveryEngine:
    """
    Get or create recovery engine.

    Supports environment variables:
        AI_SESSION_TOOLS_PROJECTS: Path to Claude projects directory
        AI_SESSION_TOOLS_RECOVERY: Path to recovery directory

    Args:
        projects_dir: Path to Claude projects directory (defaults to env var or ~/.claude/projects)
        recovery_dir: Path to recovery directory (defaults to env var or ~/.claude/recovery)
    """
    if projects_dir is None:
        projects_dir = os.getenv("AI_SESSION_TOOLS_PROJECTS", str(Path.home() / ".claude" / "projects"))
    if recovery_dir is None:
        recovery_dir = os.getenv("AI_SESSION_TOOLS_RECOVERY", str(Path.home() / ".claude" / "recovery"))
    return SessionRecoveryEngine(Path(projects_dir), Path(recovery_dir))


@app.command()
def search(
    pattern: str = typer.Option("*", "--pattern", "-p", help="File pattern (glob or regex). Default: all files"),
    min_edits: int = typer.Option(0, "--min-edits", help="Minimum number of edits. Default: 0 (no minimum)"),
    max_edits: Optional[int] = typer.Option(None, "--max-edits", help="Maximum number of edits. Default: unlimited"),
    include_types: Optional[str] = typer.Option(None, "--include-types", "-t", help="Include only these file types (comma-separated, e.g., py,md,json)"),
    exclude_types: Optional[str] = typer.Option(None, "--exclude-types", "-x", help="Exclude these file types (comma-separated, e.g., pyc,tmp)"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv, plain. Default: table"),
):
    """
    Search and list recovered session files (*.py, *.md, etc).

    This searches FILES that were recovered from Claude Code sessions,
    not the session message data (JSONL). Use 'session-messages' for message search.

    Examples:
        ai_session_tools search --pattern "*.py"
        ai_session_tools search --pattern "*.py" --min-edits 5
        ai_session_tools search --include-types py,ts,js  # Only Python, TypeScript, JavaScript
        ai_session_tools search --exclude-types pyc,tmp   # Everything except temp files
        ai_session_tools search  # Lists all files
    """
    engine = get_engine()

    # Parse optional extension filters
    include_ext_set = set(include_types.split(",")) if include_types else set()
    exclude_ext_set = set(exclude_types.split(",")) if exclude_types else set()

    filters = FilterSpec(min_edits=min_edits, max_edits=max_edits)
    if include_ext_set or exclude_ext_set:
        filters.with_extensions(include=include_ext_set if include_ext_set else None,
                                exclude=exclude_ext_set if exclude_ext_set else None)

    results = engine.search(pattern, filters)

    if not results:
        console.print("[yellow]No files found[/yellow]")
        return

    formatter = get_formatter(format, f"Recovered Files: {pattern}")
    console.print(formatter.format_many(results))
    console.print(f"\n[bold]Found {len(results)} files[/bold]")


@app.command()
def extract(
    file: str = typer.Option(..., "--file", "-f", help="Filename to extract (e.g., session_manager.py)"),
    output_dir: str = typer.Option("./recovered", "--output-dir", "-o", help="Output directory for extracted file. Default: ./recovered"),
):
    """
    Extract the final (most recent) version of a recovered file.

    This recovers a single file from Claude Code session backups.
    Use 'history' to extract all versions of a file instead.

    Examples:
        ai_session_tools extract --file session_manager.py
        ai_session_tools extract --file config.yaml --output-dir /tmp/recovery
    """
    engine = get_engine()
    path = engine.extract_final(file, Path(output_dir))

    if path:
        console.print(f"[green]✓ Extracted:[/green] {path}")
    else:
        console.print(f"[red]✗ File not found:[/red] {file}")
        raise typer.Exit(code=1)


@app.command()
def history(
    file: str = typer.Option(..., "--file", "-f", help="Filename (e.g., session_manager.py)"),
    output_dir: str = typer.Option("./recovered", "--output-dir", "-o", help="Output directory for all versions. Default: ./recovered"),
):
    """
    Extract ALL versions (edit history) of a recovered file.

    Shows the evolution of a file across all Claude Code sessions.
    Each version is saved as v[N]_line_[count].txt showing version number and line count.

    Examples:
        ai_session_tools history --file session_manager.py
        ai_session_tools history --file config.yaml --output-dir /tmp/recovery
    """
    engine = get_engine()
    versions = engine.get_versions(file)

    if not versions:
        console.print(f"[yellow]No versions found for: {file}[/yellow]")
        return

    paths = engine.extract_all(file, Path(output_dir))

    if paths:
        console.print(f"[green]✓ Extracted {len(paths)} versions to:[/green] {output_dir}")
    else:
        console.print(f"[red]✗ Failed to extract:[/red] {file}")
        raise typer.Exit(code=1)


@app.command()
def session_messages(
    session_id: str = typer.Option(..., "--session", "-s", help="Session ID to query"),
    msg_type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by message type: 'user' or 'assistant'. Default: all types"),
    limit: int = typer.Option(10, "--limit", help="Maximum number of messages to return. Default: 10"),
):
    """
    Extract messages from a Claude Code SESSION (JSONL data).

    This retrieves conversation messages from a specific session stored in JSONL format.
    Different from 'search' which finds recovered files (*.py, *.md, etc).

    Use 'search-session-messages' to search across multiple sessions for text content.

    Examples:
        ai_session_tools session-messages --session abc123
        ai_session_tools session-messages --session abc123 --type user --limit 20
        ai_session_tools session-messages --session abc123 --type assistant
    """
    engine = get_engine()
    messages_list = engine.get_messages(session_id, msg_type)[:limit]

    if not messages_list:
        console.print("[yellow]No messages found[/yellow]")
        return

    formatter = MessageFormatter()
    console.print(formatter.format_many(messages_list))
    console.print(f"\n[bold]Found {len(messages_list)} messages[/bold]")


@app.command()
def search_session_messages(
    query: str = typer.Option(..., "--query", "-q", help="Text to search for in session messages"),
    limit: int = typer.Option(10, "--limit", help="Maximum number of results to return. Default: 10"),
):
    """
    Search for text WITHIN session messages across ALL SESSIONS (JSONL data).

    This searches the content of conversation messages across all Claude Code sessions,
    not filenames. Different from 'search' which finds recovered files (*.py, *.md, etc).

    Use 'session-messages' to retrieve messages from a specific known session.

    Examples:
        ai_session_tools search-session-messages --query "function definition"
        ai_session_tools search-session-messages --query "error" --limit 20
        ai_session_tools search-session-messages --query "TODO"
    """
    engine = get_engine()
    messages_list = engine.search_messages(query)[:limit]

    if not messages_list:
        console.print("[yellow]No messages match query[/yellow]")
        return

    formatter = MessageFormatter()
    console.print(formatter.format_many(messages_list))
    console.print(f"\n[bold]Found {len(messages_list)} messages[/bold]")


@app.command()
def stats():
    """
    Show recovery statistics and summary.

    Displays overall statistics about recovered files and sessions:
    - Total sessions: unique Claude Code sessions found
    - Total files: unique recovered files
    - Total versions: total edits across all files
    - Largest file: file with most edits

    Examples:
        ai_session_tools stats
    """
    engine = get_engine()
    stats = engine.get_statistics()

    console.print(
        f"""
[bold cyan]Recovery Statistics[/bold cyan]
  Sessions:      {stats.total_sessions}
  Files:         {stats.total_files}
  Versions:      {stats.total_versions}
  Largest File:  {stats.largest_file} ({stats.largest_file_edits} edits)
"""
    )


def cli_main():
    """CLI entry point."""
    app()
