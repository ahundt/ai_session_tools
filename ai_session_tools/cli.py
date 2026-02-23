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
    pattern: str = typer.Option(..., "--pattern", "-p", help="File pattern (glob or regex)"),
    min_edits: int = typer.Option(0, "--min-edits"),
    max_edits: Optional[int] = typer.Option(None, "--max-edits"),
    format: str = typer.Option("table", "--format", "-f"),
):
    """Search for files by pattern with filtering."""
    engine = get_engine()
    filters = FilterSpec(min_edits=min_edits, max_edits=max_edits)

    results = engine.search(pattern, filters)

    if not results:
        console.print("[yellow]No files found[/yellow]")
        return

    formatter = get_formatter(format, f"Search: {pattern}")
    console.print(formatter.format_many(results))
    console.print(f"\n[bold]Found {len(results)} files[/bold]")


@app.command()
def extract(
    file: str = typer.Option(..., "--file", "-f", help="Filename to extract"),
    output_dir: str = typer.Option("./recovered", "--output-dir", "-o"),
):
    """Extract final version of a file."""
    engine = get_engine()
    path = engine.extract_final(file, Path(output_dir))

    if path:
        console.print(f"[green]✓ Extracted:[/green] {path}")
    else:
        console.print(f"[red]✗ File not found:[/red] {file}")
        raise typer.Exit(code=1)


@app.command()
def history(
    file: str = typer.Option(..., "--file", "-f", help="Filename"),
    output_dir: str = typer.Option("./recovered", "--output-dir", "-o"),
):
    """Extract version history of a file."""
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
def list_files(
    limit: int = typer.Option(20, "--limit"),
    format: str = typer.Option("table", "--format", "-f"),
):
    """List all recovered files sorted by edit count."""
    engine = get_engine()
    results = engine.search(".*", FilterSpec())[:limit]

    if not results:
        console.print("[yellow]No files found[/yellow]")
        return

    formatter = get_formatter(format, "Recovered Files")
    console.print(formatter.format_many(results))
    console.print(f"\n[bold]Showing {len(results)} files[/bold]")


@app.command()
def messages(
    session_id: str = typer.Option(..., "--session", "-s", help="Session ID"),
    msg_type: Optional[str] = typer.Option(None, "--type", "-t", help="Message type (user/assistant)"),
    limit: int = typer.Option(10, "--limit"),
):
    """Extract messages from a session."""
    engine = get_engine()
    messages_list = engine.get_messages(session_id, msg_type)[:limit]

    if not messages_list:
        console.print("[yellow]No messages found[/yellow]")
        return

    formatter = MessageFormatter()
    console.print(formatter.format_many(messages_list))
    console.print(f"\n[bold]Found {len(messages_list)} messages[/bold]")


@app.command()
def search_messages(
    query: str = typer.Option(..., "--query", "-q", help="Search query"),
    limit: int = typer.Option(10, "--limit"),
):
    """Search for text in session messages."""
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
    """Show recovery statistics."""
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
