"""
Thin CLI layer - orchestrates library components without business logic.

Supports dual-ordering: both `aise search files` and `aise files search` work.
Root commands default to broad behavior; domain groups narrow the scope.

Copyright (c) 2026 Andrew Hundt
Licensed under the Apache License, Version 2.0
"""

import json
import os
import re as _re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable as _Callable, List, Optional, Set

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
export_app = typer.Typer(
    help=(
        "Export Claude Code session messages to markdown.\n\n"
        "Use 'export session' for a single session, 'export recent' for bulk export."
    ),
)
tools_app = typer.Typer(
    help=(
        "Search tool invocations (Bash, Edit, Write, Read, etc.) from Claude Code sessions.\n\n"
        "Tool calls are stored inside assistant messages in session JSONL files."
    ),
)

config_app = typer.Typer(
    help=(
        "View and manage the ai_session_tools config file.\n\n"
        "Config file location (priority order):\n\n"
        "  1. --config CLI flag\n"
        "  2. AI_SESSION_TOOLS_CONFIG env var\n"
        "  3. OS default: ~/Library/Application Support/ai_session_tools/config.json (macOS)\n"
        "               : ~/.config/ai_session_tools/config.json (Linux)"
    ),
)

app.add_typer(files_app, name="files", rich_help_panel="Domain Groups")
app.add_typer(messages_app, name="messages", rich_help_panel="Domain Groups")
app.add_typer(export_app, name="export", rich_help_panel="Domain Groups")
app.add_typer(tools_app, name="tools", rich_help_panel="Domain Groups")
app.add_typer(config_app, name="config", rich_help_panel="Configuration")

console = Console()
err_console = Console(stderr=True)


# ── CLI rendering infrastructure ──────────────────────────────────────────────

@dataclass
class ColumnSpec:
    """One column in a Rich table: header text + optional display hints."""

    header: str
    style: str = ""
    no_wrap: bool = False
    justify: str = "left"


@dataclass
class TableSpec:
    """Render spec for a list of model objects: table, JSON, CSV, or plain text."""

    title_template: str
    columns: List[ColumnSpec]
    row_fn: _Callable
    summary_template: Optional[str] = None
    plain_fn: Optional[_Callable] = None


def _render_output(
    items: list,
    fmt: str,
    spec: "TableSpec",
    empty_msg: str = "No results found",
) -> None:
    """Render items in the requested format — eliminates repeated json/csv/table branching."""
    if not items:
        console.print(f"[yellow]{empty_msg}[/yellow]")
        return

    def _to_dict(x):
        if hasattr(x, "to_dict"):
            return x.to_dict()
        if isinstance(x, dict):
            return x
        return vars(x)

    dicts = [_to_dict(x) for x in items]
    if fmt == "json":
        # Write directly to stdout — bypasses Rich markup rendering + ANSI codes
        sys.stdout.write(json.dumps(dicts, indent=2) + "\n")
        return
    if fmt in ("csv", "plain"):
        for d in dicts:
            line = spec.plain_fn(d) if spec.plain_fn else "  ".join(str(v) for v in spec.row_fn(d))
            console.print(line)
        return
    # Default: Rich table
    from rich.table import Table
    table = Table(title=spec.title_template.format(n=len(items)))
    for col in spec.columns:
        table.add_column(
            col.header,
            style=col.style or None,
            no_wrap=col.no_wrap,
            justify=col.justify,
        )
    for d in dicts:
        table.add_row(*[str(v) for v in spec.row_fn(d)])
    console.print(table)
    if spec.summary_template:
        console.print(f"\n[bold]{spec.summary_template.format(n=len(items))}[/bold]")


def _register_alias(sub_app: "typer.Typer", func: _Callable, *names: str) -> None:
    """Register func as a command under each name in names on sub_app."""
    for name in names:
        sub_app.command(name)(func)


# ── Module-level TableSpec constants ─────────────────────────────────────────

_LIST_SPEC = TableSpec(
    title_template="Sessions ({n} found)",
    columns=[
        ColumnSpec("Session", style="cyan", no_wrap=True),
        ColumnSpec("Project", style="blue"),
        ColumnSpec("Branch", style="green"),
        ColumnSpec("Date", style="dim"),
        ColumnSpec("Messages", justify="right"),
        ColumnSpec("Summary", style="dim"),
    ],
    row_fn=lambda d: [
        (d["session_id"][:16] + "\u2026") if len(d["session_id"]) > 16 else d["session_id"],
        d["project_dir"][-20:] if len(d["project_dir"]) > 20 else d["project_dir"],
        d["git_branch"],
        d["timestamp_first"][:10],
        str(d["message_count"]),
        "\u2713" if d["has_compact_summary"] else "",
    ],
    summary_template="Found {n} sessions",
)

_CORRECTIONS_SPEC = TableSpec(
    title_template="User Corrections ({n} found)",
    columns=[
        ColumnSpec("Timestamp", style="dim", no_wrap=True),
        ColumnSpec("Session", style="cyan", no_wrap=True),
        ColumnSpec("Category", style="yellow"),
        ColumnSpec("Pattern", style="red"),
        ColumnSpec("Message"),
    ],
    row_fn=lambda d: [
        d["timestamp"][:19],
        (d["session_id"][:16] + "\u2026") if len(d["session_id"]) > 16 else d["session_id"],
        d["category"],
        d["matched_pattern"],
        d["content"][:80],
    ],
    summary_template="Found {n} corrections",
)

_PLANNING_SPEC = TableSpec(
    title_template="Planning Command Usage ({n} commands)",
    columns=[
        ColumnSpec("Command", style="cyan"),
        ColumnSpec("Count", justify="right"),
        ColumnSpec("Sessions", justify="right"),
        ColumnSpec("Projects", justify="right"),
    ],
    row_fn=lambda d: [
        d["command"],
        str(d["count"]),
        str(d["unique_sessions"]),
        str(d["unique_projects"]),
    ],
)


# Module-level overrides set by global options
_g_claude_dir: Optional[str] = None
_g_config_path: Optional[str] = None
_config_cache: Optional[dict] = None  # lazily loaded, reset per process


def load_config() -> dict:
    """Load app config from JSON file. Returns empty dict if not found or unreadable.

    Config file location priority:
      1. ``--config`` CLI flag (set on the root app callback)
      2. ``AI_SESSION_TOOLS_CONFIG`` environment variable
      3. OS-appropriate default via ``typer.get_app_dir("ai_session_tools")``:
           - macOS: ``~/Library/Application Support/ai_session_tools/config.json``
           - Linux: ``~/.config/ai_session_tools/config.json``
           - Windows: ``%APPDATA%/ai_session_tools/config.json``

    Supported keys (all optional):

    - ``correction_patterns`` (list of strings): correction patterns in
      ``"CATEGORY:REGEX"`` format; replaces built-in defaults when present.
      Example: ``["regression:you deleted", "skip_step:you forgot"]``
    - ``planning_commands`` (list of strings): slash-command regex patterns
      to count; replaces built-in defaults when present.
      Example: ``["/ar:plannew", "/ar:pn", "/mycommand"]``

    Example ``config.json``::

        {
            "correction_patterns": [
                "regression:you deleted",
                "regression:you removed",
                "skip_step:you forgot",
                "skip_step:you missed"
            ],
            "planning_commands": [
                "/ar:plannew",
                "/ar:pn",
                "/mycommand"
            ]
        }
    """
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    # Priority: --config flag > AI_SESSION_TOOLS_CONFIG env > typer.get_app_dir default
    if _g_config_path:
        config_file = Path(_g_config_path).expanduser()
    else:
        env_val = os.getenv("AI_SESSION_TOOLS_CONFIG")
        if env_val:
            config_file = Path(env_val).expanduser()
        else:
            app_dir = typer.get_app_dir("ai_session_tools")
            config_file = Path(app_dir) / "config.json"

    if config_file.exists():
        try:
            with open(config_file, encoding="utf-8") as f:
                _config_cache = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            err_console.print(f"[yellow]Warning: could not load config {config_file}: {exc}[/yellow]")
            _config_cache = {}
    else:
        _config_cache = {}

    return _config_cache


# ── Root app callback (global options) ────────────────────────────────────────

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
    config: Optional[str] = typer.Option(
        None, "--config",
        help=(
            "Path to the ai_session_tools config JSON file. "
            "Default: OS config dir / ai_session_tools / config.json "
            "(macOS: ~/Library/Application Support/ai_session_tools/config.json, "
            "Linux: ~/.config/ai_session_tools/config.json). "
            "Also overridable via AI_SESSION_TOOLS_CONFIG env var."
        ),
        envvar="AI_SESSION_TOOLS_CONFIG",
    ),
) -> None:
    global _g_claude_dir, _g_config_path, _config_cache
    _g_claude_dir = claude_dir
    if config != _g_config_path:
        _g_config_path = config
        _config_cache = None  # invalidate cache when path changes
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


def _strip_quotes(s: str) -> str:
    """Strip surrounding single or double quotes from a stripped string."""
    s = s.strip()
    if len(s) >= 2 and s[0] in ('"', "'") and s[-1] == s[0]:
        s = s[1:-1].strip()
    return s


def _parse_list_input(raw: str) -> List[str]:
    """Parse a list from raw text or a file path. Tries parsers in order:

    1. **File path** — if ``raw`` is an existing readable path, read its contents first.
    2. **JSON array** (``json.loads``) — handles ``["a", "b"]``, ``["a","b"]``, etc.
    3. **Python literal** (``ast.literal_eval``) — handles ``['a', 'b']``, mixed quotes.
    4. **CSV fallback** — handles ``a,b,c``, ``[a,b,c]`` (brackets stripped then split).

    Examples::

        "a,b,c"                         → ["a", "b", "c"]         (CSV)
        '["/ar:pn", "/ar:pr"]'          → ["/ar:pn", "/ar:pr"]    (JSON)
        "['/ar:pn', '/ar:pr']"          → ["/ar:pn", "/ar:pr"]    (Python literal)
        '[/ar:pn, "/ar:pr"]'            → ["/ar:pn", "/ar:pr"]    (CSV with brackets)
        "/path/to/list.json"            → contents parsed above   (file path)

    Empty items (trailing commas, whitespace-only) are always discarded.
    """
    import ast

    s = raw.strip()

    # 1. File path: if the string points to an existing file, read it first
    try:
        p = Path(s).expanduser()
        if p.exists() and p.is_file():
            s = p.read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        pass  # not a valid path; continue with raw string

    # 2. JSON array (handles double-quoted items, no-space format)
    if s.startswith("["):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (json.JSONDecodeError, ValueError):
            pass

        # 3. Python literal (handles single-quoted items, mixed quotes)
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (ValueError, SyntaxError):
            pass

        # 4a. CSV with brackets: strip outer brackets then split
        inner = s[1:-1] if s.endswith("]") else s[1:]
        return [_strip_quotes(item) for item in inner.split(",") if item.strip()]

    # 4b. Plain CSV
    return [_strip_quotes(item) for item in s.split(",") if item.strip()]


def _project_dir_name(path: str) -> str:
    """Convert a working directory path to the ~/.claude/projects/ directory name.

    Matches Claude Code's own encoding: replace /[^a-zA-Z0-9-]/g with '-'.
    Hyphens are preserved; slashes, dots, underscores, spaces all become '-'.

    Source: Claude Code TypeScript source, getProjectPath() in utils/path.ts

    Examples:
        /Users/alice/.claude          → -Users-alice--claude
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


def _do_extract(  # noqa: C901
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
    inc_ext = {e.strip() for e in include_extensions.split(",") if e.strip()} if include_extensions else set()
    exc_ext = {e.strip() for e in exclude_extensions.split(",") if e.strip()} if exclude_extensions else set()
    inc_sessions = {s.strip() for s in include_sessions.split(",") if s.strip()} if include_sessions else set()
    exc_sessions = {s.strip() for s in exclude_sessions.split(",") if s.strip()} if exclude_sessions else set()

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
    tool: Optional[str] = None,
    context: int = 0,
) -> None:
    """Search messages across all sessions. When context > 0, each result includes
    up to ``context`` surrounding messages from the same session file."""
    tag = f" [tool: {tool}]" if tool else ""

    if context > 0:
        ctx_results = engine.search_messages_with_context(
            query, context=context, message_type=message_type, tool=tool
        )[:limit]
        if not ctx_results:
            console.print(f"[yellow]No messages match query{tag}[/yellow]")
            return
        if fmt == "json":
            sys.stdout.write(json.dumps([r.to_dict() for r in ctx_results], indent=2) + "\n")
            return
        # Flat display: separator + before context + match + after context.
        # truncate=None means no truncation (Python slice [:None] returns full string).
        truncate = max_chars if max_chars > 0 else None
        for cm in ctx_results:
            console.print(f"[dim]{'─' * 60}[/dim]")
            for m in cm.context_before:
                console.print(f"[dim][{m.type.value}] {m.timestamp[:19]}[/dim]")
                console.print(f"[dim]{m.content[:truncate]}[/dim]\n")
            console.print(f"[bold cyan][{cm.match.type.value}] {cm.match.timestamp[:19]}[/bold cyan]")
            console.print(f"{cm.match.content[:truncate]}\n")
            for m in cm.context_after:
                console.print(f"[dim][{m.type.value}] {m.timestamp[:19]}[/dim]")
                console.print(f"[dim]{m.content[:truncate]}[/dim]\n")
        console.print(f"\n[bold]Found {len(ctx_results)} matches{tag} (context ±{context})[/bold]")
        return

    results = engine.search_messages(query, message_type, tool=tool)[:limit]

    if not results:
        console.print(f"[yellow]No messages match query{tag}[/yellow]")
        return

    if fmt in ("json", "csv", "plain"):
        # truncate=None means no limit (Python [:None] returns full string).
        # JSON format ignores row_fn entirely (uses to_dict() directly) so content
        # is always complete. csv/plain respects max_chars if set, else no truncation.
        truncate = max_chars if max_chars > 0 else None
        spec = TableSpec(
            title_template=f"Messages ({{n}} found){tag}",
            columns=[
                ColumnSpec("Timestamp", style="dim", no_wrap=True),
                ColumnSpec("Type", style="cyan"),
                ColumnSpec("Session", style="blue"),
                ColumnSpec("Content"),
            ],
            row_fn=lambda d: [
                d.get("timestamp", "")[:19],
                d.get("type", ""),
                (d.get("session_id", "")[:12] + "\u2026"),
                d.get("content", "")[:truncate],
            ],
            summary_template="Found {n} messages",
        )
        _render_output(results, fmt, spec)
        return

    formatter = MessageFormatter(max_chars=max_chars)
    console.print(formatter.format_many(results))
    console.print(f"\n[bold]Found {len(results)} messages[/bold]")


def _do_list_sessions(
    engine: SessionRecoveryEngine,
    project: Optional[str] = None,
    after: Optional[str] = None,
    before: Optional[str] = None,
    limit: Optional[int] = None,
    fmt: str = "table",
) -> None:
    """List sessions with metadata."""
    sessions = engine.get_sessions(project_filter=project, after=after, before=before)
    if limit:
        sessions = sessions[:limit]
    _render_output(sessions, fmt, _LIST_SPEC, "No sessions found")


def _parse_pattern_options(raw: List[str]) -> List[tuple]:
    """Parse correction pattern entries into engine-format tuples.

    Each entry must be ``"CATEGORY:REGEX"``. Multiple entries with the same
    category are combined (OR logic). A single entry that is a bracketed list
    (e.g. ``'["regression:you deleted", "skip_step:you forgot"]'``) is expanded
    automatically, so both ``--pattern`` repetition and Python-list syntax work:

    Examples::
        # Repeatable flags (standard):
        --pattern "skip_step:you missed" --pattern "skip_step:you forgot"
        → [("skip_step", ["you missed", "you forgot"])]

        # Single bracketed-list value (also accepted):
        --pattern '["skip_step:you missed", "skip_step:you forgot"]'
        → [("skip_step", ["you missed", "you forgot"])]

        # Config file list (same format as bracketed value items):
        ["regression:you deleted", "skip_step:you forgot"]
        → [("regression", ["you deleted"]), ("skip_step", ["you forgot"])]

    Raises:
        typer.BadParameter: If any entry lacks the required "CATEGORY:REGEX" format.
    """
    from collections import OrderedDict
    # Expand any single entry that is a Python-style bracketed list
    expanded: List[str] = []
    for entry in raw:
        s = entry.strip()
        if s.startswith("[") and s.endswith("]"):
            expanded.extend(_parse_list_input(s))
        else:
            expanded.append(entry)

    groups: "OrderedDict[str, List[str]]" = OrderedDict()
    for entry in expanded:
        if ":" not in entry:
            raise typer.BadParameter(
                f"--pattern {entry!r} must be in 'CATEGORY:REGEX' format, "
                f"e.g. --pattern 'skip_step:you missed'"
            )
        category, regex = entry.split(":", 1)
        category = category.strip()
        regex = regex.strip()
        if not category or not regex:
            raise typer.BadParameter(
                f"--pattern {entry!r}: both CATEGORY and REGEX must be non-empty"
            )
        if category not in groups:
            groups[category] = []
        groups[category].append(regex)
    return [(cat, regexes) for cat, regexes in groups.items()]


def _parse_commands_option(raw: Optional[str]) -> Optional[List[str]]:
    """Parse --commands value into a list of regex patterns.

    Returns None (use engine/config default) when raw is None or empty.

    Accepts both plain CSV and Python-style bracketed lists (see _parse_list_input):

    Examples::
        "/ar:plannew,/ar:pn,/mycommand"
        → ["/ar:plannew", "/ar:pn", "/mycommand"]

        '["/ar:plannew", "/ar:pn"]'
        → ["/ar:plannew", "/ar:pn"]
    """
    if not raw:
        return None
    return _parse_list_input(raw)


def _do_messages_corrections(
    engine: SessionRecoveryEngine,
    project: Optional[str] = None,
    after: Optional[str] = None,
    before: Optional[str] = None,
    limit: int = 20,
    fmt: str = "table",
    pattern_overrides: Optional[List[str]] = None,
) -> None:
    """Find user corrections across sessions.

    Priority for correction patterns: CLI --pattern > config file > built-in defaults.

    Args:
        pattern_overrides: Raw --pattern option values in "CATEGORY:REGEX" format.
                           When provided, replaces config/defaults entirely.
    """
    if pattern_overrides:
        # CLI flag — highest priority
        patterns = _parse_pattern_options(pattern_overrides)
    else:
        cfg = load_config()
        cfg_patterns = cfg.get("correction_patterns")
        if cfg_patterns:
            # Config file — middle priority; same format as --pattern values
            patterns = _parse_pattern_options(list(cfg_patterns))
        else:
            # Built-in defaults — lowest priority (engine uses DEFAULT_CORRECTION_PATTERNS)
            patterns = None
    corrections = engine.find_corrections(
        project_filter=project, after=after, before=before,
        limit=limit, patterns=patterns,
    )
    _render_output(corrections, fmt, _CORRECTIONS_SPEC, "No corrections found")


def _do_messages_planning(
    engine: SessionRecoveryEngine,
    project: Optional[str] = None,
    after: Optional[str] = None,
    before: Optional[str] = None,
    fmt: str = "table",
    commands_raw: Optional[str] = None,
) -> None:
    """Show planning command usage.

    Priority for command list: CLI --commands > config file > built-in defaults.

    Args:
        commands_raw: Raw --commands option value (CSV or bracketed list).
                      When provided, replaces config/defaults entirely.
    """
    if commands_raw:
        # CLI flag — highest priority
        commands = _parse_commands_option(commands_raw)
    else:
        cfg = load_config()
        cfg_commands = cfg.get("planning_commands")
        if cfg_commands:
            # Config file — middle priority; list of regex strings
            commands = list(cfg_commands)
        else:
            # Built-in defaults — lowest priority (engine uses DEFAULT_PLANNING_COMMANDS)
            commands = None
    results = engine.analyze_planning_usage(
        project_filter=project, after=after, before=before,
        commands=commands,
    )
    _render_output(results, fmt, _PLANNING_SPEC, "No planning commands found")


_ANALYZE_TOOLS_SPEC = TableSpec(
    title_template="Tool Usage ({n} tools)",
    columns=[
        ColumnSpec("Tool", style="cyan"),
        ColumnSpec("Count", justify="right"),
    ],
    row_fn=lambda d: [d["tool"], str(d["count"])],
    summary_template="Found {n} distinct tools",
)

_TIMELINE_SPEC = TableSpec(
    title_template="Session Timeline ({n} events)",
    columns=[
        ColumnSpec("Timestamp", style="dim", no_wrap=True),
        ColumnSpec("Type", style="cyan"),
        ColumnSpec("Tools", justify="right"),
        ColumnSpec("Preview"),
    ],
    row_fn=lambda d: [
        d.get("timestamp", "")[:19],
        d.get("type", ""),
        str(d.get("tool_count", 0)),
        d.get("content_preview", ""),
    ],
)


def _do_messages_analyze(
    engine: SessionRecoveryEngine,
    session_id: str,
    fmt: str = "table",
) -> None:
    """Show per-session statistics."""
    result = engine.analyze_session(session_id)
    if result is None:
        err_console.print(f"[red]No session found matching:[/red] {session_id!r}")
        raise typer.Exit(code=1)

    if fmt == "json":
        console.print(json.dumps(result.to_dict(), indent=2))
        return

    # Summary header
    console.print(f"\n[bold]Session:[/bold] {result.session_id[:16]}…")
    console.print(f"[bold]Project:[/bold] {result.project_dir}")
    console.print(f"[bold]Lines:[/bold] {result.total_lines}  "
                  f"[bold]User:[/bold] {result.user_count}  "
                  f"[bold]Assistant:[/bold] {result.assistant_count}")
    console.print(f"[bold]From:[/bold] {result.timestamp_first[:19]}  "
                  f"[bold]To:[/bold] {result.timestamp_last[:19]}\n")

    # Tool usage table
    tool_rows = [{"tool": t, "count": c}
                 for t, c in sorted(result.tool_uses_by_name.items(),
                                    key=lambda x: x[1], reverse=True)]
    if tool_rows:
        _render_output(tool_rows, fmt, _ANALYZE_TOOLS_SPEC, "No tool calls found")

    # Files touched list
    if result.files_touched:
        console.print(f"\n[bold]Files touched ({len(result.files_touched)}):[/bold]")
        for fp in result.files_touched:
            console.print(f"  {fp}")
    else:
        console.print("\n[dim]No files touched.[/dim]")


def _do_messages_timeline(
    engine: SessionRecoveryEngine,
    session_id: str,
    fmt: str = "table",
    preview_chars: int = 150,
) -> None:
    """Show chronological event timeline for a session."""
    events = engine.timeline_session(session_id, preview_chars=preview_chars)
    if not events:
        err_console.print(f"[red]No session found matching:[/red] {session_id!r}")
        raise typer.Exit(code=1)
    _render_output(events, fmt, _TIMELINE_SPEC, "No events found")


def _do_files_cross_ref(
    engine: SessionRecoveryEngine,
    file: str,
    session: Optional[str] = None,
    fmt: str = "table",
) -> None:
    """Cross-reference session edits against current file content."""
    file_path = Path(file)
    if not file_path.exists():
        err_console.print(f"[red]File not found:[/red] {file}")
        raise typer.Exit(code=1)
    current_content = file_path.read_text(errors="replace")
    results = engine.cross_reference_session(
        file_path.name, current_content, session_id=session
    )
    spec = TableSpec(
        title_template=f"Cross-reference: {file_path.name} ({{n}} edits)",
        columns=[
            ColumnSpec("Timestamp", style="dim", no_wrap=True),
            ColumnSpec("Session", style="cyan", no_wrap=True),
            ColumnSpec("Tool", style="blue"),
            ColumnSpec("Applied", justify="center"),
            ColumnSpec("Snippet"),
        ],
        row_fn=lambda d: [
            d["timestamp"][:19],
            (d["session_id"][:16] + "\u2026") if len(d["session_id"]) > 16 else d["session_id"],
            d["tool"],
            "[green]\u2713[/green]" if d["found_in_current"] else "[red]\u2717[/red]",
            d["content_snippet"][:60],
        ],
        plain_fn=lambda d: (
            "{ts}  {sid}  {tool}  {mark}  {snip}".format(
                ts=d["timestamp"][:19],
                sid=d["session_id"][:16],
                tool=d["tool"],
                mark="\u2713" if d["found_in_current"] else "\u2717",
                snip=d["content_snippet"][:60],
            )
        ),
    )
    _render_output(results, fmt, spec, "No session edits found for this file")
    if results and fmt not in ("json",):
        applied = sum(1 for r in results if r["found_in_current"])
        console.print(f"\n[bold]{applied}/{len(results)} edits found in current file[/bold]")


def _do_export_session(
    engine: SessionRecoveryEngine,
    session_id: str,
    output: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Export one session to markdown."""
    try:
        md = engine.export_session_markdown(session_id)
    except ValueError as e:
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)
    if output:
        err_console.print(f"Writing to: {output}")
        if dry_run:
            err_console.print("[yellow][dry run] no files written[/yellow]")
            return
        Path(output).write_text(md, encoding="utf-8")
    else:
        if dry_run:
            err_console.print(f"[yellow][dry run] would write {len(md)} chars to stdout[/yellow]")
            return
        err_console.print(f"Exporting session: {session_id[:8]}")
        sys.stdout.write(md)


def _do_export_recent(
    engine: SessionRecoveryEngine,
    days: int = 7,
    output: Optional[str] = None,
    project: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Export all sessions from last N days to markdown."""
    import datetime as _dt
    after = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=days)).strftime("%Y-%m-%d")
    sessions = engine.get_sessions(project_filter=project, after=after)
    if not sessions:
        console.print("[yellow]No sessions found[/yellow]")
        return
    if dry_run:
        err_console.print(f"[yellow][dry run] would export {len(sessions)} sessions[/yellow]")
        for s in sessions:
            err_console.print(f"  {s.session_id[:16]}  {s.project_dir[-20:]}  {s.timestamp_first[:10]}")
        return
    parts = []
    for s in sessions:
        try:
            parts.append(engine.export_session_markdown(s.session_id))
        except ValueError:
            continue
    combined = "\n\n".join(parts)
    if output:
        err_console.print(f"Writing {len(sessions)} sessions to: {output}")
        Path(output).write_text(combined, encoding="utf-8")
    else:
        sys.stdout.write(combined)


def _do_search(  # noqa: C901
    engine: SessionRecoveryEngine,
    domain: Optional[str],
    pattern: Optional[str],
    query: Optional[str],
    min_edits: int,
    max_edits: Optional[int],
    include_extensions: Optional[str],
    exclude_extensions: Optional[str],
    include_sessions: Optional[str],
    exclude_sessions: Optional[str],
    after: Optional[str],
    before: Optional[str],
    message_type: Optional[str],
    limit: Optional[int],
    max_chars: int,
    fmt: str,
    tool: Optional[str] = None,
) -> None:
    """Shared logic for search() and find() commands."""
    # When user explicitly says 'tools' domain, --tool is required.
    if domain == "tools" and tool is None:
        typer.echo(
            "Error: 'tools' domain requires --tool <name> (e.g. --tool Bash, --tool Write, --tool Edit).",
            err=True,
        )
        raise typer.Exit(1)

    # Normalize "tools" domain → "messages" after validation
    if domain == "tools":
        domain = "messages"

    # Validate domain (after "tools"→"messages" normalization)
    if domain is not None and domain not in ("files", "messages"):
        typer.echo(f"Error: domain must be 'files', 'messages', or 'tools', got '{domain}'", err=True)
        raise typer.Exit(1)

    # Validate flag/domain conflicts
    if domain == "messages" and pattern is not None:
        typer.echo("Error: --pattern applies to files, not messages. Use --query.", err=True)
        raise typer.Exit(1)
    if domain == "files" and query is not None:
        typer.echo("Error: --query applies to messages, not files. Use --pattern.", err=True)
        raise typer.Exit(1)
    if domain == "files" and tool is not None:
        typer.echo("Error: --tool applies to messages, not files.", err=True)
        raise typer.Exit(1)

    # Auto-detect domain from flags when not explicitly specified
    if domain is None:
        if tool is not None:
            domain = "messages"
        elif query is not None and pattern is None:
            domain = "messages"
        elif pattern is not None and query is None:
            domain = "files"
        elif pattern is not None and query is not None:
            domain = "both"
        else:
            domain = "files"

    if domain in ("files", "both"):
        _do_files_search(
            engine, pattern or "*", min_edits, max_edits,
            include_extensions, exclude_extensions,
            include_sessions, exclude_sessions,
            after, before, limit, fmt,
        )

    if domain in ("messages", "both"):
        msg_limit = limit if limit is not None else 10
        _do_messages_search(engine, query or "", message_type, msg_limit, max_chars, fmt, tool=tool)


def _do_get(
    engine: SessionRecoveryEngine,
    session_id: Optional[str],
    message_type: Optional[str] = None,
    limit: int = 10,
    max_chars: int = 0,
    fmt: str = "table",
) -> None:
    """Get messages from a specific session."""
    if not session_id:
        err_console.print("[red]Session ID is required.[/red] Use 'aise list' to find session IDs.")
        raise typer.Exit(code=1)
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


# ── export_app commands ───────────────────────────────────────────────────────

@export_app.command("session")
def export_session(
    session_id: str = typer.Argument(..., help="Session ID (prefix match, e.g. ab841016)."),
    output: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="Write to this file instead of stdout. Example: --output session.md",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Show what would be written without producing any output or writing files.",
    ),
) -> None:
    """Export a single session's messages to markdown. Outputs to stdout by default.

    Examples:
        aise export session ab841016                    # stdout
        aise export session ab841016 > session.md       # redirect to file
        aise export session ab841016 --output out.md    # write explicitly
        aise export session ab841016 --dry-run          # preview only
    """
    _do_export_session(get_engine(), session_id, output=output, dry_run=dry_run)


@export_app.command("recent")
def export_recent(
    days: int = typer.Argument(7, help="Number of days back to include (default: 7)."),
    output: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="Output file path.",
    ),
    project: Optional[str] = typer.Option(None, "--project", help="Limit to this project directory substring."),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Export all sessions from the last N days to a single markdown file.

    Examples:
        aise export recent                            # last 7 days → stdout
        aise export recent 14 --output week2.md       # last 14 days → file
        aise export recent --project myproject        # filter by project
    """
    _do_export_recent(get_engine(), days=days, output=output, project=project, dry_run=dry_run)


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


# Register 'find' as alias for 'files search'
files_app.command("find")(files_search)


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


@files_app.command("cross-ref")
def files_cross_ref(
    file: str = typer.Argument(..., help="Path to a file to compare against session edits (e.g. ./cli.py)."),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Limit to one session (prefix match)."),
    fmt: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv, plain. Default: table"),
) -> None:
    """Show which edits Claude made to a file are present in its current version.

    Examples:
        aise files cross-ref ./cli.py
        aise files cross-ref ./engine.py --session ab841016
        aise files cross-ref ./cli.py --format json
    """
    _do_files_cross_ref(get_engine(), file, session, fmt)


# Add 'find' as alias for 'files search' (registered after files_search is defined below)

# ── messages_app commands ─────────────────────────────────────────────────────

def _messages_search_cmd(
    query: Optional[str] = typer.Argument(None, help="Text to search for in messages. Use quotes for multi-word queries."),
    query_opt: Optional[str] = typer.Option(None, "--query", "-q", hidden=True),
    message_type: Optional[str] = typer.Option(None, "--type", "-t", help="Show only 'user' or 'assistant' messages. Default: both"),
    limit: int = typer.Option(10, "--limit", help="Max messages to return. Default: 10"),
    max_chars: int = typer.Option(0, "--max-chars", help="Truncate each message to this many characters. 0 = full."),
    fmt: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv, plain."),
    tool: Optional[str] = typer.Option(
        None, "--tool",
        help="Filter for tool call invocations (e.g. Bash, Edit, Write). Implies --type assistant.",
    ),
    context: int = typer.Option(
        0, "--context", "-c",
        help="Include N messages before and after each match (from the same session). Default: 0.",
    ),
) -> None:
    """Search or find conversation messages from Claude Code sessions.

    Accessible as both 'messages search' and 'messages find' (aliases).

    Examples:
        aise messages search "authentication"
        aise messages search --query "authentication"  # backward compat
        aise messages find "error"                     # find is an alias
        aise messages search "*" --tool Write          # all Write tool calls
        aise messages search "error" --context 3       # show 3 surrounding messages
    """
    q = query or query_opt
    _do_messages_search(get_engine(), q or "", message_type, limit, max_chars, fmt,
                        tool=tool, context=context)


_register_alias(messages_app, _messages_search_cmd, "search", "find")


@messages_app.command("get")
def messages_get(
    session_id: Optional[str] = typer.Argument(None, help="Session ID (prefix match, e.g. ab841016). Find IDs via 'aise list'."),
    session_opt: Optional[str] = typer.Option(None, "--session", "-s", hidden=True),
    message_type: Optional[str] = typer.Option(None, "--type", "-t", help="Show only 'user' or 'assistant' messages. Default: both"),
    limit: int = typer.Option(10, "--limit", help="Max messages to return. Default: 10"),
    max_chars: int = typer.Option(0, "--max-chars", help="Truncate each message to this many characters. 0 = full."),
    fmt: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv, plain."),
) -> None:
    """Read messages from one specific Claude Code session.

    Examples:
        aise messages get ab841016              # positional
        aise messages get --session ab841016    # flag (backward compat)
        aise messages get ab841016 --type user
    """
    sid = session_id or session_opt
    _do_get(get_engine(), sid, message_type, limit, max_chars, fmt)


@messages_app.command("corrections")
def messages_corrections(
    project: Optional[str] = typer.Option(None, "--project", help="Filter by project directory substring."),
    after: Optional[str] = typer.Option(None, "--after", help="Only corrections after this date."),
    before: Optional[str] = typer.Option(None, "--before"),
    limit: int = typer.Option(20, "--limit", help="Max corrections to return. Default: 20"),
    fmt: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv, plain. Default: table"),
    pattern: Optional[List[str]] = typer.Option(
        None, "--pattern",
        help=(
            "Custom correction pattern: 'CATEGORY:REGEX'. Repeatable. "
            "When provided, replaces built-in patterns entirely. "
            "Categories are arbitrary labels (e.g. regression, skip_step). "
            "Multiple --pattern flags with the same category are ORed together.\n\n"
            "Example: --pattern 'skip_step:you missed' --pattern 'skip_step:you forgot' "
            "--pattern 'custom:you broke it'"
        ),
    ),
) -> None:
    """Find user messages where corrections were given to Claude.

    By default detects patterns like 'you forgot', 'nono', 'that's wrong', etc.
    across four built-in categories: regression, skip_step, misunderstanding, incomplete.

    Use --pattern to supply your own patterns instead of the built-in set.
    Each --pattern value is 'CATEGORY:REGEX' where REGEX is a Python regex.
    Multiple --pattern values with the same category are combined (OR logic).
    When any --pattern is given, built-in patterns are NOT used.

    Examples:
        aise messages corrections
        aise messages corrections --limit 50 --project myproject
        aise messages corrections --format json
        aise messages corrections --pattern 'regression:you deleted' --pattern 'regression:you removed'
        aise messages corrections --pattern 'oops:nono' --pattern 'oops:that.s wrong'
    """
    _do_messages_corrections(get_engine(), project, after, before, limit, fmt,
                              pattern_overrides=pattern)


@messages_app.command("planning")
def messages_planning(
    project: Optional[str] = typer.Option(None, "--project", help="Filter by project directory substring."),
    after: Optional[str] = typer.Option(None, "--after", help="Only commands after this date."),
    before: Optional[str] = typer.Option(None, "--before", help="Only commands before this date."),
    fmt: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv, plain. Default: table"),
    commands: Optional[str] = typer.Option(
        None, "--commands",
        help=(
            "Comma-separated list of slash-command patterns to count. "
            "When provided, replaces the built-in list entirely. "
            "Each entry is a Python regex (word boundaries added automatically if absent). "
            "Example: --commands '/ar:plannew,/ar:pn,/myplanning,/plan'"
        ),
    ),
) -> None:
    """Show slash-command usage frequency across all sessions.

    By default counts the built-in planning commands:
      /ar:plannew, /ar:pn, /ar:planrefine, /ar:pr, /ar:planupdate, /ar:pu,
      /ar:planprocess, /ar:pp, /plannew, /planrefine, /planupdate, /planprocess

    These defaults reflect autorun plugin commands. Use --commands to count
    your own slash commands instead (built-in list is not used when --commands
    is given).

    Each entry in --commands is matched as a Python regex, case-insensitive.
    The display name strips trailing \\b from each pattern.

    Examples:
        aise messages planning
        aise messages planning --project myproject
        aise messages planning --format json
        aise messages planning --commands '/ar:plannew,/ar:pn'
        aise messages planning --commands '/mycommand,/mc,/plan,/p'
    """
    _do_messages_planning(get_engine(), project, after, before, fmt, commands_raw=commands)


@messages_app.command("analyze")
def messages_analyze(
    session_id: str = typer.Argument(..., help="Session ID prefix (e.g. ab841016). Find IDs via 'aise list'."),
    fmt: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv, plain. Default: table"),
) -> None:
    """Show per-session statistics: message counts, tool usage, and files touched.

    Files touched are detected from any tool_use block that includes a file_path
    input — not hardcoded to specific tool names.

    Examples:
        aise messages analyze ab841016
        aise messages analyze ab841016 --format json
    """
    _do_messages_analyze(get_engine(), session_id, fmt)


@messages_app.command("timeline")
def messages_timeline(
    session_id: str = typer.Argument(..., help="Session ID prefix (e.g. ab841016). Find IDs via 'aise list'."),
    fmt: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv, plain. Default: table"),
    preview_chars: int = typer.Option(
        150, "--preview-chars",
        help="Max characters to show in content preview column. Default: 150",
    ),
) -> None:
    """Show a chronological timeline of user/assistant events for a session.

    Each row shows the message type, timestamp, number of tool calls invoked,
    and a preview of the message content.

    Examples:
        aise messages timeline ab841016
        aise messages timeline ab841016 --format json
        aise messages timeline ab841016 --preview-chars 80
    """
    _do_messages_timeline(get_engine(), session_id, fmt, preview_chars=preview_chars)


# ── tools_app commands ────────────────────────────────────────────────────────

def _tools_search_cmd(
    tool: str = typer.Argument(..., help="Tool name (e.g. Bash, Edit, Write, Read, Glob, Grep)."),
    query: Optional[str] = typer.Argument(None, help="Optional text to match in tool input. Omit to list all uses."),
    limit: int = typer.Option(10, "--limit", help="Max results. Default: 10"),
    max_chars: int = typer.Option(0, "--max-chars", help="Truncate each result to N chars. 0=full."),
    fmt: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv, plain."),
) -> None:
    """Search or find tool invocations from Claude Code sessions.

    Accessible as both 'tools search' and 'tools find' (aliases).

    Examples:
        aise tools search Write                      # all Write calls
        aise tools search Bash "git commit"          # Bash calls with "git commit"
        aise tools find Edit "cli.py"                # find alias
        aise tools search Write --format json        # JSON output
    """
    _do_messages_search(
        get_engine(), query or "", message_type="assistant",
        limit=limit, max_chars=max_chars, fmt=fmt, tool=tool,
    )


_register_alias(tools_app, _tools_search_cmd, "search", "find")


# ── Root commands ─────────────────────────────────────────────────────────────

@app.command("list")
def list_sessions(
    project: Optional[str] = typer.Option(None, "--project", help="Filter by project directory substring."),
    after: Optional[str] = typer.Option(None, "--after", help="Only sessions after this date (e.g. 2026-01-15)."),
    before: Optional[str] = typer.Option(None, "--before", help="Only sessions before this date."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Max sessions to return. Default: unlimited."),
    fmt: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv, plain. Default: table"),
) -> None:
    """List Claude Code sessions with metadata (project, date, branch, message count).

    Examples:
        aise list                              # all sessions
        aise list --project myproject          # filter by project
        aise list --after 2026-01-01           # sessions since Jan 1
        aise list --format json                # JSON output
    """
    _do_list_sessions(get_engine(), project, after, before, limit, fmt)


def _root_search_cmd(
    domain: Optional[str] = typer.Argument(None, metavar="[files|messages|tools]",
        help="Domain: 'files', 'messages', or 'tools' (tool calls). Auto-detected from flags."),
    pattern: Optional[str] = typer.Option(None, "--pattern", "-p", help="[files] Filename glob/regex to match (e.g. '*.py', 'cli*'). Default: * (all files)"),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="[messages] Text to search for in user/assistant message content"),
    min_edits: int = typer.Option(0, "--min-edits", help="[files] Only show files edited at least this many times. Default: 0"),
    max_edits: Optional[int] = typer.Option(None, "--max-edits", help="[files] Only show files edited at most this many times. Default: unlimited"),
    include_extensions: Optional[str] = typer.Option(None, "--include-extensions", "-i", help="[files] Only these file extensions, comma-separated (e.g. py,md,json)"),
    exclude_extensions: Optional[str] = typer.Option(None, "--exclude-extensions", "-x", help="[files] Skip these file extensions, comma-separated (e.g. pyc,tmp)"),
    include_sessions: Optional[str] = typer.Option(None, "--include-sessions", help="[files] Only search these session UUIDs, comma-separated"),
    exclude_sessions: Optional[str] = typer.Option(None, "--exclude-sessions", help="[files] Skip these session UUIDs, comma-separated"),
    after: Optional[str] = typer.Option(None, "--after", help="[files/messages] Only results after this date."),
    before: Optional[str] = typer.Option(None, "--before", help="[files/messages] Only results before this date."),
    message_type: Optional[str] = typer.Option(None, "--type", "-t", help="[messages] Show only 'user' or 'assistant' messages. Default: both"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Max results to return. Default: unlimited for files, 10 for messages"),
    max_chars: int = typer.Option(0, "--max-chars", help="[messages] Truncate each message to this many characters. 0 = full message. Default: 0"),
    fmt: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv, plain. Default: table"),
    tool: Optional[str] = typer.Option(None, "--tool",
        help="[messages/tools] Filter for tool call invocations (e.g. Bash, Edit, Write). Auto-routes to messages domain."),
) -> None:
    """Search or find source files and/or conversation messages.

    Accessible as both 'search' and 'find' at the root level (aliases).

    Examples:
        aise search                                          # list all source files
        aise search files --pattern "*.py"                   # Python files only
        aise search messages --query "error"                 # messages with "error"
        aise search tools --tool Write --query "login"       # Write calls with "login"
        aise search --tool Bash --query "git commit"         # auto-routes to messages
        aise find files --pattern "*.py"                     # find is an alias
    """
    _do_search(
        get_engine(), domain, pattern, query, min_edits, max_edits,
        include_extensions, exclude_extensions, include_sessions, exclude_sessions,
        after, before, message_type, limit, max_chars, fmt, tool=tool,
    )


_register_alias(app, _root_search_cmd, "search", "find")


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
    session_id: Optional[str] = typer.Argument(None, help="Session ID (prefix match, e.g. ab841016)."),
    session_opt: Optional[str] = typer.Option(None, "--session", "-s", hidden=True),
    message_type: Optional[str] = typer.Option(None, "--type", "-t", help="Show only 'user' or 'assistant' messages. Default: both"),
    limit: int = typer.Option(10, "--limit", help="Max messages to return. Default: 10"),
    max_chars: int = typer.Option(0, "--max-chars", help="Truncate each message to this many characters. 0 = show full message. Default: 0"),
    fmt: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv, plain. Default: table"),
) -> None:
    """Read messages from one specific Claude Code session.

    Equivalent to 'aise messages get'. Session IDs are UUIDs found in ~/.claude/projects/
    or in output from 'aise list'.

    Examples:
        aise get ab841016
        aise get ab841016 --type user --limit 50
    """
    sid = session_id or session_opt
    _do_get(get_engine(), sid, message_type, limit, max_chars, fmt)


@app.command()
def stats() -> None:
    """Show counts of sessions, files, versions, and the most-edited file."""
    _do_stats(get_engine())


# ── Config app ───────────────────────────────────────────────────────────────

def _get_config_file_path() -> Path:
    """Return the resolved config file path based on current priority chain."""
    if _g_config_path:
        return Path(_g_config_path).expanduser()
    env_val = os.getenv("AI_SESSION_TOOLS_CONFIG")
    if env_val:
        return Path(env_val).expanduser()
    return Path(typer.get_app_dir("ai_session_tools")) / "config.json"


@config_app.command("path")
def config_path() -> None:
    """Print the config file path (whether or not the file exists).

    Examples:
        aise config path
        aise --config /tmp/my.json config path   # show path after override
    """
    console.print(str(_get_config_file_path()))


@config_app.command("show")
def config_show(
    fmt: str = typer.Option("table", "--format", "-f", help="Output format: table, json, plain."),
) -> None:
    """Show the effective configuration (file contents + resolved path).

    Displays all keys loaded from the config file, or a message if no file exists.

    Examples:
        aise config show
        aise config show --format json
    """
    config_file = _get_config_file_path()
    cfg = load_config()

    if fmt == "json":
        sys.stdout.write(json.dumps({
            "config_file": str(config_file),
            "exists": config_file.exists(),
            "config": cfg,
        }, indent=2) + "\n")
        return

    console.print(f"Config file: [cyan]{config_file}[/cyan]")
    if not config_file.exists():
        console.print("[yellow]File does not exist. Run 'aise config init' to create it.[/yellow]")
        return

    if not cfg:
        console.print("[dim]File exists but is empty (no keys set).[/dim]")
        return

    if fmt in ("plain", "csv"):
        for k, v in cfg.items():
            console.print(f"{k}: {v}")
        return

    from rich.table import Table
    table = Table(title=f"Config ({config_file.name})")
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    for k, v in cfg.items():
        table.add_row(k, json.dumps(v) if not isinstance(v, str) else v)
    console.print(table)


_CONFIG_INIT_TEMPLATE = {
    "correction_patterns": [
        "regression:you deleted",
        "regression:you removed",
        "skip_step:you forgot",
        "skip_step:you missed",
        "misunderstanding:that's wrong",
        "incomplete:also need",
    ],
    "planning_commands": [
        "/ar:plannew",
        "/ar:pn",
        "/ar:planrefine",
        "/ar:pr",
        "/ar:planupdate",
        "/ar:pu",
        "/ar:planprocess",
        "/ar:pp",
    ],
}


@config_app.command("init")
def config_init(
    force: bool = typer.Option(
        False, "--force",
        help="Overwrite the config file if it already exists.",
    ),
) -> None:
    """Create a starter config.json with documented default values.

    Safe by default — will NOT overwrite an existing config file unless --force is given.
    Creates any missing parent directories automatically.

    Examples:
        aise config init             # create if not exists
        aise config init --force     # overwrite existing file
    """
    config_file = _get_config_file_path()

    if config_file.exists() and not force:
        err_console.print(
            f"[yellow]Config file already exists:[/yellow] {config_file}\n"
            "Use --force to overwrite."
        )
        raise typer.Exit(code=1)

    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(json.dumps(_CONFIG_INIT_TEMPLATE, indent=2) + "\n", encoding="utf-8")

    # Invalidate cache so next command picks up the new file
    global _config_cache
    _config_cache = None

    console.print(f"[green]Created:[/green] {config_file}")
    console.print(
        "[dim]Edit the file to customize correction patterns and planning commands.[/dim]\n"
        "[dim]Run 'aise config show' to verify the active configuration.[/dim]"
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def cli_main():
    """CLI entry point."""
    app()
