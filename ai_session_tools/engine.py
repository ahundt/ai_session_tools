"""
Core session recovery engine - refactored for composition and extensibility.

Copyright (c) 2026 Andrew Hundt
Licensed under the Apache License, Version 2.0
"""

import datetime
import fnmatch
import functools
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

try:
    from orjson import loads as _json_loads
except ImportError:
    from json import loads as _json_loads  # type: ignore[assignment]

from .models import (
    ContextMatch,
    CorrectionMatch,
    FileVersion,
    FilterSpec,
    MessageType,
    PlanningCommandCount,
    RecoveredFile,
    RecoveryStatistics,
    SessionAnalysis,
    SessionInfo,
    SessionMessage,
)

#: Default correction patterns: (category, [regex_keywords...])
#: Uses \b word boundaries (matching claude_session_tools.py:103-127).
DEFAULT_CORRECTION_PATTERNS: List[tuple] = [
    ("regression",       [r"\byou deleted\b", r"\byou removed\b", r"\blost\b",
                          r"\bregressed\b", r"\brollback\b", r"\brevert\b"]),
    ("skip_step",        [r"\byou forgot\b", r"\byou missed\b", r"\byou skipped\b",
                          r"\bdon't forget\b", r"\bmissing step\b"]),
    ("misunderstanding", [r"\bwrong\b", r"\bincorrect\b", r"\bmistake\b",
                          r"\bnono\b", r"\bno,\s", r"\bthat's not correct\b"]),
    ("incomplete",       [r"\balso need\b", r"\bmust also\b", r"\bnot done\b",
                          r"\bnot finished\b", r"\bstill need\b"]),
]

#: Default planning command regex patterns.
#: Using \b word boundaries (matching claude_session_tools.py:130-136).
DEFAULT_PLANNING_COMMANDS: List[str] = [
    r"/ar:plannew\b", r"/ar:pn\b",
    r"/ar:planrefine\b", r"/ar:pr\b",
    r"/ar:planupdate\b", r"/ar:pu\b",
    r"/ar:planprocess\b", r"/ar:pp\b",
    r"/plannew\b", r"/planrefine\b", r"/planupdate\b", r"/planprocess\b",
]

#: Regex to auto-discover slash commands that START a user message.
#: Matches e.g. "/ar:plannew", "/commit", "/help" — not file paths or URLs.
#: Used when analyze_planning_usage() is called with commands=None (discovery mode).
_SLASH_CMD_DISCOVERY_RE = re.compile(r"^/(\w[\w:.-]*)")

#: System message patterns to filter from export
_EXPORT_FILTER_PATTERNS = (
    "[Request interrupted",
    "<task-notification>",
    "<system-reminder>",
)


class SessionRecoveryEngine:
    """Core recovery engine with clean separation of concerns."""

    def __init__(self, projects_dir: Path, recovery_dir: Path):
        """Initialize engine with directories.

        Args:
            projects_dir: Path to Claude Code projects directory
                          (default: ~/.claude/projects). Must contain JSONL session files.
            recovery_dir: Path to recovery output directory
                          (default: ~/.claude/recovery). Contains session_*/ subdirs
                          with extracted source files and version history.
        """
        self.projects_dir = Path(projects_dir)
        self.recovery_dir = Path(recovery_dir)
        self._file_cache: Dict[str, RecoveredFile] = {}
        self._version_cache: Dict[str, List[FileVersion]] = {}

    # ── Private JSONL iteration helpers ──────────────────────────────────────

    def _iter_all_jsonl(
        self, project_filter: Optional[str] = None
    ):
        """Yield (project_dir_name, jsonl_path) for every session across all projects.

        Handles missing directory gracefully. All multi-session scanning methods
        use this to avoid duplicating the glob + filter scaffolding.
        """
        if not self.projects_dir.exists():
            return
        for project_dir in self.projects_dir.glob("*"):
            if not project_dir.is_dir():
                continue
            if project_filter and project_filter.lower() not in project_dir.name.lower():
                continue
            for jsonl_file in project_dir.glob("*.jsonl"):
                yield project_dir.name, jsonl_file

    def _find_session_files(self, session_id: str) -> List[tuple]:
        """Return all (jsonl_path, project_dir_name) tuples for the given session ID prefix.

        Returns an empty list if no session matches.
        Results are sorted by file modification time descending (newest first),
        so callers that need a single session can use matches[0].
        """
        matches = []
        for project_dir_name, jsonl_path in self._iter_all_jsonl():
            if jsonl_path.stem.startswith(session_id):
                try:
                    mtime = jsonl_path.stat().st_mtime
                except OSError:
                    mtime = 0.0
                matches.append((mtime, jsonl_path, project_dir_name))
        # Sort newest first, then strip the mtime sort key
        matches.sort(key=lambda x: x[0], reverse=True)
        return [(path, proj) for _mtime, path, proj in matches]

    @staticmethod
    def _scan_jsonl(path: Path):
        """Yield parsed JSON dicts from a JSONL file, silently skipping malformed lines.

        Handles OSError (permission denied, missing file) by returning nothing.
        """
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        data = _json_loads(line)
                        if isinstance(data, dict):
                            yield data
                    except (json.JSONDecodeError, ValueError):
                        continue
        except OSError:
            return

    # ── End private helpers ───────────────────────────────────────────────────

    @functools.cached_property
    def _version_dirs(self) -> List[Path]:
        """Session all-versions dirs — scanned once per engine instance."""
        if not self.recovery_dir.exists():
            return []
        return [
            d for d in self.recovery_dir.iterdir()
            if d.is_dir() and d.name.startswith("session_all_versions_")
        ]

    @staticmethod
    def _compile_pattern(pattern: str) -> re.Pattern:
        """Compile pattern, auto-detecting glob vs regex.

        Args:
            pattern: Glob pattern (e.g. '*.py') or regex (e.g. 'cli.*').

        Returns:
            Compiled regex, case-insensitive.

        Raises:
            ValueError: If pattern is not a valid glob or regex.
        """
        if "*" in pattern or "?" in pattern:
            regex_pattern = fnmatch.translate(pattern)
            return re.compile(regex_pattern, re.IGNORECASE)
        try:
            return re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"Invalid search pattern {pattern!r}: {exc}") from exc

    def _get_or_create_file_info(self, file_path: Path) -> Optional[RecoveredFile]:
        """Get or create cached file info.

        Cache is keyed by filename (basename): across multiple session dirs the same
        filename is intentionally deduplicated — metadata (size, dates) comes from the
        first-encountered copy; version history comes from get_versions() which scans
        all session_all_versions_*/ dirs.

        Returns:
            RecoveredFile, or None if the file's stat() call fails (e.g. broken symlink).
        """
        if file_path.name not in self._file_cache:
            try:
                versions = self.get_versions(file_path.name)
                stat = file_path.stat()
                last_modified = datetime.datetime.fromtimestamp(
                    stat.st_mtime, tz=datetime.timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%S")
                created_date = datetime.datetime.fromtimestamp(
                    stat.st_ctime, tz=datetime.timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%S")
                self._file_cache[file_path.name] = RecoveredFile(
                    name=file_path.name,
                    path=str(file_path.resolve()),
                    location="recovery",
                    file_type=file_path.suffix[1:] or "unknown",
                    sessions=[v.session_id for v in versions],
                    edits=len(versions),
                    size_bytes=stat.st_size,
                    last_modified=last_modified,
                    created_date=created_date,
                )
            except OSError:
                return None
        return self._file_cache.get(file_path.name)

    def _apply_all_filters(self, file_info: RecoveredFile, filters: FilterSpec) -> bool:
        """Check if file passes all filters. Returns True if file should be included.

        Order: cheapest checks first (extension, size, datetime) before expensive ones
        (sessions iterates a list; edits unavoidable; location always passes after pre-check).
        """
        if not filters.matches_extension(file_info.file_type):
            return False

        if not filters.matches_size(file_info.size_bytes):
            return False

        if not filters.matches_datetime(file_info.last_modified):
            return False

        if file_info.sessions:
            if not any(filters.matches_session(s) for s in file_info.sessions):
                return False
        elif filters.include_sessions:
            return False

        if not filters.matches_edits(file_info.edits):
            return False

        if not filters.matches_location(file_info.location):
            return False

        return True

    def search(  # noqa: C901
        self,
        pattern: str,
        filters: Optional[FilterSpec] = None,
    ) -> List[RecoveredFile]:
        """Search files with optional filtering.

        Args:
            pattern: Glob or regex pattern (e.g. '*.py', 'cli.*')
            filters: Optional FilterSpec for advanced filtering

        Returns:
            List of RecoveredFile objects sorted by edits (descending).
            Returns empty list if recovery_dir does not exist.

        Raises:
            ValueError: If pattern is not a valid glob or regex.
        """
        if filters is None:
            filters = FilterSpec()

        # Compile first: raise ValueError on bad patterns before any I/O.
        pattern_re = self._compile_pattern(pattern)

        if not self.recovery_dir.exists():
            return []

        # location is always "recovery"; short-circuit if include_folders excludes it.
        if filters.include_folders and not filters.matches_location("recovery"):
            return []

        results: List[RecoveredFile] = []
        seen_names: Set[str] = set()

        with os.scandir(self.recovery_dir) as it:
            for entry in it:
                if not entry.is_dir():
                    continue
                name = entry.name
                if not name.startswith("session_") or "all_versions" in name:
                    continue

                # Session dir-level pre-filter: skip entire dir before iterating files.
                session_id_str = name[len("session_"):]
                if filters.include_sessions and session_id_str not in filters.include_sessions:
                    continue
                if filters.exclude_sessions and session_id_str in filters.exclude_sessions:
                    continue

                session_dir = Path(entry.path)

                with os.scandir(session_dir) as it2:
                    for entry2 in it2:
                        if not entry2.is_file() or not pattern_re.search(entry2.name):
                            continue
                        if entry2.name in seen_names:
                            continue

                        file_path = Path(entry2.path)

                        # Extension pre-filter: skip stat+get_versions for wrong extensions.
                        ext = file_path.suffix.lstrip(".")
                        if not filters.matches_extension(ext):
                            continue

                        # Stat pre-filter: check date/size before expensive get_versions.
                        if filters.after or filters.before or filters.min_size or (filters.max_size is not None):
                            try:
                                s = file_path.stat()
                                mtime = datetime.datetime.fromtimestamp(
                                    s.st_mtime, tz=datetime.timezone.utc
                                ).strftime("%Y-%m-%dT%H:%M:%S")
                                if not filters.matches_datetime(mtime) or not filters.matches_size(s.st_size):
                                    continue
                            except OSError:
                                continue

                        file_info = self._get_or_create_file_info(file_path)
                        if file_info is None:
                            continue

                        if self._apply_all_filters(file_info, filters):
                            results.append(file_info)
                            seen_names.add(file_path.name)

        return sorted(results, key=lambda f: f.edits, reverse=True)

    def get_versions(self, filename: str) -> List[FileVersion]:
        """Get all versions of a file across all sessions.

        Args:
            filename: Exact filename (e.g. 'cli.py')

        Returns:
            List of FileVersion objects sorted by version number (ascending).
        """
        if filename in self._version_cache:
            return self._version_cache[filename]

        versions = []
        version_pattern = r"_v(\d+)_line_(\d+)\.txt$"

        for session_dir in self._version_dirs:
            # Use raw filename in glob: glob treats '.' as a literal character,
            # not as a special meta-character, so re.escape() must NOT be used here.
            for version_file in session_dir.glob(f"{filename}_v*_line_*.txt"):
                match = re.search(version_pattern, version_file.name)
                if match:
                    version_num = int(match.group(1))
                    line_count = int(match.group(2))
                    try:
                        ts = datetime.datetime.fromtimestamp(
                            version_file.stat().st_mtime, tz=datetime.timezone.utc
                        ).strftime("%Y-%m-%d %H:%M")
                    except OSError:
                        ts = ""
                    versions.append(
                        FileVersion(
                            filename=filename,
                            version_num=version_num,
                            line_count=line_count,
                            session_id=session_dir.name.replace("session_all_versions_", ""),
                            timestamp=ts,
                        )
                    )

        versions.sort()
        self._version_cache[filename] = versions
        return versions

    def extract_final(self, filename: str, output_dir: Path) -> Optional[Path]:
        """Extract the most recent version of a file (highest version number).

        Args:
            filename: Exact filename to extract (e.g. 'cli.py')
            output_dir: Directory to write the extracted file

        Returns:
            Path to extracted file, or None if not found.
        """
        versions = self.get_versions(filename)

        if versions:
            # Use max version_num, not max line_count: a refactor that shortens a file
            # should not revert to an older longer version.
            final = max(versions, key=lambda v: v.version_num)
            session_dir = self.recovery_dir / f"session_all_versions_{final.session_id}"
            version_file = session_dir / f"{filename}_v{final.version_num:06d}_line_{final.line_count}.txt"

            if version_file.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / filename
                output_path.write_text(version_file.read_text(errors="ignore"))
                return output_path

        # Fallback: check session_*/ directories
        if self.recovery_dir.exists():
            for session_dir in self.recovery_dir.glob("session_*/"):
                if "all_versions" in session_dir.name:
                    continue

                file_path = session_dir / filename
                if file_path.exists():
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / filename
                    output_path.write_text(file_path.read_text(errors="ignore"))
                    return output_path

        return None

    def extract_all(self, filename: str, output_dir: Path) -> List[Path]:
        """Extract all recorded versions of a file.

        Args:
            filename: Exact filename to extract (e.g. 'cli.py')
            output_dir: Directory to write version files

        Returns:
            List of paths to extracted version files.
        """
        versions = self.get_versions(filename)
        output_dir.mkdir(parents=True, exist_ok=True)
        extracted = []

        if versions:
            for version in sorted(versions):
                session_dir = self.recovery_dir / f"session_all_versions_{version.session_id}"
                version_file = session_dir / f"{filename}_v{version.version_num:06d}_line_{version.line_count}.txt"

                if version_file.exists():
                    target = output_dir / f"v{version.version_num:06d}_line_{version.line_count}.txt"
                    target.write_text(version_file.read_text(errors="ignore"))
                    extracted.append(target)
        elif self.recovery_dir.exists():
            # Fallback: single copy from first session_*/ dir that has the file
            for session_dir in self.recovery_dir.glob("session_*/"):
                if "all_versions" not in session_dir.name:
                    file_path = session_dir / filename
                    if file_path.exists():
                        target = output_dir / "v000001_final.txt"
                        target.write_text(file_path.read_text(errors="ignore"))
                        extracted.append(target)
                        break

        return extracted

    @staticmethod
    def _parse_message_type(msg_type: str) -> MessageType:
        """Parse a message type string into a MessageType enum value.

        Falls back to MessageType.SYSTEM for unrecognised types.
        """
        try:
            return MessageType(msg_type)
        except ValueError:
            return MessageType.SYSTEM

    def _process_message_line(
        self, line: str, session_id: str, message_type: Optional[str]
    ) -> Optional[SessionMessage]:
        """Process a single JSONL line into a SessionMessage.

        Supports prefix matching: session_id 'ab841016' matches 'ab841016-f07b-...'.
        """
        try:
            # Fast pre-filter: session_id must appear in the raw line (no false negatives).
            if session_id not in line:
                return None
            data = _json_loads(line)
            msg_session = data.get("sessionId", "")
            # Prefix match: allow short IDs (e.g. 'ab841016') to match full UUIDs
            if msg_session != session_id and not msg_session.startswith(session_id):
                return None

            msg_type = data.get("type", "").lower()
            if message_type and msg_type != message_type.lower():
                return None

            content = self._extract_content(data)
            if not content:
                return None

            return SessionMessage(
                type=self._parse_message_type(msg_type),
                timestamp=data.get("timestamp", ""),
                content=content,
                session_id=msg_session,
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def get_messages(self, session_id: str, message_type: Optional[str] = None) -> List[SessionMessage]:
        """Extract messages from a session.

        Args:
            session_id: Full or prefix UUID (e.g. 'ab841016' or 'ab841016-f07b-...')
            message_type: Optional filter: 'user', 'assistant', or 'system'

        Returns:
            List of SessionMessage objects. Empty list if projects_dir does not exist.
        """
        messages: List[SessionMessage] = []

        if not self.projects_dir.exists():
            return messages

        for project_dir in self.projects_dir.glob("*"):
            if not project_dir.is_dir():
                continue

            for jsonl_file in project_dir.glob(f"{session_id}*.jsonl"):
                try:
                    with open(jsonl_file, encoding="utf-8", errors="replace") as f:
                        for line in f:
                            msg = self._process_message_line(line, session_id, message_type)
                            if msg:
                                messages.append(msg)
                except OSError:
                    continue

        return messages

    def search_messages(  # noqa: C901
        self,
        query: str,
        message_type: Optional[str] = None,
        tool: Optional[str] = None,
    ) -> List[SessionMessage]:
        """Search for messages across all sessions.

        Args:
            query: Text or glob/regex pattern to search for in message content.
                   When tool is set, matches against serialized tool input JSON.
            message_type: Optional filter: 'user', 'assistant', or 'system'.
                          Defaults to 'assistant' when tool is set.
            tool: Optional tool name to filter by (e.g. 'Bash', 'Write', 'Edit').
                  When set, searches assistant messages for tool_use blocks with
                  this name. query is matched against the serialized tool input.

        Returns:
            List of matching SessionMessage objects. Empty list if projects_dir does not exist.
            When tool is set, SessionMessage.content holds json.dumps(tool_input).

        Raises:
            ValueError: If query is not a valid glob or regex.
        """
        # tool_use only appears in assistant messages
        if tool is not None and message_type is None:
            message_type = "assistant"

        # Allow empty query when filtering by tool (list all invocations)
        pattern = self._compile_pattern(query) if query else None

        # Pre-compute literal pre-filter: only safe when query has no regex metacharacters.
        is_literal = not query or re.escape(query) == query
        query_lower = query.lower() if (is_literal and query) else None
        # Heuristic: message_type value must appear in the raw JSON line (no false negatives).
        msg_type_hint = f'"{message_type}"' if message_type else None
        tool_lower = tool.lower() if tool else None

        messages: List[SessionMessage] = []

        for _project_dir_name, jsonl_file in self._iter_all_jsonl():
            try:
                with open(jsonl_file, encoding="utf-8", errors="replace") as f:
                    for line in f:
                        try:
                            # Raw line pre-filters: skip json.loads when possible.
                            if query_lower and query_lower not in line.lower():
                                continue
                            if msg_type_hint and msg_type_hint not in line:
                                continue
                            # Tool pre-filter: tool name value must appear in raw line.
                            # Use f'"{tool}"' (value-only check) — robust against both
                            # json (space-separated) and orjson (no-space) serialization.
                            if tool and f'"{tool}"' not in line:
                                continue
                            data = _json_loads(line)
                            msg_type = data.get("type", "").lower()

                            if message_type and msg_type != message_type.lower():
                                continue

                            if tool is not None:
                                # Tool filtering: scan content array DIRECTLY for tool_use blocks.
                                # IMPORTANT: _extract_content() only returns type=="text" blocks
                                # and would miss tool_use entries entirely — must NOT use it here.
                                msg_content = data.get("message", {}).get("content", [])
                                if not isinstance(msg_content, list):
                                    continue
                                for item in msg_content:
                                    if (isinstance(item, dict)
                                            and item.get("type") == "tool_use"
                                            and item.get("name", "").lower() == tool_lower):
                                        # Serialize input for query matching + display
                                        input_str = json.dumps(item.get("input", {}))
                                        if not pattern or pattern.search(input_str):
                                            messages.append(SessionMessage(
                                                type=self._parse_message_type(msg_type),
                                                timestamp=data.get("timestamp", ""),
                                                content=input_str,
                                                session_id=data.get("sessionId", ""),
                                            ))
                                            break  # one match per message line
                            else:
                                # Original behavior: extract text content only
                                content = self._extract_content(data)
                                if content and (not pattern or pattern.search(content)):
                                    messages.append(
                                        SessionMessage(
                                            type=self._parse_message_type(msg_type),
                                            timestamp=data.get("timestamp", ""),
                                            content=content,
                                            session_id=data.get("sessionId", ""),
                                        )
                                    )
                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue
            except OSError:
                continue

        return messages

    def get_sessions(
        self,
        project_filter: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
    ) -> List[SessionInfo]:
        """List all sessions with metadata, sorted newest-first.

        Args:
            project_filter: Substring to match against project_dir name. None = all projects.
            after:  Only sessions with timestamp_first >= this (ISO prefix, e.g. "2026-01-15").
            before: Only sessions with timestamp_first <= this (ISO prefix, e.g. "2026-12-31").

        Returns:
            List of SessionInfo, sorted by timestamp_first descending (newest first).
        """
        sessions: List[SessionInfo] = []
        for project_dir_name, jsonl_file in self._iter_all_jsonl(project_filter):
            session_id = jsonl_file.stem
            cwd, git_branch = "", "unknown"
            ts_first, ts_last = "", ""
            message_count = 0
            has_compact = False
            for data in self._scan_jsonl(jsonl_file):
                ts = data.get("timestamp", "")
                if ts and not ts_first:
                    ts_first = ts
                if ts:
                    ts_last = ts
                if not cwd and data.get("cwd"):
                    cwd = data["cwd"]
                if git_branch == "unknown" and data.get("gitBranch"):
                    git_branch = data["gitBranch"]
                if data.get("isCompactSummary"):
                    has_compact = True
                if data.get("type") in ("user", "assistant"):
                    message_count += 1
            if after and ts_first and ts_first < after:
                continue
            if before and ts_first and ts_first > before:
                continue
            sessions.append(SessionInfo(
                session_id=session_id,
                project_dir=project_dir_name,
                cwd=cwd,
                git_branch=git_branch,
                timestamp_first=ts_first,
                timestamp_last=ts_last,
                message_count=message_count,
                has_compact_summary=has_compact,
            ))
        sessions.sort(key=lambda s: s.timestamp_first, reverse=True)
        return sessions

    def find_corrections(
        self,
        project_filter: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        patterns: Optional[List[tuple]] = None,
        limit: int = 50,
    ) -> List[CorrectionMatch]:
        """Find user messages where corrections were given to Claude.

        Args:
            project_filter: Substring to match project_dir. None = all projects.
            after:    Only messages >= this timestamp (ISO prefix).
            before:   Only messages <= this timestamp (ISO prefix).
            patterns: Override DEFAULT_CORRECTION_PATTERNS. Each tuple is
                      (category, [regex_keyword_strings]).
            limit:    Max results. Default: 50.

        Returns:
            List of CorrectionMatch, sorted by timestamp descending.
        """
        _patterns = patterns or DEFAULT_CORRECTION_PATTERNS
        # Pre-compile: one regex per category
        compiled = [
            (cat, re.compile("|".join(kws), re.IGNORECASE), kws)
            for cat, kws in _patterns
        ]
        results: List[CorrectionMatch] = []
        for project_dir_name, jsonl_file in self._iter_all_jsonl(project_filter):
            try:
                with open(jsonl_file, encoding="utf-8", errors="replace") as f:
                    for line in f:
                        try:
                            # Raw pre-filter: skip non-user lines cheaply before json.loads
                            if '"type":"user"' not in line and '"type": "user"' not in line:
                                continue
                            data = _json_loads(line)
                            if data.get("type") != "user":
                                continue
                            ts = data.get("timestamp", "")
                            if after and ts and ts < after:
                                continue
                            if before and ts and ts > before:
                                continue
                            content = self._extract_content(data)
                            if not content:
                                continue
                            for cat, regex, _kws in compiled:
                                m = regex.search(content)
                                if m:
                                    results.append(CorrectionMatch(
                                        session_id=data.get("sessionId", ""),
                                        project_dir=project_dir_name,
                                        timestamp=ts,
                                        content=content,
                                        category=cat,
                                        matched_pattern=m.group(0),
                                    ))
                                    break
                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue
            except OSError:
                continue
        results.sort(key=lambda x: x.timestamp, reverse=True)
        return results[:limit]

    def analyze_planning_usage(
        self,
        commands: Optional[List[str]] = None,
        project_filter: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
    ) -> List[PlanningCommandCount]:
        """Count slash command usage across all sessions, sorted by frequency.

        Two modes depending on whether ``commands`` is provided:

        **Discovery mode** (``commands=None``, the default):
            Auto-discovers every slash command actually used — any user message whose
            content starts with ``/word`` (e.g. ``/commit``, ``/ar:plannew``, ``/help``).
            No configuration required; works on any Claude Code workspace.

        **Pattern mode** (``commands`` is a list of regex strings):
            Counts only the commands matching the supplied regex patterns.
            Useful when you want to track a specific set of commands and apply
            word-boundary anchors for precision.

        Args:
            commands:       Regex patterns to match (pattern mode). ``None`` = auto-discover.
            project_filter: Substring to match project_dir. None = all projects.
            after:          Only messages >= this timestamp (ISO prefix).
            before:         Only messages <= this timestamp (ISO prefix).

        Returns:
            List of PlanningCommandCount, sorted by count descending.
        """
        discovery_mode = commands is None
        # Pattern mode only: compile provided patterns
        compiled = (
            []
            if discovery_mode
            else [(cmd, re.compile(cmd, re.IGNORECASE)) for cmd in commands]  # type: ignore[union-attr]
        )
        counts: Dict[str, int] = defaultdict(int)
        session_ids_by_cmd: Dict[str, set] = defaultdict(set)
        project_dirs_by_cmd: Dict[str, set] = defaultdict(set)
        for project_dir_name, jsonl_file in self._iter_all_jsonl(project_filter):
            try:
                with open(jsonl_file, encoding="utf-8", errors="replace") as f:
                    for line in f:
                        try:
                            # Raw pre-filter: discovery mode only needs user messages
                            if discovery_mode:
                                if '"type":"user"' not in line and '"type": "user"' not in line:
                                    continue
                            data = _json_loads(line)
                            if discovery_mode and data.get("type") != "user":
                                continue
                            ts = data.get("timestamp", "")
                            if after and ts and ts < after:
                                continue
                            if before and ts and ts > before:
                                continue
                            content = self._extract_content(data)
                            if not content:
                                continue
                            session_id = data.get("sessionId", "")
                            if discovery_mode:
                                # Match slash command at the very start of message content
                                m = _SLASH_CMD_DISCOVERY_RE.match(content.lstrip())
                                if m:
                                    cmd = m.group(0)  # e.g. "/ar:plannew", "/commit"
                                    counts[cmd] += 1
                                    session_ids_by_cmd[cmd].add(session_id)
                                    project_dirs_by_cmd[cmd].add(project_dir_name)
                            else:
                                for cmd, regex in compiled:
                                    if regex.search(content):
                                        counts[cmd] += 1
                                        session_ids_by_cmd[cmd].add(session_id)
                                        project_dirs_by_cmd[cmd].add(project_dir_name)
                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue
            except OSError:
                continue
        if discovery_mode:
            result = [
                PlanningCommandCount(
                    command=cmd,
                    count=counts[cmd],
                    session_ids=sorted(session_ids_by_cmd[cmd]),
                    project_dirs=sorted(project_dirs_by_cmd[cmd]),
                )
                for cmd in counts
            ]
        else:
            result = [
                PlanningCommandCount(
                    # Normalize display name: strip trailing \b regex suffix
                    command=re.sub(r"\\b$", "", cmd),
                    count=counts[cmd],
                    session_ids=sorted(session_ids_by_cmd[cmd]),
                    project_dirs=sorted(project_dirs_by_cmd[cmd]),
                )
                for cmd in (commands or []) if counts[cmd] > 0
            ]
        result.sort(key=lambda x: x.count, reverse=True)
        return result

    def cross_reference_session(
        self,
        filename: str,
        current_content: str,
        session_id: Optional[str] = None,
        snippet_chars: int = 200,
    ) -> List[dict]:
        """Find Edit/Write calls for a file and check if their content appears in current_content.

        Args:
            filename:        Basename to match (e.g. "cli.py"). Matched against file_path ending.
            current_content: Current file content to compare against.
            session_id:      Session ID prefix to restrict search. None = all sessions.
            snippet_chars:   Characters to include in content_snippet. Default: 200.

        Returns:
            List of dicts sorted by timestamp ascending, each with keys:
                session_id (str), project_dir (str), timestamp (str),
                tool ("Edit"|"Write"), file_path (str),
                content_snippet (str, first snippet_chars chars), found_in_current (bool)
        """
        results: List[dict] = []
        for project_dir_name, jsonl_file in self._iter_all_jsonl():
            try:
                with open(jsonl_file, encoding="utf-8", errors="replace") as f:
                    for line in f:
                        # Raw pre-filter: only assistant messages carry tool_use blocks
                        if '"type":"assistant"' not in line and '"type": "assistant"' not in line:
                            continue
                        try:
                            data = _json_loads(line)
                            if data.get("type") != "assistant":
                                continue
                            sid = data.get("sessionId", "")
                            if session_id and not sid.startswith(session_id):
                                continue
                            # Direct content array scan — cannot use _extract_content()
                            msg_content = data.get("message", {}).get("content", [])
                            if not isinstance(msg_content, list):
                                continue
                            for item in msg_content:
                                if not isinstance(item, dict):
                                    continue
                                tool = item.get("name", "")
                                if item.get("type") != "tool_use" or tool not in ("Edit", "Write"):
                                    continue
                                inp = item.get("input", {})
                                fp = inp.get("file_path", "")
                                if not fp.endswith(filename):
                                    continue
                                # Edit uses new_string; Write uses content
                                snippet_src = inp.get("new_string") or inp.get("content", "")
                                snippet = snippet_src[:snippet_chars]
                                results.append({
                                    "session_id": sid,
                                    "project_dir": project_dir_name,
                                    "timestamp": data.get("timestamp", ""),
                                    "tool": tool,
                                    "file_path": fp,
                                    "content_snippet": snippet,
                                    "found_in_current": bool(snippet and snippet in current_content),
                                })
                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue
            except OSError:
                continue
        results.sort(key=lambda x: x["timestamp"])
        return results

    def export_session_markdown(self, session_id: str) -> str:
        """Export one session's messages as a markdown string.

        Args:
            session_id: Session ID prefix to match. Must match exactly one session.

        Returns:
            Markdown string with metadata header + all non-system messages.

        Raises:
            ValueError: If no session matches, or multiple sessions match.
        """
        matches = self._find_session_files(session_id)
        if not matches:
            raise ValueError(f"No session found matching: {session_id!r}")
        if len(matches) > 1:
            candidates = ", ".join(m[0].stem[:16] for m in matches)
            raise ValueError(
                f"Ambiguous session ID {session_id!r} matches {len(matches)} sessions: {candidates}"
            )
        session_file, _project_dir_name = matches[0]

        # Scan file for metadata + messages
        cwd, git_branch, ts_first = "", "unknown", ""
        message_count = 0
        lines_md: List[str] = []

        for data in self._scan_jsonl(session_file):
            ts = data.get("timestamp", "")
            if ts and not ts_first:
                ts_first = ts
            if not cwd and data.get("cwd"):
                cwd = data["cwd"]
            if git_branch == "unknown" and data.get("gitBranch"):
                git_branch = data["gitBranch"]
            msg_type = data.get("type", "")
            if msg_type not in ("user", "assistant"):
                continue
            is_summary = data.get("isCompactSummary", False)
            content = self._extract_content(data)
            # Filter system noise — count only messages that are actually rendered
            if any(pat in content for pat in _EXPORT_FILTER_PATTERNS):
                continue
            if not content.strip():
                continue
            message_count += 1
            ts_short = ts[:16].replace("T", " ") if ts else "\u2014"
            if is_summary:
                lines_md.append(f"## Session Summary\n\n{content}\n\n---\n")
            else:
                lines_md.append(f"## [{msg_type}] {ts_short}\n\n{content}\n\n---\n")

        short_id = session_id[:8]
        header = (
            f"# Session {short_id}\n\n"
            f"**Date**: {ts_first}\n"
            f"**Branch**: {git_branch}\n"
            f"**Directory**: {cwd}\n"
            f"**Messages**: {message_count}\n\n"
            f"---\n\n"
        )
        return header + "\n".join(lines_md)

    def analyze_session(self, session_id: str) -> Optional[SessionAnalysis]:
        """Return per-session statistics: message counts, tool usage, and files touched.

        Files touched are detected by scanning every ``tool_use`` block for an input
        key named ``file_path`` — this is not hardcoded to Edit/Write/Read, so it
        captures any tool that operates on files, including future or custom tools.

        Args:
            session_id: Session ID prefix. Must match exactly one session JSONL file.

        Returns:
            SessionAnalysis, or None if no matching session is found.
        """
        matches = self._find_session_files(session_id)
        if not matches:
            return None
        session_file, project_dir_name = matches[0]  # newest first on ambiguous prefix

        total_lines = 0
        user_count = 0
        assistant_count = 0
        tool_uses_by_name: Dict[str, int] = {}
        files_touched_set: Set[str] = set()
        ts_first = ""
        ts_last = ""

        for data in self._scan_jsonl(session_file):
            total_lines += 1
            ts = data.get("timestamp", "")
            if ts and not ts_first:
                ts_first = ts
            if ts:
                ts_last = ts
            msg_type = data.get("type", "")
            if msg_type == "user":
                user_count += 1
            elif msg_type == "assistant":
                assistant_count += 1
                # Scan tool_use blocks — general: detect any tool with file_path input
                msg_content = data.get("message", {}).get("content", [])
                if isinstance(msg_content, list):
                    for item in msg_content:
                        if not isinstance(item, dict) or item.get("type") != "tool_use":
                            continue
                        tool_name = item.get("name", "unknown")
                        tool_uses_by_name[tool_name] = tool_uses_by_name.get(tool_name, 0) + 1
                        inp = item.get("input", {})
                        if isinstance(inp, dict) and "file_path" in inp:
                            fp = inp["file_path"]
                            if fp:
                                files_touched_set.add(fp)

        return SessionAnalysis(
            session_id=session_id,
            project_dir=project_dir_name,
            total_lines=total_lines,
            user_count=user_count,
            assistant_count=assistant_count,
            tool_uses_by_name=tool_uses_by_name,
            files_touched=sorted(files_touched_set),
            timestamp_first=ts_first,
            timestamp_last=ts_last,
        )

    def timeline_session(
        self,
        session_id: str,
        preview_chars: int = 150,
    ) -> List[dict]:
        """Return a chronological timeline of user/assistant events for one session.

        Each event includes the message type, timestamp, a content preview,
        and the number of tool_use blocks invoked (for assistant messages).

        Args:
            session_id:    Session ID prefix. Must match exactly one JSONL file.
            preview_chars: Maximum characters to include in content_preview. Default: 150.

        Returns:
            List of dicts sorted by timestamp ascending, each with keys:
                type (str), timestamp (str), content_preview (str, ≤preview_chars),
                tool_count (int)
        """
        matches = self._find_session_files(session_id)
        if not matches:
            return []
        session_file, _project_dir_name = matches[0]  # newest first on ambiguous prefix

        events: List[dict] = []
        for data in self._scan_jsonl(session_file):
            msg_type = data.get("type", "")
            if msg_type not in ("user", "assistant"):
                continue
            ts = data.get("timestamp", "")
            content = self._extract_content(data)
            tool_count = 0
            if msg_type == "assistant":
                msg_content = data.get("message", {}).get("content", [])
                if isinstance(msg_content, list):
                    tool_count = sum(
                        1 for item in msg_content
                        if isinstance(item, dict) and item.get("type") == "tool_use"
                    )
            events.append({
                "type": msg_type,
                "timestamp": ts,
                "content_preview": content[:preview_chars],
                "tool_count": tool_count,
            })
        return events

    def search_messages_with_context(
        self,
        query: str,
        context: int = 3,
        message_type: Optional[str] = None,
        tool: Optional[str] = None,
    ) -> List[ContextMatch]:
        """Search messages and return each match with surrounding context messages.

        Unlike ``search_messages``, this buffers each JSONL file in memory to
        retrieve up to ``context`` messages before and after each match from the
        same session file.

        Args:
            query:        Text or glob/regex pattern to search for.
            context:      Number of messages before AND after each match to include.
            message_type: Optional filter: 'user', 'assistant', or 'system'.
            tool:         Optional tool name filter (same semantics as search_messages).

        Returns:
            List of ContextMatch, one per matching message (not deduplicated).
        """
        if tool is not None and message_type is None:
            message_type = "assistant"

        pattern = self._compile_pattern(query) if query else None
        tool_lower = tool.lower() if tool else None

        results: List[ContextMatch] = []

        for _project_dir_name, jsonl_file in self._iter_all_jsonl():
            # Buffer all messages from this file (needed for context window)
            all_msgs: List[SessionMessage] = []
            try:
                with open(jsonl_file, encoding="utf-8", errors="replace") as f:
                    for line in f:
                        try:
                            data = _json_loads(line)
                            msg_type_raw = data.get("type", "").lower()
                            if msg_type_raw not in ("user", "assistant", "system"):
                                continue
                            if message_type and msg_type_raw != message_type.lower():
                                continue
                            if tool is not None:
                                msg_content = data.get("message", {}).get("content", [])
                                if not isinstance(msg_content, list):
                                    continue
                                for item in msg_content:
                                    if (isinstance(item, dict)
                                            and item.get("type") == "tool_use"
                                            and item.get("name", "").lower() == tool_lower):
                                        input_str = json.dumps(item.get("input", {}))
                                        all_msgs.append(SessionMessage(
                                            type=self._parse_message_type(msg_type_raw),
                                            timestamp=data.get("timestamp", ""),
                                            content=input_str,
                                            session_id=data.get("sessionId", ""),
                                        ))
                                        break
                            else:
                                content = self._extract_content(data)
                                if content:
                                    all_msgs.append(SessionMessage(
                                        type=self._parse_message_type(msg_type_raw),
                                        timestamp=data.get("timestamp", ""),
                                        content=content,
                                        session_id=data.get("sessionId", ""),
                                    ))
                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue
            except OSError:
                continue

            # Find matches and collect context windows
            for i, msg in enumerate(all_msgs):
                if not pattern or pattern.search(msg.content):
                    before = all_msgs[max(0, i - context): i]
                    after = all_msgs[i + 1: i + 1 + context]
                    results.append(ContextMatch(
                        match=msg,
                        context_before=before,
                        context_after=after,
                    ))

        return results

    def get_statistics(self) -> RecoveryStatistics:
        """Get recovery statistics across all sessions."""
        session_ids: Set[str] = set()
        total_files = 0
        total_versions = 0
        largest_file = None
        largest_edits = 0

        if not self.recovery_dir.exists():
            return RecoveryStatistics()

        # Count sessions and files from session_*/ dirs
        for session_dir in self.recovery_dir.glob("session_*"):
            if not session_dir.is_dir() or "all_versions" in session_dir.name:
                continue

            session_id = session_dir.name.replace("session_", "")
            session_ids.add(session_id)

            for file_path in session_dir.glob("*"):
                if file_path.is_file():
                    total_files += 1

        # Count versions globally (accumulate across all sessions before comparing)
        file_version_totals: Dict[str, int] = defaultdict(int)
        for session_dir in self._version_dirs:
            version_files = list(session_dir.glob("*_v*_line_*.txt"))
            total_versions += len(version_files)

            for v_file in version_files:
                filename = self._extract_filename(v_file.name)
                file_version_totals[filename] += 1

        # Find globally largest file after accumulating all sessions
        for filename, edits in file_version_totals.items():
            if edits > largest_edits:
                largest_edits = edits
                largest_file = filename

        return RecoveryStatistics(
            total_sessions=len(session_ids),
            total_files=total_files,
            total_versions=total_versions,
            largest_file=largest_file,
            largest_file_edits=largest_edits,
        )

    @staticmethod
    def _extract_filename(version_filename: str) -> str:
        """Extract original filename from versioned name (e.g. 'cli.py_v000001_line_10.txt' → 'cli.py')."""
        match = re.match(r"(.+?)_v\d+_line_\d+\.txt$", version_filename)
        return match.group(1) if match else version_filename

    @staticmethod
    def _extract_content(data: dict) -> str:
        """Extract text content from a JSONL message record.

        Handles three message formats:
          - dict with content list of {type, text} blocks (multi-part messages)
          - dict with content string (simple messages)
          - raw string message
        """
        if "message" not in data:
            return ""

        msg = data["message"]
        if isinstance(msg, dict):
            content_data = msg.get("content", "")
            if isinstance(content_data, list):
                return " ".join(
                    item.get("text", "")
                    for item in content_data
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            if isinstance(content_data, str):
                return content_data
        elif isinstance(msg, str):
            return msg

        return ""

    def get_original_path(self, filename: str) -> Optional[str]:  # noqa: C901
        """Find the most recent original path where Claude wrote or edited this filename.

        Searches project JSONL files for Write/Edit/NotebookEdit tool calls and
        toolUseResult confirmation messages.  Returns the last recorded absolute path,
        or None if not found.

        Args:
            filename: Exact filename to look up (e.g. 'cli.py')

        Returns:
            Most recently recorded absolute path string, or None.
        """
        if not self.projects_dir.exists():
            return None

        last_path: Optional[str] = None

        for project_dir in self.projects_dir.iterdir():
            if not project_dir.is_dir():
                continue
            for jsonl_file in project_dir.glob("*.jsonl"):
                try:
                    with open(jsonl_file, encoding="utf-8", errors="replace") as f:
                        for line in f:
                            # Fast pre-filter: filename must appear in the raw line.
                            if filename not in line:
                                continue
                            try:
                                data = _json_loads(line)
                                # Path 1: toolUseResult.filePath (user confirmation message)
                                tool_result = data.get("toolUseResult") or {}
                                if isinstance(tool_result, dict):
                                    fp = tool_result.get("filePath", "")
                                    if fp and Path(fp).name == filename:
                                        last_path = fp
                                        continue
                                # Path 2: message.content[].input.file_path (assistant tool_use)
                                msg = data.get("message") or {}
                                if isinstance(msg, dict):
                                    for item in msg.get("content") or []:
                                        if (
                                            isinstance(item, dict)
                                            and item.get("type") == "tool_use"
                                            and item.get("name") in ("Write", "Edit", "NotebookEdit")
                                        ):
                                            fp = (item.get("input") or {}).get("file_path", "")
                                            if fp and Path(fp).name == filename:
                                                last_path = fp
                            except (json.JSONDecodeError, KeyError, ValueError):
                                continue
                except OSError:
                    continue

        return last_path
