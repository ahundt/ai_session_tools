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
import warnings
from collections import defaultdict
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Set

try:
    from orjson import loads as _json_loads
except ImportError:
    from json import loads as _json_loads  # type: ignore[assignment]

from .models import (
    FileVersion,
    FilterSpec,
    MessageType,
    RecoveredFile,
    RecoveryStatistics,
    SessionMessage,
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

    def search(
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
                    versions.append(
                        FileVersion(
                            filename=filename,
                            version_num=version_num,
                            line_count=line_count,
                            session_id=session_dir.name.replace("session_all_versions_", ""),
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

    def search_messages(self, query: str, message_type: Optional[str] = None) -> List[SessionMessage]:
        """Search for messages across all sessions.

        Args:
            query: Text or glob/regex pattern to search for in message content
            message_type: Optional filter: 'user', 'assistant', or 'system'

        Returns:
            List of matching SessionMessage objects. Empty list if projects_dir does not exist.

        Raises:
            ValueError: If query is not a valid glob or regex.
        """
        # Compile first: raise ValueError on bad patterns before any I/O.
        pattern = self._compile_pattern(query)

        # Pre-compute literal pre-filter: only safe when query has no regex metacharacters.
        # re.escape(query) == query iff every char is literal (no . + * ? [ ] ^ $ | etc.)
        is_literal = re.escape(query) == query
        query_lower = query.lower() if is_literal else None
        # Heuristic: message_type value must appear in the raw JSON line (no false negatives).
        msg_type_hint = f'"{message_type}"' if message_type else None

        messages: List[SessionMessage] = []

        if not self.projects_dir.exists():
            return messages

        for project_dir in self.projects_dir.glob("*"):
            if not project_dir.is_dir():
                continue

            for jsonl_file in project_dir.glob("*.jsonl"):
                try:
                    with open(jsonl_file, encoding="utf-8", errors="replace") as f:
                        for line in f:
                            try:
                                # Raw line pre-filters: skip json.loads when possible.
                                if query_lower and query_lower not in line.lower():
                                    continue
                                if msg_type_hint and msg_type_hint not in line:
                                    continue
                                data = _json_loads(line)
                                msg_type = data.get("type", "").lower()

                                if message_type and msg_type != message_type.lower():
                                    continue

                                content = self._extract_content(data)
                                if content and pattern.search(content):
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

    def get_original_path(self, filename: str) -> Optional[str]:
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
