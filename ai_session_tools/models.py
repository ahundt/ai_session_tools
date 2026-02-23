"""
Data models for session recovery - using modern Python patterns.

Includes dataclasses, enums, and structured types for type safety.

Copyright (c) 2026 Andrew Hundt
Licensed under the Apache License, Version 2.0
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set


class FileLocation(str, Enum):
    """File location categories."""

    CLAUTORUN_MAIN = "clautorun/main"
    CLAUTORUN_WORKTREE = "clautorun/worktree"
    CLAUTORUN_PLANEXPORT = "clautorun/plan-export"
    AUTORUN = "autorun"
    EXTERNAL = "external"
    ALL = "all"


class FileType(str, Enum):
    """Supported file types."""

    PYTHON = "python"
    MARKDOWN = "markdown"
    JSON = "json"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    SHELL = "shell"
    ALL = "all"


class MessageType(str, Enum):
    """Session message types."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass(frozen=True)
class FileVersion:
    """Immutable file version representation."""

    filename: str
    version_num: int
    line_count: int
    session_id: str
    timestamp: str = ""

    def __lt__(self, other: "FileVersion") -> bool:
        """Compare by version number for sorting."""
        if not isinstance(other, FileVersion):
            return NotImplemented
        return self.version_num < other.version_num


@dataclass
class RecoveredFile:
    """Recovered file from Claude Code sessions with edit history.

    Attributes:
        name: Filename (e.g., "session_manager.py")
        path: Full file path
        location: Where file was saved (main repo, worktree, etc)
        file_type: File extension without dot (e.g., "py", "md", "json")
        sessions: List of session IDs that touched this file
        edits: Total number of times THIS FILE was modified across ALL sessions
               (1-2 = simple one-off file, 10+ = heavily refined project file)
        created_date: When file was first created
        last_modified: When file was last edited
        size_bytes: File size in bytes
    """

    name: str
    path: str
    location: FileLocation
    file_type: str
    sessions: List[str] = field(default_factory=list)
    edits: int = 0
    created_date: Optional[str] = None
    last_modified: Optional[str] = None
    size_bytes: int = 0

    @property
    def is_versioned(self) -> bool:
        """Check if file has version history."""
        return self.edits > 0

    @property
    def session_count(self) -> int:
        """Get number of sessions this file appears in."""
        return len(self.sessions)


@dataclass
class SessionMessage:
    """Message from a session (user or assistant)."""

    type: MessageType
    timestamp: str
    content: str
    session_id: str
    line_count: Optional[int] = None

    def preview(self, limit: int = 100) -> str:
        """Get preview of message content.

        Args:
            limit: Max characters to show. 0 = no limit (full content).
        """
        text = self.content.replace("\n", " ")
        if limit and len(text) > limit:
            return text[:limit]
        return text

    @property
    def is_long(self) -> bool:
        """Check if message is long."""
        return len(self.content) > 500


@dataclass
class SessionMetadata:
    """Rich metadata from session JSONL."""

    session_id: str
    timestamp: str
    message_type: str
    operation: Optional[str] = None
    tool_name: Optional[str] = None
    file_path: Optional[str] = None

    @property
    def has_file_operation(self) -> bool:
        """Check if this is a file operation."""
        return self.tool_name in ("Write", "Edit", "Read")


@dataclass
class FilterSpec:
    """Advanced filter specification (immutable builder pattern)."""

    # File patterns
    file_patterns: List[str] = field(default_factory=lambda: ["*"])

    # File extension filtering (e.g., "py", "md", "json")
    include_extensions: Set[str] = field(default_factory=set)
    exclude_extensions: Set[str] = field(default_factory=set)

    # Edit count range: how many times a FILE was modified across all sessions
    # (1-2 = simple one-off files, 10+ = major project files with long history)
    min_edits: int = 0
    max_edits: Optional[int] = None

    # Session filtering
    include_sessions: Set[str] = field(default_factory=set)
    exclude_sessions: Set[str] = field(default_factory=set)

    # Location filtering
    include_folders: Set[str] = field(default_factory=set)
    exclude_folders: Set[str] = field(default_factory=set)

    # Datetime range filtering (ISO format: "2026-02-22" or "2026-02-22T14:30:00")
    after: Optional[str] = None
    before: Optional[str] = None

    # File size range (bytes)
    min_size: int = 0
    max_size: Optional[int] = None

    def matches_session(self, session_id: str) -> bool:
        """Check if session matches filter."""
        if self.include_sessions and session_id not in self.include_sessions:
            return False
        if self.exclude_sessions and session_id in self.exclude_sessions:
            return False
        return True

    def matches_location(self, location: str) -> bool:
        """Check if location matches filter."""
        location_lower = location.lower()

        if self.include_folders:
            if not any(inc.lower() in location_lower for inc in self.include_folders):
                return False

        if self.exclude_folders:
            if any(exc.lower() in location_lower for exc in self.exclude_folders):
                return False

        return True

    def matches_extension(self, extension: str) -> bool:
        """Check if file extension matches filter.

        Args:
            extension: File extension (with or without leading dot, e.g., "py" or ".py")

        Returns:
            True if extension passes include/exclude filters.
        """
        # Normalize extension (remove leading dot if present)
        normalized_ext = extension.lstrip(".")

        if self.include_extensions:
            if normalized_ext not in self.include_extensions:
                return False

        if self.exclude_extensions:
            if normalized_ext in self.exclude_extensions:
                return False

        return True

    def matches_edits(self, edit_count: int) -> bool:
        """Check if edit count matches filter."""
        if edit_count < self.min_edits:
            return False
        if self.max_edits and edit_count > self.max_edits:
            return False
        return True

    def matches_size(self, size_bytes: int) -> bool:
        """Check if size matches filter."""
        if size_bytes < self.min_size:
            return False
        if self.max_size and size_bytes > self.max_size:
            return False
        return True

    def matches_datetime(self, datetime_str: Optional[str]) -> bool:
        """Check if datetime falls within after/before range.

        Accepts any ISO 8601 prefix: "2026-02-22", "2026-02-22T14:30:00", etc.
        Lexicographic comparison works for all ISO 8601 precisions.

        Args:
            datetime_str: ISO datetime string (e.g. "2026-02-22" or "2026-02-22T14:30:00"), or None.

        Returns:
            True if datetime passes filter. Files with no datetime (None) pass when no filter is set;
            are excluded when a filter IS set (conservative: unknown datetime treated as out-of-range).
        """
        if not datetime_str:
            return not (self.after or self.before)
        if self.after and datetime_str < self.after:
            return False
        if self.before and datetime_str > self.before:
            return False
        return True

    def with_pattern(self, *patterns: str) -> "FilterSpec":
        """Builder: set file patterns."""
        self.file_patterns = list(patterns)
        return self

    def with_edit_range(self, min_edits: int, max_edits: Optional[int] = None) -> "FilterSpec":
        """Builder: set edit count range."""
        self.min_edits = min_edits
        self.max_edits = max_edits
        return self

    def with_sessions(self, include: Optional[Set[str]] = None, exclude: Optional[Set[str]] = None) -> "FilterSpec":
        """Builder: set session filtering."""
        if include:
            self.include_sessions = include
        if exclude:
            self.exclude_sessions = exclude
        return self

    def with_extensions(self, include: Optional[Set[str]] = None, exclude: Optional[Set[str]] = None) -> "FilterSpec":
        """Builder: filter by file extensions.

        Args:
            include: Set of extensions to include (e.g., {"py", "md", "json"}). None means all extensions.
            exclude: Set of extensions to exclude (e.g., {"pyc", "tmp"}). Applied after include filter.

        Returns:
            Self for builder chaining.

        Examples:
            FilterSpec().with_extensions(include={"py", "ts", "js"})  # Only Python, TypeScript, JavaScript
            FilterSpec().with_extensions(exclude={"tmp", "bak"})  # Everything except temp/backup files
        """
        if include:
            self.include_extensions = include
        if exclude:
            self.exclude_extensions = exclude
        return self



@dataclass
class RecoveryStatistics:
    """Recovery operation statistics."""

    total_sessions: int = 0
    total_files: int = 0
    total_versions: int = 0
    largest_file: Optional[str] = None
    largest_file_edits: int = 0
    total_size_bytes: int = 0

    @property
    def avg_versions_per_file(self) -> float:
        """Calculate average versions per file."""
        if self.total_files == 0:
            return 0
        return self.total_versions / self.total_files

    @property
    def avg_edits_per_file(self) -> float:
        """Calculate average edits per file."""
        if self.total_files == 0:
            return 0
        return self.total_versions / self.total_files  # Simplified

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "total_sessions": self.total_sessions,
            "total_files": self.total_files,
            "total_versions": self.total_versions,
            "largest_file": self.largest_file,
            "largest_file_edits": self.largest_file_edits,
            "total_size_bytes": self.total_size_bytes,
            "avg_versions_per_file": self.avg_versions_per_file,
        }
