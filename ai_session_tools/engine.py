"""
Core session recovery engine - refactored for composition and extensibility.

Copyright (c) 2026 Andrew Hundt
Licensed under the Apache License, Version 2.0
"""

import fnmatch
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Set

from .models import (
    FileLocation,
    FileVersion,
    FilterSpec,
    MessageType,
    RecoveredFile,
    RecoveryStatistics,
    SessionMessage,
)


class SessionRecoveryEngine:
    """Core recovery engine with clean separation of concerns."""

    # Configuration constants (parameterized)
    FILE_TYPE_PATTERNS: ClassVar[Dict[str, tuple]] = {
        "python": ("*.py", r"\.py$"),
        "markdown": ("*.md", r"\.md$"),
        "json": ("*.json", r"\.json$"),
    }

    LOST_LOCATION_PATTERNS: ClassVar[List[str]] = [
        r"/tmp",
        r"/var",
        r"/private/tmp",
        r"\.cache",
        r"\.temp",
    ]

    def __init__(self, projects_dir: Path, recovery_dir: Path):
        """Initialize engine with directories."""
        self.projects_dir = Path(projects_dir)
        self.recovery_dir = Path(recovery_dir)
        self._file_cache: Dict[str, RecoveredFile] = {}
        self._version_cache: Dict[str, List[FileVersion]] = {}

    @staticmethod
    def _compile_pattern(pattern: str) -> re.Pattern:
        """Compile pattern, auto-detecting glob vs regex."""
        if "*" in pattern or "?" in pattern:
            regex_pattern = fnmatch.translate(pattern)
            return re.compile(regex_pattern, re.IGNORECASE)
        return re.compile(pattern, re.IGNORECASE)

    def _get_or_create_file_info(self, file_path: Path) -> RecoveredFile:
        """Get or create cached file info."""
        if file_path.name not in self._file_cache:
            versions = self.get_versions(file_path.name)
            self._file_cache[file_path.name] = RecoveredFile(
                name=file_path.name,
                path=str(file_path),
                location=FileLocation.CLAUTORUN_MAIN,
                file_type=file_path.suffix[1:] or "unknown",
                sessions=[v.session_id for v in versions],
                edits=len(versions),
                size_bytes=file_path.stat().st_size,
            )
        return self._file_cache[file_path.name]

    def _apply_all_filters(self, file_info: RecoveredFile, filters: FilterSpec) -> bool:
        """Check if file passes all filters. Returns True if file should be included."""
        if file_info.sessions:
            if not any(filters.matches_session(s) for s in file_info.sessions):
                return False
        elif filters.include_sessions:
            return False

        if not filters.matches_location(file_info.location.value):
            return False

        if not filters.matches_edits(file_info.edits):
            return False

        if not filters.matches_extension(file_info.file_type):
            return False

        return True

    def search(
        self,
        pattern: str,
        filters: Optional[FilterSpec] = None,
    ) -> List[RecoveredFile]:
        """
        Search files with optional filtering.

        Args:
            pattern: Glob or regex pattern
            filters: Optional FilterSpec for advanced filtering

        Returns:
            List of RecoveredFile objects sorted by edits (descending)
        """
        if filters is None:
            filters = FilterSpec()

        pattern_re = self._compile_pattern(pattern)
        results = []

        for session_dir in self.recovery_dir.glob("session_*"):
            if not session_dir.is_dir() or "all_versions" in session_dir.name:
                continue

            for file_path in session_dir.glob("*"):
                if not file_path.is_file() or not pattern_re.search(file_path.name):
                    continue

                file_info = self._get_or_create_file_info(file_path)
                if self._apply_all_filters(file_info, filters):
                    results.append(file_info)

        return sorted(results, key=lambda f: f.edits, reverse=True)

    def get_versions(self, filename: str) -> List[FileVersion]:
        """Get all versions of a file."""
        if filename in self._version_cache:
            return self._version_cache[filename]

        versions = []
        version_pattern = r"_v(\d+)_line_(\d+)\.txt$"

        for session_dir in self.recovery_dir.glob("session_all_versions_*"):
            if not session_dir.is_dir():
                continue

            for version_file in session_dir.glob(f"{re.escape(filename)}_v*_line_*.txt"):
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
        """Extract final version of a file."""
        versions = self.get_versions(filename)

        if versions:
            final = max(versions, key=lambda v: v.line_count)
            session_dir = self.recovery_dir / f"session_all_versions_{final.session_id}"
            version_file = session_dir / f"{filename}_v{final.version_num:06d}_line_{final.line_count}.txt"

            if version_file.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / filename
                output_path.write_text(version_file.read_text(errors="ignore"))
                return output_path

        # Fallback: check session_*/ directories
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
        """Extract all versions of a file."""
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
        else:
            # Fallback
            for session_dir in self.recovery_dir.glob("session_*/"):
                if "all_versions" not in session_dir.name:
                    file_path = session_dir / filename
                    if file_path.exists():
                        target = output_dir / "v000001_final.txt"
                        target.write_text(file_path.read_text(errors="ignore"))
                        extracted.append(target)
                        break

        return extracted

    def _process_message_line(self, line: str, session_id: str, message_type: Optional[str]) -> Optional[SessionMessage]:
        """Process a single JSONL line into a SessionMessage."""
        try:
            data = json.loads(line)
            if data.get("sessionId") != session_id:
                return None

            msg_type = data.get("type", "").lower()
            if message_type and msg_type != message_type.lower():
                return None

            content = self._extract_content(data)
            if not content:
                return None

            return SessionMessage(
                type=MessageType(msg_type) if msg_type in MessageType.__members__.values() else MessageType.SYSTEM,
                timestamp=data.get("timestamp", ""),
                content=content[:500],
                session_id=session_id,
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def get_messages(self, session_id: str, message_type: Optional[str] = None) -> List[SessionMessage]:
        """Extract messages from a session."""
        messages = []

        try:
            for project_dir in self.projects_dir.glob("*"):
                if not project_dir.is_dir():
                    continue

                for jsonl_file in project_dir.glob("*.jsonl"):
                    try:
                        with open(jsonl_file) as f:
                            for line in f:
                                msg = self._process_message_line(line, session_id, message_type)
                                if msg:
                                    messages.append(msg)
                    except OSError:
                        continue
        except Exception:
            pass

        return messages

    def search_messages(self, query: str, message_type: Optional[str] = None) -> List[SessionMessage]:
        """Search for messages across all sessions."""
        messages = []
        pattern = self._compile_pattern(query)

        try:
            for project_dir in self.projects_dir.glob("*"):
                if not project_dir.is_dir():
                    continue

                for jsonl_file in project_dir.glob("*.jsonl"):
                    try:
                        with open(jsonl_file) as f:
                            for line in f:
                                try:
                                    data = json.loads(line)
                                    msg_type = data.get("type", "").lower()

                                    if message_type and msg_type != message_type.lower():
                                        continue

                                    content = self._extract_content(data)
                                    if content and pattern.search(content):
                                        messages.append(
                                            SessionMessage(
                                                type=MessageType(msg_type) if msg_type in MessageType.__members__.values() else MessageType.SYSTEM,
                                                timestamp=data.get("timestamp", ""),
                                                content=content[:500],
                                                session_id=data.get("sessionId", ""),
                                            )
                                        )
                                except (json.JSONDecodeError, KeyError, ValueError):
                                    continue
                    except OSError:
                        continue
        except Exception:
            pass

        return messages

    def get_statistics(self) -> RecoveryStatistics:
        """Get recovery statistics."""
        session_ids: Set[str] = set()
        total_files = 0
        total_versions = 0
        largest_file = None
        largest_edits = 0

        # Count sessions and files
        for session_dir in self.recovery_dir.glob("session_*"):
            if not session_dir.is_dir() or "all_versions" in session_dir.name:
                continue

            session_id = session_dir.name.replace("session_", "")
            session_ids.add(session_id)

            for file_path in session_dir.glob("*"):
                if file_path.is_file():
                    total_files += 1

        # Count versions
        for session_dir in self.recovery_dir.glob("session_all_versions_*"):
            if not session_dir.is_dir():
                continue

            version_files = list(session_dir.glob("*_v*_line_*.txt"))
            total_versions += len(version_files)

            # Track largest file
            files_in_session = defaultdict(int)
            for v_file in version_files:
                filename = self._extract_filename(v_file.name)
                files_in_session[filename] += 1

            for filename, edits in files_in_session.items():
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
        """Extract original filename from versioned name."""
        match = re.match(r"(.+?)_v\d+_line_\d+\.txt$", version_filename)
        return match.group(1) if match else version_filename

    @staticmethod
    def _extract_content(data: dict) -> str:
        """Extract content from JSONL data."""
        if "message" not in data:
            return ""

        msg = data["message"]
        if isinstance(msg, dict):
            if "content" in msg:
                content_data = msg["content"]
                if isinstance(content_data, list):
                    text_parts = []
                    for item in content_data:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    return " ".join(text_parts)
                elif isinstance(content_data, str):
                    return content_data
        elif isinstance(msg, str):
            return msg

        return ""
