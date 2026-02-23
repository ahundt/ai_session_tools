"""
Extraction strategies for different data types.

Copyright (c) 2026 Andrew Hundt
Licensed under the Apache License, Version 2.0
"""

from pathlib import Path
from typing import List, Optional

from .models import SessionMessage


class FileExtractor:
    """Extract file content with strategies."""

    def __init__(self, recovery_dir: Path):
        """Initialize with recovery directory."""
        self.recovery_dir = recovery_dir

    def extract_final(self, filename: str, output_dir: Path) -> Optional[Path]:
        """Extract final version of a file."""
        # This would be implemented by the engine
        raise NotImplementedError("Use SessionRecoveryEngine.extract_final()")

    def extract_all_versions(self, filename: str, output_dir: Path) -> List[Path]:
        """Extract all versions of a file."""
        # This would be implemented by the engine
        raise NotImplementedError("Use SessionRecoveryEngine.extract_all()")

    def extract_by_session(self, filename: str, session_id: str, output_dir: Path) -> Optional[Path]:
        """Extract file from specific session."""
        raise NotImplementedError()


class MessageExtractor:
    """Extract messages from JSONL session files."""

    def __init__(self, projects_dir: Path):
        """Initialize with projects directory."""
        self.projects_dir = projects_dir

    def extract_all_from_session(self, session_id: str) -> List[SessionMessage]:
        """Extract all messages from a session."""
        raise NotImplementedError("Use SessionRecoveryEngine.get_messages()")

    def extract_by_type(self, session_id: str, message_type: str) -> List[SessionMessage]:
        """Extract messages of specific type."""
        raise NotImplementedError()

    def extract_with_content(self, session_id: str, content_pattern: str) -> List[SessionMessage]:
        """Extract messages matching content pattern."""
        raise NotImplementedError()


class BulkExtractor:
    """Extract multiple files at once."""

    def __init__(self, engine):
        """Initialize with engine."""
        self.engine = engine

    def extract_files(self, files: List[str], output_dir: Path) -> dict:
        """Extract multiple files."""
        results = {}
        for filename in files:
            path = self.engine.extract_final(filename, output_dir)
            results[filename] = path
        return results

    def extract_all_histories(self, files: List[str], output_dir: Path) -> dict:
        """Extract all histories for multiple files."""
        results = {}
        for filename in files:
            paths = self.engine.extract_all(filename, output_dir)
            results[filename] = paths
        return results
