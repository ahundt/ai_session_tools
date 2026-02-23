"""
Composable filter implementations for advanced searching.

Copyright (c) 2026 Andrew Hundt
Licensed under the Apache License, Version 2.0
"""

from typing import Any, Callable, List, Optional

from .models import MessageType, RecoveredFile, SessionMessage


class SearchFilter:
    """Composable file search filter."""

    def __init__(self):
        """Initialize filter."""
        self._predicates: List[Callable[[RecoveredFile], bool]] = []

    def by_edits(self, min_edits: int = 0, max_edits: Optional[int] = None) -> "SearchFilter":
        """Filter by edit count range."""

        def predicate(f: RecoveredFile) -> bool:
            if f.edits < min_edits:
                return False
            if max_edits and f.edits > max_edits:
                return False
            return True

        self._predicates.append(predicate)
        return self

    def by_extension(self, extension: str) -> "SearchFilter":
        """Filter by file extension (e.g., 'py', 'md', 'json')."""

        def predicate(f: RecoveredFile) -> bool:
            return f.file_type == extension

        self._predicates.append(predicate)
        return self

    def by_location(self, location: str) -> "SearchFilter":
        """Filter by location."""

        def predicate(f: RecoveredFile) -> bool:
            return location.lower() in f.location.value.lower()

        self._predicates.append(predicate)
        return self

    def by_session(self, session_id: str) -> "SearchFilter":
        """Filter by session."""

        def predicate(f: RecoveredFile) -> bool:
            return session_id in f.sessions

        self._predicates.append(predicate)
        return self

    def by_size(self, min_size: int = 0, max_size: Optional[int] = None) -> "SearchFilter":
        """Filter by file size."""

        def predicate(f: RecoveredFile) -> bool:
            if f.size_bytes < min_size:
                return False
            if max_size and f.size_bytes > max_size:
                return False
            return True

        self._predicates.append(predicate)
        return self

    def custom(self, predicate: Callable[[RecoveredFile], bool]) -> "SearchFilter":
        """Add custom filter predicate."""
        self._predicates.append(predicate)
        return self

    def apply(self, files: List[RecoveredFile]) -> List[RecoveredFile]:
        """Apply all filters to file list."""
        result = files
        for predicate in self._predicates:
            result = [f for f in result if predicate(f)]
        return result

    def __call__(self, files: List[RecoveredFile]) -> List[RecoveredFile]:
        """Support callable interface."""
        return self.apply(files)


class MessageFilter:
    """Composable message filter."""

    def __init__(self):
        """Initialize filter."""
        self._predicates: List[Callable[[SessionMessage], bool]] = []

    def by_type(self, message_type: MessageType) -> "MessageFilter":
        """Filter by message type."""

        def predicate(m: SessionMessage) -> bool:
            return m.type == message_type

        self._predicates.append(predicate)
        return self

    def by_session(self, session_id: str) -> "MessageFilter":
        """Filter by session."""

        def predicate(m: SessionMessage) -> bool:
            return m.session_id == session_id

        self._predicates.append(predicate)
        return self

    def by_content(self, pattern: str) -> "MessageFilter":
        """Filter by content pattern."""

        def predicate(m: SessionMessage) -> bool:
            return pattern.lower() in m.content.lower()

        self._predicates.append(predicate)
        return self

    def long_messages_only(self) -> "MessageFilter":
        """Filter to long messages only."""
        self._predicates.append(lambda m: m.is_long)
        return self

    def custom(self, predicate: Callable[[SessionMessage], bool]) -> "MessageFilter":
        """Add custom filter predicate."""
        self._predicates.append(predicate)
        return self

    def apply(self, messages: List[SessionMessage]) -> List[SessionMessage]:
        """Apply all filters to message list."""
        result = messages
        for predicate in self._predicates:
            result = [m for m in result if predicate(m)]
        return result

    def __call__(self, messages: List[SessionMessage]) -> List[SessionMessage]:
        """Support callable interface."""
        return self.apply(messages)


class LocationMatcher:
    """Location matching with pattern support."""

    def __init__(
        self,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ):
        """Initialize with patterns."""
        self.include_patterns = include_patterns or []
        self.exclude_patterns = exclude_patterns or []

    def matches(self, location: str) -> bool:
        """Check if location matches patterns."""
        location_lower = location.lower()

        # Check inclusions
        if self.include_patterns:
            if not any(inc.lower() in location_lower for inc in self.include_patterns):
                return False

        # Check exclusions
        if self.exclude_patterns:
            if any(exc.lower() in location_lower for exc in self.exclude_patterns):
                return False

        return True

    def filter_files(self, files: List[RecoveredFile]) -> List[RecoveredFile]:
        """Filter files by location."""
        return [f for f in files if self.matches(f.location.value)]

    def __call__(self, location: str) -> bool:
        """Support callable interface."""
        return self.matches(location)


class ChainedFilter:
    """Combine multiple filters with AND logic."""

    def __init__(self, *filters):
        """Initialize with filters."""
        self._filters = filters

    def apply(self, items: List[Any]) -> List[Any]:
        """Apply all filters in sequence."""
        result = items
        for f in self._filters:
            result = f(result) if callable(f) else f.apply(result)
        return result

    def __call__(self, items: List[Any]) -> List[Any]:
        """Support callable interface."""
        return self.apply(items)
