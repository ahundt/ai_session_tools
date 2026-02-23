"""
Type protocols for composable, extensible architecture.

Protocols allow dependency injection and multiple implementations.

Copyright (c) 2026 Andrew Hundt
Licensed under the Apache License, Version 2.0
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .models import FileVersion, FilterSpec, RecoveredFile


@runtime_checkable
class Searchable(Protocol):
    """Protocol for searchable storage backends."""

    def search(self, pattern: str, filters: Optional[FilterSpec] = None) -> List[RecoveredFile]:
        """Search for files matching pattern with optional filters."""
        ...


@runtime_checkable
class Extractable(Protocol):
    """Protocol for file extraction."""

    def extract_final(self, filename: str, output_dir: Path) -> Optional[Path]:
        """Extract final version of a file."""
        ...

    def extract_all(self, filename: str, output_dir: Path) -> List[Path]:
        """Extract all versions of a file."""
        ...


@runtime_checkable
class Filterable(Protocol):
    """Protocol for filtering operations."""

    def apply_filters(self, items: List[Any], filters: FilterSpec) -> List[Any]:
        """Apply filters to collection."""
        ...

    def matches(self, item: Any, filters: FilterSpec) -> bool:
        """Check if single item matches filters."""
        ...


@runtime_checkable
class Formatter(Protocol):
    """Protocol for output formatting."""

    def format(self, data: Any) -> str:
        """Format data for output."""
        ...

    def format_many(self, items: List[Any]) -> str:
        """Format multiple items."""
        ...


@runtime_checkable
class Reporter(Protocol):
    """Protocol for report generation."""

    def generate(self, data: Dict[str, Any]) -> str:
        """Generate a report from data."""
        ...


@runtime_checkable
class Storage(Protocol):
    """Protocol for storage backends."""

    def list_files(self) -> List[RecoveredFile]:
        """List all available files."""
        ...

    def get_versions(self, filename: str) -> List[FileVersion]:
        """Get all versions of a file."""
        ...

    def read_file(self, path: Path) -> str:
        """Read file content."""
        ...


class ComposableFilter:
    """Composable filter builder using fluent API."""

    def __init__(self):
        """Initialize filter."""
        self._filters: List[callable] = []

    def add(self, predicate: callable) -> "ComposableFilter":
        """Add a filter predicate."""
        self._filters.append(predicate)
        return self

    def apply(self, items: List[Any]) -> List[Any]:
        """Apply all filters to items."""
        result = items
        for filter_fn in self._filters:
            result = [item for item in result if filter_fn(item)]
        return result

    def __call__(self, items: List[Any]) -> List[Any]:
        """Support callable interface."""
        return self.apply(items)


class ComposableSearch:
    """Composable search builder with chaining."""

    def __init__(self, searcher: Searchable):
        """Initialize with searcher."""
        self._searcher = searcher
        self._pattern = "*"
        self._filters: Optional[FilterSpec] = None

    def pattern(self, pattern: str) -> "ComposableSearch":
        """Set search pattern."""
        self._pattern = pattern
        return self

    def with_filters(self, filters: FilterSpec) -> "ComposableSearch":
        """Add filters."""
        self._filters = filters
        return self

    def execute(self) -> List[RecoveredFile]:
        """Execute the search."""
        return self._searcher.search(self._pattern, self._filters)

    def __iter__(self):
        """Support iteration."""
        return iter(self.execute())

    def __len__(self):
        """Support len()."""
        return len(self.execute())
