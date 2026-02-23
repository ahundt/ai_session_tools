"""
Output formatters with multiple output types.

Copyright (c) 2026 Andrew Hundt
Licensed under the Apache License, Version 2.0
"""

import csv
import io
import json
from abc import ABC, abstractmethod
from typing import Any, List

from .models import RecoveredFile, SessionMessage

try:
    from rich.console import Console
    from rich.table import Table

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class ResultFormatter(ABC):
    """Base formatter protocol."""

    @abstractmethod
    def format(self, data: Any) -> str:
        """Format data for output."""
        pass

    @abstractmethod
    def format_many(self, items: List[Any]) -> str:
        """Format multiple items."""
        pass


class TableFormatter(ResultFormatter):
    """Format results as table using Rich."""

    def __init__(self, title: str = "Results"):
        """Initialize with title."""
        self.title = title
        if not HAS_RICH:
            raise ImportError("Rich library required for table formatting")

    def format(self, data: RecoveredFile) -> str:
        """Format single file."""
        lines = [
            f"Name:          {data.name}",
            f"Location:      {data.location}",
            f"Type:          {data.file_type}",
            f"Edits:         {data.edits}",
            f"Sessions:      {len(data.sessions)}",
            f"Last Modified: {data.last_modified or 'unknown'}",
            f"Created:       {data.created_date or 'unknown'}",
        ]
        return "\n".join(lines)

    def format_many(self, items: List[RecoveredFile]) -> str:
        """Format multiple files as table."""
        table = Table(title=self.title)
        table.add_column("File", style="cyan")
        table.add_column("Edits", justify="right", style="magenta")
        table.add_column("Type", style="green")
        table.add_column("Last Modified", style="blue")
        table.add_column("Location", style="yellow")
        table.add_column("Sessions", style="dim")

        for item in items:
            session_str = ", ".join(s[:8] + "\u2026" for s in item.sessions[:3])
            if len(item.sessions) > 3:
                session_str += f" (+{len(item.sessions) - 3})"
            table.add_row(item.name, str(item.edits), item.file_type, item.last_modified or "", item.location, session_str)

        console = Console()
        with console.capture() as capture:
            console.print(table)
        return capture.get()


class JsonFormatter(ResultFormatter):
    """Format results as JSON."""

    def _file_dict(self, data: RecoveredFile) -> dict:
        return {
            "name": data.name,
            "location": data.location,
            "type": data.file_type,
            "edits": data.edits,
            "sessions": data.sessions,
            "size_bytes": data.size_bytes,
            "last_modified": data.last_modified,
            "created_date": data.created_date,
        }

    def format(self, data: RecoveredFile) -> str:
        """Format single file."""
        return json.dumps(self._file_dict(data), indent=2)

    def format_many(self, items: List[RecoveredFile]) -> str:
        """Format multiple files as JSON array."""
        return json.dumps([self._file_dict(item) for item in items], indent=2)


_CSV_HEADER = ["name", "location", "type", "edits", "sessions", "size_bytes", "last_modified", "created_date"]


def _file_to_csv_row(data: RecoveredFile) -> list:
    return [
        data.name,
        data.location,
        data.file_type,
        data.edits,
        len(data.sessions),
        data.size_bytes,
        data.last_modified or "",
        data.created_date or "",
    ]


class CsvFormatter(ResultFormatter):
    """Format results as RFC 4180-compliant CSV (fields properly quoted)."""

    def format(self, data: RecoveredFile) -> str:
        """Format single file as CSV with header."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(_CSV_HEADER)
        writer.writerow(_file_to_csv_row(data))
        return buf.getvalue()

    def format_many(self, items: List[RecoveredFile]) -> str:
        """Format multiple files as CSV with header."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(_CSV_HEADER)
        for item in items:
            writer.writerow(_file_to_csv_row(item))
        return buf.getvalue()


class MessageFormatter(ResultFormatter):
    """Format session messages."""

    def __init__(self, max_chars: int = 0):
        """Initialize with max_chars. 0 = full content (no truncation)."""
        self.max_chars = max_chars

    def format(self, data: SessionMessage) -> str:
        """Format single message."""
        text = data.preview(self.max_chars) if self.max_chars else data.content.replace("\n", " ")
        return f"""Type:       {data.type.value}
Session:    {data.session_id}
Timestamp:  {data.timestamp}
Content:    {text}
Length:     {len(data.content)} chars"""

    def format_many(self, items: List[SessionMessage]) -> str:
        """Format multiple messages."""
        lines = []
        for msg in items:
            text = msg.preview(self.max_chars) if self.max_chars else msg.content.replace("\n", " ")
            lines.append(f"[{msg.type.value}] {msg.timestamp[:19]} - {text}")
        return "\n".join(lines)


class PlainFormatter(ResultFormatter):
    """Simple plain text formatter."""

    def format(self, data: Any) -> str:
        """Format single item."""
        if isinstance(data, RecoveredFile):
            return f"{data.name} ({data.edits} edits) - {data.location}"
        elif isinstance(data, SessionMessage):
            return f"[{data.type.value}] {data.preview()}"
        return str(data)

    def format_many(self, items: List[Any]) -> str:
        """Format multiple items."""
        return "\n".join(self.format(item) for item in items)


def get_formatter(format_type: str, title: str = "Results") -> ResultFormatter:
    """Factory function to get formatter by type."""
    formatters = {
        "table": TableFormatter,
        "json": JsonFormatter,
        "csv": CsvFormatter,
        "plain": PlainFormatter,
    }

    formatter_class = formatters.get(format_type.lower())
    if not formatter_class:
        raise ValueError(f"Unknown format: {format_type}")

    try:
        return formatter_class(title)
    except TypeError:
        return formatter_class()
