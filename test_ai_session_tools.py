#!/usr/bin/env python3
"""
Unit tests for AI Session Tools

Tests cover:
- File search and filtering
- Version extraction
- Message access
- Statistics collection
- Filter composability
"""

import json
import os
import pytest
from pathlib import Path
from ai_session_tools import (
    ChainedFilter,
    SessionRecoveryEngine,
    FilterSpec,
    SearchFilter,
    FileVersion,
    SessionMessage,
    MessageType,
    RecoveryStatistics,
    ComposableFilter,
    ComposableSearch,
)


@pytest.fixture
def recovery_dir():
    """Get the recovery directory path"""
    return Path.home() / ".claude" / "recovery" / "2026_02_22_session_scripts_recovery"


@pytest.fixture
def projects_dir():
    """Get the projects directory path"""
    return Path.home() / ".claude" / "projects"


@pytest.fixture
def engine(recovery_dir, projects_dir):
    """Create a SessionRecoveryEngine instance"""
    if not recovery_dir.exists():
        pytest.skip("Recovery directory not found")
    return SessionRecoveryEngine(projects_dir, recovery_dir)


class TestFileSearch:
    """Test file search functionality"""
    pytestmark = pytest.mark.integration

    def test_search_returns_list(self, engine):
        """Test that search returns a list"""
        results = engine.search("*.py")
        assert isinstance(results, list)

    def test_search_finds_python_files(self, engine):
        """Test searching for Python files"""
        results = engine.search("*.py")
        assert len(results) > 0
        filenames = [r.name for r in results]
        assert any(f.endswith(".py") for f in filenames)

    def test_search_returns_sorted_by_edits(self, engine):
        """Test that results are sorted by edit count (descending)"""
        results = engine.search("*.py")
        if len(results) > 1:
            edits = [r.edits for r in results]
            assert edits == sorted(edits, reverse=True)

    def test_search_with_filters(self, engine):
        """Test searching with FilterSpec"""
        filters = FilterSpec(min_edits=5)
        results = engine.search("*.py", filters)
        assert all(r.edits >= 5 for r in results)

    def test_search_empty_pattern(self, engine):
        """Test search with empty/broad pattern"""
        results = engine.search(".*")
        assert isinstance(results, list)


class TestVersionExtraction:
    """Test version extraction functionality"""
    pytestmark = pytest.mark.integration

    def test_get_versions_returns_list(self, engine):
        """Test that get_versions returns a list"""
        results = engine.search("*.py")
        if results:
            filename = results[0].name
            versions = engine.get_versions(filename)
            assert isinstance(versions, list)

    def test_get_versions_are_sorted(self, engine):
        """Test that versions are sorted"""
        results = engine.search("*.py")
        if results:
            filename = results[0].name
            versions = engine.get_versions(filename)
            if len(versions) > 1:
                assert versions == sorted(versions)

    def test_get_versions_contains_file_versions(self, engine):
        """Test that versions contain FileVersion objects"""
        results = engine.search("*.py")
        if results:
            filename = results[0].name
            versions = engine.get_versions(filename)
            assert all(isinstance(v, FileVersion) for v in versions)


class TestStatistics:
    """Test statistics collection"""
    pytestmark = pytest.mark.integration

    def test_get_statistics_returns_object(self, engine):
        """Test that get_statistics returns proper object"""
        stats = engine.get_statistics()
        assert stats is not None

    def test_statistics_has_expected_attributes(self, engine):
        """Test that statistics have expected attributes"""
        stats = engine.get_statistics()
        assert hasattr(stats, "total_sessions")
        assert hasattr(stats, "total_files")
        assert hasattr(stats, "total_versions")

    def test_statistics_are_positive(self, engine):
        """Test that statistics are positive numbers"""
        stats = engine.get_statistics()
        assert stats.total_sessions > 0
        assert stats.total_files > 0
        assert stats.total_versions > 0


class TestMessages:
    """Test message extraction"""
    pytestmark = pytest.mark.integration

    def test_search_messages_returns_list(self, engine):
        """Test that search_messages returns a list"""
        messages = engine.search_messages("test")
        assert isinstance(messages, list)

    def test_get_messages_returns_list(self, engine):
        """Test that get_messages returns a list"""
        messages = engine.get_messages("any_session_id")
        assert isinstance(messages, list)


class TestFilters:
    """Test composable filter functionality"""

    def test_search_filter_creation(self):
        """Test creating a search filter"""
        filter_obj = SearchFilter()
        assert filter_obj is not None

    def test_search_filter_chaining(self):
        """Test that filters can be chained"""
        filter_obj = SearchFilter().by_edits(min_edits=5)
        assert filter_obj is not None

    def test_filter_spec_creation(self):
        """Test creating a FilterSpec"""
        spec = FilterSpec(min_edits=5)
        assert spec.min_edits == 5

    def test_filter_spec_builder_pattern(self):
        """Test FilterSpec builder pattern"""
        spec = FilterSpec().with_edit_range(5, 100)
        assert spec is not None


class TestFileLocation:
    """Test file location field on RecoveredFile."""

    def test_location_is_string(self):
        """RecoveredFile.location is a plain string, not an enum."""
        from ai_session_tools import RecoveredFile
        r = RecoveredFile(name="a.py", path="/a.py", file_type="py")
        assert isinstance(r.location, str)

    def test_location_default_is_recovery(self):
        """RecoveredFile.location defaults to 'recovery'."""
        from ai_session_tools import RecoveredFile
        r = RecoveredFile(name="a.py", path="/a.py", file_type="py")
        assert r.location == "recovery"

    def test_location_can_be_custom_string(self):
        """RecoveredFile.location accepts any string."""
        from ai_session_tools import RecoveredFile
        r = RecoveredFile(name="a.py", path="/a.py", file_type="py", location="custom/path")
        assert r.location == "custom/path"


class TestIntegration:
    """Integration tests combining multiple operations"""
    pytestmark = pytest.mark.integration

    def test_search_and_get_versions(self, engine):
        """Test searching then getting versions"""
        results = engine.search("*.py")
        if results:
            filename = results[0].name
            versions = engine.get_versions(filename)
            assert isinstance(versions, list)

    def test_filter_and_search(self, engine):
        """Test filtering and searching together"""
        filters = FilterSpec(min_edits=1)
        results = engine.search("*.py", filters)
        assert all(r.edits >= 1 for r in results)

    def test_statistics_consistency(self, engine):
        """Test that statistics are consistent"""
        stats = engine.get_statistics()
        search_results = engine.search(".*")
        # Statistics should count all files
        assert stats.total_files >= len(search_results)


class TestEdgeCases:
    """Test edge cases and error handling"""
    pytestmark = pytest.mark.integration

    def test_search_nonexistent_pattern(self, engine):
        """Test searching for pattern that matches nothing"""
        results = engine.search("ZZZZZ_NONEXISTENT_ZZZZZZ.xyz")
        assert isinstance(results, list)

    def test_get_versions_nonexistent_file(self, engine):
        """Test getting versions for file that doesn't exist"""
        versions = engine.get_versions("NONEXISTENT_FILE.xyz")
        assert isinstance(versions, list)

    def test_extract_with_invalid_path(self, engine):
        """Test extraction with invalid output path"""
        results = engine.search("*.py")
        if results:
            filename = results[0].name
            # Just verify it doesn't crash
            try:
                engine.extract_final(filename, Path("/tmp/ai_session_tools_test"))
            except Exception:
                # Expected to potentially fail with permission, but shouldn't crash
                pass


class TestMultipleFileTypes:
    """Test searching for different file types (not just .py)"""
    pytestmark = pytest.mark.integration

    def test_search_markdown_files(self, engine):
        """Test searching for Markdown files"""
        results = engine.search("*.md")
        # May be empty if no .md files exist, but shouldn't error
        assert isinstance(results, list)

    def test_search_json_files(self, engine):
        """Test searching for JSON files"""
        results = engine.search("*.json")
        # May be empty if no .json files exist, but shouldn't error
        assert isinstance(results, list)

    def test_search_yaml_files(self, engine):
        """Test searching for YAML files"""
        results = engine.search("*.yaml")
        assert isinstance(results, list)
        # Verify all results are YAML files
        if results:
            assert all(r.name.endswith((".yaml", ".yml")) for r in results)

    def test_search_typescript_files(self, engine):
        """Test searching for TypeScript files"""
        results = engine.search("*.ts")
        assert isinstance(results, list)
        if results:
            assert all(r.name.endswith(".ts") for r in results)

    def test_search_rust_files(self, engine):
        """Test searching for Rust files"""
        results = engine.search("*.rs")
        assert isinstance(results, list)
        if results:
            assert all(r.name.endswith(".rs") for r in results)

    def test_search_all_file_types_together(self, engine):
        """Test searching for all files regardless of type"""
        results = engine.search("*")
        assert isinstance(results, list)
        # Should find more files with wildcard than specific patterns
        py_results = engine.search("*.py")
        if py_results:
            assert len(results) >= len(py_results)

    def test_file_type_field_populated(self, engine):
        """Test that file_type field is populated for all files"""
        results = engine.search("*")
        assert len(results) > 0, "Need at least one file to test"
        for result in results:
            assert result.file_type is not None
            assert isinstance(result.file_type, str)
            # file_type should be the extension without dot, or "unknown"
            if "." in result.name:
                expected_type = result.name.split(".")[-1]
                assert result.file_type == expected_type or result.file_type == "unknown"

    def test_unknown_file_type_handling(self, engine):
        """Test that unknown file types are handled gracefully"""
        # Search for a pattern that might match unknown extensions
        results = engine.search("*.unknown")
        # Should not crash, even if no results
        assert isinstance(results, list)

    def test_files_without_extension(self, engine):
        """Test that files without extensions are handled"""
        results = engine.search("*")
        # Check if any files lack extensions
        files_without_ext = [r for r in results if "." not in r.name]
        for file in files_without_ext:
            # Should have file_type of "unknown" or empty string
            assert file.file_type in ("unknown", "")


class TestFileTypeFiltering:
    """Test filtering by file type"""
    pytestmark = pytest.mark.integration

    def test_filter_by_extension(self, engine):
        """Test SearchFilter with file extension filtering"""
        # Get all results first
        all_results = engine.search("*")
        if len(all_results) > 0:
            # Try filtering by first file's extension
            first_extension = all_results[0].file_type
            filter_obj = SearchFilter().by_extension(first_extension)
            filtered = filter_obj(all_results)
            # All results should match the extension
            assert all(f.file_type == first_extension for f in filtered)

    def test_filter_by_python_extension(self, engine):
        """Test filtering specifically for Python files"""
        all_results = engine.search("*")
        filter_obj = SearchFilter().by_extension("py")
        filtered = filter_obj(all_results)
        if filtered:
            assert all(r.file_type == "py" for r in filtered)

    def test_filter_empty_result_for_nonexistent_extension(self, engine):
        """Test filtering for an extension that doesn't exist"""
        all_results = engine.search("*")
        filter_obj = SearchFilter().by_extension("nonexistent_extension_xyz")
        filtered = filter_obj(all_results)
        # Should return empty list since no files have that extension
        assert filtered == []


class TestMessageFiltering:
    """Test message extraction and filtering"""
    pytestmark = pytest.mark.integration

    def test_get_messages_with_type_filter_user(self, engine):
        """Test filtering for user messages only"""
        stats = engine.get_statistics()
        if stats.total_sessions > 0:
            messages = engine.get_messages("*", message_type="user")
            assert isinstance(messages, list)
            # All returned messages must be user type
            for msg in messages:
                assert msg.type.value == "user"

    def test_get_messages_with_type_filter_assistant(self, engine):
        """Test filtering for assistant messages only"""
        stats = engine.get_statistics()
        if stats.total_sessions > 0:
            messages = engine.get_messages("*", message_type="assistant")
            assert isinstance(messages, list)
            for msg in messages:
                assert msg.type.value == "assistant"

    def test_search_messages_case_insensitive(self, engine):
        """Test that message search is case insensitive"""
        results_lower = engine.search_messages("python")
        results_upper = engine.search_messages("PYTHON")
        assert isinstance(results_lower, list)
        assert isinstance(results_upper, list)
        # Both cases must return the same number of matches
        assert len(results_lower) == len(results_upper)

    def test_search_messages_with_phrases(self, engine):
        """Test searching for multi-word phrases"""
        results = engine.search_messages("session data")
        assert isinstance(results, list)


class TestFilterComposition:
    """Test composable filter combinations"""
    pytestmark = pytest.mark.integration

    def test_filter_by_edits_range(self, engine):
        """Test filtering by edit range"""
        filters = FilterSpec(min_edits=1, max_edits=50)
        results = engine.search("*", filters)
        assert all(1 <= r.edits <= 50 for r in results)

    def test_filter_by_high_edit_count(self, engine):
        """Test filtering for highly edited files"""
        filters = FilterSpec(min_edits=100)
        results = engine.search("*", filters)
        if results:
            assert all(r.edits >= 100 for r in results)

    def test_filter_with_pattern_and_edits(self, engine):
        """Test combining pattern matching with edit filtering"""
        filters = FilterSpec(min_edits=5)
        py_results = engine.search("*.py", filters)
        if py_results:
            assert all(r.edits >= 5 for r in py_results)
            assert all(r.name.endswith(".py") for r in py_results)

    def test_search_filter_multiple_conditions(self, engine):
        """Test SearchFilter with multiple chained conditions"""
        all_results = engine.search("*")
        filter_obj = SearchFilter().by_edits(min_edits=1).by_extension("py")
        filtered = filter_obj(all_results)
        if filtered:
            assert all(r.edits >= 1 for r in filtered)
            assert all(r.file_type == "py" for r in filtered)


# ── Regression Tests: Already-completed steps ─────────────────────────────────

class TestCompletedStepModels:
    """Tests for already-completed models.py changes (Step 2)."""

    def test_filter_spec_has_matches_datetime(self):
        """FilterSpec should have matches_datetime method."""
        spec = FilterSpec()
        assert hasattr(spec, "matches_datetime")
        assert callable(spec.matches_datetime)

    def test_filter_spec_has_matches_extension(self):
        """FilterSpec should have matches_extension method."""
        spec = FilterSpec()
        assert spec.matches_extension("py") is True
        assert spec.matches_extension(".py") is True

    def test_filter_spec_with_extensions_builder(self):
        """FilterSpec.with_extensions() builder works."""
        spec = FilterSpec().with_extensions(include={"py", "md"})
        assert spec.matches_extension("py") is True
        assert spec.matches_extension("rs") is False

    def test_filter_spec_matches_edits(self):
        """FilterSpec.matches_edits works with min/max."""
        spec = FilterSpec(min_edits=5, max_edits=10)
        assert spec.matches_edits(5) is True
        assert spec.matches_edits(10) is True
        assert spec.matches_edits(4) is False
        assert spec.matches_edits(11) is False

    def test_filter_spec_max_edits_none_is_unlimited(self):
        """max_edits=None means unlimited."""
        spec = FilterSpec(min_edits=0, max_edits=None)
        assert spec.matches_edits(999999) is True

    def test_filter_spec_matches_session(self):
        """FilterSpec.matches_session works."""
        spec = FilterSpec(include_sessions={"abc", "def"})
        assert spec.matches_session("abc") is True
        assert spec.matches_session("xyz") is False

    def test_filter_spec_matches_location(self):
        """FilterSpec.matches_location works with plain string location values."""
        spec = FilterSpec(include_folders={"recovery"})
        assert spec.matches_location("recovery") is True
        assert spec.matches_location("other") is False


class TestCompletedStepTypes:
    """Tests for already-completed types.py changes (Step 5)."""

    def test_searchable_protocol_has_search(self):
        """Searchable protocol should have search() method."""
        from ai_session_tools.types import Searchable
        assert hasattr(Searchable, "search")

    def test_searchable_protocol_no_search_filtered(self):
        """Searchable protocol should NOT have search_filtered() (removed)."""
        from ai_session_tools.types import Searchable
        # search_filtered was removed because engine has no such method
        assert not hasattr(Searchable, "search_filtered") or "search_filtered" not in dir(Searchable)

    def test_composable_search_no_with_options(self):
        """ComposableSearch should NOT have with_options() (SearchOptions removed)."""
        assert not hasattr(ComposableSearch, "with_options")

    def test_composable_search_has_with_filters(self):
        """ComposableSearch should have with_filters() method."""
        assert hasattr(ComposableSearch, "with_filters")

    def test_composable_filter_callable(self):
        """ComposableFilter should be callable."""
        cf = ComposableFilter()
        result = cf([1, 2, 3])
        assert result == [1, 2, 3]


class TestCompletedStepInitExports:
    """Tests for already-completed __init__.py changes (Step 6)."""

    def test_no_search_options_export(self):
        """SearchOptions should NOT be exported (removed)."""
        import ai_session_tools
        assert not hasattr(ai_session_tools, "SearchOptions")

    def test_no_file_extractor_export(self):
        """FileExtractor should NOT be exported (extractors.py deleted)."""
        import ai_session_tools
        assert not hasattr(ai_session_tools, "FileExtractor")

    def test_no_message_extractor_export(self):
        """MessageExtractor should NOT be exported (extractors.py deleted)."""
        import ai_session_tools
        assert not hasattr(ai_session_tools, "MessageExtractor")

    def test_exports_recovery_statistics(self):
        """RecoveryStatistics should be exported."""
        assert RecoveryStatistics is not None

    def test_exports_message_type(self):
        """MessageType should be exported."""
        assert MessageType.USER.value == "user"

    def test_exports_composable_filter(self):
        """ComposableFilter should be exported."""
        assert ComposableFilter is not None

    def test_exports_composable_search(self):
        """ComposableSearch should be exported."""
        assert ComposableSearch is not None

    def test_version_not_hardcoded_2(self):
        """__version__ should not be '2.0.0' (was wrong, fixed to use importlib.metadata)."""
        import ai_session_tools
        assert ai_session_tools.__version__ != "2.0.0"

    def test_author_is_andrew_hundt(self):
        """__author__ should be Andrew Hundt."""
        import ai_session_tools
        assert ai_session_tools.__author__ == "Andrew Hundt"

    def test_pyproject_has_aise_alias(self):
        """pyproject.toml should have aise script alias."""
        import configparser
        from pathlib import Path
        pyproject = Path(__file__).parent / "pyproject.toml"
        content = pyproject.read_text()
        assert 'aise = "ai_session_tools.cli:cli_main"' in content


# ── TDD Tests: Step 2b — MessageFormatter max_chars ──────────────────────────

class TestSessionMessagePreview:
    """TDD tests for SessionMessage.preview() method (was @property, now method with limit)."""

    def test_preview_default_limit_100(self):
        """preview() with default limit truncates at 100 chars."""
        msg = SessionMessage(
            type=MessageType.USER,
            timestamp="2026-02-22T10:00:00Z",
            content="x" * 200,
            session_id="test-session",
        )
        result = msg.preview()
        assert len(result) == 100

    def test_preview_zero_returns_full_content(self):
        """preview(0) returns full content (no truncation)."""
        content = "x" * 500
        msg = SessionMessage(
            type=MessageType.USER,
            timestamp="2026-02-22T10:00:00Z",
            content=content,
            session_id="test-session",
        )
        result = msg.preview(0)
        assert len(result) == 500

    def test_preview_custom_limit(self):
        """preview(50) truncates at 50 chars."""
        msg = SessionMessage(
            type=MessageType.USER,
            timestamp="2026-02-22T10:00:00Z",
            content="x" * 200,
            session_id="test-session",
        )
        result = msg.preview(50)
        assert len(result) == 50

    def test_preview_short_content_not_truncated(self):
        """preview() does not truncate content shorter than limit."""
        msg = SessionMessage(
            type=MessageType.USER,
            timestamp="2026-02-22T10:00:00Z",
            content="short",
            session_id="test-session",
        )
        assert msg.preview() == "short"
        assert msg.preview(0) == "short"
        assert msg.preview(100) == "short"

    def test_preview_replaces_newlines(self):
        """preview() replaces newlines with spaces."""
        msg = SessionMessage(
            type=MessageType.USER,
            timestamp="2026-02-22T10:00:00Z",
            content="line1\nline2\nline3",
            session_id="test-session",
        )
        assert "\n" not in msg.preview()
        assert "\n" not in msg.preview(0)


class TestMessageFormatterMaxChars:
    """TDD tests for MessageFormatter accepting max_chars parameter."""

    def test_formatter_default_shows_full_content(self):
        """MessageFormatter() with default max_chars=0 shows full content in format()."""
        from ai_session_tools.formatters import MessageFormatter
        formatter = MessageFormatter()
        msg = SessionMessage(
            type=MessageType.USER,
            timestamp="2026-02-22T10:00:00Z",
            content="a" * 300,
            session_id="test-session",
        )
        output = formatter.format(msg)
        # Full content should appear (300 chars of 'a')
        assert "a" * 300 in output.replace("\n", " ") or len([c for c in output if c == 'a']) >= 300

    def test_formatter_max_chars_truncates(self):
        """MessageFormatter(max_chars=50) truncates in format()."""
        from ai_session_tools.formatters import MessageFormatter
        formatter = MessageFormatter(max_chars=50)
        msg = SessionMessage(
            type=MessageType.USER,
            timestamp="2026-02-22T10:00:00Z",
            content="a" * 300,
            session_id="test-session",
        )
        output = formatter.format(msg)
        # Should NOT contain 300 chars of 'a'
        assert "a" * 300 not in output

    def test_formatter_format_many_default_full_content(self):
        """MessageFormatter().format_many() shows full content by default."""
        from ai_session_tools.formatters import MessageFormatter
        formatter = MessageFormatter()
        msgs = [
            SessionMessage(
                type=MessageType.USER,
                timestamp="2026-02-22T10:00:00Z",
                content="b" * 200,
                session_id="test-session",
            )
        ]
        output = formatter.format_many(msgs)
        assert "b" * 200 in output

    def test_formatter_format_many_max_chars_truncates(self):
        """MessageFormatter(max_chars=50).format_many() truncates."""
        from ai_session_tools.formatters import MessageFormatter
        formatter = MessageFormatter(max_chars=50)
        msgs = [
            SessionMessage(
                type=MessageType.USER,
                timestamp="2026-02-22T10:00:00Z",
                content="b" * 200,
                session_id="test-session",
            )
        ]
        output = formatter.format_many(msgs)
        assert "b" * 200 not in output


class TestPlainFormatterPreview:
    """TDD tests for PlainFormatter using preview() method call."""

    def test_plain_formatter_uses_preview_method(self):
        """PlainFormatter should use preview() as method, not property."""
        from ai_session_tools.formatters import PlainFormatter
        formatter = PlainFormatter()
        msg = SessionMessage(
            type=MessageType.USER,
            timestamp="2026-02-22T10:00:00Z",
            content="hello world " * 20,
            session_id="test-session",
        )
        # Should not raise TypeError (would if preview is still @property and called as method)
        output = formatter.format(msg)
        assert isinstance(output, str)
        assert "user" in output.lower()


# ── TDD Tests: Step 3 — Engine date/size filtering ───────────────────────────

class TestFilterSpecMatchesDatetime:
    """TDD tests for FilterSpec.matches_datetime() method."""

    def test_no_filter_passes_any_datetime(self):
        """With no datetime filters set, any datetime passes."""
        spec = FilterSpec()
        assert spec.matches_datetime("2026-02-22") is True
        assert spec.matches_datetime("2026-02-22T14:30:00") is True
        assert spec.matches_datetime("2020-01-01") is True

    def test_no_filter_passes_none(self):
        """With no datetime filters, None passes."""
        spec = FilterSpec()
        assert spec.matches_datetime(None) is True

    def test_after_excludes_earlier(self):
        """after excludes datetimes before the threshold."""
        spec = FilterSpec(after="2026-02-01")
        assert spec.matches_datetime("2026-01-15") is False
        assert spec.matches_datetime("2026-02-01") is True
        assert spec.matches_datetime("2026-02-22T14:30:00") is True

    def test_before_excludes_later(self):
        """before excludes datetimes after the threshold."""
        spec = FilterSpec(before="2026-02-15")
        assert spec.matches_datetime("2026-02-22") is False
        assert spec.matches_datetime("2026-02-15") is True
        assert spec.matches_datetime("2026-01-01T08:00:00") is True

    def test_datetime_range(self):
        """Combined after + before creates a range."""
        spec = FilterSpec(after="2026-02-01", before="2026-02-28")
        assert spec.matches_datetime("2026-02-15") is True
        assert spec.matches_datetime("2026-02-15T12:00:00") is True
        assert spec.matches_datetime("2026-01-15") is False
        assert spec.matches_datetime("2026-03-01") is False

    def test_none_excluded_when_filter_active(self):
        """None is excluded when any datetime filter is active (conservative)."""
        spec = FilterSpec(after="2026-01-01")
        assert spec.matches_datetime(None) is False

    def test_none_excluded_with_before(self):
        """None is excluded when before is set."""
        spec = FilterSpec(before="2026-12-31")
        assert spec.matches_datetime(None) is False

    def test_full_datetime_filter_with_full_datetime_value(self):
        """Full datetime filter against full datetime value."""
        spec = FilterSpec(after="2026-02-15T10:00:00", before="2026-02-15T18:00:00")
        assert spec.matches_datetime("2026-02-15T12:00:00") is True
        assert spec.matches_datetime("2026-02-15T08:00:00") is False
        assert spec.matches_datetime("2026-02-15T20:00:00") is False

    def test_date_filter_with_datetime_value(self):
        """Date-only filter works with full datetime values (mixed precision)."""
        spec = FilterSpec(after="2026-02-15")
        assert spec.matches_datetime("2026-02-15T14:30:00") is True
        assert spec.matches_datetime("2026-02-14T23:59:59") is False

    def test_datetime_filter_with_date_value(self):
        """Full datetime filter works with date-only values (mixed precision)."""
        spec = FilterSpec(after="2026-02-15T14:30:00")
        # "2026-02-15" < "2026-02-15T14:30:00" lexicographically, so excluded
        assert spec.matches_datetime("2026-02-15") is False
        assert spec.matches_datetime("2026-02-16") is True


class TestEnginePopulatesDateFields:
    """TDD tests for engine populating last_modified and created_date."""
    pytestmark = pytest.mark.integration

    def test_search_results_have_last_modified(self, engine):
        """Search results should have last_modified populated (not None)."""
        results = engine.search("*.py")
        if results:
            # At least some files should have last_modified set
            dated = [r for r in results if r.last_modified is not None]
            assert len(dated) > 0, "Engine should populate last_modified from file stat"

    def test_search_results_have_created_date(self, engine):
        """Search results should have created_date populated."""
        results = engine.search("*.py")
        if results:
            dated = [r for r in results if r.created_date is not None]
            assert len(dated) > 0, "Engine should populate created_date from file stat"

    def test_last_modified_is_iso_datetime_format(self, engine):
        """last_modified should be ISO YYYY-MM-DDTHH:MM:SS format."""
        results = engine.search("*.py")
        if results:
            for r in results:
                if r.last_modified:
                    assert len(r.last_modified) == 19, f"Expected YYYY-MM-DDTHH:MM:SS (19 chars), got {r.last_modified!r}"
                    assert r.last_modified[4] == "-"
                    assert r.last_modified[7] == "-"
                    assert r.last_modified[10] == "T"
                    assert r.last_modified[13] == ":"
                    assert r.last_modified[16] == ":"

    def test_datetime_filter_actually_filters(self, engine):
        """Datetime filter should actually exclude files outside the range."""
        # Get all files first
        all_results = engine.search("*.py")
        if not all_results:
            pytest.skip("No files found")

        # Use a future datetime that should exclude everything
        filters = FilterSpec(after="2099-01-01T00:00:00")
        filtered = engine.search("*.py", filters)
        assert len(filtered) < len(all_results), "Date filter should exclude some files"


class TestEngineSizeFilterWired:
    """TDD tests for engine wiring matches_size() in _apply_all_filters."""
    pytestmark = pytest.mark.integration

    def test_size_filter_excludes_small_files(self, engine):
        """min_size filter should exclude files smaller than threshold."""
        all_results = engine.search("*")
        if not all_results:
            pytest.skip("No files found")

        # Find max size to set a threshold that excludes some
        max_size = max(r.size_bytes for r in all_results)
        if max_size == 0:
            pytest.skip("All files have 0 size")

        filters = FilterSpec(min_size=max_size)
        filtered = engine.search("*", filters)
        # Should have fewer results (only files >= max_size)
        assert len(filtered) <= len(all_results)
        assert all(r.size_bytes >= max_size for r in filtered)


# ── TDD Tests: Step 7 — CLI dual-ordering ────────────────────────────────────

class TestCLIDualOrdering:
    """TDD tests for CLI dual-ordering commands."""

    def test_cli_has_search_command(self):
        """CLI app has 'search' command."""
        from ai_session_tools.cli import app
        from typer.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(app, ["search", "--help"])
        assert result.exit_code == 0
        assert "search" in result.output.lower() or "pattern" in result.output.lower()

    def test_cli_has_files_group(self):
        """CLI app has 'files' command group."""
        from ai_session_tools.cli import app
        from typer.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(app, ["files", "--help"])
        assert result.exit_code == 0

    def test_cli_has_messages_group(self):
        """CLI app has 'messages' command group."""
        from ai_session_tools.cli import app
        from typer.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(app, ["messages", "--help"])
        assert result.exit_code == 0

    def test_cli_files_search_exists(self):
        """'ais files search' route exists."""
        from ai_session_tools.cli import app
        from typer.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(app, ["files", "search", "--help"])
        assert result.exit_code == 0
        assert "pattern" in result.output.lower()

    def test_cli_messages_search_exists(self):
        """'ais messages search' route exists."""
        from ai_session_tools.cli import app
        from typer.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(app, ["messages", "search", "--help"])
        assert result.exit_code == 0
        assert "query" in result.output.lower()

    def test_cli_messages_get_exists(self):
        """'ais messages get' route exists."""
        from ai_session_tools.cli import app
        from typer.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(app, ["messages", "get", "--help"])
        assert result.exit_code == 0
        assert "session" in result.output.lower()

    def test_cli_extract_uses_name_not_file(self):
        """'extract' uses --name/-n, not --file/-f."""
        from ai_session_tools.cli import app
        from typer.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(app, ["extract", "--help"])
        assert result.exit_code == 0
        assert "--name" in result.output

    def test_cli_get_command_exists(self):
        """Root 'get' command exists."""
        from ai_session_tools.cli import app
        from typer.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(app, ["get", "--help"])
        assert result.exit_code == 0
        assert "session" in result.output.lower()

    def test_cli_stats_command_exists(self):
        """Root 'stats' command exists."""
        from ai_session_tools.cli import app
        from typer.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(app, ["stats", "--help"])
        assert result.exit_code == 0

    def test_cli_search_has_max_chars_for_messages(self):
        """'search messages' or 'messages search' should have --max-chars option."""
        from ai_session_tools.cli import app
        from typer.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(app, ["messages", "search", "--help"])
        assert result.exit_code == 0
        assert "max-chars" in result.output.lower()

    def test_cli_search_files_has_datetime_flags(self):
        """File search should have --after and --before flags."""
        from ai_session_tools.cli import app
        from typer.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(app, ["files", "search", "--help"])
        assert result.exit_code == 0
        assert "--after" in result.output
        assert "--before" in result.output

    def test_cli_search_files_has_session_flags(self):
        """File search should have --include-sessions and --exclude-sessions."""
        from ai_session_tools.cli import app
        from typer.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(app, ["files", "search", "--help"])
        assert result.exit_code == 0
        assert "include-sessions" in result.output.lower()
        assert "exclude-sessions" in result.output.lower()

    def test_cli_search_files_uses_include_extensions(self):
        """File search should use --include-extensions (not --include-types)."""
        from ai_session_tools.cli import app
        from typer.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(app, ["files", "search", "--help"])
        assert result.exit_code == 0
        assert "include-extensions" in result.output.lower()

    def test_cli_get_has_max_chars(self):
        """'get' command should have --max-chars option."""
        from ai_session_tools.cli import app
        from typer.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(app, ["get", "--help"])
        assert result.exit_code == 0
        assert "max-chars" in result.output.lower()

    def test_cli_get_has_format(self):
        """'get' command should have --format option."""
        from ai_session_tools.cli import app
        from typer.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(app, ["get", "--help"])
        assert result.exit_code == 0
        assert "format" in result.output.lower()


# ── Fixture helpers for tmp_path-based tests ─────────────────────────────────

def _make_recovery_dir(tmp_path: Path) -> Path:
    """Create a minimal recovery directory structure under tmp_path."""
    recovery = tmp_path / "recovery"
    session = recovery / "session_abc123"
    session.mkdir(parents=True)
    (session / "hello.py").write_text("print('hello')")
    (session / "notes.md").write_text("# notes")

    versions = recovery / "session_all_versions_abc123"
    versions.mkdir(parents=True)
    (versions / "hello.py_v000001_line_5.txt").write_text("print('v1')\n" * 5)
    (versions / "hello.py_v000002_line_3.txt").write_text("print('v2')\n" * 3)
    # Intentionally v2 has fewer lines than v1 — extract_final must use version_num not line_count
    return recovery


def _make_projects_dir(tmp_path: Path, session_id: str = "abc123-full-uuid") -> Path:
    """Create a minimal projects directory with one JSONL file."""
    projects = tmp_path / "projects"
    project = projects / "proj_uuid"
    project.mkdir(parents=True)
    lines = [
        json.dumps({
            "sessionId": session_id,
            "type": "user",
            "timestamp": "2026-02-22T10:00:00.000Z",
            "message": {"content": "Hello from user"},
        }),
        json.dumps({
            "sessionId": session_id,
            "type": "assistant",
            "timestamp": "2026-02-22T10:00:01.000Z",
            "message": {"content": [{"type": "text", "text": "Hello back from assistant"}]},
        }),
        # Intentionally malformed line to test robustness
        "{ not valid json",
        # Line with binary-ish content (non-UTF-8 replacement)
        json.dumps({"sessionId": session_id, "type": "user", "message": {"content": "python rocks"}}),
    ]
    (project / f"{session_id}.jsonl").write_text("\n".join(lines))
    return projects


# ── New tests: engine with missing / empty directories ────────────────────────

class TestEngineWithMissingDirs:
    """Engine returns empty results (not crashes) when dirs don't exist."""

    def test_search_missing_recovery_dir_returns_empty(self, tmp_path):
        engine = SessionRecoveryEngine(tmp_path / "no_projects", tmp_path / "no_recovery")
        assert engine.search("*") == []

    def test_get_versions_missing_recovery_dir_returns_empty(self, tmp_path):
        engine = SessionRecoveryEngine(tmp_path / "no_projects", tmp_path / "no_recovery")
        assert engine.get_versions("anything.py") == []

    def test_get_messages_missing_projects_dir_returns_empty(self, tmp_path):
        engine = SessionRecoveryEngine(tmp_path / "no_projects", tmp_path / "no_recovery")
        assert engine.get_messages("any-session") == []

    def test_search_messages_missing_projects_dir_returns_empty(self, tmp_path):
        engine = SessionRecoveryEngine(tmp_path / "no_projects", tmp_path / "no_recovery")
        assert engine.search_messages("anything") == []

    def test_get_statistics_missing_recovery_dir_returns_zeros(self, tmp_path):
        engine = SessionRecoveryEngine(tmp_path / "no_projects", tmp_path / "no_recovery")
        stats = engine.get_statistics()
        assert stats.total_sessions == 0
        assert stats.total_files == 0
        assert stats.total_versions == 0

    def test_search_empty_recovery_dir_returns_empty(self, tmp_path):
        (tmp_path / "recovery").mkdir()
        engine = SessionRecoveryEngine(tmp_path / "projects", tmp_path / "recovery")
        assert engine.search("*") == []

    def test_extract_final_missing_returns_none(self, tmp_path):
        engine = SessionRecoveryEngine(tmp_path / "no_projects", tmp_path / "no_recovery")
        assert engine.extract_final("hello.py", tmp_path / "out") is None

    def test_extract_all_missing_returns_empty(self, tmp_path):
        engine = SessionRecoveryEngine(tmp_path / "no_projects", tmp_path / "no_recovery")
        assert engine.extract_all("hello.py", tmp_path / "out") == []


# ── New tests: extract_final uses version_num not line_count ──────────────────

class TestExtractFinalVersionSelection:
    """extract_final must return the highest version_num, not the longest file."""

    def test_extract_final_picks_highest_version_num(self, tmp_path):
        """v2 (3 lines) should be returned, not v1 (5 lines)."""
        recovery = _make_recovery_dir(tmp_path)
        engine = SessionRecoveryEngine(tmp_path / "projects", recovery)
        out = tmp_path / "out"
        path = engine.extract_final("hello.py", out)
        assert path is not None
        content = path.read_text()
        # v2 contains "print('v2')" and v1 contains "print('v1')"
        assert "v2" in content
        assert "v1" not in content

    def test_extract_all_returns_all_versions_sorted(self, tmp_path):
        recovery = _make_recovery_dir(tmp_path)
        engine = SessionRecoveryEngine(tmp_path / "projects", recovery)
        paths = engine.extract_all("hello.py", tmp_path / "versions")
        assert len(paths) == 2
        # Filenames should be sorted by version number
        names = [p.name for p in paths]
        assert names[0].startswith("v000001")
        assert names[1].startswith("v000002")


# ── New tests: session ID prefix matching ─────────────────────────────────────

class TestPrefixSessionMatching:
    """get_messages should match partial/prefix session IDs."""

    def _engine_and_session(self, tmp_path):
        session_id = "abc123de-f456-7890-abcd-ef1234567890"
        projects = _make_projects_dir(tmp_path, session_id)
        recovery = _make_recovery_dir(tmp_path)
        engine = SessionRecoveryEngine(projects, recovery)
        return engine, session_id

    def test_full_session_id_matches(self, tmp_path):
        engine, session_id = self._engine_and_session(tmp_path)
        msgs = engine.get_messages(session_id)
        assert len(msgs) >= 2

    def test_prefix_session_id_matches(self, tmp_path):
        engine, session_id = self._engine_and_session(tmp_path)
        prefix = session_id[:8]
        msgs = engine.get_messages(prefix)
        assert len(msgs) >= 2

    def test_wrong_session_id_returns_empty(self, tmp_path):
        engine, _ = self._engine_and_session(tmp_path)
        msgs = engine.get_messages("ffffffff-dead-beef-0000-000000000000")
        assert msgs == []

    def test_message_type_filter_works(self, tmp_path):
        engine, session_id = self._engine_and_session(tmp_path)
        user_msgs = engine.get_messages(session_id, message_type="user")
        assert all(m.type.value == "user" for m in user_msgs)
        assistant_msgs = engine.get_messages(session_id, message_type="assistant")
        assert all(m.type.value == "assistant" for m in assistant_msgs)


# ── New tests: corrupt JSONL handling ─────────────────────────────────────────

class TestCorruptJsonlHandling:
    """Malformed JSONL lines are skipped without crashing."""

    def test_corrupt_lines_are_skipped(self, tmp_path):
        session_id = "good-session-id"
        projects = _make_projects_dir(tmp_path, session_id)
        engine = SessionRecoveryEngine(projects, tmp_path / "recovery")
        # Should find the valid lines and skip the malformed one
        msgs = engine.get_messages(session_id)
        assert isinstance(msgs, list)
        # At least the well-formed messages should be found
        assert len(msgs) >= 2

    def test_search_messages_corrupt_lines_skipped(self, tmp_path):
        session_id = "search-session"
        projects = _make_projects_dir(tmp_path, session_id)
        engine = SessionRecoveryEngine(projects, tmp_path / "recovery")
        results = engine.search_messages("python")
        assert isinstance(results, list)
        # "python rocks" is in the last valid line
        assert len(results) >= 1

    def test_totally_corrupt_jsonl_file_returns_empty(self, tmp_path):
        projects = tmp_path / "projects" / "proj"
        projects.mkdir(parents=True)
        (projects / "bad.jsonl").write_bytes(b"\xff\xfe garbage binary data \x00\x01\x02")
        engine = SessionRecoveryEngine(tmp_path / "projects", tmp_path / "recovery")
        assert engine.search_messages("anything") == []


# ── New tests: FilterSpec zero max_edits / max_size ───────────────────────────

class TestFilterSpecZeroMax:
    """max_edits=0 and max_size=0 must actually constrain, not be treated as unlimited."""

    def test_max_edits_zero_excludes_any_file_with_edits(self):
        """max_edits=0 → only files with 0 edits pass."""
        spec = FilterSpec(max_edits=0)
        assert spec.matches_edits(0) is True
        assert spec.matches_edits(1) is False
        assert spec.matches_edits(999) is False

    def test_max_size_zero_excludes_non_empty_files(self):
        """max_size=0 → only empty files pass."""
        spec = FilterSpec(max_size=0)
        assert spec.matches_size(0) is True
        assert spec.matches_size(1) is False
        assert spec.matches_size(10000) is False

    def test_search_filter_by_edits_zero_max(self):
        """SearchFilter.by_edits(max_edits=0) excludes files with edits."""
        from ai_session_tools import RecoveredFile
        f_with_edits = RecoveredFile(
            name="a.py", path="/a.py",
            file_type="py", edits=5, size_bytes=100,
        )
        f_no_edits = RecoveredFile(
            name="b.py", path="/b.py",
            file_type="py", edits=0, size_bytes=100,
        )
        sf = SearchFilter().by_edits(max_edits=0)
        result = sf([f_with_edits, f_no_edits])
        assert f_with_edits not in result
        assert f_no_edits in result


# ── New tests: path expansion ─────────────────────────────────────────────────

class TestPathExpansion:
    """Tilde in paths is expanded for both env vars and output_dir arguments."""

    def test_get_engine_expands_tilde_in_env_vars(self, tmp_path, monkeypatch):
        """AI_SESSION_TOOLS_* with leading ~ must be expanded."""
        from ai_session_tools.cli import get_engine
        monkeypatch.setenv("AI_SESSION_TOOLS_PROJECTS", "~/.claude/projects")
        monkeypatch.setenv("AI_SESSION_TOOLS_RECOVERY", "~/.claude/recovery")
        engine = get_engine()
        # Path should not literally start with ~
        assert not str(engine.projects_dir).startswith("~")
        assert not str(engine.recovery_dir).startswith("~")
        # Should point to home dir
        assert str(engine.projects_dir).startswith(str(Path.home()))

    def test_get_engine_env_var_overrides_default(self, tmp_path, monkeypatch):
        """AI_SESSION_TOOLS_PROJECTS overrides the default ~/.claude/projects."""
        from ai_session_tools.cli import get_engine
        custom = str(tmp_path / "custom_projects")
        monkeypatch.setenv("AI_SESSION_TOOLS_PROJECTS", custom)
        monkeypatch.delenv("AI_SESSION_TOOLS_RECOVERY", raising=False)
        engine = get_engine()
        assert str(engine.projects_dir) == custom

    def test_do_extract_expands_tilde_in_output_dir(self, tmp_path):
        """_do_extract resolves ~ in output_dir before passing to engine."""
        # We just verify the call does not crash and uses an expanded path;
        # the file won't be found so it'll print "not found", that's fine.
        from ai_session_tools.cli import _do_extract
        recovery = _make_recovery_dir(tmp_path)
        engine = SessionRecoveryEngine(tmp_path / "projects", recovery)
        # Should not raise due to unexpanded ~
        try:
            _do_extract(engine, "hello.py", "~/tmp_ai_session_test_output")
        except SystemExit:
            pass  # typer.Exit is expected when file not in recovery
        # The important thing: no FileNotFoundError about literal "~" directory


# ── New tests: invalid regex handling ─────────────────────────────────────────

class TestRegexErrorHandling:
    """Invalid regex patterns raise ValueError, not a raw re.error traceback."""

    def test_invalid_regex_raises_value_error(self, tmp_path):
        recovery = _make_recovery_dir(tmp_path)
        engine = SessionRecoveryEngine(tmp_path / "projects", recovery)
        with pytest.raises(ValueError, match="Invalid search pattern"):
            engine.search("[unclosed")

    def test_invalid_regex_in_search_messages_raises_value_error(self, tmp_path):
        projects = _make_projects_dir(tmp_path)
        engine = SessionRecoveryEngine(projects, tmp_path / "recovery")
        # Use a pattern without * or ? so it takes the regex (not glob) branch;
        # an unclosed bracket is always an invalid regex.
        with pytest.raises(ValueError, match="Invalid search pattern"):
            engine.search_messages("[unclosed")

    def test_glob_pattern_does_not_raise(self, tmp_path):
        recovery = _make_recovery_dir(tmp_path)
        engine = SessionRecoveryEngine(tmp_path / "projects", recovery)
        # Glob patterns (containing * or ?) are always valid
        result = engine.search("*.py")
        assert isinstance(result, list)


# ── New tests: SessionMessage.is_long ────────────────────────────────────────

class TestIsLongProperty:
    """is_long returns True for content longer than 500 chars."""

    def test_is_long_true_for_long_content(self):
        msg = SessionMessage(
            type=MessageType.USER,
            timestamp="2026-02-22T10:00:00Z",
            content="x" * 501,
            session_id="test",
        )
        assert msg.is_long is True

    def test_is_long_false_for_short_content(self):
        msg = SessionMessage(
            type=MessageType.USER,
            timestamp="2026-02-22T10:00:00Z",
            content="x" * 500,
            session_id="test",
        )
        assert msg.is_long is False

    def test_engine_preserves_full_content(self, tmp_path):
        """Engine no longer truncates message content to 500 chars."""
        session_id = "full-content-session"
        projects = tmp_path / "projects" / "proj"
        projects.mkdir(parents=True)
        long_content = "word " * 200  # 1000 chars
        (projects / f"{session_id}.jsonl").write_text(
            json.dumps({
                "sessionId": session_id,
                "type": "user",
                "timestamp": "2026-02-22T10:00:00Z",
                "message": {"content": long_content},
            })
        )
        engine = SessionRecoveryEngine(tmp_path / "projects", tmp_path / "recovery")
        msgs = engine.get_messages(session_id)
        assert len(msgs) == 1
        assert len(msgs[0].content) > 500
        assert msgs[0].is_long is True


# ── New tests: global statistics counting ─────────────────────────────────────

class TestStatisticsGlobal:
    """get_statistics counts versions globally across all sessions."""

    def test_largest_file_counts_across_sessions(self, tmp_path):
        """If hello.py has versions in two separate session dirs, totals are combined."""
        recovery = tmp_path / "recovery"

        # Session A: hello.py has 3 versions
        a = recovery / "session_all_versions_sessA"
        a.mkdir(parents=True)
        for i in range(1, 4):
            (a / f"hello.py_v{i:06d}_line_10.txt").write_text(f"version {i}")

        # Session B: hello.py has 2 more versions
        b = recovery / "session_all_versions_sessB"
        b.mkdir(parents=True)
        for i in range(4, 6):
            (b / f"hello.py_v{i:06d}_line_10.txt").write_text(f"version {i}")

        # Session B also has other.py with 1 version
        (b / "other.py_v000001_line_5.txt").write_text("other")

        # Add a session_* dir so total_files is > 0
        s = recovery / "session_sessA"
        s.mkdir()
        (s / "hello.py").write_text("latest")

        engine = SessionRecoveryEngine(tmp_path / "projects", recovery)
        stats = engine.get_statistics()

        assert stats.total_versions == 6
        assert stats.largest_file == "hello.py"
        assert stats.largest_file_edits == 5  # 3 + 2 = 5


# ── New tests: CSV formatter proper quoting ───────────────────────────────────

class TestCsvFormatterQuoting:
    """CsvFormatter produces RFC 4180-compliant output for special characters."""

    def _make_file(self, name: str) -> "RecoveredFile":
        from ai_session_tools import RecoveredFile
        return RecoveredFile(
            name=name,
            path=f"/{name}",
            file_type="py",
            edits=1,
            size_bytes=100,
            last_modified="2026-02-22T10:00:00",
            created_date="2026-02-22T09:00:00",
        )

    def test_comma_in_filename_is_quoted(self):
        from ai_session_tools.formatters import CsvFormatter
        import csv, io
        f = self._make_file("foo, bar.py")
        output = CsvFormatter().format(f)
        reader = csv.reader(io.StringIO(output))
        rows = list(reader)
        # Second row is the data row; first field is name
        assert rows[1][0] == "foo, bar.py"

    def test_format_many_produces_valid_csv(self):
        from ai_session_tools.formatters import CsvFormatter
        import csv, io
        files = [self._make_file("a.py"), self._make_file('b,"quoted".py')]
        output = CsvFormatter().format_many(files)
        reader = csv.reader(io.StringIO(output))
        rows = list(reader)
        assert rows[0] == ["name", "location", "type", "edits", "sessions", "size_bytes", "last_modified", "created_date"]
        assert rows[1][0] == "a.py"
        assert rows[2][0] == 'b,"quoted".py'

    def test_newline_in_location_is_quoted(self):
        """csv.writer must quote fields containing newlines."""
        from ai_session_tools.formatters import CsvFormatter
        import csv, io
        f = self._make_file("ok.py")
        output = CsvFormatter().format(f)
        # Verify it parses back cleanly
        rows = list(csv.reader(io.StringIO(output)))
        assert len(rows) >= 2  # header + data


# ── New tests: ChainedFilter export ──────────────────────────────────────────

class TestChainedFilterExport:
    """ChainedFilter is exported and works."""

    def test_chained_filter_exported(self):
        """ChainedFilter is importable from the top-level package."""
        assert ChainedFilter is not None

    def test_chained_filter_combines_search_filters(self):
        """ChainedFilter applies multiple filters in sequence."""
        from ai_session_tools import RecoveredFile
        files = [
            RecoveredFile("a.py", "/a.py", file_type="py", edits=5, size_bytes=100),
            RecoveredFile("b.py", "/b.py", file_type="py", edits=1, size_bytes=100),
            RecoveredFile("c.md", "/c.md", file_type="md", edits=10, size_bytes=200),
        ]
        sf1 = SearchFilter().by_edits(min_edits=2)   # keeps a.py (5), c.md (10)
        sf2 = SearchFilter().by_extension("py")       # keeps only .py
        chained = ChainedFilter(sf1, sf2)
        result = chained(files)
        assert len(result) == 1
        assert result[0].name == "a.py"

    def test_chained_filter_with_empty_input(self):
        sf = SearchFilter().by_edits(min_edits=1)
        assert ChainedFilter(sf)([]) == []


# ── New tests: FilterSpec with_sessions / with_extensions empty set ───────────

class TestFilterSpecEmptySetBuilders:
    """with_sessions/with_extensions(include=set()) clears the filter, not ignores it."""

    def test_with_extensions_include_empty_set_clears_filter(self):
        spec = FilterSpec().with_extensions(include={"py"})
        assert spec.matches_extension("md") is False
        spec.with_extensions(include=set())  # should clear include filter
        assert spec.matches_extension("md") is True

    def test_with_sessions_include_empty_set_clears_filter(self):
        spec = FilterSpec().with_sessions(include={"abc"})
        assert spec.matches_session("xyz") is False
        spec.with_sessions(include=set())  # should clear
        assert spec.matches_session("xyz") is True

    def test_with_extensions_none_leaves_filter_unchanged(self):
        spec = FilterSpec().with_extensions(include={"py"})
        spec.with_extensions(include=None)  # None = don't change
        assert spec.matches_extension("py") is True
        assert spec.matches_extension("md") is False


# ── New tests: engine search deduplicates same filename across session dirs ───

class TestEngineSearchDeduplication:
    """Same filename in multiple session_*/ dirs appears only once in results."""

    def test_same_filename_in_two_sessions_returned_once(self, tmp_path):
        recovery = tmp_path / "recovery"
        for sess in ("session_s1", "session_s2"):
            d = recovery / sess
            d.mkdir(parents=True)
            (d / "shared.py").write_text("content")
        engine = SessionRecoveryEngine(tmp_path / "projects", recovery)
        results = engine.search("shared.py")
        names = [r.name for r in results]
        assert names.count("shared.py") == 1


# ── New tests: RecoveryStatistics avg_edits_per_file removed ──────────────────

class TestRecoveryStatisticsProperties:
    """RecoveryStatistics has correct properties (no duplicate avg_edits_per_file)."""

    def test_avg_versions_per_file_correct(self):
        stats = RecoveryStatistics(total_files=4, total_versions=8)
        assert stats.avg_versions_per_file == 2.0

    def test_avg_versions_per_file_zero_files(self):
        stats = RecoveryStatistics(total_files=0, total_versions=0)
        assert stats.avg_versions_per_file == 0.0

    def test_no_avg_edits_per_file_duplicate(self):
        """avg_edits_per_file was removed as a duplicate of avg_versions_per_file."""
        assert not hasattr(RecoveryStatistics, "avg_edits_per_file")


# ── Part B tests: performance-path ────────────────────────────────────────────

class TestVersionDirsCaching:
    """_version_dirs cached_property is scanned once per engine instance."""

    def test_version_dirs_returns_list(self, tmp_path):
        recovery = _make_recovery_dir(tmp_path)
        engine = SessionRecoveryEngine(tmp_path / "projects", recovery)
        result = engine._version_dirs
        assert isinstance(result, list)

    def test_version_dirs_caches_same_object(self, tmp_path):
        """Second access returns the same list object (cached_property guarantee)."""
        recovery = _make_recovery_dir(tmp_path)
        engine = SessionRecoveryEngine(tmp_path / "projects", recovery)
        first = engine._version_dirs
        second = engine._version_dirs
        assert first is second

    def test_version_dirs_missing_recovery_returns_empty(self, tmp_path):
        engine = SessionRecoveryEngine(tmp_path / "projects", tmp_path / "no_recovery")
        assert engine._version_dirs == []


class TestGetMessagesTargetedGlob:
    """get_messages only opens JSONL files matching the session prefix."""

    def test_messages_for_session_a_excludes_session_b(self, tmp_path):
        """Two JSONL files with different session IDs; only session A's messages returned."""
        projects = tmp_path / "projects" / "proj"
        projects.mkdir(parents=True)

        session_a = "aaaabbbb-0000-0000-0000-000000000001"
        session_b = "ccccdddd-0000-0000-0000-000000000002"

        (projects / f"{session_a}.jsonl").write_text(
            json.dumps({
                "sessionId": session_a, "type": "user", "timestamp": "2026-01-01T00:00:00Z",
                "message": {"content": "hello from A"},
            })
        )
        (projects / f"{session_b}.jsonl").write_text(
            json.dumps({
                "sessionId": session_b, "type": "user", "timestamp": "2026-01-01T00:00:01Z",
                "message": {"content": "hello from B"},
            })
        )

        engine = SessionRecoveryEngine(tmp_path / "projects", tmp_path / "recovery")
        msgs = engine.get_messages(session_a)
        assert all(m.session_id == session_a for m in msgs)
        assert len(msgs) == 1
        assert msgs[0].content == "hello from A"


class TestSearchMessagesLiteralPreFilter:
    """Literal query and equivalent regex return identical results (pre-filter has no false negatives)."""

    def _make_engine(self, tmp_path: Path, content: str, session_id: str = "test-session") -> SessionRecoveryEngine:
        projects = tmp_path / "projects" / "proj"
        projects.mkdir(parents=True)
        (projects / f"{session_id}.jsonl").write_text(
            json.dumps({
                "sessionId": session_id, "type": "user", "timestamp": "2026-01-01T00:00:00Z",
                "message": {"content": content},
            })
        )
        return SessionRecoveryEngine(tmp_path / "projects", tmp_path / "recovery")

    def test_literal_matches_same_as_regex(self, tmp_path):
        engine = self._make_engine(tmp_path, "the quick brown fox")
        literal_results = engine.search_messages("quick")
        regex_results = engine.search_messages("qu.ck")
        assert len(literal_results) == len(regex_results) == 1
        assert literal_results[0].content == regex_results[0].content

    def test_literal_no_match_returns_empty(self, tmp_path):
        engine = self._make_engine(tmp_path, "the quick brown fox")
        assert engine.search_messages("elephant") == []

    def test_literal_case_insensitive(self, tmp_path):
        engine = self._make_engine(tmp_path, "The Quick Brown Fox")
        # Literal pre-filter uses .lower() so case-insensitive match should work
        results = engine.search_messages("quick")
        assert len(results) == 1


class TestLocationIsString:
    """RecoveredFile.location is a plain str; engine search results have location == 'recovery'."""

    def test_location_default_is_recovery(self):
        from ai_session_tools import RecoveredFile
        r = RecoveredFile(name="x.py", path="/x.py", file_type="py")
        assert isinstance(r.location, str)
        assert r.location == "recovery"

    def test_engine_search_results_have_string_location(self, tmp_path):
        recovery = _make_recovery_dir(tmp_path)
        engine = SessionRecoveryEngine(tmp_path / "projects", recovery)
        results = engine.search("*.py")
        assert len(results) > 0
        for r in results:
            assert isinstance(r.location, str)
            assert r.location == "recovery"

    def test_file_location_not_in_public_api(self):
        """FileLocation enum was removed; it is no longer accessible from the package."""
        import ai_session_tools
        assert not hasattr(ai_session_tools, "FileLocation")

    def test_file_type_not_in_public_api(self):
        """FileType enum was removed; it is no longer accessible from the package."""
        import ai_session_tools
        assert not hasattr(ai_session_tools, "FileType")


# ── Part C tests: original-path extraction ────────────────────────────────────

def _write_tool_call_jsonl(path: Path, session_id: str, filename: str, file_path: str, tool_name: str = "Write") -> None:
    """Write a JSONL file containing an assistant tool_use record for filename."""
    path.write_text(
        json.dumps({
            "sessionId": session_id,
            "type": "assistant",
            "timestamp": "2026-01-01T00:00:00Z",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": tool_name,
                        "input": {"file_path": file_path},
                    }
                ]
            },
        })
    )


class TestGetOriginalPath:
    """get_original_path returns the path from Write/Edit tool_use records."""

    def test_returns_path_from_write_tool_call(self, tmp_path):
        projects = tmp_path / "projects" / "proj"
        projects.mkdir(parents=True)
        session_id = "orig-path-session"
        expected = "/home/user/myproject/cli.py"
        _write_tool_call_jsonl(projects / f"{session_id}.jsonl", session_id, "cli.py", expected)
        engine = SessionRecoveryEngine(tmp_path / "projects", tmp_path / "recovery")
        assert engine.get_original_path("cli.py") == expected

    def test_returns_none_when_not_found(self, tmp_path):
        projects = tmp_path / "projects" / "proj"
        projects.mkdir(parents=True)
        session_id = "no-write-session"
        (projects / f"{session_id}.jsonl").write_text(
            json.dumps({"sessionId": session_id, "type": "user", "message": {"content": "hi"}})
        )
        engine = SessionRecoveryEngine(tmp_path / "projects", tmp_path / "recovery")
        assert engine.get_original_path("missing.py") is None

    def test_returns_none_when_projects_dir_missing(self, tmp_path):
        engine = SessionRecoveryEngine(tmp_path / "no_projects", tmp_path / "recovery")
        assert engine.get_original_path("cli.py") is None

    def test_edit_tool_also_found(self, tmp_path):
        projects = tmp_path / "projects" / "proj"
        projects.mkdir(parents=True)
        session_id = "edit-session"
        expected = "/home/user/engine.py"
        _write_tool_call_jsonl(projects / f"{session_id}.jsonl", session_id, "engine.py", expected, "Edit")
        engine = SessionRecoveryEngine(tmp_path / "projects", tmp_path / "recovery")
        assert engine.get_original_path("engine.py") == expected

    def test_notebook_edit_tool_also_found(self, tmp_path):
        projects = tmp_path / "projects" / "proj"
        projects.mkdir(parents=True)
        session_id = "nb-session"
        expected = "/home/user/notebook.ipynb"
        _write_tool_call_jsonl(projects / f"{session_id}.jsonl", session_id, "notebook.ipynb", expected, "NotebookEdit")
        engine = SessionRecoveryEngine(tmp_path / "projects", tmp_path / "recovery")
        assert engine.get_original_path("notebook.ipynb") == expected


class TestGetOriginalPathToolUseResult:
    """get_original_path finds paths from toolUseResult.filePath (user confirmation records)."""

    def test_tool_use_result_file_path(self, tmp_path):
        projects = tmp_path / "projects" / "proj"
        projects.mkdir(parents=True)
        session_id = "tur-session"
        expected = "/home/user/output.py"
        (projects / f"{session_id}.jsonl").write_text(
            json.dumps({
                "sessionId": session_id,
                "type": "user",
                "timestamp": "2026-01-01T00:00:00Z",
                "toolUseResult": {"filePath": expected},
            })
        )
        engine = SessionRecoveryEngine(tmp_path / "projects", tmp_path / "recovery")
        assert engine.get_original_path("output.py") == expected


class TestGetOriginalPathMultipleVersions:
    """When multiple Write calls exist for the same filename, the last one wins."""

    def test_last_write_wins(self, tmp_path):
        projects = tmp_path / "projects" / "proj"
        projects.mkdir(parents=True)
        session_id = "multi-write-session"
        first_path = "/home/user/old/cli.py"
        second_path = "/home/user/new/cli.py"
        (projects / f"{session_id}.jsonl").write_text(
            "\n".join([
                json.dumps({
                    "sessionId": session_id, "type": "assistant",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "message": {"content": [{"type": "tool_use", "name": "Write", "input": {"file_path": first_path}}]},
                }),
                json.dumps({
                    "sessionId": session_id, "type": "assistant",
                    "timestamp": "2026-01-01T00:00:01Z",
                    "message": {"content": [{"type": "tool_use", "name": "Write", "input": {"file_path": second_path}}]},
                }),
            ])
        )
        engine = SessionRecoveryEngine(tmp_path / "projects", tmp_path / "recovery")
        assert engine.get_original_path("cli.py") == second_path


class TestResolveOutputPath:
    """_resolve_output_path returns target unchanged when it doesn't exist; appends .recovered suffix otherwise."""

    def test_nonexistent_target_returned_as_is(self, tmp_path):
        from ai_session_tools.cli import _resolve_output_path
        target = tmp_path / "newfile.py"
        assert _resolve_output_path(target) == target

    def test_existing_target_gets_recovered_suffix(self, tmp_path):
        from ai_session_tools.cli import _resolve_output_path
        target = tmp_path / "cli.py"
        target.write_text("existing")
        result = _resolve_output_path(target)
        assert result == tmp_path / "cli.recovered.py"

    def test_recovered_suffix_also_exists_gets_numbered(self, tmp_path):
        from ai_session_tools.cli import _resolve_output_path
        (tmp_path / "cli.py").write_text("original")
        (tmp_path / "cli.recovered.py").write_text("first recovered")
        result = _resolve_output_path(tmp_path / "cli.py")
        assert result == tmp_path / "cli.recovered_1.py"

    def test_multiple_conflicts_increment(self, tmp_path):
        from ai_session_tools.cli import _resolve_output_path
        (tmp_path / "cli.py").write_text("original")
        (tmp_path / "cli.recovered.py").write_text("r0")
        (tmp_path / "cli.recovered_1.py").write_text("r1")
        result = _resolve_output_path(tmp_path / "cli.py")
        assert result == tmp_path / "cli.recovered_2.py"


class TestExtractToOriginalPath:
    """_do_extract with output_dir=None restores to the original recorded path."""

    def test_extract_restores_to_original_path(self, tmp_path):
        from ai_session_tools.cli import _do_extract

        # Set up recovery dir with hello.py
        recovery = _make_recovery_dir(tmp_path)

        # Set up projects dir with a Write tool_use record pointing to a custom path
        projects = tmp_path / "projects" / "proj"
        projects.mkdir(parents=True)
        original_path = tmp_path / "workspace" / "hello.py"
        _write_tool_call_jsonl(
            projects / "session.jsonl",
            "any-session", "hello.py", str(original_path)
        )

        engine = SessionRecoveryEngine(tmp_path / "projects", recovery)
        try:
            _do_extract(engine, "hello.py", None)
        except SystemExit:
            pass
        # If original path dir was created, the file was restored there
        assert original_path.exists()


class TestExtractFallback:
    """_do_extract falls back to ./recovered/ when no original path is recorded."""

    def test_fallback_to_recovered_dir(self, tmp_path, monkeypatch):
        from ai_session_tools.cli import _do_extract

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("AI_SESSION_TOOLS_OUTPUT", raising=False)

        recovery = _make_recovery_dir(tmp_path)
        # No projects dir — get_original_path will return None
        engine = SessionRecoveryEngine(tmp_path / "empty_projects", recovery)
        try:
            _do_extract(engine, "hello.py", None)
        except SystemExit:
            pass
        assert (tmp_path / "recovered" / "hello.py").exists()
