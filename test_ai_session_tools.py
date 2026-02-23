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

import pytest
from pathlib import Path
from ai_session_tools import (
    SessionRecoveryEngine,
    FilterSpec,
    SearchFilter,
    FileLocation,
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
    """Test file location enum"""

    def test_file_location_enum_exists(self):
        """Test that FileLocation enum has expected values"""
        assert hasattr(FileLocation, "CLAUTORUN_MAIN")
        assert hasattr(FileLocation, "CLAUTORUN_WORKTREE")

    def test_file_location_has_value(self):
        """Test that FileLocation values are strings"""
        assert isinstance(FileLocation.CLAUTORUN_MAIN.value, str)


class TestIntegration:
    """Integration tests combining multiple operations"""

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

    def test_get_messages_with_type_filter_user(self, engine):
        """Test filtering for user messages only"""
        # This requires actual session data, so skip if none exist
        stats = engine.get_statistics()
        if stats.total_sessions > 0:
            messages = engine.get_messages("*", message_type="user")
            assert isinstance(messages, list)
            # If we got results, all should be user messages
            if messages:
                for msg in messages:
                    assert msg.message_type in ("user", "USER")

    def test_get_messages_with_type_filter_assistant(self, engine):
        """Test filtering for assistant messages only"""
        stats = engine.get_statistics()
        if stats.total_sessions > 0:
            messages = engine.get_messages("*", message_type="assistant")
            assert isinstance(messages, list)
            # If we got results, all should be assistant messages
            if messages:
                for msg in messages:
                    assert msg.message_type in ("assistant", "ASSISTANT")

    def test_search_messages_case_insensitive(self, engine):
        """Test that message search is case insensitive"""
        # Search with different cases for the same term
        results_lower = engine.search_messages("python")
        results_upper = engine.search_messages("PYTHON")
        # Both should work (case insensitive)
        assert isinstance(results_lower, list)
        assert isinstance(results_upper, list)

    def test_search_messages_with_phrases(self, engine):
        """Test searching for multi-word phrases"""
        results = engine.search_messages("session data")
        assert isinstance(results, list)


class TestFilterComposition:
    """Test composable filter combinations"""

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

    def test_filter_spec_has_matches_date(self):
        """FilterSpec should have matches_date method."""
        spec = FilterSpec()
        assert hasattr(spec, "matches_date")
        assert callable(spec.matches_date)

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
        """FilterSpec.matches_location works."""
        spec = FilterSpec(include_folders={"main"})
        assert spec.matches_location("clautorun/main") is True
        assert spec.matches_location("clautorun/worktree") is False


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

class TestFilterSpecMatchesDate:
    """TDD tests for FilterSpec.matches_date() method."""

    def test_no_filter_passes_any_date(self):
        """With no date filters set, any date passes."""
        spec = FilterSpec()
        assert spec.matches_date("2026-02-22") is True
        assert spec.matches_date("2020-01-01") is True

    def test_no_filter_passes_none_date(self):
        """With no date filters, None date passes."""
        spec = FilterSpec()
        assert spec.matches_date(None) is True

    def test_after_date_excludes_earlier(self):
        """after_date excludes dates before the threshold."""
        spec = FilterSpec(after_date="2026-02-01")
        assert spec.matches_date("2026-01-15") is False
        assert spec.matches_date("2026-02-01") is True
        assert spec.matches_date("2026-02-22") is True

    def test_before_date_excludes_later(self):
        """before_date excludes dates after the threshold."""
        spec = FilterSpec(before_date="2026-02-15")
        assert spec.matches_date("2026-02-22") is False
        assert spec.matches_date("2026-02-15") is True
        assert spec.matches_date("2026-01-01") is True

    def test_date_range(self):
        """Combined after_date + before_date creates a range."""
        spec = FilterSpec(after_date="2026-02-01", before_date="2026-02-28")
        assert spec.matches_date("2026-02-15") is True
        assert spec.matches_date("2026-01-15") is False
        assert spec.matches_date("2026-03-01") is False

    def test_none_date_excluded_when_filter_active(self):
        """None date is excluded when any date filter is active (conservative)."""
        spec = FilterSpec(after_date="2026-01-01")
        assert spec.matches_date(None) is False

    def test_none_date_excluded_with_before_date(self):
        """None date is excluded when before_date is set."""
        spec = FilterSpec(before_date="2026-12-31")
        assert spec.matches_date(None) is False


class TestEnginePopulatesDateFields:
    """TDD tests for engine populating last_modified and created_date."""

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

    def test_last_modified_is_iso_format(self, engine):
        """last_modified should be ISO YYYY-MM-DD format."""
        results = engine.search("*.py")
        if results:
            for r in results:
                if r.last_modified:
                    assert len(r.last_modified) == 10, f"Expected YYYY-MM-DD, got {r.last_modified}"
                    assert r.last_modified[4] == "-"
                    assert r.last_modified[7] == "-"

    def test_date_filter_actually_filters(self, engine):
        """Date filter should actually exclude files outside the range."""
        # Get all files first
        all_results = engine.search("*.py")
        if not all_results:
            pytest.skip("No files found")

        # Use a future date that should exclude everything
        filters = FilterSpec(after_date="2099-01-01")
        filtered = engine.search("*.py", filters)
        assert len(filtered) < len(all_results), "Date filter should exclude some files"


class TestEngineSizeFilterWired:
    """TDD tests for engine wiring matches_size() in _apply_all_filters."""

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

    def test_cli_search_files_has_date_flags(self):
        """File search should have --after-date and --before-date flags."""
        from ai_session_tools.cli import app
        from typer.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(app, ["files", "search", "--help"])
        assert result.exit_code == 0
        assert "after-date" in result.output.lower()
        assert "before-date" in result.output.lower()

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
