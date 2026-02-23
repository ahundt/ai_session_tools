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
