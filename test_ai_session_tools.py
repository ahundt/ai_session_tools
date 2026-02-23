#!/usr/bin/env python3
"""
Unit tests for AI Session Tools

Tests cover:
- File search and filtering
- Version extraction
- Content search
- File location detection
- Metadata collection
"""

import pytest
from pathlib import Path
from ai_session_tools import (
    SessionRecoveryEngine,
    FileLocation,
    FileVersion,
)


@pytest.fixture
def recovery_dir():
    """Get the recovery directory path"""
    return Path.home() / ".claude" / "recovery" / "2026_02_22_session_scripts_recovery"


@pytest.fixture
def engine(recovery_dir):
    """Create a SessionRecoveryEngine instance"""
    projects_dir = Path.home() / ".claude" / "projects"
    if not recovery_dir.exists():
        pytest.skip("Recovery directory not found")
    return SessionRecoveryEngine(projects_dir, recovery_dir)


class TestPatternCompilation:
    """Test pattern matching and compilation"""

    def test_glob_pattern_with_asterisks(self):
        """Test glob pattern conversion"""
        pattern = _compile_pattern("*session*.py")
        assert pattern.search("session_manager.py")
        assert pattern.search("test_session_start_handler.py")
        assert not pattern.search("main.py")

    def test_glob_pattern_with_question_mark(self):
        """Test glob pattern with ? wildcard"""
        pattern = _compile_pattern("*.p?")
        assert pattern.search("file.py")
        assert pattern.search("file.ps")
        assert not pattern.search("file.txt")

    def test_regex_pattern(self):
        """Test regex pattern matching"""
        # Regex patterns work best without glob wildcards
        pattern = _compile_pattern("session")
        assert pattern.search("session_manager")
        assert pattern.search("test_session_handler")
        assert not pattern.search("main")

    def test_case_insensitive_matching(self):
        """Test that patterns are case insensitive"""
        pattern = _compile_pattern("*SESSION*.py")
        assert pattern.search("session_manager.py")
        assert pattern.search("SESSION_MANAGER.py")


class TestFileSearch:
    """Test file search functionality"""

    def test_search_by_glob_pattern(self, engine):
        """Test searching with glob pattern"""
        results = engine.search_files("*session*.py")
        filenames = [r.name for r in results]

        # Should find session-related files
        assert len(filenames) > 0
        assert any("session" in f for f in filenames)

    def test_search_returns_sorted_by_edits(self, engine):
        """Test that results are sorted by edit count"""
        results = engine.search_files("*.py")

        # Results should be sorted by edits descending
        edits = [r.edits for r in results]
        assert edits == sorted(edits, reverse=True)

    def test_min_edits_filter(self, engine):
        """Test filtering by minimum edits"""
        results_all = engine.search_files("*.py", min_edits=0)
        results_filtered = engine.search_files("*.py", min_edits=1)

        # Filtered results should be subset of all
        assert len(results_filtered) <= len(results_all)

    def test_file_type_detection(self, engine):
        """Test that file types are correctly detected"""
        results = engine.search_files("*.py")

        # All results should be Python files
        for result in results:
            if result.name.endswith(".py"):
                assert result.file_type == "py"


class TestVersionExtraction:
    """Test version history extraction"""

    def test_get_file_versions(self, engine):
        """Test getting versions of a file"""
        # Find a file that has versions
        results = engine.search_files("*.py")

        for result in results:
            versions = engine.get_file_versions(result.name)
            # If versions exist, they should be valid
            if versions:
                for v in versions:
                    assert isinstance(v, FileVersion)
                    assert v.filename == result.name
                    assert v.version_num > 0
                    assert v.line_count > 0
                break

    def test_version_ordering(self, engine):
        """Test that versions are in correct order"""
        results = engine.search_files("*.py")

        for result in results:
            versions = engine.get_file_versions(result.name)
            if len(versions) > 1:
                # Versions should have increasing version numbers
                version_nums = [v.version_num for v in versions]
                assert version_nums == sorted(version_nums)
                break

    def test_extract_final_version(self, engine, tmp_path):
        """Test extracting final version of a file"""
        results = engine.search_files("*.py")

        if results:
            test_file = results[0].name
            output_path = engine.extract_final_version(test_file, tmp_path)

            # Should return a Path or None
            if output_path:
                assert output_path.exists()
                assert output_path.name == test_file


class TestStatistics:
    """Test recovery statistics"""

    def test_get_statistics_returns_dict(self, engine):
        """Test that statistics returns required fields"""
        stats = engine.get_statistics()

        assert isinstance(stats, dict)
        assert "total_sessions" in stats
        assert "total_files" in stats
        assert "total_versions" in stats
        assert "largest_file" in stats
        assert "largest_file_edits" in stats

    def test_statistics_are_positive(self, engine):
        """Test that statistics contain positive values"""
        stats = engine.get_statistics()

        assert stats["total_sessions"] > 0
        assert stats["total_files"] > 0
        assert stats["total_versions"] > 0


class TestContentSearch:
    """Test searching file contents"""

    def test_search_content_pattern(self, engine):
        """Test searching for pattern in file contents"""
        results = engine.search_content("def ")

        # Should find Python function definitions
        assert isinstance(results, dict)

    def test_content_search_returns_dict(self, engine):
        """Test that content search returns structured data"""
        results = engine.search_content("import")

        assert isinstance(results, dict)
        # Results should map filenames to matches
        for filename, matches in results.items():
            assert isinstance(filename, str)
            assert isinstance(matches, list)


class TestLocationDetection:
    """Test file location categorization"""

    def test_build_file_info_includes_location(self, engine):
        """Test that file info includes location"""
        results = engine.search_files("*.py")

        if results:
            file_info = results[0]
            assert hasattr(file_info, 'location')
            assert isinstance(file_info.location, FileLocation)

    def test_location_is_valid_enum(self, engine):
        """Test that locations are valid enum values"""
        results = engine.search_files("*.py")

        valid_locations = {loc.value for loc in FileLocation}
        for result in results[:5]:
            assert result.location.value in valid_locations


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_nonexistent_file(self, engine):
        """Test handling of nonexistent files"""
        versions = engine.get_file_versions("nonexistent_file_xyz.py")

        # Should return empty list, not crash
        assert versions == []

    def test_empty_pattern_search(self, engine):
        """Test searching with very broad pattern"""
        results = engine.search_files(".*")

        # Should return results
        assert isinstance(results, list)
        assert len(results) > 0

    def test_special_characters_in_pattern(self, engine):
        """Test patterns with special characters"""
        # Should handle special regex characters safely
        pattern = _compile_pattern("test_*.py")
        assert pattern is not None

    def test_extract_with_invalid_session(self, engine, tmp_path):
        """Test extraction with invalid session ID"""
        results = engine.search_files("*.py")

        if results:
            test_file = results[0].name
            # Extract with non-existent session ID
            output_path = engine.extract_final_version(
                test_file, tmp_path, session_id="nonexistent_session"
            )

            # Should return None, not crash
            assert output_path is None


class TestIntegration:
    """Integration tests combining multiple operations"""

    def test_search_extract_workflow(self, engine, tmp_path):
        """Test complete search and extract workflow"""
        # Search for files
        results = engine.search_files("*session*.py")
        assert len(results) > 0

        # Extract a result
        test_file = results[0].name
        output_path = engine.extract_final_version(test_file, tmp_path)

        # Verify extraction
        if output_path:
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_statistics_match_search_results(self, engine):
        """Test that statistics contain valid data"""
        stats = engine.get_statistics()

        # Statistics should have reasonable values
        assert stats["total_sessions"] > 0
        assert stats["total_files"] > 0
        assert stats["total_versions"] > stats["total_files"]  # More versions than files


if __name__ == "__main__":
    # Run tests with: pytest test_ai_session_tools.py -v
    pytest.main([__file__, "-v", "--tb=short"])
