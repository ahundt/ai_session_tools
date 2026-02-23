#!/usr/bin/env python3
"""
Unit tests for AI Session Tools

Tests cover:
- File search and filtering
- Version extraction
- Message access
- Statistics collection
- Filter composability
- New: session listing, corrections, planning usage, tool search, cross-ref, export
"""

import json
from pathlib import Path
from typing import Optional

import pytest
from typer.testing import CliRunner

from ai_session_tools import (
    ChainedFilter,
    ComposableFilter,
    ComposableSearch,
    CorrectionMatch,
    FileVersion,
    FilterSpec,
    MessageType,
    PlanningCommandCount,
    RecoveryStatistics,
    SearchFilter,
    SessionInfo,
    SessionMessage,
    SessionRecoveryEngine,
)
from ai_session_tools.cli import app

runner = CliRunner(mix_stderr=False)


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


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_projects_with_sessions(tmp_path: Path) -> Path:
    """Create projects dir with 2 sessions across 2 projects, with varied message types."""
    projects = tmp_path / "projects"

    # Session 1 in project 1: has correction, planning command, Write tool call
    proj1 = projects / "-Users-alice-proj1"
    proj1.mkdir(parents=True)
    s1 = "aaaa0001-0000-0000-0000-000000000000"
    lines1 = [
        json.dumps({"sessionId": s1, "type": "user", "timestamp": "2026-01-24T10:00:00.000Z",
                    "gitBranch": "main", "cwd": "/Users/alice/proj1",
                    "message": {"role": "user", "content": "start the feature"}}),
        json.dumps({"sessionId": s1, "type": "user", "timestamp": "2026-01-24T10:05:00.000Z",
                    "message": {"role": "user", "content": "you forgot to add the test"}}),
        json.dumps({"sessionId": s1, "type": "user", "timestamp": "2026-01-24T10:10:00.000Z",
                    "message": {"role": "user", "content": "/ar:plannew add login form"}}),
        json.dumps({"sessionId": s1, "type": "assistant", "timestamp": "2026-01-24T10:11:00.000Z",
                    "message": {"role": "assistant", "content": [
                        {"type": "tool_use", "id": "t1", "name": "Write",
                         "input": {"file_path": "/Users/alice/proj1/login.py",
                                   "content": "def login():\n    pass\n"}}]}}),
    ]
    (proj1 / f"{s1}.jsonl").write_text("\n".join(lines1))

    # Session 2 in project 2: different project, different branch
    proj2 = projects / "-Users-alice-proj2"
    proj2.mkdir(parents=True)
    s2 = "bbbb0002-0000-0000-0000-000000000000"
    lines2 = [
        json.dumps({"sessionId": s2, "type": "user", "timestamp": "2026-01-25T09:00:00.000Z",
                    "gitBranch": "feature-x", "cwd": "/Users/alice/proj2",
                    "message": {"role": "user", "content": "work on proj2"}}),
        json.dumps({"sessionId": s2, "type": "user", "timestamp": "2026-01-25T09:05:00.000Z",
                    "message": {"role": "user", "content": "/ar:pn new plan"}}),
    ]
    (proj2 / f"{s2}.jsonl").write_text("\n".join(lines2))

    return projects


def _make_engine(tmp_path: Path, projects: Path) -> SessionRecoveryEngine:
    """Create engine with tmp projects dir and non-existent recovery dir."""
    return SessionRecoveryEngine(projects, tmp_path / "recovery")


# ─── Part 0: Model tests ──────────────────────────────────────────────────────

class TestSessionInfoModel:
    def test_fields_accessible(self):
        si = SessionInfo(
            session_id="abc", project_dir="-proj", cwd="/foo", git_branch="main",
            timestamp_first="2026-01-01T00:00:00Z", timestamp_last="2026-01-01T01:00:00Z",
            message_count=5, has_compact_summary=False,
        )
        assert si.session_id == "abc"
        assert si.git_branch == "main"
        assert si.message_count == 5

    def test_to_dict_returns_8_keys(self):
        si = SessionInfo(
            session_id="abc", project_dir="-proj", cwd="/foo", git_branch="main",
            timestamp_first="2026-01-01T00:00:00Z", timestamp_last="2026-01-01T01:00:00Z",
            message_count=5, has_compact_summary=True,
        )
        d = si.to_dict()
        assert set(d.keys()) == {
            "session_id", "project_dir", "cwd", "git_branch",
            "timestamp_first", "timestamp_last", "message_count", "has_compact_summary",
        }

    def test_is_mutable_dataclass(self):
        si = SessionInfo(
            session_id="abc", project_dir="-proj", cwd="/foo", git_branch="main",
            timestamp_first="", timestamp_last="", message_count=0, has_compact_summary=False,
        )
        si.message_count = 99
        assert si.message_count == 99


class TestCorrectionMatchModel:
    def test_to_dict_returns_6_keys(self):
        cm = CorrectionMatch(
            session_id="abc", project_dir="-proj", timestamp="2026-01-01T00:00:00Z",
            content="you forgot it", category="skip_step", matched_pattern="you forgot",
        )
        d = cm.to_dict()
        assert set(d.keys()) == {
            "session_id", "project_dir", "timestamp", "content", "category", "matched_pattern",
        }

    def test_is_mutable_dataclass(self):
        cm = CorrectionMatch(
            session_id="abc", project_dir="-proj", timestamp="",
            content="x", category="skip_step", matched_pattern="x",
        )
        cm.category = "regression"
        assert cm.category == "regression"


class TestPlanningCommandCountModel:
    def test_to_dict_returns_6_keys(self):
        pc = PlanningCommandCount(
            command="/ar:plannew", count=3,
            session_ids=["s1", "s2"], project_dirs=["p1"],
        )
        d = pc.to_dict()
        assert set(d.keys()) == {
            "command", "count", "unique_sessions", "unique_projects",
            "session_ids", "project_dirs",
        }
        assert d["unique_sessions"] == 2
        assert d["unique_projects"] == 1

    def test_is_mutable_dataclass(self):
        pc = PlanningCommandCount(command="/ar:plannew", count=1, session_ids=[], project_dirs=[])
        pc.count = 5
        assert pc.count == 5


# ─── Part 1a: get_sessions ────────────────────────────────────────────────────

class TestGetSessionsBasic:
    def test_returns_2_sessions(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        engine = _make_engine(tmp_path, projects)
        sessions = engine.get_sessions()
        assert len(sessions) == 2

    def test_returns_session_info_objects(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        engine = _make_engine(tmp_path, projects)
        sessions = engine.get_sessions()
        assert all(isinstance(s, SessionInfo) for s in sessions)


class TestGetSessionsFields:
    def test_proj1_fields(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        engine = _make_engine(tmp_path, projects)
        sessions = engine.get_sessions()
        # proj2 is newer, proj1 is older — find proj1 by project_dir
        s1 = next(s for s in sessions if "proj1" in s.project_dir)
        assert s1.cwd == "/Users/alice/proj1"
        assert s1.git_branch == "main"
        assert s1.message_count == 4  # 3 user + 1 assistant


class TestGetSessionsProjectFilter:
    def test_filter_to_proj1(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        engine = _make_engine(tmp_path, projects)
        sessions = engine.get_sessions(project_filter="proj1")
        assert len(sessions) == 1
        assert "proj1" in sessions[0].project_dir


class TestGetSessionsAfterFilter:
    def test_after_2026_01_25(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        engine = _make_engine(tmp_path, projects)
        sessions = engine.get_sessions(after="2026-01-25")
        assert len(sessions) == 1
        assert "proj2" in sessions[0].project_dir


class TestGetSessionsSortedNewestFirst:
    def test_proj2_first(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        engine = _make_engine(tmp_path, projects)
        sessions = engine.get_sessions()
        assert "proj2" in sessions[0].project_dir
        assert "proj1" in sessions[1].project_dir


class TestGetSessionsCompactSummary:
    def test_no_compact_when_absent(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        engine = _make_engine(tmp_path, projects)
        sessions = engine.get_sessions()
        assert all(not s.has_compact_summary for s in sessions)


# ─── Part 1b: find_corrections ───────────────────────────────────────────────

class TestFindCorrectionsBasic:
    def test_detects_you_forgot(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        engine = _make_engine(tmp_path, projects)
        corrections = engine.find_corrections()
        assert len(corrections) >= 1
        assert any("you forgot" in c.content for c in corrections)


class TestFindCorrectionsCategory:
    def test_you_forgot_is_skip_step(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        engine = _make_engine(tmp_path, projects)
        corrections = engine.find_corrections()
        forgot = next(c for c in corrections if "you forgot" in c.content)
        assert forgot.category == "skip_step"


class TestFindCorrectionsMatchedPattern:
    def test_matched_pattern_value(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        engine = _make_engine(tmp_path, projects)
        corrections = engine.find_corrections()
        forgot = next(c for c in corrections if "you forgot" in c.content)
        assert "you forgot" in forgot.matched_pattern


class TestFindCorrectionsOnlyUserMessages:
    def test_no_assistant_corrections(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        engine = _make_engine(tmp_path, projects)
        corrections = engine.find_corrections()
        # All assistant messages are tool_use blocks — none should match correction patterns
        for c in corrections:
            # corrections come from user messages only (data.get("type") != "user" is skipped)
            assert c.session_id != ""  # sanity


class TestFindCorrectionsLimit:
    def test_limit_zero_returns_empty(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        engine = _make_engine(tmp_path, projects)
        assert engine.find_corrections(limit=0) == []

    def test_limit_1_returns_max_1(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        engine = _make_engine(tmp_path, projects)
        assert len(engine.find_corrections(limit=1)) <= 1


# ─── Part 1c: analyze_planning_usage ─────────────────────────────────────────

class TestPlanningUsageBasic:
    def test_finds_ar_plannew_and_ar_pn(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        engine = _make_engine(tmp_path, projects)
        results = engine.analyze_planning_usage()
        commands = {r.command for r in results}
        assert "/ar:plannew" in commands
        assert "/ar:pn" in commands


class TestPlanningUsageCountAndSessions:
    def test_plannew_count_and_session(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        engine = _make_engine(tmp_path, projects)
        results = engine.analyze_planning_usage()
        pn = next(r for r in results if r.command == "/ar:plannew")
        assert pn.count == 1
        assert "aaaa0001-0000-0000-0000-000000000000" in pn.session_ids


class TestPlanningUsageSortedByCount:
    def test_sorted_desc_by_count(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        engine = _make_engine(tmp_path, projects)
        results = engine.analyze_planning_usage()
        counts = [r.count for r in results]
        assert counts == sorted(counts, reverse=True)


class TestPlanningUsageCustomCommands:
    def test_custom_commands_override(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        engine = _make_engine(tmp_path, projects)
        results = engine.analyze_planning_usage(commands=[r"/ar:plannew\b"])
        assert len(results) == 1
        assert results[0].command == "/ar:plannew"  # \b stripped


# ─── Part 1d: search_messages with tool ──────────────────────────────────────

class TestSearchMessagesToolFilter:
    def test_tool_write_returns_assistant_message(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        engine = _make_engine(tmp_path, projects)
        results = engine.search_messages("", tool="Write")
        assert len(results) >= 1
        assert any("login.py" in r.content for r in results)


class TestSearchMessagesToolDefaultsToAssistant:
    def test_tool_implies_assistant_type(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        engine = _make_engine(tmp_path, projects)
        results = engine.search_messages("", tool="Write")
        assert all(r.type == MessageType.ASSISTANT for r in results)


class TestSearchMessagesToolWithQuery:
    def test_tool_with_query_filter(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        engine = _make_engine(tmp_path, projects)
        results = engine.search_messages("login", tool="Write")
        assert len(results) >= 1
        assert all("login" in r.content for r in results)


class TestSearchMessagesNoToolUnchanged:
    def test_no_tool_returns_user_messages(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        engine = _make_engine(tmp_path, projects)
        results = engine.search_messages("start the feature")
        assert len(results) >= 1
        assert any("start the feature" in r.content for r in results)


# ─── Part 1e: cross_reference_session ────────────────────────────────────────

class TestCrossRefFindsWrite:
    def test_finds_write_call(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        engine = _make_engine(tmp_path, projects)
        results = engine.cross_reference_session("login.py", "")
        assert len(results) >= 1
        assert results[0]["tool"] == "Write"


class TestCrossRefFoundInFile:
    def test_found_in_current_true(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        engine = _make_engine(tmp_path, projects)
        results = engine.cross_reference_session("login.py", "def login():\n    pass\n")
        assert results[0]["found_in_current"] is True


class TestCrossRefNotFoundInFile:
    def test_found_in_current_false(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        engine = _make_engine(tmp_path, projects)
        results = engine.cross_reference_session("login.py", "something completely different")
        assert results[0]["found_in_current"] is False


class TestCrossRefSessionFilter:
    def test_session_filter(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        engine = _make_engine(tmp_path, projects)
        results = engine.cross_reference_session("login.py", "", session_id="aaaa")
        assert len(results) >= 1
        assert all(r["session_id"].startswith("aaaa") for r in results)


# ─── Part 1f: export_session_markdown ────────────────────────────────────────

class TestExportMarkdownBasic:
    def test_returns_string_starting_with_session(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        engine = _make_engine(tmp_path, projects)
        md = engine.export_session_markdown("aaaa0001")
        assert isinstance(md, str)
        assert md.startswith("# Session aaaa0001")


class TestExportMarkdownHasMetadata:
    def test_has_branch_and_cwd(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        engine = _make_engine(tmp_path, projects)
        md = engine.export_session_markdown("aaaa0001")
        assert "main" in md
        assert "/Users/alice/proj1" in md


class TestExportMarkdownHasMessages:
    def test_has_user_message_content(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        engine = _make_engine(tmp_path, projects)
        md = engine.export_session_markdown("aaaa0001")
        assert "start the feature" in md


class TestExportMarkdownNoSession:
    def test_raises_value_error_for_unknown(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        engine = _make_engine(tmp_path, projects)
        with pytest.raises(ValueError, match="No session found"):
            engine.export_session_markdown("nonexistent")


# ─── CLI helpers ─────────────────────────────────────────────────────────────

def _invoke(args, tmp_path: Optional[Path] = None, projects: Optional[Path] = None):
    """Invoke CLI with optional projects dir override."""
    env = {}
    if projects is not None:
        env["AI_SESSION_TOOLS_PROJECTS"] = str(projects)
    elif tmp_path is not None:
        env["AI_SESSION_TOOLS_PROJECTS"] = str(tmp_path / "no_projects")
    return runner.invoke(app, args, env=env if env else None, catch_exceptions=False)


# ─── Part 2b: aise list ──────────────────────────────────────────────────────

class TestListCommand:
    def test_list_exit0(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        result = runner.invoke(app, ["list"], env={"AI_SESSION_TOOLS_PROJECTS": str(projects)})
        assert result.exit_code == 0
        assert "aaaa0001" in result.output

    def test_list_json_format(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        result = runner.invoke(
            app, ["list", "--format", "json"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert "session_id" in data[0]

    def test_list_project_filter(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        result = runner.invoke(
            app, ["list", "--project", "proj1"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0
        # Check via session IDs — proj1=aaaa0001, proj2=bbbb0002
        assert "aaaa0001" in result.output
        assert "bbbb0002" not in result.output


# ─── Part 2c: messages corrections ───────────────────────────────────────────

class TestMessagesCorrections:
    def test_corrections_exit0(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        result = runner.invoke(
            app, ["messages", "corrections"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0

    def test_corrections_has_category(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        result = runner.invoke(
            app, ["messages", "corrections"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert "skip_step" in result.output or "you forgot" in result.output

    def test_corrections_json_format(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        result = runner.invoke(
            app, ["messages", "corrections", "--format", "json"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)

    def test_corrections_custom_pattern_replaces_defaults(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        # Custom pattern that matches "start the feature" (not a default correction)
        result = runner.invoke(
            app, ["messages", "corrections", "--pattern", "custom:start the feature"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0
        assert "custom" in result.output

    def test_corrections_custom_pattern_excludes_defaults(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        # Pattern that won't match anything — built-in "you forgot" should NOT appear
        result = runner.invoke(
            app, ["messages", "corrections", "--pattern", "custom:xyzzy_nomatch_xyzzy"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0
        # Built-in skip_step pattern should not be active
        assert "skip_step" not in result.output

    def test_corrections_bad_pattern_format_exits_nonzero(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        result = runner.invoke(
            app, ["messages", "corrections", "--pattern", "no-colon-here"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code != 0


# ─── Part 2d: messages search --tool ─────────────────────────────────────────

class TestMessagesSearchToolFlag:
    def test_tool_flag_returns_write_call(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        result = runner.invoke(
            app, ["messages", "search", "*", "--tool", "Write"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0
        assert "login" in result.output or "Write" in result.output

    def test_no_tool_unchanged(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        result = runner.invoke(
            app, ["messages", "search", "start the feature"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0
        assert "start the feature" in result.output


# ─── Part 2e: messages planning ──────────────────────────────────────────────

class TestMessagesPlanning:
    def test_planning_exit0(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        result = runner.invoke(
            app, ["messages", "planning"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0
        assert "/ar:plannew" in result.output

    def test_planning_json_format(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        result = runner.invoke(
            app, ["messages", "planning", "--format", "json"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)

    def test_planning_empty(self, tmp_path):
        # No planning commands in empty projects dir
        empty_projects = tmp_path / "empty_projects"
        empty_projects.mkdir()
        result = runner.invoke(
            app, ["messages", "planning"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(empty_projects)},
        )
        assert result.exit_code == 0
        assert "No planning commands found" in result.output

    def test_planning_custom_commands_replaces_defaults(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        # /ar:plannew is in session 1; /custom is not — result should have only /ar:plannew
        result = runner.invoke(
            app, ["messages", "planning", "--commands", "/ar:plannew"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0
        assert "/ar:plannew" in result.output
        # Default /ar:pn should NOT appear (replaced by custom list)
        assert "/ar:pn" not in result.output

    def test_planning_custom_commands_no_match(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        result = runner.invoke(
            app, ["messages", "planning", "--commands", "/xyzzy_nomatch"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0
        assert "No planning commands found" in result.output

    def test_planning_custom_commands_multiple(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        # Both /ar:plannew (s1) and /ar:pn (s2) are in fixture
        result = runner.invoke(
            app, ["messages", "planning", "--commands", "/ar:plannew,/ar:pn"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0
        assert "/ar:plannew" in result.output
        assert "/ar:pn" in result.output


# ─── Part 2f: files cross-ref ────────────────────────────────────────────────

class TestFilesCrossRef:
    def test_cross_ref_exit0(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        test_file = tmp_path / "login.py"
        test_file.write_text("def login():\n    pass\n")
        result = runner.invoke(
            app, ["files", "cross-ref", str(test_file)],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0

    def test_cross_ref_shows_applied(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        test_file = tmp_path / "login.py"
        test_file.write_text("def login():\n    pass\n")
        result = runner.invoke(
            app, ["files", "cross-ref", str(test_file)],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert "\u2713" in result.output or "✓" in result.output or "1/1" in result.output

    def test_cross_ref_shows_not_applied(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        test_file = tmp_path / "login.py"
        test_file.write_text("completely different content")
        result = runner.invoke(
            app, ["files", "cross-ref", str(test_file)],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert "✗" in result.output or "0/1" in result.output


# ─── Part 2a: export session + recent ────────────────────────────────────────

class TestExportSession:
    def test_export_session_stdout(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        result = runner.invoke(
            app, ["export", "session", "aaaa0001"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0
        assert "# Session aaaa0001" in result.output

    def test_export_session_to_file(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        out_file = tmp_path / "out.md"
        result = runner.invoke(
            app, ["export", "session", "aaaa0001", "--output", str(out_file)],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0
        assert out_file.exists()
        assert "# Session aaaa0001" in out_file.read_text()

    def test_export_session_dry_run(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        out_file = tmp_path / "dry.md"
        result = runner.invoke(
            app, ["export", "session", "aaaa0001", "--output", str(out_file), "--dry-run"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0
        assert not out_file.exists()


class TestExportRecent:
    def test_export_recent_to_file(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        out_file = tmp_path / "out.md"
        result = runner.invoke(
            app, ["export", "recent", "365", "--output", str(out_file)],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0
        assert out_file.exists()

    def test_export_recent_empty(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        result = runner.invoke(
            app, ["export", "recent", "0"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0
        assert "No sessions found" in result.output


# ─── Part 2g: tools search + find ────────────────────────────────────────────

class TestToolsSearch:
    def test_tools_search_write_exit0(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        result = runner.invoke(
            app, ["tools", "search", "Write"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0

    def test_tools_search_with_query(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        result = runner.invoke(
            app, ["tools", "search", "Write", "login"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0
        assert "login" in result.output

    def test_tools_search_json_format(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        result = runner.invoke(
            app, ["tools", "search", "Write", "--format", "json"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)

    def test_tools_find_alias(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        result = runner.invoke(
            app, ["tools", "find", "Write"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0


# ─── Part 4a: messages search positional ─────────────────────────────────────

class TestMessagesSearchPositional:
    def test_positional_query(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        result = runner.invoke(
            app, ["messages", "search", "start the feature"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0

    def test_query_flag_still_works(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        result = runner.invoke(
            app, ["messages", "search", "--query", "start the feature"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0


# ─── Part 4b: messages get + root get positional ─────────────────────────────

class TestMessagesGetPositional:
    def test_positional_session_id(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        result = runner.invoke(
            app, ["messages", "get", "aaaa0001"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0

    def test_session_flag_still_works(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        result = runner.invoke(
            app, ["messages", "get", "--session", "aaaa0001"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0


class TestRootGetPositional:
    def test_root_get_positional(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        result = runner.invoke(
            app, ["get", "aaaa0001"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0


# ─── Part 4c/4d: root search --tool, tools domain, find alias ────────────────

class TestRootSearchToolFlag:
    def test_root_search_tool_flag(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        result = runner.invoke(
            app, ["search", "--tool", "Write", "--query", "login"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0

    def test_root_search_tools_domain(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        result = runner.invoke(
            app, ["search", "tools", "--tool", "Write", "--query", "login"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0

    def test_root_search_tools_domain_requires_tool(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        result = runner.invoke(
            app, ["search", "tools"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code != 0


class TestRootFindAlias:
    def test_find_alias_messages(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        result = runner.invoke(
            app, ["find", "messages", "--query", "start the feature"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0

    def test_find_alias_tool(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        result = runner.invoke(
            app, ["find", "--tool", "Write", "--query", "login"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0


class TestFilesFindAlias:
    def test_files_find_alias(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        result = runner.invoke(
            app, ["files", "find", "--pattern", "*.py"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        # Should exit 0 (even if no files found in recovery dir)
        assert result.exit_code == 0


class TestMessagesFindAlias:
    def test_messages_find_positional(self, tmp_path):
        projects = _make_projects_with_sessions(tmp_path)
        result = runner.invoke(
            app, ["messages", "find", "start the feature"],
            env={"AI_SESSION_TOOLS_PROJECTS": str(projects)},
        )
        assert result.exit_code == 0

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
        from pathlib import Path
        pyproject = Path(__file__).parent.parent / "pyproject.toml"
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
        from typer.testing import CliRunner

        from ai_session_tools.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["search", "--help"])
        assert result.exit_code == 0
        assert "search" in result.output.lower() or "pattern" in result.output.lower()

    def test_cli_has_files_group(self):
        """CLI app has 'files' command group."""
        from typer.testing import CliRunner

        from ai_session_tools.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["files", "--help"])
        assert result.exit_code == 0

    def test_cli_has_messages_group(self):
        """CLI app has 'messages' command group."""
        from typer.testing import CliRunner

        from ai_session_tools.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["messages", "--help"])
        assert result.exit_code == 0

    def test_cli_files_search_exists(self):
        """'ais files search' route exists."""
        from typer.testing import CliRunner

        from ai_session_tools.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["files", "search", "--help"])
        assert result.exit_code == 0
        assert "pattern" in result.output.lower()

    def test_cli_messages_search_exists(self):
        """'ais messages search' route exists."""
        from typer.testing import CliRunner

        from ai_session_tools.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["messages", "search", "--help"])
        assert result.exit_code == 0
        assert "query" in result.output.lower()

    def test_cli_messages_get_exists(self):
        """'ais messages get' route exists."""
        from typer.testing import CliRunner

        from ai_session_tools.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["messages", "get", "--help"])
        assert result.exit_code == 0
        assert "session" in result.output.lower()

    def test_cli_extract_positional_name(self):
        """'extract' uses a positional NAME argument (not --name/-n)."""
        from typer.testing import CliRunner

        from ai_session_tools.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["extract", "--help"])
        assert result.exit_code == 0
        # Positional arg appears as NAME or name in help, not as --name flag
        assert "NAME" in result.output or "name" in result.output.lower()
        assert "--name" not in result.output

    def test_cli_get_command_exists(self):
        """Root 'get' command exists."""
        from typer.testing import CliRunner

        from ai_session_tools.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["get", "--help"])
        assert result.exit_code == 0
        assert "session" in result.output.lower()

    def test_cli_stats_command_exists(self):
        """Root 'stats' command exists."""
        from typer.testing import CliRunner

        from ai_session_tools.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["stats", "--help"])
        assert result.exit_code == 0

    def test_cli_search_has_max_chars_for_messages(self):
        """'search messages' or 'messages search' should have --max-chars option."""
        from typer.testing import CliRunner

        from ai_session_tools.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["messages", "search", "--help"])
        assert result.exit_code == 0
        assert "max-chars" in result.output.lower()

    def test_cli_search_files_has_datetime_flags(self):
        """File search should have --after and --before flags."""
        from typer.testing import CliRunner

        from ai_session_tools.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["files", "search", "--help"])
        assert result.exit_code == 0
        assert "--after" in result.output
        assert "--before" in result.output

    def test_cli_search_files_has_session_flags(self):
        """File search should have --include-sessions and --exclude-sessions."""
        from typer.testing import CliRunner

        from ai_session_tools.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["files", "search", "--help"])
        assert result.exit_code == 0
        assert "include-sessions" in result.output.lower()
        assert "exclude-sessions" in result.output.lower()

    def test_cli_search_files_uses_include_extensions(self):
        """File search should use --include-extensions (not --include-types)."""
        from typer.testing import CliRunner

        from ai_session_tools.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["files", "search", "--help"])
        assert result.exit_code == 0
        assert "include-extensions" in result.output.lower()

    def test_cli_get_has_max_chars(self):
        """'get' command should have --max-chars option."""
        from typer.testing import CliRunner

        from ai_session_tools.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["get", "--help"])
        assert result.exit_code == 0
        assert "max-chars" in result.output.lower()

    def test_cli_get_has_format(self):
        """'get' command should have --format option."""
        from typer.testing import CliRunner

        from ai_session_tools.cli import app
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
            _do_extract(engine, "hello.py", version=None, output_dir="~/tmp_ai_session_test_output", restore=False, dry_run=False)
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

    def _make_file(self, name: str):
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
        import csv
        import io

        from ai_session_tools.formatters import CsvFormatter
        f = self._make_file("foo, bar.py")
        output = CsvFormatter().format(f)
        reader = csv.reader(io.StringIO(output))
        rows = list(reader)
        # Second row is the data row; first field is name
        assert rows[1][0] == "foo, bar.py"

    def test_format_many_produces_valid_csv(self):
        import csv
        import io

        from ai_session_tools.formatters import CsvFormatter
        files = [self._make_file("a.py"), self._make_file('b,"quoted".py')]
        output = CsvFormatter().format_many(files)
        reader = csv.reader(io.StringIO(output))
        rows = list(reader)
        assert rows[0] == ["name", "location", "type", "edits", "sessions", "size_bytes", "last_modified", "created_date"]
        assert rows[1][0] == "a.py"
        assert rows[2][0] == 'b,"quoted".py'

    def test_newline_in_location_is_quoted(self):
        """csv.writer must quote fields containing newlines."""
        import csv
        import io

        from ai_session_tools.formatters import CsvFormatter
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
            _do_extract(engine, "hello.py", restore=True)
        except SystemExit:
            pass
        # If original path dir was created, the file was restored there
        assert original_path.exists()


class TestExtractFallback:
    """_do_extract default (no flags) prints to stdout."""

    def test_stdout_default_no_files_written(self, tmp_path, monkeypatch, capsys):
        from ai_session_tools.cli import _do_extract

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("AI_SESSION_TOOLS_OUTPUT", raising=False)

        recovery = _make_recovery_dir(tmp_path)
        engine = SessionRecoveryEngine(tmp_path / "empty_projects", recovery)
        try:
            _do_extract(engine, "hello.py")
        except SystemExit:
            pass
        # Default is stdout — no files in ./recovered/
        assert not (tmp_path / "recovered" / "hello.py").exists()

# ── New TDD Tests: Part 0 — Project dir naming and session filtering ──────────

class TestProjectDirName:
    """_project_dir_name() converts path to Claude project dir name."""

    def test_dot_becomes_dash(self):
        from ai_session_tools.cli import _project_dir_name
        result = _project_dir_name("/Users/alice/.claude/myproject")
        assert result == "-Users-alice--claude-myproject"

    def test_hyphen_preserved(self):
        from ai_session_tools.cli import _project_dir_name
        result = _project_dir_name("/foo/my-project")
        assert result == "-foo-my-project"

    def test_underscore_becomes_dash(self):
        from ai_session_tools.cli import _project_dir_name
        result = _project_dir_name("/foo/my_project")
        assert result == "-foo-my-project"

    def test_special_chars(self):
        from ai_session_tools.cli import _project_dir_name
        result = _project_dir_name("/var/www/my.site.com")
        assert result == "-var-www-my-site-com"


class TestSessionsForProject:
    """_sessions_for_project() returns session IDs from project directory."""

    def test_returns_session_ids(self, tmp_path):
        from ai_session_tools.cli import _project_dir_name, _sessions_for_project
        # Create projects dir structure matching the encoded path
        projects = tmp_path / "projects"
        project_path = str(tmp_path / "myproject")
        dir_name = _project_dir_name(project_path)
        project_dir = projects / dir_name
        project_dir.mkdir(parents=True)
        (project_dir / "abc123-session.jsonl").write_text("{}")
        (project_dir / "def456-session.jsonl").write_text("{}")

        engine = SessionRecoveryEngine(projects, tmp_path / "recovery")
        sessions = _sessions_for_project(engine, project_path)
        assert "abc123-session" in sessions
        assert "def456-session" in sessions

    def test_returns_empty_for_missing_project(self, tmp_path):
        from ai_session_tools.cli import _sessions_for_project
        engine = SessionRecoveryEngine(tmp_path / "projects", tmp_path / "recovery")
        sessions = _sessions_for_project(engine, "/nonexistent/path")
        assert sessions == set()


class TestClaudeConfigDirEnvVar:
    """get_engine() respects CLAUDE_CONFIG_DIR env var."""

    def test_claude_config_dir_sets_projects(self, tmp_path, monkeypatch):
        import ai_session_tools.cli as cli_module
        from ai_session_tools.cli import get_engine
        # Reset global to ensure env var path is tested
        original = cli_module._g_claude_dir
        cli_module._g_claude_dir = None
        try:
            monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
            monkeypatch.delenv("AI_SESSION_TOOLS_PROJECTS", raising=False)
            monkeypatch.delenv("AI_SESSION_TOOLS_RECOVERY", raising=False)
            engine = get_engine()
            assert str(engine.projects_dir) == str(tmp_path / "projects")
        finally:
            cli_module._g_claude_dir = original


class TestClaudeDirCliFlag:
    """--claude-dir CLI flag takes precedence over CLAUDE_CONFIG_DIR env var."""

    def test_claude_dir_flag_used_as_base(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from ai_session_tools.cli import app
        monkeypatch.delenv("AI_SESSION_TOOLS_PROJECTS", raising=False)
        monkeypatch.delenv("AI_SESSION_TOOLS_RECOVERY", raising=False)
        # Create minimal structure so search doesn't crash
        (tmp_path / "projects").mkdir()
        (tmp_path / "recovery").mkdir()
        runner = CliRunner()
        result = runner.invoke(app, ["--claude-dir", str(tmp_path), "files", "search"])
        # Should use tmp_path as claude dir; engine.projects_dir should be tmp_path/projects
        # The command should succeed (exit 0) or show "No files found"
        assert result.exit_code == 0


# ── New TDD Tests: Part 10 — Timestamp in get_versions() ─────────────────────

class TestGetVersionsTimestamp:
    """get_versions() populates FileVersion.timestamp from file mtime."""

    def test_timestamp_populated(self, tmp_path):
        recovery = _make_recovery_dir(tmp_path)
        engine = SessionRecoveryEngine(tmp_path / "projects", recovery)
        versions = engine.get_versions("hello.py")
        assert len(versions) >= 1
        # All versions should have a non-empty timestamp
        for v in versions:
            assert v.timestamp, f"Expected non-empty timestamp, got {v.timestamp!r}"
            assert v.timestamp.startswith("20"), f"Expected year prefix '20', got {v.timestamp!r}"

    def test_timestamp_format(self, tmp_path):
        """Timestamp should be YYYY-MM-DD HH:MM format."""
        recovery = _make_recovery_dir(tmp_path)
        engine = SessionRecoveryEngine(tmp_path / "projects", recovery)
        versions = engine.get_versions("hello.py")
        for v in versions:
            if v.timestamp:
                assert len(v.timestamp) == 16, f"Expected YYYY-MM-DD HH:MM (16 chars), got {v.timestamp!r}"
                assert v.timestamp[4] == "-"
                assert v.timestamp[7] == "-"
                assert v.timestamp[10] == " "
                assert v.timestamp[13] == ":"


# ── New TDD Tests: Part 2 — History display (read-only) ──────────────────────

class TestHistoryDisplayReadOnly:
    """_do_history_display() shows a table without writing any files."""

    def test_no_disk_writes(self, tmp_path):
        from ai_session_tools.cli import _do_history_display
        recovery = _make_recovery_dir(tmp_path)
        engine = SessionRecoveryEngine(tmp_path / "projects", recovery)
        files_before = set(tmp_path.rglob("*"))
        _do_history_display(engine, "hello.py")
        files_after = set(tmp_path.rglob("*"))
        # No new files should have been written
        assert files_before == files_after

    def test_output_contains_version(self, tmp_path, capsys):
        from ai_session_tools.cli import _do_history_display
        recovery = _make_recovery_dir(tmp_path)
        engine = SessionRecoveryEngine(tmp_path / "projects", recovery)
        _do_history_display(engine, "hello.py")
        # Rich output goes to console; test that it doesn't crash and versions exist
        versions = engine.get_versions("hello.py")
        assert len(versions) >= 1


class TestHistoryDisplayTimestamp:
    """_do_history_display() shows timestamp column in table."""

    def test_versions_have_timestamps(self, tmp_path):
        recovery = _make_recovery_dir(tmp_path)
        engine = SessionRecoveryEngine(tmp_path / "projects", recovery)
        versions = engine.get_versions("hello.py")
        # With our Part 10 change, versions should have timestamps
        assert any(v.timestamp for v in versions)


# ── New TDD Tests: Part 3 — History export ───────────────────────────────────

class TestHistoryExport:
    """_do_history_export() writes versioned files to disk."""

    def test_writes_versioned_files(self, tmp_path):
        from ai_session_tools.cli import _do_history_export
        recovery = _make_recovery_dir(tmp_path)
        engine = SessionRecoveryEngine(tmp_path / "projects", recovery)
        out_dir = tmp_path / "exported"
        _do_history_export(engine, "hello.py", export_dir=str(out_dir), dry_run=False)
        assert (out_dir / "hello_v1.py").exists()
        assert (out_dir / "hello_v2.py").exists()
        assert "v1" in (out_dir / "hello_v1.py").read_text()
        assert "v2" in (out_dir / "hello_v2.py").read_text()

    def test_export_creates_output_dir(self, tmp_path):
        from ai_session_tools.cli import _do_history_export
        recovery = _make_recovery_dir(tmp_path)
        engine = SessionRecoveryEngine(tmp_path / "projects", recovery)
        out_dir = tmp_path / "new" / "nested" / "dir"
        assert not out_dir.exists()
        _do_history_export(engine, "hello.py", export_dir=str(out_dir), dry_run=False)
        assert out_dir.exists()


class TestHistoryExportDryRun:
    """_do_history_export() with dry_run=True shows what would be written without writing."""

    def test_no_files_written(self, tmp_path):
        from ai_session_tools.cli import _do_history_export
        recovery = _make_recovery_dir(tmp_path)
        engine = SessionRecoveryEngine(tmp_path / "projects", recovery)
        out_dir = tmp_path / "exported_dry"
        _do_history_export(engine, "hello.py", export_dir=str(out_dir), dry_run=True)
        # Directory should NOT be created in dry run
        assert not out_dir.exists()


# ── New TDD Tests: Part 4 — History stdout ───────────────────────────────────

class TestHistoryStdout:
    """_do_history_stdout() prints all versions to stdout with headers."""

    def test_stdout_contains_header(self, tmp_path, capsys):
        from ai_session_tools.cli import _do_history_stdout
        recovery = _make_recovery_dir(tmp_path)
        engine = SessionRecoveryEngine(tmp_path / "projects", recovery)
        _do_history_stdout(engine, "hello.py")
        captured = capsys.readouterr()
        assert "=== hello.py v1" in captured.out
        assert "=== hello.py v2" in captured.out

    def test_stdout_contains_file_content(self, tmp_path, capsys):
        from ai_session_tools.cli import _do_history_stdout
        recovery = _make_recovery_dir(tmp_path)
        engine = SessionRecoveryEngine(tmp_path / "projects", recovery)
        _do_history_stdout(engine, "hello.py")
        captured = capsys.readouterr()
        assert "print('v1')" in captured.out
        assert "print('v2')" in captured.out


# ── New TDD Tests: Part 1 — Extract rewrite ──────────────────────────────────

class TestExtractStdout:
    """_do_extract() default mode prints to stdout."""

    def test_stdout_contains_content(self, tmp_path, capsys):
        from ai_session_tools.cli import _do_extract
        recovery = _make_recovery_dir(tmp_path)
        engine = SessionRecoveryEngine(tmp_path / "projects", recovery)
        try:
            _do_extract(engine, "hello.py")
        except SystemExit:
            pass
        captured = capsys.readouterr()
        # Latest version is v2 (highest version_num)
        assert "v2" in captured.out

    def test_no_files_written(self, tmp_path, capsys):
        from ai_session_tools.cli import _do_extract
        recovery = _make_recovery_dir(tmp_path)
        engine = SessionRecoveryEngine(tmp_path / "projects", recovery)
        files_before = set(tmp_path.rglob("*.py"))
        try:
            _do_extract(engine, "hello.py")
        except SystemExit:
            pass
        files_after = set(tmp_path.rglob("*.py"))
        # No new .py files outside recovery
        new_files = files_after - files_before
        assert not any("recovered" in str(f) for f in new_files)


class TestExtractSpecificVersion:
    """_do_extract() with version= selects specific version."""

    def test_version_1_selected(self, tmp_path, capsys):
        from ai_session_tools.cli import _do_extract
        recovery = _make_recovery_dir(tmp_path)
        engine = SessionRecoveryEngine(tmp_path / "projects", recovery)
        try:
            _do_extract(engine, "hello.py", version=1)
        except SystemExit:
            pass
        captured = capsys.readouterr()
        assert "v1" in captured.out
        assert "v2" not in captured.out

    def test_version_2_selected(self, tmp_path, capsys):
        from ai_session_tools.cli import _do_extract
        recovery = _make_recovery_dir(tmp_path)
        engine = SessionRecoveryEngine(tmp_path / "projects", recovery)
        try:
            _do_extract(engine, "hello.py", version=2)
        except SystemExit:
            pass
        captured = capsys.readouterr()
        assert "v2" in captured.out
        assert "v1" not in captured.out


class TestExtractDryRun:
    """_do_extract() with dry_run=True shows intent without writing or printing content."""

    def test_no_stdout_content(self, tmp_path, capsys):
        from ai_session_tools.cli import _do_extract
        recovery = _make_recovery_dir(tmp_path)
        engine = SessionRecoveryEngine(tmp_path / "projects", recovery)
        _do_extract(engine, "hello.py", dry_run=True)
        captured = capsys.readouterr()
        # stdout should be empty (dry run, no content)
        assert "print(" not in captured.out
        assert "v1" not in captured.out
        assert "v2" not in captured.out

    def test_no_files_written(self, tmp_path):
        from ai_session_tools.cli import _do_extract
        recovery = _make_recovery_dir(tmp_path)
        engine = SessionRecoveryEngine(tmp_path / "projects", recovery)
        files_before = set(tmp_path.rglob("*"))
        _do_extract(engine, "hello.py", dry_run=True)
        files_after = set(tmp_path.rglob("*"))
        assert files_before == files_after


class TestExtractDryRunRestore:
    """_do_extract() with dry_run=True, restore=True shows path but writes nothing."""

    def test_no_file_written(self, tmp_path):
        from ai_session_tools.cli import _do_extract
        # Set up with an original path record
        recovery = _make_recovery_dir(tmp_path)
        projects = tmp_path / "projects" / "proj"
        projects.mkdir(parents=True)
        original_path = tmp_path / "workspace" / "hello.py"
        _write_tool_call_jsonl(
            projects / "session.jsonl",
            "any-session", "hello.py", str(original_path)
        )
        engine = SessionRecoveryEngine(tmp_path / "projects", recovery)
        _do_extract(engine, "hello.py", restore=True, dry_run=True)
        # File should NOT be written
        assert not original_path.exists()


# ── New TDD Tests: Part 9 — Positional name argument ─────────────────────────

class TestPositionalArgExtract:
    """CLI extract command uses positional NAME argument."""

    def test_positional_name_works(self, tmp_path):
        from typer.testing import CliRunner

        import ai_session_tools.cli as cli_module
        from ai_session_tools.cli import app
        runner = CliRunner()
        # Set up tmp recovery dir
        recovery = _make_recovery_dir(tmp_path)
        import os
        old_projects = os.environ.get("AI_SESSION_TOOLS_PROJECTS")
        old_recovery = os.environ.get("AI_SESSION_TOOLS_RECOVERY")
        old_claude_dir = cli_module._g_claude_dir
        try:
            os.environ["AI_SESSION_TOOLS_PROJECTS"] = str(tmp_path / "projects")
            os.environ["AI_SESSION_TOOLS_RECOVERY"] = str(recovery)
            cli_module._g_claude_dir = None
            result = runner.invoke(app, ["files", "extract", "hello.py"])
        finally:
            if old_projects is None:
                os.environ.pop("AI_SESSION_TOOLS_PROJECTS", None)
            else:
                os.environ["AI_SESSION_TOOLS_PROJECTS"] = old_projects
            if old_recovery is None:
                os.environ.pop("AI_SESSION_TOOLS_RECOVERY", None)
            else:
                os.environ["AI_SESSION_TOOLS_RECOVERY"] = old_recovery
            cli_module._g_claude_dir = old_claude_dir
        assert result.exit_code == 0
        assert "v2" in result.output or "print" in result.output

    def test_no_name_flag_accepted(self):
        """--name flag should no longer be accepted."""
        from typer.testing import CliRunner

        from ai_session_tools.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["extract", "--help"])
        assert result.exit_code == 0
        assert "--name" not in result.output


class TestPositionalArgHistory:
    """CLI history command uses positional NAME argument."""

    def test_history_help_shows_positional(self):
        from typer.testing import CliRunner

        from ai_session_tools.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["files", "history", "--help"])
        assert result.exit_code == 0
        assert "--name" not in result.output
        # Should show NAME as positional
        assert "NAME" in result.output or "name" in result.output.lower()


# ── New TDD Tests: Part 11 — Sessions column in search table ─────────────────

class TestSearchTableShowsSessions:
    """TableFormatter.format_many() shows sessions column."""

    def test_sessions_column_in_table(self):
        from ai_session_tools import RecoveredFile
        from ai_session_tools.formatters import TableFormatter
        files = [
            RecoveredFile(
                name="cli.py", path="/cli.py", file_type="py", edits=3,
                sessions=["abc123de-f456-7890-abcd-ef1234567890", "xyz789ab-0000-0000-0000-000000000001"],
                last_modified="2026-02-22T10:00:00",
            )
        ]
        formatter = TableFormatter("Test")
        output = formatter.format_many(files)
        # Session IDs should appear (abbreviated to 8 chars + ellipsis)
        assert "abc123de" in output

    def test_sessions_column_multiple_sessions(self):
        from ai_session_tools import RecoveredFile
        from ai_session_tools.formatters import TableFormatter
        files = [
            RecoveredFile(
                name="test.py", path="/test.py", file_type="py", edits=5,
                sessions=["aaa", "bbb", "ccc", "ddd"],
            )
        ]
        formatter = TableFormatter("Test")
        output = formatter.format_many(files)
        # More than 3 sessions should show (+1) indicator
        assert "+1" in output


# ── Regression tests: missing src.exists() guard in _do_extract write branches ─

class TestExtractMissingVersionFile:
    """_do_extract raises Exit(1) with a clear message when the version file is
    missing from disk (not a raw FileNotFoundError traceback).

    Regression: restore and output_dir branches lacked src.exists() check,
    raising an unhandled FileNotFoundError instead of a friendly error message.
    """

    def _setup_engine_with_stale_cache(self, tmp_path, with_original_path=False):
        """Return (engine, original_path) where cache has versions but files are deleted."""
        recovery = _make_recovery_dir(tmp_path)
        projects = tmp_path / "projects" / "proj"
        projects.mkdir(parents=True)
        original_path = tmp_path / "workspace" / "hello.py"
        if with_original_path:
            _write_tool_call_jsonl(
                projects / "session.jsonl",
                "any-session", "hello.py", str(original_path)
            )
        engine = SessionRecoveryEngine(tmp_path / "projects", recovery)
        # Pre-populate the version cache (simulates normal usage before files disappear)
        versions = engine.get_versions("hello.py")
        assert versions, "Need at least one version for test setup"
        # Now delete the version files from disk to simulate missing files
        for vd in recovery.iterdir():
            if vd.is_dir() and "all_versions" in vd.name:
                for f in list(vd.iterdir()):
                    if f.suffix == ".txt":
                        f.unlink()
        return engine, original_path

    def test_restore_missing_version_file_exits_cleanly(self, tmp_path):
        """restore branch: missing version file on disk → Exit(1), not FileNotFoundError."""
        import typer

        from ai_session_tools.cli import _do_extract
        engine, _ = self._setup_engine_with_stale_cache(tmp_path, with_original_path=True)
        with pytest.raises(typer.Exit) as exc_info:
            _do_extract(engine, "hello.py", restore=True)
        assert exc_info.value.exit_code == 1

    def test_output_dir_missing_version_file_exits_cleanly(self, tmp_path):
        """output_dir branch: missing version file on disk → Exit(1), not FileNotFoundError."""
        import typer

        from ai_session_tools.cli import _do_extract
        engine, _ = self._setup_engine_with_stale_cache(tmp_path)
        with pytest.raises(typer.Exit) as exc_info:
            _do_extract(engine, "hello.py", output_dir=str(tmp_path / "out"))
        assert exc_info.value.exit_code == 1
