# Releasing ai_session_tools

## Version number locations (keep in sync)

| File | Location | Example |
|------|----------|---------|
| `pyproject.toml` | line 7: `version = "X.Y.Z"` | `version = "0.3.0"` |
| `ai_session_tools/__init__.py` | line 22: fallback `__version__ = "X.Y.Z"` | `__version__ = "0.3.0"` |

The canonical version is read at runtime via `importlib.metadata.version("ai_session_tools")` (set by the installed package metadata from `pyproject.toml`). The fallback in `__init__.py` is only used when the package is not installed (e.g. running directly from source without `uv tool install`).

## Version bump checklist

1. Update `pyproject.toml` `version = "X.Y.Z"`
2. Update `ai_session_tools/__init__.py` fallback `__version__ = "X.Y.Z"`
3. Run full tests: `uv run pytest tests/ -q -m 'not integration'`
4. Commit: `git commit -m "chore(version): bump to X.Y.Z"`
5. Reinstall: `uv tool install -e .`
6. Verify: `aise --version`  # → "aise X.Y.Z"
