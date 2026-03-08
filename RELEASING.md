# Releasing ai_session_tools

## Version number locations (keep in sync)

| File | Location |
|------|----------|
| `pyproject.toml` | line 7: `version = "X.Y.Z"` |
| `ai_session_tools/__init__.py` | line 61: fallback `__version__ = "X.Y.Z"` |

The canonical version is read at runtime via `importlib.metadata.version("ai_session_tools")` (set by the installed package metadata from `pyproject.toml`). The fallback in `__init__.py` is only used when the package is not installed (e.g. running directly from source without `uv tool install`).

## Version bump checklist

1. Update `pyproject.toml` `version = "X.Y.Z"`
2. Update `ai_session_tools/__init__.py` fallback `__version__ = "X.Y.Z"`
3. Run full tests: `uv run pytest tests/ -q -m 'not integration'`
4. Commit: `git commit -m "chore(version): bump to X.Y.Z"`
5. Reinstall: `uv tool install -e .`
6. Verify: `aise --version`  # → "aise X.Y.Z"
7. Tag: `git tag vX.Y.Z`
8. Push: `git push origin main --tags`
