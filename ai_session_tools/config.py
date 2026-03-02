"""Canonical config loading for ai_session_tools.

Respects priority: --config CLI flag > AI_SESSION_TOOLS_CONFIG env var > OS default.
Used by ALL modules (cli.py, analysis/*, sources/*) for consistent config handling.

Copyright (c) 2026 Andrew Hundt
Licensed under the Apache License, Version 2.0
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

# Module-level state set by CLI app_callback
_g_config_path: str | None = None
_config_cache: dict | None = None
_config_cache_key: str | None = None  # resolved path that produced the cache


def _resolved_config_key() -> str:
    """Return a string key identifying the current config source (for cache validation).

    Changes when --config flag, AI_SESSION_TOOLS_CONFIG env var, or OS default changes.
    load_config() compares this to _config_cache_key and auto-invalidates on mismatch.
    """
    if _g_config_path:
        return f"cli:{_g_config_path}"
    env_p = os.getenv("AI_SESSION_TOOLS_CONFIG")
    if env_p:
        return f"env:{env_p}"
    return "default"


def set_config_path(path: str | None) -> None:
    """Set config path from --config CLI flag. Called from app_callback.

    When path changes, cache is invalidated so load_config() re-reads.
    """
    global _g_config_path, _config_cache, _config_cache_key
    if path != _g_config_path:
        _g_config_path = path
        _config_cache = None       # invalidate cache on path change
        _config_cache_key = None


def invalidate_config_cache() -> None:
    """Force load_config() to re-read from disk on the next call.

    Call this after writing a new config file (e.g. after config init or source add)
    so the next load_config() picks up the updated file contents.
    """
    global _config_cache, _config_cache_key
    _config_cache = None
    _config_cache_key = None


def load_config() -> dict:
    """Load config respecting priority: --config flag > env var > OS default.

    Priority order:
      1. _g_config_path (from --config CLI flag, set by app_callback)
      2. AI_SESSION_TOOLS_CONFIG env var (treated as FILE path)
      3. OS-appropriate default via typer.get_app_dir():
         - macOS: ~/Library/Application Support/ai_session_tools/config.json
         - Linux: ~/.config/ai_session_tools/config.json
         - Windows: %APPDATA%/ai_session_tools/config.json

    Returns empty dict {} if config file does not exist or is unreadable.
    Uses in-memory cache to avoid repeated file reads within a process.
    """
    global _config_cache, _config_cache_key
    current_key = _resolved_config_key()
    if _config_cache is not None and _config_cache_key == current_key:
        return _config_cache

    if _g_config_path:
        config_path = Path(_g_config_path).expanduser()
    elif env_p := os.getenv("AI_SESSION_TOOLS_CONFIG"):
        # env var is the FILE path (not directory)
        config_path = Path(env_p).expanduser()
    else:
        # OS-appropriate default via typer.get_app_dir
        import typer
        config_dir = Path(typer.get_app_dir("ai_session_tools"))
        config_path = config_dir / "config.json"

    with contextlib.suppress(OSError, json.JSONDecodeError):
        content = config_path.read_text(encoding="utf-8")
        _config_cache = json.loads(content)
        _config_cache_key = current_key
        return _config_cache

    # File missing/unreadable: cache empty dict keyed to this path so we don't re-hit disk
    _config_cache = {}
    _config_cache_key = current_key
    return {}


def get_config_path() -> Path:
    """Return the config file path using the same priority chain as load_config().

    Priority: --config CLI flag > AI_SESSION_TOOLS_CONFIG env var > OS default.
    The returned path may not exist yet (caller creates it on first write).
    """
    if _g_config_path:
        return Path(_g_config_path).expanduser()
    if env_p := os.getenv("AI_SESSION_TOOLS_CONFIG"):
        return Path(env_p).expanduser()
    import typer
    return Path(typer.get_app_dir("ai_session_tools")) / "config.json"


def write_config(cfg: dict) -> None:
    """Write cfg to the config file and update the in-process cache.

    Creates parent directories if needed. Updates _config_cache so the next
    load_config() call within this process returns the written dict without
    an extra disk read.

    Callers that need to persist auto-discovered source paths or clear stale
    cache entries should use this function rather than writing directly.
    """
    global _config_cache, _config_cache_key
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    _config_cache = cfg                      # keep in-process cache current
    _config_cache_key = _resolved_config_key()  # update key so cache stays valid


def get_config_section(key: str, default=None):
    """Return config[key] if present and non-empty, else default.

    Empty containers ([], {}) and empty strings are treated as absent so callers
    fall through to their file-based or module-level defaults.
    """
    cfg = load_config()
    if key in cfg:
        val = cfg[key]
        if val not in (None, "", [], {}):
            return val
    return default
