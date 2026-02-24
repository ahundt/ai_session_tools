"""
Shared codebook and keyword loading utilities.

DRY: imported by analyzer.py, orchestrator.py, and vocab.py.
Never duplicate these functions across analysis modules.

Copyright (c) 2026 Andrew Hundt
Licensed under the Apache License, Version 2.0
"""
from __future__ import annotations

import contextlib
import json
import re
from collections import Counter
from pathlib import Path


def load_codebook(org_dir: Path) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Parse CODEBOOK.md → (tech_codes, role_codes). Non-destructive read only.

    Returns:
        (tech_codes, role_codes): dicts mapping code_name -> [marker_strings]
    """
    tech_codes: dict[str, list[str]] = {}
    role_codes: dict[str, list[str]] = {}
    codebook_path = org_dir / "CODEBOOK.md"
    if not codebook_path.exists():
        return tech_codes, role_codes

    in_roles = False
    with contextlib.suppress(OSError):
        for line in codebook_path.read_text(encoding="utf-8").splitlines():
            if "## 2. Expert Role" in line:
                in_roles = True
            if "|" not in line or "`" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 4:
                continue
            code = parts[1].replace("`", "").strip()
            markers_raw = parts[3] if len(parts) > 3 else ""
            markers = [m.strip().replace("`", "") for m in markers_raw.split(",") if m.strip()]
            if code and markers:
                if in_roles:
                    role_codes[code] = markers
                else:
                    tech_codes[code] = markers
    return tech_codes, role_codes


def load_keyword_maps(org_dir: Path) -> dict[str, dict[str, list[str]]]:
    """Load all external keyword map files. WOLOG: no hardcoding, all from config files.

    Files: task_categories.json, writing_methods.json, project_map.json, workflow_map.json
    """
    maps: dict[str, dict[str, list[str]]] = {}
    for name in ("task_categories", "writing_methods", "project_map", "workflow_map"):
        path = org_dir / f"{name}.json"
        with contextlib.suppress(OSError, json.JSONDecodeError):
            maps[name] = json.loads(path.read_text(encoding="utf-8"))
    return maps


def compile_codes(codes: dict[str, list[str]], min_marker_len: int = 5) -> dict[str, re.Pattern[str]]:
    """Pre-compile codebook markers to regex patterns with word boundaries.

    Complexity: O(K*M) once at startup, then O(K*T) per session.
    min_marker_len: skip markers shorter than this to avoid false positives.
    Word boundaries (\b) prevent partial-word matches.
    """
    patterns = {}
    for code, markers in codes.items():
        valid = [m for m in markers if len(m.strip()) >= min_marker_len]
        if valid:
            pattern_str = "|".join(r"\b" + re.escape(m) for m in valid)
            patterns[code] = re.compile(pattern_str, re.IGNORECASE)
    return patterns


def get_ngrams(text: str, n: int) -> list[str]:
    """Extract n-word phrases from text. Cleans JSON escape sequences."""
    text = text.replace("\\n", " ").replace("\\u0027", "'").replace('\\"', '"')
    words = re.findall(r"[a-z][a-z0-9\-]*", text.lower())
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


_STOP_WORDS = frozenset({
    "the", "this", "that", "and", "is", "for", "with", "from", "you", "are",
    "into", "of", "to", "a", "in", "it", "as", "be", "an", "or", "on", "at",
    "by", "we", "i", "can", "but", "not", "so", "if", "do", "its", "all",
    "my", "me", "have", "has", "was", "will", "your", "our", "they", "them",
    "also", "then", "than", "when", "just", "up", "out", "about",
})


def is_meaningful(phrase: str) -> bool:
    """Filter out phrases that start with or consist entirely of stop words."""
    words = phrase.split()
    if not words:
        return False
    if words[0] in _STOP_WORDS:
        return False
    if all(w in _STOP_WORDS for w in words):
        return False
    return True
