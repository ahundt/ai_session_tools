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


_DEFAULT_STOP_WORDS = frozenset({
    "the", "this", "that", "and", "is", "for", "with", "from", "you", "are",
    "into", "of", "to", "a", "in", "it", "as", "be", "an", "or", "on", "at",
    "by", "we", "i", "can", "but", "not", "so", "if", "do", "its", "all",
    "my", "me", "have", "has", "was", "will", "your", "our", "they", "them",
    "also", "then", "than", "when", "just", "up", "out", "about",
})


def load_stop_words(org_dir: Path) -> frozenset[str]:
    """Load stop words from stop_words.json. Returns module default if absent.

    File: <org_dir>/stop_words.json  — list of lowercase words to exclude from n-grams.
    """
    path = org_dir / "stop_words.json"
    with contextlib.suppress(OSError, json.JSONDecodeError):
        data = json.loads(path.read_text(encoding="utf-8"))
        words = data.get("stop_words", [])
        if words:
            return frozenset(w.lower() for w in words)
    return _DEFAULT_STOP_WORDS


def is_meaningful(phrase: str, stop_words: frozenset[str] | None = None) -> bool:
    """Filter out phrases that start with or consist entirely of stop words.

    stop_words: loaded via load_stop_words(org_dir). Falls back to _DEFAULT_STOP_WORDS.
    """
    sw = stop_words if stop_words is not None else _DEFAULT_STOP_WORDS
    words = phrase.split()
    if not words:
        return False
    if words[0] in sw:
        return False
    if all(w in sw for w in words):
        return False
    return True


# Patterns that identify code/config content — auto-detected, no hardcoded keywords
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_INDENTED_CODE_RE = re.compile(r"(?:^|\n)((?:    |\t)[^\n]+(?:\n(?:    |\t)[^\n]+)*)")
_CODE_LINE_RE = re.compile(
    r"^\s*(?:"
    r"(?:import|from|def|class|return|if|elif|else|for|while|try|except|with|async|await)\b"
    r"|[a-zA-Z_]\w*\s*[=\(]"  # assignment or function call
    r"|\{|\[|</"             # JSON/dict/array start or closing HTML/XML tag
    r"|#\s*\w"               # comment line
    r")",
    re.MULTILINE,
)


def extract_prose(text: str) -> str:
    """Strip code blocks and code-heavy lines; return prose-only text.

    Auto-detects code via:
    - Fenced code blocks (``` ... ```)
    - Indented blocks (4 spaces or tab)
    - Lines with Python/JS/config syntax signatures

    No configuration knobs — detection is structural, not keyword-based.
    """
    # Remove fenced code blocks first (most reliable signal)
    prose = _FENCED_CODE_RE.sub(" ", text)
    # Remove indented code blocks
    prose = _INDENTED_CODE_RE.sub(" ", prose)
    # Filter remaining lines that look like code
    lines = []
    for line in prose.splitlines():
        if line.strip() and _CODE_LINE_RE.match(line):
            continue  # skip code-looking lines
        lines.append(line)
    return "\n".join(lines)


def prose_fraction(text: str) -> float:
    """Return fraction of characters that are prose (not code), 0.0-1.0.

    Returns 1.0 for empty text (no code to detect).
    """
    if not text:
        return 1.0
    prose = extract_prose(text)
    return len(prose) / len(text)


def load_continuation_config(org_dir: Path) -> tuple[list[str], int]:
    """Load continuation prompt detection config from continuation_markers.json.

    Returns (prefix_markers, min_initial_len) — both from config file.
    File: <org_dir>/continuation_markers.json

    If absent, returns empty list + 0 so caller falls back to length-only detection.
    Putting markers in a JSON file makes them discoverable, editable, and version-controlled
    without touching source code.
    """
    path = org_dir / "continuation_markers.json"
    with contextlib.suppress(OSError, json.JSONDecodeError):
        data = json.loads(path.read_text(encoding="utf-8"))
        markers = data.get("prefix_markers", [])
        min_len = int(data.get("min_initial_len", 0))
        return markers, min_len
    return [], 0


def classify_prompt_role(
    message_text: str,
    is_first_in_session: bool = True,
    continuation_markers: list[str] | None = None,
    min_initial_len: int = 0,
) -> str:
    """Classify a user message as 'initial', 'continuation', or 'standalone'.

    Returns:
        'initial'      — substantive opening prompt (first in session, long enough, no continuation signal)
        'continuation' — builds on prior context (short or starts with configured marker)
        'standalone'   — caller should set directly; not computed here

    Parameters:
        continuation_markers: list of word/phrase strings loaded from continuation_markers.json.
                               If empty/None, only message length is used as a signal.
        min_initial_len:       minimum chars for a message to be classified 'initial'.
                               0 = use length signal only if continuation_markers also matches.
                               Loaded from continuation_markers.json["min_initial_len"].
    """
    text = message_text.strip()

    is_short = (min_initial_len > 0 and len(text) < min_initial_len)

    is_continuation_opener = False
    if continuation_markers:
        pattern = re.compile(
            r"^\s*(?:" + "|".join(r"\b" + re.escape(m) for m in continuation_markers) + r")\b",
            re.IGNORECASE,
        )
        is_continuation_opener = bool(pattern.match(text))

    if is_first_in_session and not is_continuation_opener and not is_short:
        return "initial"
    if is_short or is_continuation_opener:
        return "continuation"
    return "initial"
