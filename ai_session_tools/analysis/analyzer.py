"""
Session Content Analysis Engine - backs `aise analyze`.

Single-pass streaming pipeline: qualitative coding + vocabulary mining simultaneously.
Reads all sessions from all three source directories. Writes session_db.json + VOCABULARY_ANALYSIS.md.

METHODOLOGICAL REFERENCES:
- Hsieh & Shannon (2005): https://journals.sagepub.com/doi/10.1177/1049732305276687
- Wei et al. (2022): https://arxiv.org/abs/2201.11903

Copyright (c) 2026 Andrew Hundt
Licensed under the Apache License, Version 2.0
"""
from __future__ import annotations

import contextlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

from ai_session_tools.sources.aistudio import AiStudioSource
from ai_session_tools.config import load_config
from ai_session_tools.analysis.codebook import (
    load_codebook, load_keyword_maps, compile_codes, get_ngrams, is_meaningful
)


@dataclass
class SessionRecord:
    """Analysis record for one session. user_text excluded from DB serialization.

    user_text: in-memory only during pipeline. NOT serialized to session_db.json.
    Use to_db_dict() for persistent storage.
    Memory: O(text_len) per session, GC'd after coding + vocabulary accumulation.
    """
    name: str
    source_dir: str
    filepath: str
    source_format: str       # 'aistudio_json' | 'markdown' | 'gemini_cli' | 'claude_jsonl'
    user_text: str           # in-memory only
    chunk_count: int
    user_chunk_count: int
    techniques: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    task_categories: list[str] = field(default_factory=list)
    writing_methods: list[str] = field(default_factory=list)
    rigor_score: int = 0
    utility: int = 0
    version_num: int | None = None
    is_branch: bool = False
    is_copy: bool = False
    graph_parent: str | None = None
    era: str = ""
    has_srt: bool = False
    has_transcript: bool = False
    project_hash: str = ""

    @property
    def user_text_full(self) -> str:
        return self.user_text

    def user_text_sample(self, max_chars: int) -> str:
        return self.user_text[:max_chars]

    def to_db_dict(self) -> dict:
        """Serialize for session_db.json — excludes user_text. Stores ~/... paths (no PII)."""
        d = asdict(self)
        d.pop("user_text", None)
        home = str(Path.home())
        for key in ("source_dir", "filepath"):
            val = d.get(key, "")
            if val and val.startswith(home):
                d[key] = "~" + val[len(home):]
        return d


def _detect_era(name: str, user_text: str) -> str:
    """Detect era from session filename or content year signals.

    Strategy (priority order):
    1. 4-digit year at start of name (e.g. "2024-03-meeting")
    2. Standalone year anywhere in name (e.g. "Meeting Notes 2024")
    3. Year in first 2000 chars of user_text context (e.g. "as of 2024")
    4. Default: 2025-2026
    """
    # Priority 1: year prefix in name
    m = re.match(r"(202[3-6])", name)
    if m:
        yr = m.group(1)
        if yr in ("2023", "2024"):
            return yr
        return "2025-2026"

    # Priority 2: standalone year anywhere in name
    m2 = re.search(r"\b(202[3-6])\b", name)
    if m2:
        yr = m2.group(1)
        if yr in ("2023", "2024"):
            return yr
        return "2025-2026"

    # Priority 3: year in early content
    sample = user_text[:2000]
    for yr in ("2024", "2023"):
        if re.search(r"\b" + yr + r"\b", sample):
            return yr

    return "2025-2026"


def apply_codes(
    rec: SessionRecord,
    tech_patterns: dict[str, re.Pattern[str]],
    role_patterns: dict[str, re.Pattern[str]],
    keyword_maps: dict[str, dict[str, list[str]]],
    scoring_weights: dict[str, int],
    marker_window: int = 25_000,
) -> None:
    """Apply codebook codes using pre-compiled regex patterns.

    Complexity: O(K*T) per session (K=codes, T=marker_window chars).
    All weights from scoring_weights dict (not hardcoded).
    Implements Directed Content Analysis (Hsieh & Shannon, 2005).
    """
    text = rec.user_text_full[:marker_window]
    lower = text.lower()

    w_technique = scoring_weights.get("technique", 20)
    w_role = scoring_weights.get("role", 15)
    w_thinking = scoring_weights.get("thinking_budget", 30)
    w_anti_ai = scoring_weights.get("anti_ai", 35)
    w_version = scoring_weights.get("version_multiplier", 10)
    w_corrected = scoring_weights.get("corrected_bonus", 25)

    for tech, pattern in tech_patterns.items():
        if pattern.search(text):
            rec.techniques.append(tech)
            rec.rigor_score += w_technique

    for role, pattern in role_patterns.items():
        if pattern.search(text):
            rec.roles.append(role)
            rec.rigor_score += w_role

    # Task categories from external task_categories.json
    for cat, kws in keyword_maps.get("task_categories", {}).items():
        if any(k in lower for k in kws):
            rec.task_categories.append(cat)

    # Writing methods from external writing_methods.json
    for method, kws in keyword_maps.get("writing_methods", {}).items():
        if any(k in lower for k in kws):
            rec.writing_methods.append(method)

    if "thinkingbudget" in lower or "thinking_budget" in lower:
        rec.rigor_score += w_thinking
    if "anti-ai" in lower or "wikipedia_signs_of_ai" in lower:
        rec.rigor_score += w_anti_ai

    v_match = re.search(r"\bv(\d+)\b", rec.name, re.I)
    if v_match:
        rec.version_num = int(v_match.group(1))
        rec.rigor_score += rec.version_num * w_version

    if "corrected" in rec.name.lower() or "improved" in rec.name.lower():
        rec.rigor_score += w_corrected

    name_lower = rec.name.lower()
    if "branch of " in name_lower:
        rec.is_branch = True
        rec.graph_parent = re.sub(r"(?i)branch of\s*", "", rec.name).strip()
    elif "copy of " in name_lower:
        rec.is_copy = True
        rec.graph_parent = re.sub(r"(?i)copy of\s*", "", rec.name).strip()
    else:
        v_chain = re.search(r"^(.*?)\s+v(\d+)\s*$", rec.name, re.I)
        if v_chain and int(v_chain.group(2)) > 1:
            prev_v = int(v_chain.group(2)) - 1
            rec.graph_parent = f"{v_chain.group(1).strip()} v{prev_v}"

    rec.utility = rec.rigor_score


def compute_descendant_boost(records: list[SessionRecord], boost_per_descendant: int = 15) -> None:
    """Add utility boost to ROOT sessions that spawned descendants.

    Implements provenance-based scoring (SAGE/Nature digital archiving).
    Older roots of version chains are valued MORE, not less (MSG 128).
    """
    name_to_rec = {r.name: r for r in records}
    for rec in records:
        if rec.graph_parent and rec.graph_parent in name_to_rec:
            name_to_rec[rec.graph_parent].utility += boost_per_descendant


def write_vocab_report(tri: Counter[str], quad: Counter[str], output_file: Path) -> None:
    """Write vocabulary analysis to markdown. No arbitrary truncation."""
    tri_rows = [(freq, phrase) for phrase, freq in tri.most_common()
                if freq >= 3 and is_meaningful(phrase)]
    quad_rows = [(freq, phrase) for phrase, freq in quad.most_common()
                 if freq >= 3 and is_meaningful(phrase)]

    lines = [
        "# Vocabulary Analysis: Recurring Prompt Phrases\n\n",
        "N-gram analysis of user turns across all AI Studio sessions.\n",
        "Source: PromptEngineering.org — https://www.promptengineering.org/building-a-reusable-prompt-library/\n\n",
        f"## 3-Word Phrases ({len(tri_rows)} total with freq >= 3)\n\n",
        "| Count | Phrase |\n| :--- | :--- |\n",
    ]
    lines.extend(f"| {freq} | {phrase} |\n" for freq, phrase in tri_rows)
    lines += [
        f"\n## 4-Word Phrases ({len(quad_rows)} total with freq >= 3)\n\n",
        "| Count | Phrase |\n| :--- | :--- |\n",
    ]
    lines.extend(f"| {freq} | {phrase} |\n" for freq, phrase in quad_rows)

    output_file.write_text("".join(lines), encoding="utf-8")
    print(f"Vocabulary: {len(tri_rows)} trigrams, {len(quad_rows)} quadgrams -> {output_file}")


def _filter_config_by_source(cfg: dict, source_filter: str) -> dict:
    """Filter config to include only specified source backend.

    Args:
        cfg: Configuration dict
        source_filter: 'aistudio', 'gemini', or None (all)

    Returns:
        Filtered config dict with only requested sources in source_dirs
    """
    filtered = dict(cfg)
    if not source_filter:
        return filtered

    sd = filtered.get("source_dirs", {})
    new_sd = {}

    if source_filter in ("aistudio", "all"):
        if "aistudio" in sd:
            new_sd["aistudio"] = sd["aistudio"]

    if source_filter in ("gemini", "all"):
        if "gemini_cli" in sd:
            new_sd["gemini_cli"] = sd["gemini_cli"]

    filtered["source_dirs"] = new_sd
    return filtered


def run_analysis(
    marker_window: int | None = None,
    source_filter: str | None = None,
    config: dict | None = None,
) -> list[SessionRecord]:
    """Single-pass streaming analysis: code + vocabulary simultaneously.

    Args:
        marker_window: Chars for marker matching (0 = from config)
        source_filter: Narrow to one backend: 'aistudio', 'gemini', or None (all)
        config: Config dict (if None, loads from config.json)

    WOLOG: all paths from config; no hardcoded truncation.
    Single source scan, O(1) memory per session (generator).
    """
    if config is None:
        cfg = load_config()
    else:
        cfg = config

    # Filter sources based on source_filter parameter
    if source_filter:
        cfg = _filter_config_by_source(cfg, source_filter)
    source_dirs_cfg = cfg.get("source_dirs", {}).get("aistudio", [])
    if isinstance(source_dirs_cfg, str):
        source_dirs_cfg = [source_dirs_cfg]
    source_dirs = [Path(p) for p in source_dirs_cfg]
    org_dir_str = cfg.get("org_dir")
    if not org_dir_str:
        raise RuntimeError(
            "org_dir not configured. Run 'aise config init' or set org_dir in config.json"
        )
    org_dir = Path(org_dir_str)
    db_file = org_dir / "session_db.json"
    vocab_output = org_dir / cfg.get("vocab_output_filename", "VOCABULARY_ANALYSIS.md")
    mw = marker_window or cfg.get("marker_window", 25_000)
    md_mw = cfg.get("md_marker_window", 2_000)  # .md files include model responses; limit window

    # Load scoring weights
    scoring_weights: dict[str, int] = {
        "technique": 20, "role": 15, "thinking_budget": 30,
        "anti_ai": 35, "version_multiplier": 10, "corrected_bonus": 25,
        "descendant_boost": 15,
    }
    sw_path = org_dir / "scoring_weights.json"
    with contextlib.suppress(OSError, json.JSONDecodeError):
        scoring_weights.update(json.loads(sw_path.read_text(encoding="utf-8")))

    tech_codes, role_codes = load_codebook(org_dir)
    keyword_maps = load_keyword_maps(org_dir)
    tech_patterns = compile_codes(tech_codes)
    role_patterns = compile_codes(role_codes)

    source = AiStudioSource(source_dirs=source_dirs)
    records: list[SessionRecord] = []
    tri: Counter[str] = Counter()
    quad: Counter[str] = Counter()

    print(f"Analyzing sessions from {len(source_dirs)} source directories...")
    for session_info in source.stream_sessions():
        with contextlib.suppress(Exception):
            messages = source.read_session(session_info)
            if not messages:
                continue

            user_text = " ".join(m.content for m in messages if m.type.value == "user")
            name = session_info.session_id
            era = _detect_era(name, user_text)
            lower_sample = user_text[:5000].lower()

            # Detect source format from content
            source_format = "markdown" if name.endswith(".md") else "aistudio_json"

            rec = SessionRecord(
                name=name,
                source_dir=session_info.project_dir,
                filepath=str(Path(session_info.project_dir) / name),
                source_format=source_format,
                user_text=user_text,
                chunk_count=len(messages),
                user_chunk_count=sum(1 for m in messages if m.type.value == "user"),
                era=era,
                has_srt="srt" in lower_sample,
                has_transcript="transcript" in lower_sample,
            )

            effective_mw = md_mw if source_format == "markdown" else mw
            apply_codes(rec, tech_patterns, role_patterns, keyword_maps, scoring_weights, marker_window=effective_mw)
            records.append(rec)

            # Vocabulary: full text, no limit
            tri.update(get_ngrams(user_text, 3))
            quad.update(get_ngrams(user_text, 4))

    # Also process Gemini CLI sessions if configured (Gap 2 fix)
    gemini_cli_dir_cfg = cfg.get("source_dirs", {}).get("gemini_cli", "")
    if gemini_cli_dir_cfg:
        from ai_session_tools.sources.gemini_cli import GeminiCliSource
        gemini_source = GeminiCliSource(Path(gemini_cli_dir_cfg))
        for session_info in gemini_source.stream_sessions():
            with contextlib.suppress(Exception):
                messages = gemini_source.read_session(session_info)
                if not messages:
                    continue
                user_text = " ".join(m.content for m in messages if m.type.value == "user")
                if not user_text.strip():
                    continue
                name = session_info.session_id
                era = _detect_era(name, user_text)
                lower_sample = user_text[:5000].lower()
                rec = SessionRecord(
                    name=name,
                    source_dir=gemini_cli_dir_cfg,
                    filepath=str(Path(gemini_cli_dir_cfg) / name),
                    source_format="gemini_cli",
                    user_text=user_text,
                    chunk_count=len(messages),
                    user_chunk_count=sum(1 for m in messages if m.type.value == "user"),
                    era=era,
                    has_srt="srt" in lower_sample,
                    has_transcript="transcript" in lower_sample,
                    project_hash=getattr(session_info, "project_hash", ""),
                )
                apply_codes(rec, tech_patterns, role_patterns, keyword_maps, scoring_weights, marker_window=mw)
                records.append(rec)
                tri.update(get_ngrams(user_text, 3))
                quad.update(get_ngrams(user_text, 4))
    print(f"Total after all sources: {len(records)} sessions")

    compute_descendant_boost(records, scoring_weights.get("descendant_boost", 15))

    # Write DB (metadata only, no user_text)
    db = [r.to_db_dict() for r in records]
    with open(db_file, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)
    print(f"Analysis complete: {len(records)} sessions -> {db_file}")

    write_vocab_report(tri, quad, vocab_output)
    return records


def main(source_filter: str | None = None, marker_window: int | None = None) -> None:
    """Entry point for `aise analyze` CLI command.

    Args:
        source_filter: Narrow to one backend: 'aistudio', 'gemini', or None (all)
        marker_window: Chars for marker matching (0 = from config)
    """
    run_analysis(marker_window=marker_window, source_filter=source_filter)


if __name__ == "__main__":
    main()
