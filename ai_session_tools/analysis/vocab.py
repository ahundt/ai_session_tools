"""
Vocabulary Miner: Recurring Phrase Analysis - backs `aise vocab`.

Standalone fallback; normally vocabulary is mined inline by `aise analyze`.

Source: PromptEngineering.org
https://www.promptengineering.org/building-a-reusable-prompt-library/

Copyright (c) 2026 Andrew Hundt
Licensed under the Apache License, Version 2.0
"""
from __future__ import annotations

import contextlib
import json
from collections import Counter
from pathlib import Path

from ai_session_tools.sources.aistudio import AiStudioSource, load_config
from ai_session_tools.analysis.codebook import get_ngrams, is_meaningful


def mine_all() -> tuple[Counter[str], Counter[str]]:
    """Stream full text from AI Studio sources. O(1) memory per session."""
    cfg = load_config()
    source_dirs_cfg = cfg.get("source_dirs", {}).get("aistudio", [])
    if isinstance(source_dirs_cfg, str):
        source_dirs_cfg = [source_dirs_cfg]
    source_dirs = [Path(p) for p in source_dirs_cfg]

    source = AiStudioSource(source_dirs=source_dirs)
    tri: Counter[str] = Counter()
    quad: Counter[str] = Counter()
    total = 0

    for session_info in source.stream_sessions():
        with contextlib.suppress(Exception):
            messages = source.read_session(session_info)
            user_text = " ".join(m.content for m in messages if m.type.value == "user")
            if len(user_text) < 50:
                continue
            tri.update(get_ngrams(user_text, 3))
            quad.update(get_ngrams(user_text, 4))
            total += 1

    print(f"Mined {total} sessions")
    return tri, quad


def write_report(tri: Counter[str], quad: Counter[str]) -> None:
    """Write vocabulary report. No arbitrary truncation."""
    cfg = load_config()
    org_dir_str = cfg.get("org_dir")
    if not org_dir_str:
        raise RuntimeError(
            "org_dir not configured. Run 'aise config init' or set org_dir in config.json"
        )
    org_dir = Path(org_dir_str)
    output_file = org_dir / cfg.get("vocab_output_filename", "VOCABULARY_ANALYSIS.md")

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


def main() -> None:
    """Entry point for `aise vocab` CLI command."""
    tri, quad = mine_all()
    write_report(tri, quad)


if __name__ == "__main__":
    main()
