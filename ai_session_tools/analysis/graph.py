"""
Multi-Format Session Graph Builder - backs `aise graph`.

Detects session lineage edges, outputs SESSION_GRAPH.json with bi-temporal fields.

METHODOLOGICAL REFERENCES:
- Rasmussen (2025): Zep Temporal KG: https://arxiv.org/abs/2501.13956
- Edge et al. (2024): GraphRAG: https://arxiv.org/abs/2404.16130

Copyright (c) 2026 Andrew Hundt
Licensed under the Apache License, Version 2.0
"""
from __future__ import annotations

import contextlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from ai_session_tools.sources.aistudio import load_config


@dataclass
class GraphNode:
    id: str
    source_format: str
    title: str
    project_hash: str = ""
    event_time: str = ""
    ingest_time: str = ""
    utility: int = 0
    era: str = ""
    techniques: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    filepath: str = ""


@dataclass
class GraphEdge:
    source: str
    target: str
    edge_type: str
    confidence: float
    detection_method: str


class LineageStrategy(Protocol):
    def detect(self, nodes: list[GraphNode]) -> list[GraphEdge]: ...


class AiStudioFilenameStrategy:
    """Detects 'Branch of X', 'Copy of X', 'Name vN' patterns. Confidence: HIGH."""

    def detect(self, nodes: list[GraphNode]) -> list[GraphEdge]:
        name_idx = {n.id.lower(): n for n in nodes}
        edges = []
        for node in nodes:
            parent_id, edge_type = self._find_parent(node.id, name_idx)
            if parent_id:
                edges.append(GraphEdge(parent_id, node.id, edge_type, 0.95, "filename_pattern"))
        return edges

    def _find_parent(self, name: str, idx: dict) -> tuple[str | None, str]:
        lower = name.lower()
        if m := re.match(r"branch of (.+)", lower):
            parent = m.group(1).strip()
            return (idx[parent].id if parent in idx else None), "branch"
        if m := re.match(r"copy of (.+)", lower):
            parent = m.group(1).strip()
            return (idx[parent].id if parent in idx else None), "copy"
        if m := re.search(r"^(.*?)\s+v(\d+)$", name, re.I):
            base, v = m.group(1).strip().lower(), int(m.group(2))
            if v > 1:
                for candidate in (f"{base} v{v-1}", base):
                    if candidate in idx:
                        return idx[candidate].id, "version"
        return None, ""


class GeminiProjectHashStrategy:
    """Groups sessions by projectHash — project_group edges. Confidence: HIGH."""

    def detect(self, nodes: list[GraphNode]) -> list[GraphEdge]:
        by_project: dict[str, list[GraphNode]] = defaultdict(list)
        for n in nodes:
            if n.source_format == "gemini_cli" and n.project_hash:
                by_project[n.project_hash].append(n)
        edges = []
        for group in by_project.values():
            sorted_group = sorted(group, key=lambda n: n.event_time or "")
            for i in range(1, len(sorted_group)):
                edges.append(GraphEdge(
                    sorted_group[i - 1].id, sorted_group[i].id,
                    "project_group", 0.80, "project_hash"
                ))
        return edges


class TfIdfSimilarityStrategy:
    """Cross-format topical links via TF-IDF cosine similarity of session titles.
    Only creates edges above threshold (configurable). O(N^2 * V) on titles (short).
    """

    def __init__(self, threshold: float = 0.50) -> None:
        self.threshold = threshold

    def detect(self, nodes: list[GraphNode]) -> list[GraphEdge]:
        tokenize = lambda t: re.findall(r"[a-z]+", t.lower())
        doc_tokens = [tokenize(n.title) for n in nodes]
        N = len(nodes)
        if N < 2:
            return []
        df: Counter[str] = Counter(tok for toks in doc_tokens for tok in set(toks))
        idf = {tok: math.log((N + 1) / (cnt + 1)) for tok, cnt in df.items()}

        def tfidf(tokens: list[str]) -> dict[str, float]:
            tf = Counter(tokens)
            return {tok: (cnt / max(len(tokens), 1)) * idf.get(tok, 0.0) for tok, cnt in tf.items()}

        def cosine(a: dict, b: dict) -> float:
            common = set(a) & set(b)
            if not common:
                return 0.0
            dot = sum(a[t] * b[t] for t in common)
            mag = (math.sqrt(sum(v ** 2 for v in a.values())) *
                   math.sqrt(sum(v ** 2 for v in b.values())))
            return dot / mag if mag else 0.0

        vecs = [tfidf(toks) for toks in doc_tokens]
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                sim = cosine(vecs[i], vecs[j])
                if sim >= self.threshold:
                    edges.append(GraphEdge(
                        nodes[i].id, nodes[j].id, "topical",
                        round(sim, 3), "tfidf_similarity"
                    ))
        return edges


def build_graph(records: list[dict], strategies: list | None = None, config: dict | None = None) -> dict:
    """Build SESSION_GRAPH.json from session_db.json records.

    strategies: pluggable LineageStrategy list. None -> load from config.
    config: scoring_weights content. None -> load from org_dir.
    """
    if config is None:
        config = {}

    if strategies is None:
        threshold = config.get("tfidf_similarity_threshold", 0.50)
        strategies = [
            AiStudioFilenameStrategy(),
            GeminiProjectHashStrategy(),
            TfIdfSimilarityStrategy(threshold=threshold),
        ]

    now = datetime.now(timezone.utc).isoformat()
    nodes = [
        GraphNode(
            id=r["name"],
            source_format=r.get("source_format", "aistudio_json"),
            title=r["name"],
            project_hash=r.get("project_hash", ""),
            event_time=r.get("era", ""),
            ingest_time=now,
            utility=r.get("utility", 0),
            era=r.get("era", ""),
            techniques=r.get("techniques", []),
            roles=r.get("roles", []),
            filepath=r.get("filepath", ""),
        )
        for r in records
    ]

    all_edges: list[GraphEdge] = []
    for strategy in strategies:
        with contextlib.suppress(Exception):
            all_edges.extend(strategy.detect(nodes))

    return {
        "generated_at": now,
        "node_count": len(nodes),
        "edge_count": len(all_edges),
        "nodes": [asdict(n) for n in nodes],
        "edges": [asdict(e) for e in all_edges],
    }


def main() -> None:
    """Entry point for `aise graph` CLI command."""
    cfg = load_config()
    org_dir = Path(cfg.get("org_dir", str(Path.home() / "Downloads/aistudio_sessions/organized")))
    db_file = org_dir / "session_db.json"
    out_file = org_dir / "SESSION_GRAPH.json"

    if not db_file.exists():
        raise FileNotFoundError(f"Run `aise analyze` first to generate {db_file}")

    records = json.loads(db_file.read_text(encoding="utf-8"))

    scoring_weights: dict = {}
    with contextlib.suppress(OSError, json.JSONDecodeError):
        scoring_weights = json.loads((org_dir / "scoring_weights.json").read_text(encoding="utf-8"))

    graph = build_graph(records, config=scoring_weights)
    out_file.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    print(f"SESSION_GRAPH.json: {graph['node_count']} nodes, {graph['edge_count']} edges -> {out_file}")


if __name__ == "__main__":
    main()
