"""
AI Studio Knowledge Base Orchestrator - backs `aise organize`.

Reads session_db.json, creates non-destructive symlink taxonomy, writes INDEX.md,
SESSIONS_FULL.md, and KNOWLEDGE_GRAPH.md.

METHODOLOGICAL REFERENCES:
- Hsieh & Shannon (2005): https://journals.sagepub.com/doi/10.1177/1049732305276687
- SAGE/Nature archiving: https://journals.sagepub.com/doi/full/10.1177/00016993211051521

Copyright (c) 2026 Andrew Hundt
Licensed under the Apache License, Version 2.0
"""
from __future__ import annotations

import contextlib
import json
import os
from collections import defaultdict
from pathlib import Path

from ai_session_tools.config import load_config
from ai_session_tools.analysis.codebook import load_keyword_maps

# Files that must NEVER be deleted or overwritten
PERMANENT_FILES = frozenset({
    "CODEBOOK.md", "REFERENCES.md", "ORGANIZATION_TASK_INSTRUCTIONS.md",
    "USER_INSTRUCTIONS_CLEAN.md", "extract_verbatim_history.py",
    "analyze_sessions.py", "orchestrate_kb.py", "vocabulary_miner.py",
    "session_db.json", ".git", "VOCABULARY_ANALYSIS.md",
})


def make_symlink(source_path: str, link_path: Path) -> bool:
    """Create symlink non-destructively. Skip if already exists. Returns True if created."""
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        return False
    with contextlib.suppress(OSError):
        rel = os.path.relpath(source_path, link_path.parent)
        link_path.symlink_to(rel)
        return True
    return False


def assign_taxonomy(rec: dict, project_map: dict, workflow_map: dict) -> dict[str, list[str]]:
    """Return mapping of taxonomy_dir -> [sub_categories] for this session."""
    assignments: defaultdict[str, list[str]] = defaultdict(list)

    for tech in rec.get("techniques", []):
        assignments["03_by_technique"].append(tech)

    for role in rec.get("roles", []):
        assignments["05_by_expert_role"].append(role)

    for cat in rec.get("task_categories", []):
        assignments["04_by_task"].append(cat)

    for method in rec.get("writing_methods", []):
        assignments["06_by_writing_method"].append(method)

    era = rec.get("era", "2025-2026")
    assignments["07_by_era"].append(era)

    # 01_by_project: keyword match on name (filename signals valid for project mapping)
    name_lower = rec.get("name", "").lower()
    matched = False
    for proj, kws in project_map.items():
        if any(k.lower() in name_lower for k in kws):
            assignments["01_by_project"].append(proj)
            matched = True
    if not matched:
        assignments["01_by_project"].append("misc_research")

    # 02_by_workflow: derived from techniques
    techs = set(rec.get("techniques", []))
    for workflow, markers in workflow_map.items():
        if techs & set(markers):
            assignments["02_by_workflow"].append(workflow)

    return dict(assignments)


def build_symlinks(
    records: list[dict],
    org_dir: Path,
    project_map: dict,
    workflow_map: dict,
) -> dict[str, list[str]]:
    """Create symlinks across all taxonomy dimensions. Non-destructive."""
    session_paths: dict[str, list[str]] = {}
    created = 0

    for rec in records:
        raw_fp = rec.get("filepath", "")
        if not raw_fp:
            continue
        filepath = str(Path(raw_fp).expanduser())
        if not Path(filepath).exists():
            continue
        name = rec["name"]
        assignments = assign_taxonomy(rec, project_map, workflow_map)
        session_paths[name] = []

        for dim, categories in assignments.items():
            for cat in categories:
                link_path = org_dir / dim / cat / name
                if make_symlink(filepath, link_path):
                    created += 1
                session_paths[name].append(f"{dim}/{cat}")

    print(f"Created {created} new symlinks")
    return session_paths


def write_index(records: list[dict], session_paths: dict[str, list[str]], org_dir: Path) -> None:
    """Write INDEX.md with correct per-session links. No hardcoded misc_research."""
    sorted_recs = sorted(records, key=lambda r: r.get("utility", 0), reverse=True)
    all_ranked = [r for r in sorted_recs if r.get("utility", 0) >= 20]

    lines = [
        "# AI Studio Knowledge Base: Integrated Dashboard\n\n",
        "Ranked by utility score. Grounded in Directed Content Analysis "
        "(Hsieh & Shannon, 2005) and Chain-of-Thought scoring (Wei et al., 2022).\n\n",
        "## Hall of Fame: Top Sessions by Utility\n\n",
        "| Rank | Utility | Session | Technique | Role | Era |\n",
        "| :--- | :--- | :--- | :--- | :--- | :--- |\n",
    ]

    for count, rec in enumerate(all_ranked, 1):
        name = rec["name"]
        primary_paths = session_paths.get(name, ["01_by_project/misc_research"])
        link_target = next(
            (p for p in primary_paths if "07_by_era" not in p and "misc_research" not in p),
            primary_paths[0] if primary_paths else "01_by_project/misc_research"
        )
        encoded = name.replace(" ", "%20").replace("&", "%26")
        link = f"[{name}]({link_target}/{encoded})"
        tech = (rec.get("techniques") or ["—"])[0]
        role = (rec.get("roles") or ["—"])[0]
        era = rec.get("era", "—")
        util = rec.get("utility", 0)
        lines.append(f"| {count} | {util} | {link} | {tech} | {role} | {era} |\n")

    lines += [
        f"\n*Full list: {len(all_ranked)} sessions ranked. See [SESSIONS_FULL.md](SESSIONS_FULL.md).*\n\n",
        "## Taxonomy\n\n",
        "- [01 By Project](01_by_project/)\n",
        "- [02 By Workflow](02_by_workflow/)\n",
        "- [03 By Technique](03_by_technique/)\n",
        "- [04 By Task](04_by_task/)\n",
        "- [05 By Expert Role](05_by_expert_role/)\n",
        "- [06 By Writing Method](06_by_writing_method/)\n",
        "- [07 By Era](07_by_era/)\n\n",
        "## Governance\n\n",
        "- [Codebook](CODEBOOK.md)\n",
        "- [References](REFERENCES.md)\n",
        "- [User Instructions](USER_INSTRUCTIONS_CLEAN.md)\n",
        "- [Knowledge Graph](KNOWLEDGE_GRAPH.md)\n",
        "- [Vocabulary Analysis](VOCABULARY_ANALYSIS.md)\n",
        "- [All Sessions Ranked](SESSIONS_FULL.md)\n",
    ]

    with open(org_dir / "INDEX.md", "w", encoding="utf-8") as f:
        f.writelines(lines)

    # SESSIONS_FULL.md: ALL ranked sessions, no truncation (MSG 133)
    full_lines = [
        "# All Sessions: Complete Ranked List\n\n",
        f"{len(all_ranked)} sessions with utility >= 20, ranked by score.\n\n",
        "| Rank | Utility | Session | Technique | Role | Era |\n",
        "| :--- | :--- | :--- | :--- | :--- | :--- |\n",
    ]
    for i, rec in enumerate(all_ranked, 1):
        name = rec["name"]
        paths = session_paths.get(name, ["01_by_project/misc_research"])
        lp = next(
            (p for p in paths if "07_by_era" not in p and "misc_research" not in p),
            paths[0] if paths else "01_by_project/misc_research"
        )
        enc = name.replace(" ", "%20").replace("&", "%26")
        full_lines.append(
            f"| {i} | {rec.get('utility', 0)} | [{name}]({lp}/{enc}) "
            f"| {(rec.get('techniques') or ['—'])[0]} | {(rec.get('roles') or ['—'])[0]} "
            f"| {rec.get('era', '—')} |\n"
        )

    with open(org_dir / "SESSIONS_FULL.md", "w", encoding="utf-8") as f:
        f.writelines(full_lines)

    print(f"INDEX.md: {len(all_ranked)} entries; SESSIONS_FULL.md: {len(all_ranked)} total")


def write_knowledge_graph(records: list[dict], org_dir: Path) -> None:
    """Write session lineage graph in Mermaid markdown."""
    name_to_rec = {r["name"]: r for r in records}
    children: defaultdict[str, list[str]] = defaultdict(list)
    roots = []

    for rec in records:
        parent = rec.get("graph_parent")
        if parent and parent in name_to_rec:
            children[parent].append(rec["name"])
        elif not parent:
            roots.append(rec["name"])

    def emit_node(node: str, out: list[str], depth: int = 0, max_depth: int = 3) -> None:
        safe = node.replace('"', "'").replace("(", "[").replace(")", "]")[:60]
        for child in children.get(node, [])[:8]:
            child_safe = child.replace('"', "'").replace("(", "[").replace(")", "]")[:60]
            out.append(f'    "{safe}" --> "{child_safe}"\n')
            if depth < max_depth:
                emit_node(child, out, depth + 1, max_depth)

    lines = [
        "# Knowledge Graph: Session Lineage\n\n",
        "Maps provenance relationships: ROOT -> VERSION chains and BRANCH derivations.\n",
        "Source: https://journals.sagepub.com/doi/full/10.1177/00016993211051521\n\n",
        "## Major Lineage Chains\n\n",
    ]

    significant_roots = [r for r in roots if children.get(r)]
    for root in sorted(significant_roots, key=lambda n: name_to_rec.get(n, {}).get("utility", 0), reverse=True)[:30]:
        util = name_to_rec.get(root, {}).get("utility", 0)
        lines.append(f"### {root} (utility: {util})\n\n")
        lines.append("```mermaid\ngraph TD\n")
        emit_node(root, lines)
        lines.append("```\n\n")

    orphans = [r for r in roots if not children.get(r)]
    lines.append(f"## Standalone Sessions\n\n{len(orphans)} sessions with no detected lineage.\n\n")

    with open(org_dir / "KNOWLEDGE_GRAPH.md", "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("KNOWLEDGE_GRAPH.md written")


def run_orchestration() -> None:
    """Main orchestration: read session_db.json, create symlinks, write index files."""
    cfg = load_config()
    org_dir = Path(cfg.get("org_dir", str(Path.home() / "Downloads/aistudio_sessions/organized")))
    db_file = org_dir / "session_db.json"

    if not db_file.exists():
        raise FileNotFoundError(f"Run `aise analyze` first: {db_file}")

    records = json.loads(db_file.read_text(encoding="utf-8"))
    print(f"Loaded {len(records)} session records")

    keyword_maps = load_keyword_maps(org_dir)
    project_map = keyword_maps.get("project_map", {})
    workflow_map = keyword_maps.get("workflow_map", {})

    session_paths = build_symlinks(records, org_dir, project_map, workflow_map)
    write_index(records, session_paths, org_dir)
    write_knowledge_graph(records, org_dir)

    # Verify permanent files are intact
    for pf in PERMANENT_FILES:
        path = org_dir / pf
        if pf != "session_db.json" and not path.exists():
            print(f"WARNING: Permanent file missing: {pf}")

    print("Orchestration complete.")


def main() -> None:
    """Entry point for `aise organize` CLI command."""
    run_orchestration()


if __name__ == "__main__":
    main()
