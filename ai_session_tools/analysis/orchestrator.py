"""
AI Studio Knowledge Base Orchestrator - backs `aise organize`.

Reads session_db.json, creates taxonomy output in one or more configurable formats,
writes INDEX.md, SESSIONS_FULL.md, and KNOWLEDGE_GRAPH.md.

Output formats (config.json["organize_formats"] or --format CLI flag):
  "symlinks"  — non-destructive symlink taxonomy dirs (default)
  "json"      — SESSION_TAXONOMY.json  {name: {dim: [cats], utility, era}}
  "markdown"  — TAXONOMY.md  grouped by dimension

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
from ai_session_tools.analysis.codebook import load_keyword_maps, load_scoring_weights

VALID_FORMATS: frozenset[str] = frozenset({"symlinks", "json", "markdown"})


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
    """Return mapping of taxonomy_dir -> [sub_categories] for a single session record."""
    assignments: defaultdict[str, list[str]] = defaultdict(list)

    for tech in rec.get("techniques", []):
        assignments["03_by_technique"].append(tech)

    for role in rec.get("roles", []):
        assignments["05_by_expert_role"].append(role)

    for cat in rec.get("task_categories", []):
        assignments["04_by_task"].append(cat)

    for method in rec.get("writing_methods", []):
        assignments["06_by_writing_method"].append(method)

    era = rec.get("era", "unknown")
    assignments["07_by_era"].append(era)

    # 01_by_project: keyword match on name
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


def build_taxonomy(
    records: list[dict],
    project_map: dict,
    workflow_map: dict,
) -> dict[str, dict[str, list[str]]]:
    """Compute taxonomy assignments for all records.

    Returns {session_name: {taxonomy_dim: [categories]}}.
    Pure computation — no filesystem side effects.
    """
    result: dict[str, dict[str, list[str]]] = {}
    for rec in records:
        name = rec.get("name")
        if name:
            result[name] = assign_taxonomy(rec, project_map, workflow_map)
    return result


def taxonomy_to_session_paths(taxonomy: dict[str, dict[str, list[str]]]) -> dict[str, list[str]]:
    """Flatten taxonomy to session_paths {name: ["dim/cat", ...]} for write_index."""
    return {
        name: [f"{dim}/{cat}" for dim, cats in dims.items() for cat in cats]
        for name, dims in taxonomy.items()
    }


def apply_symlinks(
    records: list[dict],
    org_dir: Path,
    taxonomy: dict[str, dict[str, list[str]]],
) -> int:
    """Create symlinks from pre-computed taxonomy. Non-destructive. Returns new symlink count."""
    created = 0
    for rec in records:
        raw_fp = rec.get("filepath", "")
        if not raw_fp:
            continue
        filepath = str(Path(raw_fp).expanduser())
        if not Path(filepath).exists():
            continue
        name = rec["name"]
        for dim, categories in taxonomy.get(name, {}).items():
            for cat in categories:
                link_path = org_dir / dim / cat / name
                if make_symlink(filepath, link_path):
                    created += 1
    return created


def write_taxonomy_json(
    taxonomy: dict[str, dict[str, list[str]]],
    records: list[dict],
    org_dir: Path,
) -> None:
    """Write SESSION_TAXONOMY.json: {name: {taxonomy, utility, era}} for all sessions."""
    name_to_rec = {r["name"]: r for r in records}
    output = {
        name: {
            "taxonomy": dims,
            "utility": name_to_rec.get(name, {}).get("utility", 0),
            "era": name_to_rec.get(name, {}).get("era", "unknown"),
        }
        for name, dims in taxonomy.items()
    }
    path = org_dir / "SESSION_TAXONOMY.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"SESSION_TAXONOMY.json: {len(taxonomy)} sessions")


def write_taxonomy_markdown(
    taxonomy: dict[str, dict[str, list[str]]],
    records: list[dict],
    org_dir: Path,
) -> None:
    """Write TAXONOMY.md: sessions grouped by taxonomy dimension and category."""
    name_to_rec = {r["name"]: r for r in records}
    sw = load_scoring_weights()
    min_utility = int(sw.get("min_utility_for_index", 20))

    # {dim: {cat: [names]}}
    dim_cat_names: defaultdict[str, defaultdict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for name, dims in taxonomy.items():
        for dim, cats in dims.items():
            for cat in cats:
                dim_cat_names[dim][cat].append(name)

    lines = ["# Session Taxonomy\n\n"]
    for dim in sorted(dim_cat_names):
        lines.append(f"## {dim}\n\n")
        for cat in sorted(dim_cat_names[dim]):
            names = dim_cat_names[dim][cat]
            qualifying = [
                n for n in names
                if name_to_rec.get(n, {}).get("utility", 0) >= min_utility
            ]
            if not qualifying:
                continue
            lines.append(f"### {cat} ({len(qualifying)} sessions)\n\n")
            lines.append("| Session | Utility | Era |\n| :--- | :--- | :--- |\n")
            for name in sorted(
                qualifying,
                key=lambda n: name_to_rec.get(n, {}).get("utility", 0),
                reverse=True,
            ):
                util = name_to_rec.get(name, {}).get("utility", 0)
                era = name_to_rec.get(name, {}).get("era", "—")
                lines.append(f"| {name} | {util} | {era} |\n")
            lines.append("\n")

    path = org_dir / "TAXONOMY.md"
    path.write_text("".join(lines), encoding="utf-8")
    print(f"TAXONOMY.md: {len(dim_cat_names)} dimensions")


def write_index(records: list[dict], session_paths: dict[str, list[str]], org_dir: Path) -> None:
    """Write INDEX.md and SESSIONS_FULL.md. Always written regardless of format.

    min_utility_for_index loaded from scoring_weights (default 20).
    """
    sw = load_scoring_weights(org_dir)
    min_utility = int(sw.get("min_utility_for_index", 20))

    sorted_recs = sorted(records, key=lambda r: r.get("utility", 0), reverse=True)
    all_ranked = [r for r in sorted_recs if r.get("utility", 0) >= min_utility]

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

    # SESSIONS_FULL.md: all ranked sessions, no truncation
    full_lines = [
        "# All Sessions: Complete Ranked List\n\n",
        f"{len(all_ranked)} sessions with utility >= {min_utility}, ranked by score.\n\n",
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
    for root in sorted(
        significant_roots,
        key=lambda n: name_to_rec.get(n, {}).get("utility", 0),
        reverse=True,
    )[:30]:
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


def _resolve_formats(cfg: dict, formats: list[str] | None) -> list[str]:
    """Resolve output formats: parameter > config > default ["symlinks"].

    Accepts a list or comma-separated string from config.
    Raises ValueError for unknown format names.
    """
    if formats is not None:
        resolved = formats
    else:
        cfg_val = cfg.get("organize_formats")
        if isinstance(cfg_val, str):
            resolved = [f.strip() for f in cfg_val.split(",") if f.strip()]
        elif isinstance(cfg_val, list):
            resolved = cfg_val
        else:
            resolved = ["symlinks"]

    bad = set(resolved) - VALID_FORMATS
    if bad:
        raise ValueError(
            f"Unknown organize format(s): {sorted(bad)}. "
            f"Valid: {sorted(VALID_FORMATS)}"
        )
    return resolved


def run_orchestration(formats: list[str] | None = None) -> None:
    """Read session_db.json, produce taxonomy output, write index files.

    Args:
        formats: Output formats to produce. None reads from config.json["organize_formats"].
                 Valid values: "symlinks", "json", "markdown" (combinable as a list).
                 Default when unconfigured: ["symlinks"].

    INDEX.md and SESSIONS_FULL.md are always written regardless of formats.
    """
    cfg = load_config()
    org_dir_str = cfg.get("org_dir")
    if not org_dir_str:
        raise RuntimeError(
            "org_dir not configured. Run 'aise config init' or set org_dir in config.json"
        )
    org_dir = Path(org_dir_str).expanduser()
    db_file = org_dir / "session_db.json"

    if not db_file.exists():
        raise FileNotFoundError(f"Run `aise analyze` first: {db_file}")

    records = json.loads(db_file.read_text(encoding="utf-8"))
    print(f"Loaded {len(records)} session records")

    active_formats = _resolve_formats(cfg, formats)
    print(f"Output formats: {', '.join(active_formats)}")

    keyword_maps = load_keyword_maps()
    project_map = keyword_maps.get("project_map", {})
    workflow_map = keyword_maps.get("workflow_map", {})

    # Always compute taxonomy — needed for all format outputs and write_index
    taxonomy = build_taxonomy(records, project_map, workflow_map)
    session_paths = taxonomy_to_session_paths(taxonomy)

    if "symlinks" in active_formats:
        created = apply_symlinks(records, org_dir, taxonomy)
        print(f"Created {created} new symlinks")

    if "json" in active_formats:
        write_taxonomy_json(taxonomy, records, org_dir)

    if "markdown" in active_formats:
        write_taxonomy_markdown(taxonomy, records, org_dir)

    # Index files always written
    write_index(records, session_paths, org_dir)
    write_knowledge_graph(records, org_dir)
    print("Orchestration complete.")


def main() -> None:
    """Entry point for `aise organize` CLI command."""
    run_orchestration()


if __name__ == "__main__":
    main()
