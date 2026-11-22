"""Queries over the extracted dependency graph (definitions, calls, imports)."""

from __future__ import annotations

from .store import Store


def find_definitions(store: Store, name: str, limit: int = 50) -> list[dict]:
    """Match by simple name, or by qualified name when `name` contains a dot."""
    col = "d.qualified" if "." in name else "d.name"
    rows = store.conn.execute(
        "SELECT d.id, d.name, d.qualified, d.kind, d.start_line, d.end_line, f.path, f.id AS file_id "
        f"FROM definitions d JOIN files f ON f.id = d.file_id WHERE {col} = ? "
        "ORDER BY f.path, d.start_line LIMIT ?",
        (name, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def callers(store: Store, name: str, limit: int = 100) -> list[dict]:
    """Call sites referencing `name` (by last identifier), with enclosing chunk."""
    simple = name.split(".")[-1].split("::")[-1]
    rows = store.conn.execute(
        "SELECT r.line, r.qualified AS call, f.path, "
        "  (SELECT d.qualified FROM definitions d WHERE d.file_id = r.file_id "
        "     AND r.line BETWEEN d.start_line AND d.end_line "
        "   ORDER BY d.end_line - d.start_line LIMIT 1) AS in_definition, "
        "  c.symbol AS chunk_symbol, c.start_line AS chunk_start, c.end_line AS chunk_end "
        "FROM refs r JOIN files f ON f.id = r.file_id LEFT JOIN chunks c ON c.id = r.chunk_id "
        "WHERE r.name = ? ORDER BY f.path, r.line LIMIT ?",
        (simple, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def resolve_name(store: Store, name: str, from_file_id: int) -> list[dict]:
    """Rank candidate definitions of `name` as seen from a file:
    same file > files it imports > anywhere else."""
    rows = store.conn.execute(
        "SELECT d.qualified, d.kind, d.start_line, f.path, d.file_id, "
        "  CASE WHEN d.file_id = :src THEN 0 "
        "       WHEN d.file_id IN (SELECT dst_file_id FROM file_deps WHERE src_file_id = :src) THEN 1 "
        "       ELSE 2 END AS scope "
        "FROM definitions d JOIN files f ON f.id = d.file_id WHERE d.name = :name "
        "ORDER BY scope, f.path, d.start_line",
        {"src": from_file_id, "name": name},
    ).fetchall()
    if not rows:
        return []
    best = rows[0]["scope"]
    return [dict(r) for r in rows if r["scope"] == best]


def callees(store: Store, name: str, limit: int = 200) -> list[dict]:
    """Calls made inside the definition(s) of `name`, each resolved to likely
    target definitions."""
    out = []
    for d in find_definitions(store, name):
        rows = store.conn.execute(
            "SELECT r.name, r.qualified, r.line FROM refs r "
            "WHERE r.file_id = ? AND r.line BETWEEN ? AND ? ORDER BY r.line LIMIT ?",
            (d["file_id"], d["start_line"], d["end_line"], limit),
        ).fetchall()
        for r in rows:
            targets = resolve_name(store, r["name"], d["file_id"])
            out.append({
                "from": f"{d['path']}:{d['qualified']}",
                "line": r["line"],
                "call": r["qualified"],
                "targets": [f"{t['path']}:{t['start_line']} {t['qualified']}" for t in targets[:5]],
            })
    return out


def file_dependencies(store: Store, path: str) -> dict:
    row = store.conn.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()
    if row is None:
        raise KeyError(f"file not indexed: {path}")
    fid = row[0]
    imports = store.conn.execute(
        "SELECT i.line, i.module, i.name, GROUP_CONCAT(f.path, ', ') AS resolved "
        "FROM imports i LEFT JOIN file_deps fd ON fd.import_id = i.id "
        "LEFT JOIN files f ON f.id = fd.dst_file_id WHERE i.file_id = ? "
        "GROUP BY i.id ORDER BY i.line",
        (fid,),
    ).fetchall()
    importers = store.conn.execute(
        "SELECT DISTINCT f.path FROM file_deps fd JOIN files f ON f.id = fd.src_file_id "
        "WHERE fd.dst_file_id = ? ORDER BY f.path",
        (fid,),
    ).fetchall()
    return {
        "path": path,
        "imports": [dict(r) for r in imports],
        "imported_by": [r[0] for r in importers],
    }
