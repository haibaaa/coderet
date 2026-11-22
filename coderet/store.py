"""SQLite storage for files, chunks, full-text index, dependency graph and the
embedding cache.

Embeddings are keyed by (content_hash, model_key), so
* unchanged chunks are never re-embedded,
* identical chunks in different files share one vector,
* vectors from several models can coexist, which lets you switch checkpoints
  back and forth without recomputation.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    language TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    syntax_errors INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    kind TEXT NOT NULL,
    symbol TEXT,
    text TEXT NOT NULL,
    -- sha256 of the exact string given to the embedding model
    content_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_file ON chunks(file_id);
CREATE INDEX IF NOT EXISTS chunks_hash ON chunks(content_hash);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    path, symbol, body, tokenize = "unicode61 tokenchars '_'"
);

CREATE TABLE IF NOT EXISTS embeddings (
    content_hash TEXT NOT NULL,
    model_key TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,  -- float32, L2-normalised
    PRIMARY KEY (content_hash, model_key)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS definitions (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    chunk_id INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
    name TEXT NOT NULL,
    qualified TEXT NOT NULL,
    kind TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS definitions_name ON definitions(name);
CREATE INDEX IF NOT EXISTS definitions_file ON definitions(file_id);

CREATE TABLE IF NOT EXISTS imports (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    module TEXT NOT NULL,
    name TEXT,
    line INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS imports_file ON imports(file_id);

-- resolved file-level dependency edges (rebuilt after each indexing run)
CREATE TABLE IF NOT EXISTS file_deps (
    src_file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    dst_file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    import_id INTEGER NOT NULL REFERENCES imports(id) ON DELETE CASCADE,
    PRIMARY KEY (src_file_id, dst_file_id, import_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS file_deps_dst ON file_deps(dst_file_id);

CREATE TABLE IF NOT EXISTS refs (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    chunk_id INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
    name TEXT NOT NULL,
    qualified TEXT NOT NULL,
    line INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS refs_name ON refs(name);
CREATE INDEX IF NOT EXISTS refs_chunk ON refs(chunk_id);
"""


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- metadata ----------------------------------------------------------- #

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    # -- files -------------------------------------------------------------- #

    def file_states(self) -> dict[str, tuple[int, str, int, int]]:
        """path -> (id, sha256, size, mtime_ns)"""
        return {
            r["path"]: (r["id"], r["sha256"], r["size"], r["mtime_ns"])
            for r in self.conn.execute("SELECT id, path, sha256, size, mtime_ns FROM files")
        }

    def delete_file(self, file_id: int) -> None:
        self.conn.execute(
            "DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE file_id = ?)",
            (file_id,),
        )
        self.conn.execute("DELETE FROM files WHERE id = ?", (file_id,))

    def stats(self, model_key: str | None = None) -> dict[str, int]:
        q = lambda sql, *a: self.conn.execute(sql, a).fetchone()[0]
        out = {
            "files": q("SELECT COUNT(*) FROM files"),
            "chunks": q("SELECT COUNT(*) FROM chunks"),
            "definitions": q("SELECT COUNT(*) FROM definitions"),
            "imports": q("SELECT COUNT(*) FROM imports"),
            "resolved_file_deps": q("SELECT COUNT(*) FROM file_deps"),
            "references": q("SELECT COUNT(*) FROM refs"),
            "cached_embeddings_all_models": q("SELECT COUNT(*) FROM embeddings"),
        }
        if model_key is not None:
            out["chunks_missing_embedding"] = q(
                "SELECT COUNT(*) FROM chunks c WHERE NOT EXISTS ("
                "SELECT 1 FROM embeddings e WHERE e.content_hash = c.content_hash AND e.model_key = ?)",
                model_key,
            )
        return out

    def prune_embeddings(self, keep_model_keys: list[str] | None = None) -> int:
        """Delete cached vectors not referenced by any chunk (optionally also those
        of models not listed). Returns rows deleted."""
        sql = "DELETE FROM embeddings WHERE content_hash NOT IN (SELECT content_hash FROM chunks)"
        args: list[str] = []
        if keep_model_keys:
            sql += f" OR model_key NOT IN ({','.join('?' * len(keep_model_keys))})"
            args = list(keep_model_keys)
        cur = self.conn.execute(sql, args)
        self.conn.commit()
        return cur.rowcount
