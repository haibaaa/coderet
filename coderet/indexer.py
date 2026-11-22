"""Incremental repository indexing.

Files are processed one at a time and committed in small batches, so memory use
does not grow with repository size and an interrupted run can simply be
restarted. Change detection: (size, mtime) fast path, then sha256. Embedding is
a separate, resumable pass over chunks lacking a vector for the active model.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import INDEX_FORMAT_VERSION, __version__
from .config import Config
from .embedder import Embedder, code_subtokens
from .languages import language_for
from .parsing import Chunk, ParsedFile, parse_source
from .resolve import resolve_dependencies
from .store import Store

Log = Callable[[str], None]


@dataclass
class IndexReport:
    discovered: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    skipped: list[str] = field(default_factory=list)
    chunks_written: int = 0
    dependency_edges: int = 0
    embedded: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["skipped"] = len(self.skipped)
        return d


# --------------------------------------------------------------------------- #
# Text representations
# --------------------------------------------------------------------------- #

def embed_text(path: str, symbol: str | None, text: str, header: bool) -> str:
    """The exact string handed to the embedding model (before the model prefix)."""
    if not header:
        return text
    head = f"File: {path}\n"
    if symbol:
        head += f"Symbol: {symbol}\n"
    return head + "\n" + text


def content_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()


def fts_expand(text: str) -> str:
    """Append camelCase/snake_case sub-tokens so lexical search matches parts of
    identifiers (e.g. 'parse' matches parseConfigFile)."""
    subs = [t for t in code_subtokens(text) if "_" not in t]
    return text + "\n" + " ".join(subs)


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #

def discover_files(repo: Path, cfg: Config) -> list[str]:
    icfg = cfg.index
    exclude = set(icfg.exclude_dirs)
    allowed = [e.lower() for e in icfg.extensions] if icfg.extensions else None
    paths: list[str] | None = None
    if icfg.use_git and (repo / ".git").exists():
        try:
            out = subprocess.run(
                ["git", "-C", str(repo), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                check=True, capture_output=True,
            ).stdout
            paths = [p for p in out.decode("utf-8", errors="replace").split("\0") if p]
        except (OSError, subprocess.CalledProcessError):
            paths = None
    if paths is None:
        paths = []
        for root, dirs, files in os.walk(repo):
            dirs[:] = sorted(d for d in dirs if d not in exclude and not d.startswith(".") or d in {".github"})
            rel_root = Path(root).relative_to(repo).as_posix()
            for f in files:
                paths.append(f if rel_root == "." else f"{rel_root}/{f}")
    result = []
    for p in paths:
        if any(part in exclude for part in p.split("/")[:-1]):
            continue
        if language_for(p, allowed) is None:
            continue
        result.append(p)
    return sorted(set(result))


def _looks_binary(data: bytes) -> bool:
    return b"\0" in data[:8192]


# --------------------------------------------------------------------------- #
# Indexer
# --------------------------------------------------------------------------- #

class Indexer:
    def __init__(self, repo: Path, cfg: Config, store: Store, log: Log | None = None):
        self.repo = repo.resolve()
        self.cfg = cfg
        self.store = store
        self.log = log or (lambda msg: None)

    def format_fingerprint(self) -> str:
        """Anything that changes chunk boundaries / stored text forces re-parsing."""
        blob = json.dumps({"format": INDEX_FORMAT_VERSION, "chunking": asdict(self.cfg.chunking)}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    # -- scanning ----------------------------------------------------------- #

    def scan(self, report: IndexReport) -> None:
        store, cfg = self.store, self.cfg
        fingerprint = self.format_fingerprint()
        full_rebuild = store.get_meta("format_fingerprint") != fingerprint
        if full_rebuild and store.get_meta("format_fingerprint") is not None:
            self.log("chunking settings or index format changed: re-parsing all files")

        allowed = [e.lower() for e in cfg.index.extensions] if cfg.index.extensions else None
        paths = discover_files(self.repo, cfg)
        report.discovered = len(paths)
        known = store.file_states()
        seen: set[str] = set()
        pending = 0

        for n, rel in enumerate(paths, 1):
            seen.add(rel)
            full = self.repo / rel
            try:
                st = full.stat()
            except OSError:
                report.skipped.append(rel)
                continue
            if st.st_size > cfg.index.max_file_bytes:
                report.skipped.append(rel)
                if rel in known:
                    store.delete_file(known[rel][0])
                continue
            prev = known.get(rel)
            if prev and not full_rebuild and prev[2] == st.st_size and prev[3] == st.st_mtime_ns:
                report.unchanged += 1
                continue
            try:
                data = full.read_bytes()
            except OSError:
                report.skipped.append(rel)
                continue
            if _looks_binary(data):
                report.skipped.append(rel)
                continue
            sha = hashlib.sha256(data).hexdigest()
            if prev and not full_rebuild and prev[1] == sha:
                store.conn.execute("UPDATE files SET mtime_ns = ?, size = ? WHERE id = ?",
                                   (st.st_mtime_ns, st.st_size, prev[0]))
                report.unchanged += 1
                continue
            lang = language_for(rel, allowed)
            parsed = parse_source(data, lang, cfg.chunking)
            if prev:
                store.delete_file(prev[0])
                report.updated += 1
            else:
                report.added += 1
            report.chunks_written += self._write_file(rel, sha, st, parsed)
            del data, parsed
            pending += 1
            if pending >= cfg.index.file_batch:
                store.conn.commit()
                pending = 0
                self.log(f"  parsed {n}/{len(paths)} files")

        for rel, (fid, *_rest) in known.items():
            if rel not in seen:
                store.delete_file(fid)
                report.removed += 1
        store.set_meta("format_fingerprint", fingerprint)
        store.conn.commit()

    def _write_file(self, rel: str, sha: str, st: os.stat_result, parsed: ParsedFile) -> int:
        conn = self.store.conn
        header = self.cfg.chunking.embed_header
        cur = conn.execute(
            "INSERT INTO files(path, language, sha256, size, mtime_ns, syntax_errors) VALUES (?,?,?,?,?,?)",
            (rel, parsed.language, sha, st.st_size, st.st_mtime_ns, int(parsed.syntax_errors)),
        )
        file_id = cur.lastrowid
        spans: list[tuple[int, int, int]] = []  # (start, end, chunk_id)
        path_tokens = fts_expand(rel.replace("/", " ").replace(".", " "))
        for ch in parsed.chunks:
            h = content_hash(embed_text(rel, ch.symbol, ch.text, header))
            cid = conn.execute(
                "INSERT INTO chunks(file_id, start_line, end_line, kind, symbol, text, content_hash) "
                "VALUES (?,?,?,?,?,?,?)",
                (file_id, ch.start_line, ch.end_line, ch.kind, ch.symbol, ch.text, h),
            ).lastrowid
            conn.execute(
                "INSERT INTO chunks_fts(rowid, path, symbol, body) VALUES (?,?,?,?)",
                (cid, path_tokens, fts_expand(ch.symbol or ""), fts_expand(ch.text)),
            )
            spans.append((ch.start_line, ch.end_line, cid))

        def chunk_for(line: int) -> int | None:
            best = None
            for s, e, cid in spans:
                if s <= line <= e and (best is None or e - s < best[0]):
                    best = (e - s, cid)
            return best[1] if best else None

        conn.executemany(
            "INSERT INTO definitions(file_id, chunk_id, name, qualified, kind, start_line, end_line) "
            "VALUES (?,?,?,?,?,?,?)",
            [(file_id, chunk_for(d.start_line), d.name, d.qualified, d.kind, d.start_line, d.end_line)
             for d in parsed.definitions],
        )
        conn.executemany(
            "INSERT INTO imports(file_id, module, name, line) VALUES (?,?,?,?)",
            [(file_id, i.module, i.name, i.line) for i in parsed.imports],
        )
        conn.executemany(
            "INSERT INTO refs(file_id, chunk_id, name, qualified, line) VALUES (?,?,?,?,?)",
            [(file_id, chunk_for(r.line), r.name, r.qualified, r.line) for r in parsed.references],
        )
        return len(parsed.chunks)

    # -- embedding ---------------------------------------------------------- #

    def embed_pending(self, embedder: Embedder) -> int:
        """Embed chunks lacking a vector for `embedder.model_key`, in batches.
        Resumable: each batch is committed immediately."""
        store, cfg = self.store, self.cfg
        key = embedder.model_key
        header = cfg.chunking.embed_header
        total = store.stats(key)["chunks_missing_embedding"]
        if total == 0:
            return 0
        self.log(f"embedding {total} chunks (deduplicated by content) with {key}")
        done = 0
        t0 = time.time()
        while True:
            # one representative chunk per missing content hash; longest first so
            # memory problems surface early and batches have similar lengths
            rows = store.conn.execute(
                "SELECT c.content_hash, f.path, c.symbol, c.text FROM chunks c "
                "JOIN files f ON f.id = c.file_id "
                "WHERE NOT EXISTS (SELECT 1 FROM embeddings e "
                "                  WHERE e.content_hash = c.content_hash AND e.model_key = ?) "
                "GROUP BY c.content_hash ORDER BY LENGTH(c.text) DESC LIMIT ?",
                (key, cfg.index.embed_fetch),
            ).fetchall()
            if not rows:
                break
            texts = [embed_text(r["path"], r["symbol"], r["text"], header) for r in rows]
            vecs = embedder.encode_documents(texts).astype(np.float32, copy=False)
            store.conn.executemany(
                "INSERT OR REPLACE INTO embeddings(content_hash, model_key, dim, vector) VALUES (?,?,?,?)",
                [(r["content_hash"], key, int(v.shape[0]), v.tobytes()) for r, v in zip(rows, vecs)],
            )
            store.conn.commit()
            done += len(rows)
            rate = done / max(time.time() - t0, 1e-9)
            self.log(f"  embedded {done}/{total} unique chunks ({rate:.2f}/s)")
        return done

    # -- full run ----------------------------------------------------------- #

    def run(self, embedder: Embedder | None) -> IndexReport:
        report = IndexReport()
        t0 = time.time()
        self.log(f"scanning {self.repo}")
        self.scan(report)
        self.log(f"files: +{report.added} ~{report.updated} -{report.removed} "
                 f"={report.unchanged} (skipped {len(report.skipped)})")
        report.dependency_edges = resolve_dependencies(self.store, self.repo)
        self.log(f"resolved {report.dependency_edges} file dependency edges")
        if embedder is not None:
            report.embedded = self.embed_pending(embedder)
            self.store.set_meta("last_model_key", embedder.model_key)
            self.store.set_meta(f"model:{embedder.model_key}", json.dumps(embedder.describe(), sort_keys=True))
        self.store.set_meta("coderet_version", __version__)
        self.store.set_meta("config", json.dumps(self.cfg.to_dict(), sort_keys=True))
        self.store.set_meta("config_fingerprint", self.cfg.fingerprint())
        self.store.set_meta("repo", str(self.repo))
        self.store.conn.commit()
        report.seconds = round(time.time() - t0, 2)
        return report
