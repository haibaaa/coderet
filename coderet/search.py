"""Hybrid retrieval: exact dense scan + BM25 (SQLite FTS5), fused with weighted
reciprocal rank fusion, optionally rescored by a cross-encoder.

The dense scan is exhaustive (no approximate index), so ranking is exact. It
streams embeddings from SQLite in `scan_batch` rows at a time, keeping memory
bounded regardless of index size.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .config import Config
from .embedder import Embedder, code_subtokens
from .store import Store


@dataclass
class SearchResult:
    chunk_id: int
    path: str
    start_line: int
    end_line: int
    kind: str
    symbol: str | None
    language: str
    text: str
    score: float
    dense_rank: int | None = None
    dense_score: float | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None
    rerank_score: float | None = None

    def to_dict(self, with_text: bool = True) -> dict:
        d = asdict(self)
        if not with_text:
            d.pop("text")
        return d


def _filter_sql(path_glob: str | None, language: str | None) -> tuple[str, list]:
    sql, args = "", []
    if path_glob:
        sql += " AND f.path GLOB ?"
        args.append(path_glob)
    if language:
        sql += " AND f.language = ?"
        args.append(language)
    return sql, args


def fts_query(query: str) -> str | None:
    seen: list[str] = []
    for tok in code_subtokens(query):
        if len(tok) > 1 and tok not in seen:
            seen.append(tok)
    if not seen:
        return None
    return " OR ".join(f'"{t}"' for t in seen)


class Searcher:
    def __init__(self, store: Store, cfg: Config, embedder: Embedder | None = None, reranker=None):
        self.store = store
        self.cfg = cfg
        self.embedder = embedder
        self.reranker = reranker
        self.warnings: list[str] = []

    # -- individual retrievers --------------------------------------------- #

    def dense(self, query: str, k: int, path_glob: str | None = None,
              language: str | None = None) -> list[tuple[int, float]]:
        if self.embedder is None:
            raise RuntimeError("dense retrieval requires an embedder")
        key = self.embedder.model_key
        missing = self.store.stats(key)["chunks_missing_embedding"]
        total = self.store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if total and missing == total:
            raise RuntimeError(
                f"no chunks are embedded with model '{key}'. Run `coderet embed` (or `coderet index`) first."
            )
        if missing:
            self.warnings.append(f"{missing}/{total} chunks have no embedding for '{key}' and are "
                                 f"invisible to dense search; run `coderet embed`.")
        q = self.embedder.encode_queries([query])[0].astype(np.float32)
        fsql, fargs = _filter_sql(path_glob, language)
        cur = self.store.conn.execute(
            "SELECT c.id, e.vector FROM chunks c "
            "JOIN files f ON f.id = c.file_id "
            "JOIN embeddings e ON e.content_hash = c.content_hash AND e.model_key = ? "
            f"WHERE 1 = 1 {fsql} ORDER BY c.id",
            [key, *fargs],
        )
        best_ids = np.empty(0, dtype=np.int64)
        best_scores = np.empty(0, dtype=np.float32)
        while True:
            rows = cur.fetchmany(self.cfg.search.scan_batch)
            if not rows:
                break
            ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
            mat = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), -1)
            if mat.shape[1] != q.shape[0]:
                raise RuntimeError(f"embedding dimension mismatch ({mat.shape[1]} vs query {q.shape[0]})")
            scores = mat @ q
            best_ids = np.concatenate([best_ids, ids])
            best_scores = np.concatenate([best_scores, scores])
            if best_ids.shape[0] > k:
                keep = np.argpartition(-best_scores, k - 1)[:k]
                best_ids, best_scores = best_ids[keep], best_scores[keep]
        # deterministic order: score desc, then chunk id asc
        order = np.lexsort((best_ids, -best_scores))
        return [(int(best_ids[i]), float(best_scores[i])) for i in order[:k]]

    def lexical(self, query: str, k: int, path_glob: str | None = None,
                language: str | None = None) -> list[tuple[int, float]]:
        match = fts_query(query)
        if match is None:
            return []
        fsql, fargs = _filter_sql(path_glob, language)
        # column weights: path, symbol, body
        rows = self.store.conn.execute(
            "SELECT c.id, bm25(chunks_fts, 2.0, 4.0, 1.0) AS s FROM chunks_fts "
            "JOIN chunks c ON c.id = chunks_fts.rowid JOIN files f ON f.id = c.file_id "
            f"WHERE chunks_fts MATCH ? {fsql} ORDER BY s, c.id LIMIT ?",
            [match, *fargs, k],
        ).fetchall()
        return [(int(r[0]), -float(r[1])) for r in rows]

    # -- combined ----------------------------------------------------------- #

    def search(self, query: str, top_k: int | None = None, mode: str | None = None,
               path_glob: str | None = None, language: str | None = None,
               rerank: bool | None = None) -> list[SearchResult]:
        scfg = self.cfg.search
        top_k = top_k or scfg.top_k
        mode = mode or scfg.mode
        pool = max(scfg.candidate_pool, top_k)
        self.warnings = []

        dense = self.dense(query, pool, path_glob, language) if mode in {"dense", "hybrid"} else []
        lexical = self.lexical(query, pool, path_glob, language) if mode in {"lexical", "hybrid"} else []

        fused: dict[int, float] = {}
        info: dict[int, dict] = {}
        for weight, hits, label in ((scfg.dense_weight, dense, "dense"), (scfg.lexical_weight, lexical, "lexical")):
            for rank, (cid, s) in enumerate(hits, 1):
                if mode == "hybrid":
                    fused[cid] = fused.get(cid, 0.0) + weight / (scfg.rrf_k + rank)
                else:
                    fused[cid] = s
                info.setdefault(cid, {})[f"{label}_rank"] = rank
                info[cid][f"{label}_score"] = s

        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
        use_rerank = self.cfg.rerank.enabled if rerank is None else rerank
        n_fetch = max(top_k, self.cfg.rerank.pool) if use_rerank else top_k
        results = self._hydrate(ranked[:n_fetch], info)

        if use_rerank and results:
            if self.reranker is None:
                from .embedder import CrossEncoderReranker
                self.reranker = CrossEncoderReranker(self.cfg.rerank, seed=self.cfg.seed)
            scores = self.reranker.score(query, [r.text for r in results])
            for r, s in zip(results, scores):
                r.rerank_score = s
                r.score = s
            results.sort(key=lambda r: (-r.score, r.chunk_id))
        return results[:top_k]

    def _hydrate(self, ranked: list[tuple[int, float]], info: dict[int, dict]) -> list[SearchResult]:
        if not ranked:
            return []
        ids = [cid for cid, _ in ranked]
        rows = self.store.conn.execute(
            "SELECT c.id, f.path, c.start_line, c.end_line, c.kind, c.symbol, f.language, c.text "
            f"FROM chunks c JOIN files f ON f.id = c.file_id WHERE c.id IN ({','.join('?' * len(ids))})",
            ids,
        ).fetchall()
        by_id = {r[0]: r for r in rows}
        out = []
        for cid, score in ranked:
            r = by_id.get(cid)
            if r is None:
                continue
            out.append(SearchResult(
                chunk_id=cid, path=r[1], start_line=r[2], end_line=r[3], kind=r[4],
                symbol=r[5], language=r[6], text=r[7], score=score, **info.get(cid, {}),
            ))
        return out
