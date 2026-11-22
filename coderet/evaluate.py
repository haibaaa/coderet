"""Retrieval evaluation against a labelled query set.

Query file: JSON Lines, one object per line:

    {"id": "q1", "query": "parse the config file", "relevant": ["src/config.py::load_config"]}

Relevance targets:
    "path/to/file.py"              any chunk of that file
    "path/to/file.py::Symbol"      chunk overlapping the definition of Symbol
                                   (qualified name, e.g. Class.method)
    "path/to/file.py:120-160"      chunk overlapping those lines

Symbol targets are resolved to line spans through the index's definitions
table, so a label stays valid when chunking settings change (a method may live
in its own chunk, inside its class's chunk, or across several windows). If a
symbol cannot be resolved, matching falls back to the chunk's symbol name and
a warning is recorded in the report.

Each target is credited at most once (its best-ranked hit), so duplicate chunks
of one relevant file do not inflate the metrics.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path

from . import __version__
from .search import Searcher, SearchResult
from .store import Store

_LINES = re.compile(r"^(.*):(\d+)-(\d+)$")


@dataclass(frozen=True)
class Target:
    raw: str
    path: str
    symbol: str | None = None
    start: int | None = None
    end: int | None = None
    # line spans of the symbol's definition(s), filled in by resolve_target
    spans: tuple[tuple[int, int], ...] = ()

    @classmethod
    def parse(cls, raw: str) -> "Target":
        if "::" in raw:
            path, sym = raw.split("::", 1)
            return cls(raw, path, symbol=sym)
        m = _LINES.match(raw)
        if m:
            return cls(raw, m.group(1), start=int(m.group(2)), end=int(m.group(3)))
        return cls(raw, raw)

    def matches(self, r: SearchResult) -> bool:
        if r.path != self.path:
            return False
        if self.spans:
            return any(r.start_line <= e and r.end_line >= s for s, e in self.spans)
        if self.symbol is not None:
            return r.symbol is not None and (r.symbol == self.symbol or r.symbol.startswith(self.symbol + "."))
        if self.start is not None:
            return r.start_line <= self.end and r.end_line >= self.start
        return True


def resolve_target(store: Store, target: Target) -> Target:
    """Attach definition line spans to a symbol target (no-op otherwise)."""
    if target.symbol is None:
        return target
    rows = store.conn.execute(
        "SELECT d.start_line, d.end_line FROM definitions d JOIN files f ON f.id = d.file_id "
        "WHERE f.path = ? AND d.qualified = ? ORDER BY d.start_line",
        (target.path, target.symbol),
    ).fetchall()
    return replace(target, spans=tuple((r[0], r[1]) for r in rows)) if rows else target


def load_queries(path: str | Path) -> list[dict]:
    queries = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            if "query" not in obj or not obj.get("relevant"):
                raise ValueError(f"{path}:{n}: each line needs 'query' and non-empty 'relevant'")
            obj.setdefault("id", f"q{n}")
            queries.append(obj)
    return queries


def score_query(results: list[SearchResult], targets: list[Target], ks: list[int]) -> dict:
    credited: set[Target] = set()
    gains: list[int] = []  # binary gain per rank position
    first_rank = None
    for rank, r in enumerate(results, 1):
        hit = next((t for t in targets if t not in credited and t.matches(r)), None)
        if hit is not None:
            credited.add(hit)
            gains.append(1)
            first_rank = first_rank or rank
        else:
            gains.append(0)
    out: dict = {"first_relevant_rank": first_rank}
    n_rel = len(targets)
    for k in ks:
        hits_k = sum(gains[:k])
        out[f"recall@{k}"] = hits_k / n_rel
        out[f"hit@{k}"] = 1.0 if hits_k else 0.0
        dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))
        idcg = sum(1 / math.log2(i + 2) for i in range(min(n_rel, k)))
        out[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
    kmax = max(ks)
    out[f"mrr@{kmax}"] = 1.0 / first_rank if first_rank and first_rank <= kmax else 0.0
    return out


def evaluate(searcher: Searcher, queries: list[dict], modes: list[str], ks: list[int],
             rerank: bool | None = None) -> dict:
    ks = sorted(set(ks))
    depth = max(ks)
    report: dict = {
        "coderet_version": __version__,
        "config_fingerprint": searcher.cfg.fingerprint(),
        "config": searcher.cfg.to_dict(),
        "model_key": getattr(searcher.embedder, "model_key", None),
        "model": searcher.embedder.describe() if searcher.embedder else None,
        "index_stats": searcher.store.stats(getattr(searcher.embedder, "model_key", None)),
        "num_queries": len(queries),
        "ks": ks,
        "modes": {},
        "warnings": [],
    }
    resolved: dict[str, list[Target]] = {}
    for q in queries:
        targets = [resolve_target(searcher.store, Target.parse(t)) for t in q["relevant"]]
        for t in targets:
            if t.symbol is not None and not t.spans:
                report["warnings"].append(f"{q['id']}: no definition found for {t.raw}; "
                                          "falling back to chunk symbol matching")
        resolved[q["id"]] = targets
    for mode in modes:
        per_query = []
        for q in queries:
            targets = resolved[q["id"]]
            results = searcher.search(q["query"], top_k=depth, mode=mode, rerank=rerank)
            m = score_query(results, targets, ks)
            per_query.append({
                "id": q["id"], "query": q["query"], **m,
                "top": [f"{r.path}:{r.start_line}-{r.end_line} {r.symbol or ''}".strip() for r in results[:5]],
            })
        metric_names = [k for k in per_query[0] if k not in {"id", "query", "top", "first_relevant_rank"}] if per_query else []
        summary = {name: round(sum(p[name] for p in per_query) / len(per_query), 4) for name in metric_names}
        report["modes"][mode] = {"summary": summary, "per_query": per_query}
    return report
