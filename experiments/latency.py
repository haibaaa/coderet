"""Measure query latency on the Flask index (model load excluded and reported separately).

    FLASK=~/r/flask uv run python experiments/latency.py
Writes experiments/results/latency.json.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

from coderet.config import load_config
from coderet.embedder import make_embedder
from coderet.evaluate import load_queries
from coderet.search import Searcher
from coderet.store import Store

ROOT = Path(__file__).resolve().parent.parent
repo = Path(os.environ.get("FLASK", Path.home() / "r" / "flask"))
cfg = load_config(None, repo)
store = Store(repo / cfg.index.path)
queries = [q["query"] for q in load_queries(ROOT / "examples" / "flask_queries.jsonl")]

t0 = time.perf_counter()
emb = make_embedder(cfg.model, seed=cfg.seed)
emb.encode_queries(["warm-up"])  # loads the model
load_s = time.perf_counter() - t0

searcher = Searcher(store, cfg, emb)
out = {"model_load_seconds": round(load_s, 2), "queries": len(queries)}
for mode in ("lexical", "dense", "hybrid"):
    times = []
    for q in queries:
        t = time.perf_counter()
        searcher.search(q, top_k=10, mode=mode)
        times.append(time.perf_counter() - t)
    out[mode] = {"median_ms": round(1000 * statistics.median(times), 1),
                 "p90_ms": round(1000 * sorted(times)[int(0.9 * len(times)) - 1], 1)}
(ROOT / "experiments" / "results" / "latency.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
