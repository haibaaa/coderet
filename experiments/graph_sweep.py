"""Tune the test-aware graph reranker on the DEVELOPMENT queries only, then
evaluate the selected setting once on the held-out queries.

    FLASK=~/r/flask uv run python experiments/graph_sweep.py

Selection rule (fixed before running): highest dev MRR@10 in dense mode;
ties broken by dev Acc@1, then by smaller graph_beta, then smaller graph_top,
then test_weight closer to 1 (simpler = less intervention).
"""

from __future__ import annotations

import copy
import itertools
import json
import os
from pathlib import Path

from coderet.config import load_config
from coderet.embedder import make_embedder
from coderet.evaluate import evaluate, load_queries
from coderet.search import Searcher
from coderet.store import Store

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "results"
repo = Path(os.environ.get("FLASK", Path.home() / "r" / "flask"))
base_cfg = load_config(None, repo)
store = Store(repo / base_cfg.index.path)
emb = make_embedder(base_cfg.model, seed=base_cfg.seed)
dev = load_queries(ROOT / "examples" / "flask_queries.jsonl")
heldout = load_queries(ROOT / "examples" / "flask_queries_heldout.jsonl")
KS = [1, 3, 5, 10]


def run(queries, modes, **search_overrides):
    cfg = copy.deepcopy(base_cfg)
    for k, v in search_overrides.items():
        setattr(cfg.search, k, v)
    return evaluate(Searcher(store, cfg, emb), queries, modes, KS)


grid = list(itertools.product([0.5, 1.0, 2.0, 4.0], [5, 10], [1.0, 0.5]))
sweep = []
baseline_dev = run(dev, ["dense", "hybrid"])
for beta, top, tw in grid:
    rep = run(dev, ["dense", "hybrid"], graph_rerank=True, graph_beta=beta, graph_top=top, test_weight=tw)
    row = {"graph_beta": beta, "graph_top": top, "test_weight": tw,
           **{f"{m}_{k}": rep["modes"][m]["summary"][k] for m in ("dense", "hybrid")
              for k in ("hit@1", "recall@5", "mrr@10", "ndcg@10")}}
    sweep.append(row)
    print(row, flush=True)

best = sorted(sweep, key=lambda r: (-r["dense_mrr@10"], -r["dense_hit@1"], r["graph_beta"],
                                    r["graph_top"], abs(1 - r["test_weight"])))[0]
chosen = {k: best[k] for k in ("graph_beta", "graph_top", "test_weight")}
print("chosen on dev:", chosen, flush=True)

(OUT / "graph-sweep-dev.json").write_text(json.dumps(
    {"baseline": {m: baseline_dev["modes"][m]["summary"] for m in ("dense", "hybrid")},
     "sweep": sweep, "chosen": chosen}, indent=2))

# Dev reports with full per-query data for the chosen setting.
(OUT / "graph-dev.json").write_text(json.dumps(
    run(dev, ["dense", "hybrid"], graph_rerank=True, **chosen), indent=2))

# Held-out: baseline and chosen setting, evaluated once.
(OUT / "heldout-baseline.json").write_text(json.dumps(
    run(heldout, ["dense", "lexical", "hybrid"]), indent=2))
(OUT / "heldout-graph.json").write_text(json.dumps(
    run(heldout, ["dense", "lexical", "hybrid"], graph_rerank=True, **chosen), indent=2))
print("done", flush=True)
