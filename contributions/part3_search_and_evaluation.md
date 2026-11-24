# Part 3: Search, Ranking and Evaluation

**Share of the implementation: about one third (≈ 33%)**
Code: `coderet/search.py`, `coderet/evaluate.py`, `coderet/pretty.py`, the
test-aware part of `coderet/graph.py` (`is_test_path`,
`callee_definition_chunks`), the `search` and `eval` commands in
`coderet/cli.py`, the benchmark query sets in `examples/`, the experiment
scripts (`experiments/run_flask.sh`, `experiments/graph_sweep.py`,
`experiments/latency.py`), the significance test, and their tests (≈ 780 lines
of code, plus 72 hand-written benchmark queries, each checked against the Flask
source). This is the part that
answers queries and proves, with numbers, how well the system works.

---

## 1. Project overview

CodeRet lets a developer search a code repository with plain-English questions
such as *"where is the session cookie signed?"* and get back the exact
functions that implement the behaviour, together with who calls them and what
they import. It runs entirely on a CPU.

The system is a pipeline of three stages, each built by one team member:

```
 source files ──► [1] CODE ANALYSIS ──► [2] INDEXING & EMBEDDINGS ──► [3] SEARCH & EVALUATION ──► results
                  parse, chunk,          store, cache, embed            dense + BM25 search,
                  extract dependency     chunks in SQLite               fusion, reranking,
                  graph                  (incremental)                  accuracy measurement
```

| Part | Input | Output |
|---|---|---|
| 1. Code analysis | raw bytes of one source file | chunks, definitions, imports, call references |
| 2. Indexing & embeddings | the repository + Part 1's output | a SQLite index with chunks, keyword index and one vector per chunk |
| **3. Search & evaluation (this part)** | the index + a query | ranked chunks; accuracy metrics on labelled queries |

**Where this part sits:** at the end. It takes a query, finds the best chunks in
the index built by Part 2, and presents them. It also contains the evaluation
framework and the benchmark that measure all three parts.

---

## 2. What this part does

1. **Dense (semantic) search:** an exact cosine-similarity scan over all chunk
   vectors, with bounded memory.
2. **Lexical (keyword) search:** BM25 through SQLite's FTS5, with weighted
   fields.
3. **Fusion:** combines the two rankings with weighted reciprocal rank fusion.
4. **Test-aware graph reranking:** a new reranker that follows calls from
   highly ranked tests to the implementation they exercise.
5. **Evaluation:** labelled benchmarks (42 development + 30 held-out queries),
   metrics (precision, recall, F1, accuracy, MRR, nDCG) and experiment scripts.
6. **Presentation:** syntax-highlighted, scrollable output (`--pretty`).

### The interface to Part 2

This part only needs:
- the database tables `chunks`, `chunks_fts`, `embeddings` (and, for the
  reranker, the graph tables `refs` and `definitions`);
- `embedder.encode_queries([query])` and `embedder.model_key`;
- the `Config` object (`search` section).

It doesn't need to know how chunks were produced or how vectors were cached.

---

## 3. How it works in detail

### 3.1 Dense search (`Searcher.dense`)

1. Encode the query into a 768-dimensional unit vector `q` (via Part 2's
   embedder).
2. Stream `(chunk_id, vector)` rows for the active model from SQLite, **2048
   rows at a time**.
3. For each block `E` (2048 × 768), compute scores `E · q`. Vectors are
   normalised, so this is cosine similarity.
4. Merge each block into a **running top-k** with `numpy.argpartition`, which
   keeps only the best k scores so far.
5. Sort the final top-k, breaking ties by chunk ID so results are deterministic.

Properties: **exact** (every vector is compared, so there's no approximate-index
recall loss); memory is `O(block size × 768 + k)` regardless of corpus size;
time is linear in the number of chunks. It also checks the vector dimension and
warns if some chunks haven't been embedded yet. Optional filters (`--glob`,
`--lang`) are applied in SQL.

### 3.2 Lexical search (`Searcher.lexical`)

- The query is split into words **and** identifier sub-tokens
  (`getUserProfile` → `getuserprofile`, `get`, `user`, `profile`), then
  combined with `OR`.
- SQLite FTS5 ranks matches with **BM25**:

  `score(q,c) = Σ_t IDF(t) · f(t,c)(k1+1) / ( f(t,c) + k1(1 − b + b·|c|/avgdl) )`,
  with `k1 = 1.2` and `b = 0.75`.

- **Field weights:** path × 2, symbol name × 4, body × 1. A word in a function's
  name counts most.

### 3.3 Weighted reciprocal rank fusion (`Searcher.search`)

Each retriever returns its top 100 candidates. The fused score uses only
ranks, so the two very different score scales never need to be calibrated:

```
score(c) = λ_dense / (60 + rank_dense(c))  +  λ_lex / (60 + rank_lex(c))
```

A chunk missing from one list gets no contribution from that list. Three modes
are available: `dense`, `lexical` and `hybrid` (the default).

### 3.4 Test-aware graph reranking (new)

**Observation from error analysis:** tests and documentation often describe a
behaviour in words closer to the query than the implementation itself. For
example, `test_max_cookie_size` ranked above the `Response` class that
implements the cookie-size warning.

**Idea:** if a test ranks highly, the code it calls is probably relevant.
Part 1's call graph tells us exactly what each test calls.

```
b(c) = 1 / (60 + rank(c))                          # rank-based score (works in every mode)
for each test chunk t in the top m results:
    D(t) = non-test definitions called by t        # call names resolved with Part 1's scope rule
    for each c in D(t):  score(c) += β · b(t) / |D(t)|
test chunks themselves:  score(t) = γ · b(t)       # optional demotion, γ ≤ 1
```

- Test code is detected by convention: `tests/`, `test_*.py`, `*_test.go`,
  `*.spec.ts`, `conftest.py`, `*Test.java`, ...
- The boost is split across a test's callees, so a test that calls 20
  functions spreads its evidence thinly.
- A chunk *outside* the original list can enter the ranking if a highly ranked
  test calls it.
- It uses no extra model and costs one database query per test in the top m.

### 3.5 Evaluation framework (`evaluate.py`)

**Query file format** (one JSON object per line):

```json
{"id": "session-sign", "query": "create the serializer that signs the session cookie with the secret key",
 "relevant": ["src/flask/sessions.py::SecureCookieSessionInterface.get_signing_serializer"]}
```

**Labels that don't depend on chunking.** A target `path::Symbol` is resolved
through the definitions table to that symbol's **line span**. Any result in the
same file that overlaps the span counts as a hit. The same label is therefore
correct whether a method is its own chunk, part of its class's chunk, or split
across windows. Without this, comparing chunking strategies would be impossible:
the first version of the evaluator matched symbol names and silently failed on
one label when a small class was stored as a single chunk.

**Metrics** (each relevant target is credited at most once):

| Metric | Definition |
|---|---|
| Precision@k | relevant hits in top k / k |
| Recall@k | relevant targets found in top k / number of targets |
| F1@k | harmonic mean of precision@k and recall@k |
| Accuracy@1 | 1 if the first result is relevant |
| MRR@10 | 1 / rank of the first relevant result (0 if beyond 10) |
| nDCG@10 | rank-discounted gain, normalised by the ideal ranking |

Each report stores the configuration, its fingerprint, the model, index
statistics and per-query results, so any number can be traced back.

**Benchmark construction.**
- *Development set:* 42 queries over Flask, written to paraphrase behaviour
  rather than quote identifiers, and checked to resolve to real definitions.
- *Held-out set:* 30 more queries, written **before** the reranker existed, with
  no overlap with the development set. The file was frozen and its SHA-256
  recorded (`14459c77…`). Tuning used only the development set; the chosen
  setting was run on the held-out set exactly once.

**Statistics.** Differences are tested with a paired randomisation test on
per-query reciprocal rank (10,000 sign flips), and wins and losses are reported.

### 3.6 Result display (`pretty.py`)

`--pretty` prints syntax-highlighted code with real file line numbers in a
scrollable pager. It was redesigned after testing in a narrow terminal:

- no boxes; headers wrap instead of being cut off;
- code never wraps (wrapping destroys indentation); it scrolls sideways in
  `less -S`;
- common indentation is removed, but line numbers stay correct;
- colours follow the terminal's own theme;
- when piped, output is plain text with no colour codes.

### 3.7 Experiment scripts

- `experiments/run_flask.sh`: the main comparison, the fusion-weight sweep, and
  two ablations (no header, fixed windows), each in a separate index.
- `experiments/graph_sweep.py`: the 16-setting reranker sweep on the development
  set, then a single held-out evaluation.
- `experiments/latency.py`: query speed per mode, with the one-time model load
  measured separately.

### 3.8 Tests that cover this part

`test_index_and_search` (all three modes find the right function),
`test_eval_metrics`, `test_symbol_target_independent_of_chunking`,
`test_is_test_path`, `test_graph_rerank_promotes_code_called_by_tests`.

---

## 4. Results attributable to this part

**Main comparison** (development set, 42 queries):

| Mode | Accuracy@1 | Recall@5 | MRR@10 | nDCG@10 |
|---|---|---|---|---|
| BM25 | 0.286 | 0.726 | 0.474 | 0.565 |
| **Dense** | **0.714** | **0.940** | **0.813** | **0.847** |
| Hybrid (equal weights) | 0.595 | 0.893 | 0.741 | 0.789 |

- Dense beats BM25 by +0.339 MRR (p < 0.001).
- **Equal-weight fusion hurts** (−0.072 MRR, p = 0.054). Fusion only becomes
  neutral when the keyword weight is reduced to 0.05.
- **Held-out replication** (30 queries): dense 0.823 > hybrid 0.769 > BM25 0.551 MRR@10.
- **Test-aware reranking:** on the development set, no setting helped (best:
  0.813 → 0.801). On the held-out set, the chosen setting improved dense
  MRR@10 from 0.823 to 0.879 and accuracy@1 from 0.767 to 0.833, with 4
  queries improved and none worsened. **Not statistically significant**
  (p = 0.13 held-out, p = 0.33 pooled over 72 queries).
- **Speed:** 177 ms per dense query and 3.8 ms per BM25 query, after a one-time
  model load.

---

## 5. How to present this part

### 5.1 Five-minute talk outline

1. **The problem (30 s).** "Given a plain-English question, find the right
   function among 1307 chunks, and prove how often we get it right."
2. **Two retrievers (60 s).** Dense means meaning-based (vectors); BM25 means
   word-based. Draw the streaming top-k scan: exact, and memory stays constant.
3. **Demo (60 s).**
   ```bash
   uv run coderet search "where is the session cookie signed" --repo ../flask --pretty
   uv run coderet search "session cookie" --repo ../flask --mode lexical -k 3
   ```
4. **Evaluation method (60 s).** Show one labelled query and explain why labels
   are resolved to line spans: the same test set works for any chunking.
   Mention the frozen held-out set.
5. **Findings (60 s).** Show the main table. The surprise: combining both
   retrievers equally makes results *worse*.
6. **The new reranker (30 s).** The `test_max_cookie_size` → `Response`
   example. Say honestly that it is promising but not significant.

### 5.2 Likely questions and answers

- **Why does hybrid lose when the literature says fusion helps?**
  Rank fusion assumes both rankers are similarly good. Here BM25 is much weaker
  (MRR 0.47 vs 0.81) because our queries paraphrase behaviour instead of naming
  identifiers. When BM25 ranks a wrong chunk first, fusion lets it displace a
  correct dense result. The sweep shows accuracy rising steadily as the keyword
  weight drops.
- **Isn't your query set biased towards dense search?**
  Yes, and we say so. The queries avoid identifiers, which favours semantic
  search. Queries that quote function names or error messages would favour
  BM25. That's why the conclusion is stated for this query distribution only.
- **Why precision@5 so low (≈ 0.19)?**
  Most queries have one correct answer, so precision@5 can be at most 0.2.
  Accuracy@1, recall, MRR and nDCG are the informative metrics here.
- **How do you avoid overfitting the reranker?**
  The held-out queries were written and frozen (hash recorded) before the
  reranker existed. All tuning used the development set with a selection rule
  fixed in advance, and the held-out set was evaluated once.
- **Then why isn't the reranker clearly better?**
  It only changes queries where a test outranks the implementation, a minority.
  With 72 queries there are too few such cases for significance, and the two
  sets point in different directions. We report it as promising but
  unconfirmed.
- **Why an exact scan instead of an approximate index?**
  At this scale it takes milliseconds, gives exact and reproducible rankings,
  and needs no tuning. Approximate indexes would only be needed beyond roughly
  a million chunks.
- **What does the significance test do?**
  For each query we take the difference in reciprocal rank between two
  systems. We then randomly flip the signs of those differences 10,000 times
  and see how often a mean at least as large appears by chance. That fraction
  is the p-value.

### 5.3 What to emphasise, and what not to claim

- Emphasise: exact bounded-memory search, the chunking-independent evaluation
  method, the held-out protocol, honest statistics, and the fusion finding.
- Don't claim hybrid search, BM25 or rank fusion are new; they're standard.
  Don't claim the reranker "works": the correct statement is that it improved
  the held-out set without hurting any held-out query, but the effect isn't
  statistically significant.
