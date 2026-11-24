# Part 2: Indexing, Storage and Embeddings

**Share of the implementation: about one third (≈ 33%)**
Code: `coderet/config.py`, `coderet/store.py`, `coderet/embedder.py`,
`coderet/indexer.py`, the command-line skeleton and the `index`, `embed`,
`stats`, `config` and `prune` commands in `coderet/cli.py`, the uv project
setup (`pyproject.toml`, `uv.lock`), and their tests (≈ 850 lines of code). This is the systems-engineering part: it makes the tool
correct, incremental, resumable and reproducible on a CPU.

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
| **2. Indexing & embeddings (this part)** | the repository + Part 1's output | a SQLite index with chunks, keyword index and one vector per chunk |
| 3. Search & evaluation | the index + a query | ranked chunks; accuracy metrics on labelled queries |

**Where this part sits:** in the middle. It walks the repository, calls Part 1
for each changed file, stores everything in one database, and turns every chunk
into a vector with a neural embedding model. Part 3 reads from that database.

---

## 2. What this part does

1. **Finds the files** to index, respecting `.gitignore`.
2. **Detects changes**, so only new or edited files are processed again.
3. **Stores** chunks, the keyword-search index and the dependency graph in a
   single SQLite file.
4. **Embeds** each chunk with CodeRankEmbed, a 137M-parameter model trained for
   code retrieval, in small batches on the CPU.
5. **Caches** every vector under a hash of its content, so no unchanged chunk is
   ever embedded twice, even across configuration changes or model switches.
6. **Makes runs reproducible:** typed configuration, a configuration
   fingerprint, a pinned model revision and a locked environment.

### The interfaces

- **From Part 1** this part calls `parse_source(...)` and
  `resolve_dependencies(...)` and stores the returned objects.
- **To Part 3** it provides the database (`chunks`, `chunks_fts`,
  `embeddings`, graph tables), an embedder with `encode_queries(texts)` and its
  `model_key`, and the `Config` object.

---

## 3. How it works in detail

### 3.1 Configuration (`config.py`)

Settings are typed Python dataclasses in five sections: `model`, `rerank`,
`chunking`, `index` and `search`. They are loaded from a TOML file.

- **Unknown keys are rejected** (`unknown config key 'model.nmae'`), so a typo
  can never silently change an experiment.
- `validate()` checks value ranges, e.g. that the window overlap is smaller
  than the window size.
- `fingerprint()` is the SHA-256 of the whole configuration as sorted JSON.
  Every evaluation report records it, so two results can always be traced to
  their exact settings.

### 3.2 Storage (`store.py`)

One SQLite file (`.coderet/index.db`) holds everything:

| Table | Contents |
|---|---|
| `files` | path, language, SHA-256, size, modification time |
| `chunks` | file, line range, kind, symbol, text, **content hash** |
| `chunks_fts` | FTS5 full-text index (path, symbol, body) for BM25 search |
| `embeddings` | `(content_hash, model_key) → vector` (float32 bytes) |
| `definitions`, `imports`, `refs`, `file_deps` | dependency graph (filled from Part 1) |
| `meta` | index format, configuration, model description |

Design choices:

- **Foreign keys with cascade delete.** Deleting a file deletes its chunks,
  definitions, imports and references automatically. The FTS table can't
  cascade, so its rows are deleted explicitly first.
- **WAL journal mode**, so searches can read while indexing writes.
- **The embedding table is keyed by content, not by chunk.** Vectors belong to
  `(content_hash, model_key)`, not to a chunk ID, which is what makes the cache
  work (section 3.5).
- The FTS tokenizer keeps `_` inside words, so `get_user` stays one token, and
  Part 1's identifier splitting adds the parts `get` and `user`.

### 3.3 File discovery and change detection (`indexer.py`)

**Discovery.** In a git repository it runs
`git ls-files --cached --others --exclude-standard`, which respects
`.gitignore`. Otherwise it walks the directory tree, skipping `.git`,
`node_modules`, `.venv`, `build`, `dist`, and so on. Paths are sorted, so the
processing order is deterministic. Binary files (a NUL byte in the first 8 KB)
and files over 1 MB are skipped.

**Change detection** for every file:

```
same size AND same modification time as stored?  ── yes ──► unchanged (no read)
        │ no
        ▼
read file, compute SHA-256 — same as stored?     ── yes ──► update mtime only
        │ no
        ▼
re-parse with Part 1, delete old rows, insert new rows
```

Files that disappeared from the repository are deleted from the index. A
**format fingerprint** (the index format version plus the chunking settings) is
stored too. If chunking settings change, every file is re-parsed, because the
chunks would differ.

**Bounded memory.** Files are processed one at a time and committed every 32
files, so memory use doesn't grow with repository size.

### 3.4 Writing a file (`Indexer._write_file`)

For each chunk it:
1. Builds the **exact text the model will see** (`embed_text`):
   `File: <path>\nSymbol: <symbol>\n\n<code>`.
2. Computes `content_hash = SHA-256(that text)`.
3. Inserts the chunk, and its FTS row with identifier sub-tokens added
   (`getUserProfile` → `get user profile`).
4. Inserts definitions, imports and references, each linked to the smallest
   chunk containing its line.

### 3.5 The embedding cache and resumable embedding

```
embedding key = ( SHA-256(text given to model) , model_key )
model_key     = "st|<model name>@<revision>|len=<max tokens>|dp=<hash of document prefix>"
```

Consequences:

- **Unchanged chunks are never re-embedded**, even after a full re-parse,
  because the hash is the same.
- **Identical chunks share one vector** (the same code in two places is embedded
  once).
- **Several models coexist.** Switching to another checkpoint embeds only the
  missing vectors; switching back costs nothing.
- **Anything that changes the vector changes the key:** model, revision,
  maximum length or document prefix.

`embed_pending` repeatedly:
1. selects up to 64 distinct content hashes that lack a vector for the active
   model, **longest first**, so memory problems show up immediately and
   batches contain similar lengths;
2. encodes them in model batches of 8;
3. **commits immediately**.

If the run is interrupted (Ctrl-C, crash, power loss), running it again simply
continues with what is still missing.

### 3.6 Embedding backends (`embedder.py`)

- **`SentenceTransformerEmbedder`** loads the model lazily, on first use.
  It runs on the CPU in full 32-bit precision; there is no quantisation,
  because quality is the priority. It L2-normalises every vector, so cosine
  similarity becomes a dot product. Queries get the model's instruction prefix
  (`"Represent this query for searching relevant code: "`). For
  reproducibility, the torch seed is fixed and deterministic algorithms are
  requested.
- **`HashingEmbedder`** is a deterministic, model-free stand-in that hashes
  identifier sub-tokens into a vector. It lets the whole pipeline be tested in
  under a second without downloading anything. It is used only for tests.
- **`CrossEncoderReranker`** loads the optional reranking model used by Part 3.

### 3.7 Command-line interface and packaging

- `cli.py` uses argparse with one sub-command per action and shared flags
  (`--repo`, `--config`, `--index`). Progress goes to stderr and results to
  stdout, so output can be piped.
- This part owns `index`, `embed`, `stats` (counts, missing embeddings, cached
  models, configuration fingerprints), `config` (prints effective settings) and
  `prune` (deletes vectors no chunk uses).
- **uv project:** `uv sync` installs everything from a lockfile. torch is always
  taken from the CPU-only wheel index, so no CUDA download is needed.
- **Compatibility bug found and fixed:** CodeRankEmbed's model code calls a
  function that transformers 5.x removed (`get_extended_attention_mask`).
  We diagnosed it and pinned `transformers>=4.48,<5`.

### 3.8 Tests that cover this part

`test_incremental_and_cache` (modify one file and delete another: exactly 1
update, 1 removal and 1 new embedding), `test_chunking_change_forces_reparse`,
`test_config_rejects_unknown_keys`, plus the end-to-end `test_index_and_search`.
All tests use the hashing embedder, so they need no network access.

---

## 4. Results attributable to this part (Flask, 194 files, 2-core laptop CPU)

| Measurement | Value |
|---|---|
| Embedding 1307 chunks | about 18 minutes (1.22 chunks/s on average) |
| Re-index after a configuration change forced all 194 files to be re-parsed | **4.7 s, 0 chunks re-embedded**, instead of ~18 min |
| Model load (once per process) | 12.1 s |
| Index size on disk | 13.5 MB |
| Interrupted run | the first run on our own repository crashed during embedding (the transformers bug); the rerun found all 15 files unchanged, skipped parsing, and embedded only what was missing |
| Deduplication | 156 chunks in our own repository needed only 155 vectors (two identical chunks share one) |
| Memory | bounded by the model, one batch and one 2048-row scan block; independent of repository size |

---

## 5. How to present this part

### 5.1 Five-minute talk outline

1. **The problem (30 s).** "Embedding code on a CPU is slow, about one chunk
   per second. If every change meant re-embedding everything, the tool would be
   unusable for experiments."
2. **The pipeline (60 s).** Walk through the change-detection flowchart: size
   and time, then hash, then re-parse. Mention that deleted files disappear.
3. **The key idea, content-addressed caching (90 s).** Write the cache key on
   the board: `(SHA-256 of model input, model_key)`. Show three consequences:
   unchanged chunks are free, duplicates are shared, and models can be
   switched.
4. **Demo (60 s).**
   ```bash
   uv run coderet index ../flask      # second run: everything "unchanged", 0 embedded
   uv run coderet stats --repo ../flask
   ```
   Point at `chunks_missing_embedding: 0` and the configuration fingerprint.
5. **Evidence (45 s).** "After a change that forced a full re-parse of 194
   files, re-indexing took 4.7 seconds and embedded nothing, instead of 18
   minutes."
6. **Reproducibility (15 s).** Strict configuration, fingerprint, pinned model,
   locked environment.

### 5.2 Likely questions and answers

- **Why SQLite instead of a vector database such as FAISS?**
  The corpus is thousands to tens of thousands of chunks. An exact scan is
  fast enough (about 0.18 s per query including encoding), avoids the recall
  loss of approximate indexes, and keeps the chunks, keyword index, graph and
  vectors in one file with transactions.
- **Why hash the text instead of using the file hash?**
  A file edit usually changes only one function. Hashing each chunk's model
  input means only that chunk is re-embedded.
- **What if the model changes but the text doesn't?**
  The `model_key` changes, so new vectors are computed and stored alongside the
  old ones.
- **Why check size and modification time before hashing?**
  Reading and hashing every file on every run is wasted I/O. Size plus
  nanosecond mtime is a safe first filter, and the hash confirms real changes.
- **Is it deterministic?**
  File order is sorted, configuration is fingerprinted, the model revision can
  be pinned, torch is seeded, and ties in search are broken by chunk ID. The
  evaluation numbers reproduced exactly between runs.
- **Why no quantisation for speed?**
  The project prioritises retrieval quality over speed, and memory wasn't a
  constraint, so full precision was kept.

### 5.3 What to emphasise, and what not to claim

- Emphasise: incremental and resumable indexing, the content-addressed cache
  and its measured effect, strict reproducibility, and the CPU-only packaging.
- Don't claim caching or incremental indexing are new ideas; they're standard
  engineering. The contribution is applying them correctly, so that results
  stay reproducible, and measuring the effect.
