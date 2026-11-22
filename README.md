# coderet

An offline, CPU-only code retrieval tool for research and experiments. It
splits a repository into syntax-aware chunks with tree-sitter, extracts a
dependency graph (definitions, calls, imports), stores everything in SQLite,
and answers natural-language queries with exact dense search plus BM25.

The design follows the constraints in `CLAUDE.md`: retrieval quality comes
first, the default model runs in full precision on CPU, embeddings are cached,
indexing is incremental and resumable, and memory stays bounded.

## Install

This is a [uv](https://docs.astral.sh/uv/) project:

```bash
uv sync                      # creates .venv from uv.lock (CPU-only torch)
uv run coderet --help        # or: source .venv/bin/activate
```

`uv.lock` pins every dependency version, which keeps experiments
reproducible. torch always comes from the CPU wheel index (see
`[tool.uv.sources]` in `pyproject.toml`), so no CUDA runtime is downloaded.

> **Note:** CodeRankEmbed's remote modeling code does not work with
> transformers 5.x, so `pyproject.toml` pins `transformers>=4.48,<5`.

## Usage

```bash
coderet index path/to/repo            # parse, resolve deps, embed (incremental)
coderet index path/to/repo --no-embed # parse only; run `coderet embed` later
coderet search "where is the retry backoff computed" --repo path/to/repo
coderet search "parse config" --mode lexical --glob 'src/*' --lang python --json
coderet symbol load_config            # definitions
coderet callers load_config           # call sites + enclosing definition
coderet callees Indexer.scan          # calls made, resolved to definitions
coderet deps src/app.py               # imports (resolved) and importers
coderet stats
coderet eval queries.jsonl --modes dense,lexical,hybrid --output report.json
coderet prune [--only-active]         # drop unused cached vectors
```

All commands take `--repo`, `--config` and `--index`. Progress goes to stderr
and results go to stdout.

## How it works

| Stage | Details |
|---|---|
| Discovery | `git ls-files` (respects `.gitignore`), or a directory walk with excluded dirs. Binary and oversized files are skipped. |
| Change detection | A (size, mtime) fast path, then sha256. Changing the chunking settings forces a full re-parse. Deleted files are removed. |
| Chunking | One chunk per function or class. An oversized class is split into its methods plus a header chunk. An oversized function without nested definitions becomes overlapping line windows. Code outside definitions becomes `module` chunks. Languages without a grammar use line windows. |
| Languages (syntax-aware) | Python, JavaScript, TypeScript/TSX, Java, Go, Rust, C, C++, Ruby, C#, PHP. Markdown, YAML, shell and similar files are indexed as text. |
| Dependency graph | Definitions with qualified names, call references, and imports. Imports are resolved to files using per-language rules (Python relative/absolute, JS relative, C includes, Go via `go.mod`, Java packages, Rust `crate::`/`super::`, Ruby, PHP). External packages are left unresolved rather than guessed. |
| Embedding | Batched (`model.batch_size`), committed every `index.embed_fetch` chunks, so an interrupted run resumes where it stopped. The cache key is (sha256 of the embedded text, model key), so identical chunks share one vector and you can switch between checkpoints without recomputing. |
| Dense search | An exact (exhaustive) cosine scan streamed from SQLite `scan_batch` rows at a time. No approximate index, so no recall loss. |
| Lexical search | SQLite FTS5 BM25 with column weights (path 2, symbol 4, body 1). Identifiers are also indexed as camelCase/snake_case sub-tokens. |
| Fusion | Weighted reciprocal rank fusion (`rrf_k`, `dense_weight`, `lexical_weight`). Ties break by chunk id, so results are deterministic. |
| Reranking | Optional cross-encoder (`[rerank] enabled = true`) over the top `pool` fused candidates. |

## Reproducibility

- `coderet eval` reports include the full config, a config fingerprint, the
  model key, library versions and index statistics.
- Pin `model.revision` to a commit hash for exact model reproducibility.
- Torch runs with a fixed seed and `use_deterministic_algorithms(warn_only=True)`.
- `INDEX_FORMAT_VERSION` in `coderet/__init__.py` invalidates stale indexes
  when the parsing logic changes.

## Evaluation format

One JSON object per line in the query file:

```json
{"id": "q1", "query": "parse the config file", "relevant": ["src/config.py::load_config"]}
```

A target can be `path`, `path::Symbol` (matches `Symbol` and anything nested
under it) or `path:120-160` (line overlap). The metrics are recall@k, hit@k,
nDCG@k and MRR. Each target is credited only once.

`examples/self_queries.jsonl` is a small benchmark built from this repo's own
code. On it, CodeRankEmbed scored MRR@10 0.93 dense, 0.83 hybrid and
0.60 lexical. Ten queries are too few to tune defaults on, so build a labelled
set for your target repositories and compare `--modes` and weights there.

## Tests

```bash
uv run pytest   # uses the model-free "hashing" backend; no downloads needed
```
