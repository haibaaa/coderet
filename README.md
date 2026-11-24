# coderet

Search a code repository with natural-language queries. Runs on a CPU.

## Setup

1. Install [uv](https://docs.astral.sh/uv/):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh        # macOS / Linux
   # Windows (PowerShell): irm https://astral.sh/uv/install.ps1 | iex
   ```

2. Install the project:

   ```bash
   git clone https://github.com/haibaaa/coderet.git
   cd coderet
   uv sync
   ```

3. Check that it works:

   ```bash
   uv run pytest
   ```

## Usage

Index a repository, then search it:

```bash
uv run coderet index path/to/repo
uv run coderet search "where is the session cookie signed" --repo path/to/repo
```

The first `index` downloads the embedding model (~550 MB). Indexing is
incremental: running it again only processes changed files.

## Commands

| Command | What it does |
|---|---|
| `index [REPO]` | Parse, chunk and embed a repository |
| `embed` | Embed chunks that are not embedded yet |
| `search QUERY` | Search the index |
| `symbol NAME` | Find where a function or class is defined |
| `callers NAME` | Find where a function is called |
| `callees NAME` | List the functions a function calls |
| `deps PATH` | Show a file's imports and the files that import it |
| `stats` | Show index statistics |
| `eval QUERIES` | Evaluate search accuracy on a labelled query file |
| `config` | Print the active configuration |
| `prune` | Delete cached embeddings that are no longer used |

Run `uv run coderet <command> --help` for details.

## Flags

**All commands**

| Flag | Meaning |
|---|---|
| `--repo PATH` | Repository to use (default: current directory) |
| `--config FILE` | Config file (default: `<repo>/coderet.toml` if it exists) |
| `--index FILE` | Index database (default: `<repo>/.coderet/index.db`) |

**`index`**

| Flag | Meaning |
|---|---|
| `--no-embed` | Parse only; run `embed` later |

**`search`**

| Flag | Meaning |
|---|---|
| `-k N` | Number of results |
| `--mode dense\|lexical\|hybrid` | Search method (default: `hybrid`) |
| `--glob PATTERN` | Only search matching paths, e.g. `'src/*'` |
| `--lang LANG` | Only search one language, e.g. `python` |
| `--pretty` | Syntax-highlighted results in a scrollable pager |
| `--no-pager` | With `--pretty`: print directly instead of opening `less` |
| `--theme NAME` | With `--pretty`: colour theme (default `ansi_dark`; also `ansi_light`, `monokai`, ...) |
| `--lines N` | Lines shown per result (default: 8, or all with `--pretty`) |
| `--full` | Show whole chunks |
| `--with-deps` | Show where each result is called from |
| `--json` | JSON output |
| `--rerank` / `--no-rerank` | Turn the cross-encoder reranker on or off |

**`eval`**

| Flag | Meaning |
|---|---|
| `--modes LIST` | Modes to compare, e.g. `dense,lexical,hybrid` |
| `--k LIST` | Cut-offs, e.g. `1,5,10` |
| `--output FILE` | Save the full JSON report |
| `--rerank` / `--no-rerank` | Turn the cross-encoder reranker on or off |

**`callers`**: `--limit N` caps the number of call sites.
**`prune`**: `--only-active` also deletes embeddings from other models.

## Examples

```bash
uv run coderet search "parse config" --repo ../flask --mode dense -k 5
uv run coderet search "session cookie" --repo ../flask --pretty --with-deps
uv run coderet callers load_config --repo ../flask
uv run coderet deps src/flask/app.py --repo ../flask
uv run coderet eval examples/flask_queries.jsonl --repo ../flask --modes dense,lexical,hybrid
```

## Configuration

Copy `coderet.example.toml` to `<repo>/coderet.toml` to change the model,
chunk sizes or search settings. Every option is listed in that file.
