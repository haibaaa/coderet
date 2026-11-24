# Part 1: Code Analysis (Parsing, Chunking and the Dependency Graph)

**Share of the implementation: about one third (≈ 34%)**
Code: `coderet/languages.py`, `coderet/parsing.py`, `coderet/resolve.py`,
the dependency queries in `coderet/graph.py` (`find_definitions`, `callers`,
`resolve_name`, `callees`, `file_dependencies`), the `symbol`, `callers`,
`callees` and `deps` commands in `coderet/cli.py`, and their tests
(≈ 850 lines of code). This is the most algorithm-heavy part: it has to
understand the syntax of 12 programming languages.

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
| **1. Code analysis (this part)** | raw bytes of one source file | chunks, definitions, imports, call references; resolved file-to-file edges |
| 2. Indexing & embeddings | the repository + Part 1's output | a SQLite index with chunks, keyword index and one vector per chunk |
| 3. Search & evaluation | the index + a query | ranked chunks; accuracy metrics on labelled queries |

**Where this part sits:** it is the first stage. It decides *what a searchable
unit is* (a chunk) and *how code is connected* (the dependency graph). Every
later stage depends on the quality of these decisions: a chunk that cuts a
function in half can never be retrieved well, however good the search is.

---

## 2. What this part does

1. **Recognises languages.** It maps file extensions to 12 tree-sitter grammars
   (Python, JavaScript, TypeScript/TSX, Java, Go, Rust, C, C++, Ruby, C#, PHP)
   and treats other text files (Markdown, YAML, shell, ...) as plain text.
2. **Chunks code along its syntax.** Each function, method or class becomes
   one chunk, with its qualified name (`SecureCookieSessionInterface.save_session`)
   and exact line range.
3. **Extracts the dependency graph.** It records every definition, every call
   site and every import statement.
4. **Resolves imports to files.** For example, it works out that
   `from .util import read_text` in `pkg/config.py` refers to `pkg/util.py`,
   while `import os` refers to nothing inside the repository.
5. **Answers graph questions.** "Where is `load_config` defined?", "Who calls
   it?", "What does it call?" and "What does this file import?"

### The interface to Part 2

Part 2 only needs two functions from this part:

```python
parse_source(source: bytes, language: str, cfg: ChunkingConfig) -> ParsedFile
#   ParsedFile.chunks       list of Chunk(start_line, end_line, kind, symbol, text)
#   ParsedFile.definitions  list of Definition(name, qualified, kind, start_line, end_line)
#   ParsedFile.imports      list of Import(module, name, line)
#   ParsedFile.references   list of Reference(name, qualified, line)

resolve_dependencies(store, repo) -> int     # fills the file_deps table, returns #edges
```

Part 2 stores these objects; it never needs to know how they were produced.

---

## 3. How it works in detail

### 3.1 Language registry (`languages.py`)

Every supported language is described by a small declarative record, `LangSpec`:

| Field | Meaning | Python example |
|---|---|---|
| `definitions` | syntax-tree node types that define a named unit | `function_definition`, `class_definition` |
| `wrappers` | nodes that wrap a definition and must stay with it | `decorated_definition` (decorators) |
| `var_declarations` | declarations whose value can be a function | JS/TS `const f = () => ...` |
| `imports` | import statement node types | `import_statement`, `import_from_statement` |
| `calls` | call node type → field holding the callee | `call` → `function` |
| `import_calls` | calls that are really imports | JS `require`, Ruby `require_relative` |

Adding a language means adding one record, not writing new code. Parsers are
created once per language and cached (`get_parser`, `lru_cache`).

### 3.2 Syntax-aware chunking (`parsing.py`)

The file is parsed once with tree-sitter. The chunker then applies a
**recursive algorithm** with a size budget `L = 3000` characters:

```
ChunkScope(node, lines s..e, parent_name):
    for each OUTERMOST definition d below node:        # does not descend into d
        name = parent_name + "." + name(d)
        if size(d) <= L:            emit d as one chunk
        elif d contains definitions: ChunkScope(d, lines(d), name)   # e.g. a big class
        else:                        split d into overlapping line windows
    lines of s..e not covered by any definition  ──►  "module" / "header" chunks
```

What this produces:

| Situation | Result |
|---|---|
| Small function or class | one chunk, e.g. `load_config` |
| Large class | one chunk per method (`Big.method_1`, ...) plus a `header` chunk with the class line, docstring and attributes |
| Huge function with no nested definitions | 60-line windows overlapping by 10 lines, all keeping the function's name |
| Imports, globals, top-level scripts | contiguous `module` chunks |
| Markdown, YAML, unsupported languages | 60-line windows |

Details that matter for correctness:

- **Wrappers.** A decorated Python function or an exported JS function is
  chunked *with* its decorator or `export` keyword, but named after the inner
  definition.
- **Naming.** `definition_name` handles each language's quirks. C/C++ function
  names sit inside a chain of declarators, so it walks the chain. Rust `impl`
  blocks are named `Trait for Type`, and Go methods are named `Receiver.method`.
- **Windows.** If a window exceeds the character budget (very long lines), it
  is shrunk. A single gigantic line (minified code) is truncated, and chunks
  shorter than 20 characters are dropped.
- **Line numbers** are 1-based and inclusive. tree-sitter reports a node ending
  at column 0 of the next line, which would add a phantom line; that case is
  corrected.

### 3.3 Dependency extraction (`parsing.py`)

The same syntax tree is traversed depth-first while tracking the enclosing
definition, which produces qualified names:

- **Definitions:** `Definition(name="m1", qualified="Big.m1", kind, start, end)`.
- **Call references:** the callee expression's text, e.g. `self.foo.bar` or
  `std::vector<int>::push_back`. Generic arguments `<...>` are stripped, and the
  *last identifier* (`bar`) is stored as the name used for matching.
- **Imports:** each language's import syntax is normalised to a module string:

| Language | Source | Stored as |
|---|---|---|
| Python | `from .util import helper, other` | (`.util`, `helper`), (`.util`, `other`) |
| JS/TS | `import x from "./lib/x"`; `require('fs')` | `./lib/x`; `fs` |
| Go | `import "github.com/a/b/pkg"` | `github.com/a/b/pkg` |
| Java | `import static a.b.C.m;` | `a.b.C.m` |
| C/C++ | `#include "util.h"` | `util.h` |
| Rust | `use crate::foo::{bar, baz};` | `crate::foo::{bar, baz}` |

References are later attached to the smallest chunk and definition that contain
their line, which is what makes "who calls X, and from which function"
possible.

### 3.4 Import resolution (`resolve.py`)

An import string is mapped to repository files using per-language rules. A
`_FileIndex` is built once per run with lookup tables for exact paths, every
path suffix (`a/b/c.py`, `b/c.py`, `c.py`) and every directory suffix.

| Language | Rule |
|---|---|
| Python, relative (`..pkg.mod`) | go up one directory per extra dot, then `mod.py` or `mod/__init__.py` |
| Python, absolute | suffix match, but only if the file could really be imported under that name (see below) |
| Python, `from pkg import sub` | first try `pkg/sub.py` (a submodule), then `pkg` |
| JS/TS | only relative paths; try the exact file, each extension (`.ts`, `.tsx`, `.js`, ...), then `index.*` |
| C/C++ | relative to the including file, then suffix match |
| Go | strip the module path from `go.mod`; otherwise a suffix of at least two path segments |
| Java | package path → `a/b/C.java`; `a.b.*` → every file in that directory |
| Rust | `crate::`, `self::`, `super::` paths → `x.rs` or `x/mod.rs` |
| Ruby, PHP | relative path, or namespace → path suffix |

**Design principle: never guess.** External packages (standard library, npm,
crates.io) stay unresolved instead of being matched to a similarly named local
file. When a suffix matches several files, the one sharing the longest
directory prefix with the importing file wins.

**A correctness bug found and fixed.** Auditing Flask showed that naive suffix
matching resolved `import typing` inside Flask to Flask's own
`src/flask/typing.py`. That is wrong: Python can only import that file as
`flask.typing`. The fix encodes Python's actual rule. A file matches an
absolute import only if the directory containing the top-level package is
**not itself a package** (has no `__init__.py`), i.e. it could be on
`sys.path`:

```
import typing  → candidate src/flask/typing.py → root "src/flask" has __init__.py → reject ✔
import flask.app → candidate src/flask/app.py  → root "src" is not a package     → accept ✔
```

### 3.5 Graph queries (`graph.py`, CLI)

| Command | What it answers | How |
|---|---|---|
| `coderet symbol NAME` | where is NAME defined? | `definitions` table, by name or qualified name |
| `coderet callers NAME` | where is NAME called, and from which function? | `refs` by last identifier, plus the smallest enclosing definition |
| `coderet callees NAME` | what does NAME call? | refs inside NAME's line range, each resolved to a definition |
| `coderet deps PATH` | what does a file import, and who imports it? | `imports` + resolved `file_deps` edges |

`resolve_name` decides which of several same-named definitions a call refers
to, preferring the **same file**, then **files this file imports**, then
**anywhere**.

### 3.6 Tests that cover this part

`test_dependencies`, `test_python_absolute_import_not_shadowed_by_package_module`,
`test_large_class_split`, `test_oversized_function_windows`,
`test_window_strategy_keeps_dependencies`. Before the unit tests existed, the
parser was also smoke-tested on a hand-written sample in each of the 12 languages.

---

## 4. Results attributable to this part (Flask, 194 files)

| Measurement | Value |
|---|---|
| Time to parse, chunk, extract and resolve all 194 files | 0.6 s |
| Chunks / definitions / call references / imports | 1307 / 1622 / 3961 / 648 |
| Chunk types | 752 functions, 51 classes, 30 class headers, 99 module chunks, 375 windows (mostly docs) |
| Files with syntax errors | 0 |
| Import edges, naive suffix matching | 343, of which **28 (8.2%) wrong** |
| Import edges, package-aware rule | 316, **0 wrong**; no in-repo import left unresolved; no ambiguous matches |
| **Chunking ablation:** syntax-aware vs fixed 60-line windows | **+8.3 Recall@5 points**, top-1 accuracy 0.714 vs 0.524, MRR@10 0.813 vs 0.651 (p = 0.017) |

The ablation is the strongest evidence for this part. With everything else held
fixed, replacing these chunks with naive line windows makes search clearly and
significantly worse. A published study of AST-based chunking (cAST) found a
+4.3 Recall@5 gain in the same direction.

---

## 5. How to present this part

### 5.1 Five-minute talk outline

1. **The problem (30 s).** "Search can only return what it has indexed. If a
   function is split across two chunks, or glued to its neighbours, it can't be
   found. And a list of snippets doesn't tell you who calls what."
2. **The idea (60 s).** Show a Python class and how it becomes chunks: one per
   method, plus a header. Say that one recursive rule, "fit it whole, otherwise
   split along nested definitions, otherwise use windows", handles every case.
3. **Multi-language design (45 s).** Show one `LangSpec` record. "Adding a
   language means writing a table, not code."
4. **Dependency graph (60 s).** Live demo:
   ```bash
   uv run coderet callers load_config --repo ../flask
   uv run coderet deps src/flask/sessions.py --repo ../flask
   ```
5. **Correctness (60 s).** Tell the `import typing` story: we audited the
   graph, found 8.2% false edges, and fixed them by encoding Python's real
   import rule. Emphasise that we measured before claiming.
6. **Evidence (45 s).** The ablation table: syntax-aware chunking beats windows
   by 8.3 Recall@5 points (p = 0.017), consistent with published work.

### 5.2 Likely questions and answers

- **Why tree-sitter rather than Python's `ast` module?**
  `ast` only parses Python. tree-sitter gives one uniform interface for 12+
  languages, tolerates syntax errors (it still produces a tree), and reports
  exact positions.
- **Why 3000 characters?**
  The embedding model reads up to 1024 tokens; 3000 characters of code is
  roughly 750–1000 tokens, so a chunk is almost never truncated by the model.
  The value is configurable, and changing it automatically re-parses the index.
- **What happens with a 500-line function?**
  It has no nested definitions, so it becomes overlapping 60-line windows that
  all carry the function's name. The 10-line overlap means a statement near a
  boundary appears whole in at least one window.
- **How accurate are call edges?**
  They match by name, so `obj.save()` cannot tell which class's `save` is
  meant without type inference. We reduce errors with the scope preference
  (same file > imported files > elsewhere). This is a known limitation, stated
  in the report.
- **How do you know import resolution is correct?**
  We audited every Python import in Flask. Every resolved edge points to a
  repository module, no in-repository import is unresolved, and nothing
  resolves ambiguously. The regression test reproduces the `typing` bug.
- **Why not resolve third-party imports?**
  They aren't in the repository, and guessing produces exactly the false edges
  we removed.

### 5.3 What to emphasise, and what not to claim

- Emphasise: one recursive chunking rule, a declarative multi-language design,
  a measured correctness fix, and the ablation result.
- Don't claim AST chunking itself is new; cAST and others use it. Our
  contribution is the implementation across 12 languages, the graph built in
  the same pass, and the measurement on our benchmark.
