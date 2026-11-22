import json
import os
import time
from pathlib import Path

import pytest

from coderet import graph
from coderet.config import ChunkingConfig, Config, load_config
from coderet.embedder import make_embedder
from coderet.evaluate import Target, evaluate, score_query
from coderet.indexer import Indexer
from coderet.parsing import parse_source
from coderet.search import Searcher, SearchResult
from coderet.store import Store

FILES = {
    "pkg/__init__.py": "",
    "pkg/config.py": '''
import json
from .util import read_text


def load_config(path):
    """Parse the JSON configuration file at path."""
    return json.loads(read_text(path))
''',
    "pkg/util.py": '''
def read_text(path):
    with open(path) as fh:
        return fh.read()


def unrelated_helper(x):
    return x * 2
''',
    "pkg/app.py": '''
from pkg.config import load_config


class Application:
    def __init__(self, cfg_path):
        self.cfg = load_config(cfg_path)

    def run(self):
        return self.cfg["name"]
''',
    "web/client.js": '''
import { fetchJson } from "./http";
export function getUserProfile(userId) {
  return fetchJson(`/users/${userId}`);
}
''',
    "web/http.js": '''
export async function fetchJson(url) {
  const res = await fetch(url);
  return res.json();
}
''',
}


def make_cfg() -> Config:
    cfg = Config()
    cfg.model.backend = "hashing"
    cfg.model.query_prefix = ""
    cfg.index.use_git = False
    return cfg


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    for rel, text in FILES.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text.lstrip("\n"))
    return tmp_path


def build(repo: Path, cfg: Config | None = None):
    cfg = cfg or make_cfg()
    store = Store(repo / cfg.index.path)
    emb = make_embedder(cfg.model)
    report = Indexer(repo, cfg, store).run(emb)
    return cfg, store, emb, report


def test_index_and_search(repo):
    cfg, store, emb, report = build(repo)
    assert report.added == len(FILES)  # empty __init__.py is tracked even with no chunks
    s = Searcher(store, cfg, emb)
    for mode in ("lexical", "dense", "hybrid"):
        res = s.search("load config json file", top_k=3, mode=mode)
        assert res, mode
        assert res[0].path == "pkg/config.py" and res[0].symbol == "load_config", mode
    res = s.search("user profile", top_k=1, mode="lexical")
    assert res[0].symbol == "getUserProfile"  # camelCase split for BM25


def test_incremental_and_cache(repo):
    cfg, store, emb, report = build(repo)
    first_embedded = report.embedded
    assert first_embedded > 0
    # second run: nothing changes, nothing embedded
    report2 = Indexer(repo, cfg, store).run(emb)
    assert report2.added == report2.updated == report2.removed == 0
    assert report2.embedded == 0
    # modify one file, delete another
    p = repo / "pkg/util.py"
    p.write_text(p.read_text() + "\n\ndef brand_new_function():\n    return 'fresh content'\n")
    os.utime(p, ns=(time.time_ns(), time.time_ns() + 10**9))
    (repo / "web/http.js").unlink()
    report3 = Indexer(repo, cfg, store).run(emb)
    assert report3.updated == 1 and report3.removed == 1
    # only the new chunk needs a vector; unchanged chunks hit the cache
    assert report3.embedded == 1
    assert store.stats(emb.model_key)["chunks_missing_embedding"] == 0


def test_chunking_change_forces_reparse(repo):
    cfg, store, emb, _ = build(repo)
    cfg2 = make_cfg()
    cfg2.chunking.embed_header = False
    report = Indexer(repo, cfg2, store).run(emb)
    assert report.updated == len(FILES) and report.unchanged == 0


def test_dependencies(repo):
    cfg, store, emb, _ = build(repo)
    deps = graph.file_dependencies(store, "pkg/config.py")
    resolved = {i["module"]: i["resolved"] for i in deps["imports"]}
    assert resolved[".util"] == "pkg/util.py"
    assert resolved["json"] is None  # stdlib left unresolved
    assert "pkg/app.py" in deps["imported_by"]
    assert graph.file_dependencies(store, "web/client.js")["imports"][0]["resolved"] == "web/http.js"

    callers = graph.callers(store, "load_config")
    assert any(c["path"] == "pkg/app.py" and c["in_definition"] == "Application.__init__" for c in callers)
    callees = graph.callees(store, "load_config")
    targets = [t for c in callees for t in c["targets"]]
    assert any(t.startswith("pkg/util.py") and "read_text" in t for t in targets)


def test_large_class_split():
    methods = "\n".join(
        f"    def method_{i}(self):\n" + "".join(f"        x{j} = {j}\n" for j in range(10)) + "        return 0\n"
        for i in range(10)
    )
    src = f"class Big:\n    '''doc'''\n    attr = 1\n\n{methods}".encode()
    parsed = parse_source(src, "python", ChunkingConfig(max_chunk_chars=400))
    symbols = [c.symbol for c in parsed.chunks]
    assert "Big.method_0" in symbols and "Big.method_9" in symbols
    header = [c for c in parsed.chunks if c.kind == "header"]
    assert header and "attr = 1" in header[0].text
    assert all(len(c.text) <= 400 for c in parsed.chunks)


def test_oversized_function_windows():
    body = "".join(f"    v{i} = {i}\n" for i in range(200))
    parsed = parse_source(f"def huge():\n{body}".encode(), "python",
                          ChunkingConfig(max_chunk_chars=500, window_lines=30, window_overlap=5))
    assert len(parsed.chunks) > 3
    assert all(c.symbol == "huge" for c in parsed.chunks)
    # windows cover every line
    covered = set()
    for c in parsed.chunks:
        covered.update(range(c.start_line, c.end_line + 1))
    assert covered >= set(range(1, 202))


def test_eval_metrics(repo):
    t = [Target.parse("a.py::f"), Target.parse("b.py")]
    mk = lambda p, s: SearchResult(0, p, 1, 5, "function", s, "python", "", 0.0)
    m = score_query([mk("x.py", None), mk("a.py", "f"), mk("a.py", "f"), mk("b.py", "g")], t, [1, 5])
    assert m["first_relevant_rank"] == 2
    assert m["recall@1"] == 0 and m["recall@5"] == 1.0
    assert m["mrr@5"] == 0.5

    cfg, store, emb, _ = build(repo)
    q = [{"id": "a", "query": "parse JSON configuration", "relevant": ["pkg/config.py::load_config"]}]
    rep = evaluate(Searcher(store, cfg, emb), q, ["lexical", "hybrid"], [1, 5])
    assert rep["modes"]["lexical"]["summary"]["hit@1"] == 1.0
    json.dumps(rep)  # serialisable


def test_symbol_target_independent_of_chunking(repo):
    """A Class.method label must match whether or not the class was split."""
    cfg, store, emb, _ = build(repo)
    from coderet.evaluate import resolve_target
    t = resolve_target(store, Target.parse("pkg/app.py::Application.run"))
    assert t.spans  # resolved via definitions table
    whole_class = SearchResult(0, "pkg/app.py", 4, 10, "class_definition", "Application", "python", "", 0.0)
    other_file = SearchResult(0, "pkg/util.py", 4, 10, "function_definition", "x", "python", "", 0.0)
    assert t.matches(whole_class) and not t.matches(other_file)


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "coderet.toml"
    p.write_text('[model]\nnmae = "x"\n')
    with pytest.raises(ValueError, match="model.nmae"):
        load_config(p)
    p.write_text('[model]\nname = "org/other-model"\nbatch_size = 4\n')
    cfg = load_config(p)
    assert cfg.model.name == "org/other-model" and cfg.model.batch_size == 4
