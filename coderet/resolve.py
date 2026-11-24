"""Resolve import statements to indexed files, producing `file_deps` edges.

Resolution is heuristic but conservative: external packages (stdlib, npm,
crates.io, ...) are left unresolved rather than guessed. When a suffix match is
ambiguous, the candidates sharing the longest directory prefix with the
importing file are kept.
"""

from __future__ import annotations

import posixpath
import re
from collections import defaultdict
from pathlib import Path

from .store import Store

_JS_EXTS = [".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts", ".d.ts"]


class _FileIndex:
    def __init__(self, rows: list[tuple[int, str]]):
        self.by_path: dict[str, int] = {}
        self.by_suffix: dict[str, list[int]] = defaultdict(list)
        self.by_dir_suffix: dict[str, list[int]] = defaultdict(list)
        self.path_of: dict[int, str] = {}
        # directories that are Python packages (contain __init__.py)
        self.py_packages: set[str] = {
            posixpath.dirname(p) for _, p in rows if posixpath.basename(p) == "__init__.py"
        }
        for fid, path in rows:
            self.by_path[path] = fid
            self.path_of[fid] = path
            parts = path.split("/")
            for k in range(len(parts)):
                self.by_suffix["/".join(parts[k:])].append(fid)
            dparts = parts[:-1]
            for k in range(len(dparts)):
                self.by_dir_suffix["/".join(dparts[k:])].append(fid)
            if not dparts:
                self.by_dir_suffix[""].append(fid)

    def exact(self, path: str) -> list[int]:
        path = posixpath.normpath(path)
        fid = self.by_path.get(path)
        return [fid] if fid is not None else []

    def suffix(self, suffix: str, src: str) -> list[int]:
        return self._closest(self.by_suffix.get(posixpath.normpath(suffix), []), src)

    def dir_files(self, dir_suffix: str, src: str, ext: str) -> list[int]:
        cands = [f for f in self.by_dir_suffix.get(dir_suffix, []) if self.path_of[f].endswith(ext)]
        if not cands:
            return []
        # keep only files of the single closest matching directory
        dirs = {posixpath.dirname(self.path_of[f]) for f in cands}
        best = max(dirs, key=lambda d: (_common_prefix_len(d, src), -len(d)))
        return [f for f in cands if posixpath.dirname(self.path_of[f]) == best]

    def python_absolute(self, parts: list[str], src: str) -> list[int]:
        """Files importable as absolute module `parts` (e.g. ["flask", "app"]).

        A file matches only if the directory holding the top-level package is
        not itself a package, i.e. it could be an import root on sys.path.
        This stops `import typing` from resolving to `src/flask/typing.py`,
        which is only importable as `flask.typing`.
        """
        rel = "/".join(parts)
        for suffix in (rel + ".py", rel + "/__init__.py"):
            ok = []
            for f in self.by_suffix.get(suffix, []):
                path = self.path_of[f]
                root = path[: len(path) - len(suffix)].rstrip("/")
                if root not in self.py_packages:
                    ok.append(f)
            if ok:
                return self._closest(ok, src)
        return []

    def _closest(self, cands: list[int], src: str) -> list[int]:
        if len(cands) <= 1:
            return list(cands)
        score = {f: _common_prefix_len(posixpath.dirname(self.path_of[f]), src) for f in cands}
        best = max(score.values())
        return [f for f in cands if score[f] == best]


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a.split("/"), posixpath.dirname(b).split("/")):
        if x != y:
            break
        n += 1
    return n


def _go_module_path(repo: Path) -> str | None:
    gomod = repo / "go.mod"
    if not gomod.is_file():
        return None
    m = re.search(r"^module\s+(\S+)", gomod.read_text(errors="replace"), re.M)
    return m.group(1) if m else None


def _first(*candidate_lists: list[int]) -> list[int]:
    for c in candidate_lists:
        if c:
            return c
    return []


def resolve_import(idx: _FileIndex, lang: str, src: str, module: str, name: str | None,
                   go_module: str | None) -> list[int]:
    d = posixpath.dirname(src)
    if lang == "python":
        level = len(module) - len(module.lstrip("."))
        rest = module[level:]
        def cands(mod_parts: list[str]) -> list[int]:
            rel = "/".join(mod_parts)
            if level:
                base = d
                for _ in range(level - 1):
                    base = posixpath.dirname(base)
                p = posixpath.join(base, rel) if rel else base
                return _first(idx.exact(p + ".py"), idx.exact(p + "/__init__.py"))
            if not rel:
                return []
            return idx.python_absolute(mod_parts, src)
        parts = rest.split(".") if rest else []
        if name and name != "*":
            hit = cands(parts + [name])  # `from pkg import submodule`
            if hit:
                return hit
        return cands(parts)

    if lang in {"javascript", "typescript", "tsx"}:
        if not module.startswith("."):
            return []
        p = posixpath.normpath(posixpath.join(d, module))
        stem = re.sub(r"\.(js|jsx|mjs|cjs)$", "", p)
        return _first(
            idx.exact(p),
            *[idx.exact(stem + e) for e in _JS_EXTS],
            *[idx.exact(p + "/index" + e) for e in _JS_EXTS],
        )

    if lang in {"c", "cpp"}:
        return _first(idx.exact(posixpath.join(d, module)), idx.suffix(module, src))

    if lang == "go":
        if go_module and (module == go_module or module.startswith(go_module + "/")):
            rel = module[len(go_module):].lstrip("/")
            return idx.dir_files(rel, src, ".go")
        segs = module.split("/")
        if len(segs) < 3:  # stdlib or very short path: don't guess
            return []
        # drop the host/owner prefix progressively; keep at least two segments
        for k in range(1, len(segs) - 1):
            hit = idx.dir_files("/".join(segs[k:]), src, ".go")
            if hit:
                return hit
        return []

    if lang == "java":
        segs = module.split(".")
        if segs[-1] == "*":
            return idx.dir_files("/".join(segs[:-1]), src, ".java")
        hit = idx.suffix("/".join(segs) + ".java", src)
        if not hit and len(segs) > 2:  # static import of a member
            hit = idx.suffix("/".join(segs[:-1]) + ".java", src)
        return hit

    if lang == "rust":
        path = module.split("::{")[0].split(" as ")[0].strip()
        segs = [s for s in path.split("::") if s]
        if not segs or segs[0] not in {"crate", "self", "super"}:
            return []
        head, segs = segs[0], segs[1:]
        if head == "crate":
            look = lambda rel: _first(idx.suffix(rel + ".rs", src), idx.suffix(rel + "/mod.rs", src))
        else:
            base = d if head == "self" else posixpath.dirname(d)
            look = lambda rel: _first(idx.exact(posixpath.join(base, rel) + ".rs"),
                                      idx.exact(posixpath.join(base, rel, "mod.rs")))
        for k in range(len(segs), 0, -1):
            hit = look("/".join(segs[:k]))
            if hit:
                return hit
        return []

    if lang == "ruby":
        if module.startswith("."):
            p = posixpath.join(d, module)
            return _first(idx.exact(p), idx.exact(p + ".rb"))
        return idx.suffix(module.removesuffix(".rb") + ".rb", src)

    if lang == "php":
        segs = module.split("\\")
        for k in range(0, max(1, len(segs) - 1)):
            hit = idx.suffix("/".join(segs[k:]) + ".php", src)
            if hit:
                return hit
        return []

    return []


def resolve_dependencies(store: Store, repo: Path, batch: int = 2000) -> int:
    conn = store.conn
    idx = _FileIndex([(r[0], r[1]) for r in conn.execute("SELECT id, path FROM files")])
    go_module = _go_module_path(repo)
    conn.execute("DELETE FROM file_deps")
    cur = conn.execute(
        "SELECT i.id, i.file_id, i.module, i.name, f.path, f.language "
        "FROM imports i JOIN files f ON f.id = i.file_id ORDER BY i.id"
    )
    edges = 0
    while True:
        rows = cur.fetchmany(batch)
        if not rows:
            break
        out = []
        for imp_id, fid, module, name, path, lang in rows:
            for dst in resolve_import(idx, lang, path, module, name, go_module):
                if dst != fid:
                    out.append((fid, dst, imp_id))
        conn.executemany("INSERT OR IGNORE INTO file_deps VALUES (?, ?, ?)", out)
        edges += len(out)
    conn.commit()
    return edges
