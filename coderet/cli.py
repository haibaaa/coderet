"""Command-line interface: `coderet <command> ...`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config, load_config
from .store import Store


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _setup(args, must_exist: bool = True) -> tuple[Path, Config, Store]:
    repo = Path(args.repo).resolve()
    cfg = load_config(args.config, repo)
    db = Path(args.index) if args.index else repo / cfg.index.path
    if must_exist and not db.is_file():
        raise SystemExit(f"error: no index at {db}. Run `coderet index {args.repo}` first.")
    return repo, cfg, Store(db)


def _embedder(cfg: Config):
    from .embedder import make_embedder
    return make_embedder(cfg.model, seed=cfg.seed)


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------- #

def cmd_index(args) -> None:
    from .indexer import Indexer
    repo, cfg, store = _setup(args, must_exist=False)
    with store:
        embedder = None if args.no_embed else _embedder(cfg)
        report = Indexer(repo, cfg, store, log=_log).run(embedder)
        _print_json({**report.to_dict(), "index": str(store.path)})


def cmd_embed(args) -> None:
    from .indexer import Indexer
    repo, cfg, store = _setup(args)
    with store:
        embedder = _embedder(cfg)
        n = Indexer(repo, cfg, store, log=_log).embed_pending(embedder)
        store.set_meta("last_model_key", embedder.model_key)
        store.set_meta(f"model:{embedder.model_key}", json.dumps(embedder.describe(), sort_keys=True))
        store.conn.commit()
        _print_json({"embedded": n, "model_key": embedder.model_key})


def cmd_search(args) -> None:
    from . import graph
    from .search import Searcher
    _, cfg, store = _setup(args)
    with store:
        mode = args.mode or cfg.search.mode
        embedder = _embedder(cfg) if mode in {"dense", "hybrid"} else None
        searcher = Searcher(store, cfg, embedder)
        rerank = True if args.rerank else (False if args.no_rerank else None)
        results = searcher.search(args.query, top_k=args.k, mode=mode, path_glob=args.glob,
                                  language=args.lang, rerank=rerank)
        for w in searcher.warnings:
            _log(f"warning: {w}")
        if args.json:
            out = []
            for r in results:
                d = r.to_dict(with_text=args.full)
                if args.with_deps and r.symbol:
                    d["callers"] = graph.callers(store, r.symbol, limit=10)
                out.append(d)
            _print_json(out)
            return
        if args.pretty:
            from .pretty import render_results
            # pretty output scrolls, so show whole chunks unless --lines is given
            render_results(
                args.query, results,
                max_lines=None if args.full or args.lines is None else args.lines,
                callers_for=(lambda r: graph.callers(store, r.symbol, limit=5)) if args.with_deps else None,
                theme=args.theme, pager=not args.no_pager,
            )
            return
        from .pretty import score_label
        n_lines = 8 if args.lines is None else args.lines
        for i, r in enumerate(results, 1):
            print(f"{i:>2}. {r.path}:{r.start_line}-{r.end_line}  {r.symbol or '(' + r.kind + ')'}  "
                  f"[{score_label(r)}]")
            lines = r.text.split("\n")
            shown = lines if args.full else lines[:n_lines]
            for line in shown:
                print(f"      {line}")
            if len(shown) < len(lines):
                print(f"      ... ({len(lines) - len(shown)} more lines)")
            if args.with_deps and r.symbol:
                cs = graph.callers(store, r.symbol, limit=5)
                if cs:
                    print("      called from: " + ", ".join(f"{c['path']}:{c['line']}" for c in cs))
            print()


def cmd_symbol(args) -> None:
    from . import graph
    _, _, store = _setup(args)
    with store:
        _print_json(graph.find_definitions(store, args.name))


def cmd_callers(args) -> None:
    from . import graph
    _, _, store = _setup(args)
    with store:
        _print_json(graph.callers(store, args.name, limit=args.limit))


def cmd_callees(args) -> None:
    from . import graph
    _, _, store = _setup(args)
    with store:
        _print_json(graph.callees(store, args.name))


def cmd_deps(args) -> None:
    from . import graph
    repo, _, store = _setup(args)
    p = Path(args.path)
    rel = (p.resolve().relative_to(repo) if p.is_absolute() or p.exists() else p).as_posix()
    with store:
        try:
            _print_json(graph.file_dependencies(store, rel))
        except KeyError as e:
            raise SystemExit(f"error: {e.args[0]}")


def cmd_stats(args) -> None:
    _, cfg, store = _setup(args)
    with store:
        from .embedder import make_embedder
        key = make_embedder(cfg.model, seed=cfg.seed).model_key  # no model load
        models = [r[0] for r in store.conn.execute("SELECT DISTINCT model_key FROM embeddings")]
        _print_json({
            **store.stats(key),
            "active_model_key": key,
            "cached_model_keys": models,
            "index": str(store.path),
            "coderet_version": store.get_meta("coderet_version"),
            "config_fingerprint_at_last_index": store.get_meta("config_fingerprint"),
            "config_fingerprint_now": cfg.fingerprint(),
        })


def cmd_eval(args) -> None:
    from .evaluate import evaluate, load_queries
    from .search import Searcher
    _, cfg, store = _setup(args)
    modes = [m.strip() for m in args.modes.split(",")] if args.modes else [cfg.search.mode]
    ks = [int(k) for k in args.k.split(",")]
    with store:
        embedder = _embedder(cfg) if any(m in {"dense", "hybrid"} for m in modes) else None
        rerank = True if args.rerank else (False if args.no_rerank else None)
        report = evaluate(Searcher(store, cfg, embedder), load_queries(args.queries), modes, ks, rerank)
    for w in report["warnings"]:
        _log(f"warning: {w}")
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        _log(f"wrote {args.output}")
    for mode, data in report["modes"].items():
        print(f"[{mode}] " + "  ".join(f"{k}={v:.4f}" for k, v in data["summary"].items()))


def cmd_config(args) -> None:
    repo = Path(args.repo).resolve()
    cfg = load_config(args.config, repo)
    _print_json({**cfg.to_dict(), "_fingerprint": cfg.fingerprint()})


def cmd_prune(args) -> None:
    _, cfg, store = _setup(args)
    with store:
        keep = None
        if args.only_active:
            keep = [_embedder(cfg).model_key]
        n = store.prune_embeddings(keep)
        store.conn.execute("VACUUM")
        _print_json({"deleted_embeddings": n})


# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo", default=".", help="repository root (default: .)")
    common.add_argument("--config", help="config TOML (default: <repo>/coderet.toml if present)")
    common.add_argument("--index", help="index database path (default: <repo>/.coderet/index.db)")

    p = argparse.ArgumentParser(prog="coderet", description="Offline code retrieval research tool.")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("index", parents=[common], help="incrementally index a repository (and embed)")
    s.add_argument("repo_pos", nargs="?", metavar="REPO", help="repository root (same as --repo)")
    s.add_argument("--no-embed", action="store_true", help="parse and store only; embed later")
    s.set_defaults(func=cmd_index)

    s = sub.add_parser("embed", parents=[common], help="embed chunks missing vectors for the configured model")
    s.set_defaults(func=cmd_embed)

    s = sub.add_parser("search", parents=[common], help="search the index")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=None, help="number of results")
    s.add_argument("--mode", choices=["hybrid", "dense", "lexical"])
    s.add_argument("--glob", help="restrict to paths matching a GLOB, e.g. 'src/*'")
    s.add_argument("--lang", help="restrict to a language, e.g. python")
    s.add_argument("--rerank", action="store_true", help="force cross-encoder reranking")
    s.add_argument("--no-rerank", action="store_true", help="disable reranking")
    s.add_argument("--json", action="store_true")
    s.add_argument("--full", action="store_true", help="print/return full chunk text")
    s.add_argument("--lines", type=int, default=None,
                   help="snippet lines per result (default: 8, or the whole chunk with --pretty)")
    s.add_argument("--pretty", action="store_true",
                   help="syntax-highlighted output in a scrollable pager (for humans)")
    s.add_argument("--no-pager", action="store_true", help="with --pretty: print directly, no pager")
    s.add_argument("--theme", default="monokai",
                   help="with --pretty: pygments colour theme, e.g. monokai, github-dark, solarized-light")
    s.add_argument("--with-deps", action="store_true", help="show call sites of each result's symbol")
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("symbol", parents=[common], help="find definitions by name")
    s.add_argument("name")
    s.set_defaults(func=cmd_symbol)

    s = sub.add_parser("callers", parents=[common], help="call sites of a symbol")
    s.add_argument("name")
    s.add_argument("--limit", type=int, default=100)
    s.set_defaults(func=cmd_callers)

    s = sub.add_parser("callees", parents=[common], help="calls made by a symbol, resolved to definitions")
    s.add_argument("name")
    s.set_defaults(func=cmd_callees)

    s = sub.add_parser("deps", parents=[common], help="imports of a file and files importing it")
    s.add_argument("path")
    s.set_defaults(func=cmd_deps)

    s = sub.add_parser("stats", parents=[common], help="index statistics")
    s.set_defaults(func=cmd_stats)

    s = sub.add_parser("eval", parents=[common], help="evaluate retrieval on a labelled JSONL query set")
    s.add_argument("queries")
    s.add_argument("--modes", help="comma-separated, e.g. dense,lexical,hybrid")
    s.add_argument("--k", default="1,5,10", help="cut-offs (default 1,5,10)")
    s.add_argument("--rerank", action="store_true")
    s.add_argument("--no-rerank", action="store_true")
    s.add_argument("--output", help="write full JSON report here")
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("config", parents=[common], help="print the effective configuration")
    s.set_defaults(func=cmd_config)

    s = sub.add_parser("prune", parents=[common], help="drop cached embeddings no chunk uses")
    s.add_argument("--only-active", action="store_true", help="also drop vectors of other models")
    s.set_defaults(func=cmd_prune)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if getattr(args, "repo_pos", None):
        args.repo = args.repo_pos
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        raise SystemExit(f"error: {e}")


if __name__ == "__main__":
    main()
