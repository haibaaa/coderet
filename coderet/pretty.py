"""Human-friendly rendering of search results: syntax highlighting with real
line numbers, one panel per result, and a scrollable pager.

Only used with `coderet search --pretty`; the default output stays plain text
so it can be piped and parsed.
"""

from __future__ import annotations

import os
import sys
from typing import Callable

from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from .search import SearchResult

# coderet language id -> pygments lexer name
_LEXERS = {
    "python": "python", "javascript": "javascript", "typescript": "typescript",
    "tsx": "tsx", "java": "java", "go": "go", "rust": "rust", "c": "c", "cpp": "cpp",
    "ruby": "ruby", "csharp": "csharp", "php": "php", "kotlin": "kotlin",
    "scala": "scala", "swift": "swift", "lua": "lua", "shell": "bash", "sql": "sql",
    "markdown": "markdown", "rst": "rst", "toml": "toml", "yaml": "yaml", "ini": "ini",
}


def score_label(r: SearchResult) -> str:
    parts = [f"{r.score:.4f}"]
    if r.dense_rank:
        parts.append(f"dense#{r.dense_rank}")
    if r.lexical_rank:
        parts.append(f"bm25#{r.lexical_rank}")
    if r.rerank_score is not None:
        parts.append("reranked")
    return " ".join(parts)


def _lexer(language: str):
    name = _LEXERS.get(language, "text")
    if name == "php":  # chunks rarely start with "<?php"; highlight anyway
        from pygments.lexers import PhpLexer
        return PhpLexer(startinline=True)
    return name


def render_results(
    query: str,
    results: list[SearchResult],
    *,
    max_lines: int | None,
    callers_for: Callable[[SearchResult], list[dict]] | None = None,
    theme: str = "monokai",
    pager: bool = True,
) -> None:
    console = Console(highlight=False)
    items: list = [Text.assemble(("Query: ", "bold"), query, (f"   {len(results)} results", "dim"))]
    for i, r in enumerate(results, 1):
        lines = r.text.split("\n")
        shown = lines if max_lines is None else lines[:max_lines]
        body: list = [Syntax(
            "\n".join(shown), _lexer(r.language), theme=theme, line_numbers=True,
            start_line=r.start_line, word_wrap=True, background_color="default",
        )]
        if len(shown) < len(lines):
            body.append(Text(f"... {len(lines) - len(shown)} more lines", style="dim"))
        if callers_for and r.symbol:
            calls = callers_for(r)
            if calls:
                body.append(Text("called from: " + ", ".join(f"{c['path']}:{c['line']}" for c in calls),
                                 style="cyan"))
        title = Text.assemble(
            (f"{i}. ", "bold"), (r.path, "bold magenta"), (f":{r.start_line}-{r.end_line}", "magenta"),
            "  ", (r.symbol or f"({r.kind})", "bold green"),
        )
        items.append(Panel(Group(*body), title=title, title_align="left",
                           subtitle=Text(score_label(r), style="dim"), subtitle_align="right",
                           border_style="blue"))

    if pager and sys.stdout.isatty():
        # -R: pass colours through, -F: quit if it fits on one screen,
        # -X: leave the output on screen after quitting. Respect an existing $LESS.
        os.environ.setdefault("LESS", "-RFX")
        with console.pager(styles=True):
            console.print(*items)
    else:
        console.print(*items)
