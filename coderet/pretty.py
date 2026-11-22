"""Human-friendly rendering of search results (`coderet search --pretty`).

Layout is designed to stay readable in narrow terminals:
* no boxes; each result has a plain header that wraps rather than truncates,
* code is never soft-wrapped (wrapping destroys indentation). In the pager,
  long lines extend to the right and are scrolled with the arrow keys
  (`less -S`); without a pager they are cropped at the terminal edge,
* the chunk's common indentation is removed; line numbers stay the real ones,
* the default theme uses the terminal's own ANSI palette, so it matches
  whatever colour scheme the terminal has.

The default (non-pretty) output stays plain text so it can be piped and parsed.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import textwrap
from typing import Callable

from rich.console import Console
from rich.rule import Rule
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

# -R colours, -S don't wrap long lines (scroll sideways with arrow keys),
# -F exit immediately if everything fits on one screen, -#8 sideways step.
# No -X: kitty and most terminals only turn the mouse wheel into scrolling
# inside the pager's alternate screen.
_LESS_FLAGS = ["-R", "-S", "-F", "-#8"]


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


def _code(r: SearchResult, max_lines: int | None) -> tuple[str, int]:
    lines = r.text.expandtabs(4).split("\n")
    hidden = 0
    if max_lines is not None and len(lines) > max_lines:
        hidden = len(lines) - max_lines
        lines = lines[:max_lines]
    return textwrap.dedent("\n".join(lines)), hidden


def _renderables(query, results, max_lines, callers_for, theme):
    """Returns ([(renderable, wide)], widest_code_line). `wide` items (code) may
    extend past the terminal edge; everything else is wrapped to the terminal."""
    items: list = [
        (Text.assemble(("Query: ", "bold"), (query, "bold")), False),
        (Text(f"{len(results)} result{'' if len(results) == 1 else 's'}", style="dim"), False),
    ]
    widest = 0
    for i, r in enumerate(results, 1):
        code, hidden = _code(r, max_lines)
        gutter = len(str(r.end_line)) + 3
        widest = max(widest, gutter + max((len(l) for l in code.split("\n")), default=0))
        items.append((Rule(style="dim"), False))
        items.append((Text.assemble(
            (f"{i}. ", "bold"), (r.path, "bold magenta"),
            (f":{r.start_line}-{r.end_line}", "magenta"),
        ), False))
        items.append((Text.assemble(
            (r.symbol or f"({r.kind})", "bold green"), "  ", (score_label(r), "dim"),
        ), False))
        items.append((Text(""), False))
        items.append((Syntax(
            code, _lexer(r.language), theme=theme, line_numbers=True,
            start_line=r.start_line, word_wrap=False, background_color="default",
        ), True))
        if hidden:
            items.append((Text(f"   ... {hidden} more lines (drop --lines to see all)", style="dim"), False))
        if callers_for and r.symbol:
            calls = callers_for(r)
            if calls:
                items.append((Text("called from: " + ", ".join(f"{c['path']}:{c['line']}" for c in calls),
                                   style="cyan"), False))
        items.append((Text(""), False))
    return items, widest


def _emit(console: Console, items, term_width: int) -> None:
    for renderable, wide in items:
        if wide:
            console.print(renderable, crop=True, overflow="crop")
        else:
            console.print(renderable, width=term_width)


def _pager_command() -> list[str] | None:
    """$CODERET_PAGER, else less with our flags, else None (print directly)."""
    custom = os.environ.get("CODERET_PAGER")
    if custom:
        return shlex.split(custom)
    less = shutil.which("less")
    return [less, *_LESS_FLAGS] if less else None


def render_results(
    query: str,
    results: list[SearchResult],
    *,
    max_lines: int | None,
    callers_for: Callable[[SearchResult], list[dict]] | None = None,
    theme: str = "ansi_dark",
    pager: bool = True,
) -> None:
    items, widest = _renderables(query, results, max_lines, callers_for, theme)
    term_width = shutil.get_terminal_size((100, 24)).columns
    pager_cmd = _pager_command() if pager and sys.stdout.isatty() else None

    if pager_cmd is None:
        # Direct output: fit the terminal; long code lines are cropped, not wrapped.
        width = term_width if sys.stdout.isatty() else max(widest, 80)
        console = Console(highlight=False, width=width)
        _emit(console, items, width)
        return

    # Pager: render wide enough that no code line is cut; less scrolls sideways.
    console = Console(highlight=False, force_terminal=True, width=max(term_width, widest + 1),
                      color_system=Console().color_system or "256")
    with console.capture() as cap:
        _emit(console, items, term_width)
    out = cap.get()
    try:
        subprocess.run(pager_cmd, input=out.encode("utf-8"), check=False)
    except (OSError, KeyboardInterrupt):
        sys.stdout.write(out)
