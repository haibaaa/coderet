"""Syntax-aware chunking and dependency extraction (one tree-sitter pass per file).

Chunking strategy
-----------------
* Each top-level definition (function, class, ...) becomes one chunk if it fits
  in `max_chunk_chars`.
* An oversized definition that contains nested definitions (e.g. a large class)
  is split recursively: each nested definition becomes a chunk (symbol
  `Outer.inner`) and the remaining lines (signature, fields, docstring) become a
  `header` chunk.
* An oversized definition without nested definitions is split into overlapping
  line windows that keep the symbol name.
* Code outside definitions (imports, globals, scripts) is grouped into
  contiguous `module` chunks.
* Files without a grammar are split into overlapping line windows.

Line numbers are 1-based and inclusive throughout.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import ChunkingConfig
from .languages import SPECS, LangSpec, get_parser


@dataclass
class Chunk:
    start_line: int
    end_line: int
    kind: str  # definition node type, "header", "module", "window", ...
    symbol: str | None
    text: str


@dataclass
class Definition:
    name: str
    qualified: str
    kind: str
    start_line: int
    end_line: int


@dataclass
class Import:
    module: str
    name: str | None
    line: int


@dataclass
class Reference:
    name: str  # last identifier of the callee, e.g. "bar" for self.foo.bar()
    qualified: str  # full callee text (truncated)
    line: int


@dataclass
class ParsedFile:
    language: str
    chunks: list[Chunk] = field(default_factory=list)
    definitions: list[Definition] = field(default_factory=list)
    imports: list[Import] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    syntax_errors: bool = False


_IDENT = re.compile(r"[A-Za-z_$][\w$]*")
_ANGLE = re.compile(r"<[^<>]*>")
_STRING = re.compile(r"""['"`]([^'"`]+)['"`]""")


def parse_source(source: bytes, language: str, cfg: ChunkingConfig) -> ParsedFile:
    text = source.decode("utf-8", errors="replace")
    lines = text.split("\n")
    spec = SPECS.get(language)
    parser = get_parser(language) if spec else None
    result = ParsedFile(language=language)
    if parser is None or spec is None:
        result.chunks = _window_chunks(lines, 1, len(lines), "window", None, cfg)
        return result

    tree = parser.parse(source)
    root = tree.root_node
    result.syntax_errors = root.has_error
    extractor = _Extractor(spec, source, lines, cfg)
    result.chunks = extractor.chunk_scope(root, 1, len(lines), None, "module")
    extractor.collect_dependencies(root, result)
    return result


# --------------------------------------------------------------------------- #
# Plain line windows
# --------------------------------------------------------------------------- #

def _window_chunks(
    lines: list[str], start: int, end: int, kind: str, symbol: str | None, cfg: ChunkingConfig
) -> list[Chunk]:
    """Split lines[start..end] (1-based inclusive) into overlapping windows,
    additionally bounded by max_chunk_chars."""
    chunks: list[Chunk] = []
    size, overlap = cfg.window_lines, cfg.window_overlap
    i = start
    while i <= end:
        j = min(end, i + size - 1)
        # shrink window if too many characters (very long lines)
        while j > i and sum(len(l) + 1 for l in lines[i - 1:j]) > cfg.max_chunk_chars:
            j = i + max(1, (j - i) // 2)
        body = "\n".join(lines[i - 1:j])
        if len(body.strip()) >= cfg.min_chunk_chars:
            if len(body) > cfg.max_chunk_chars:  # single giant line
                body = body[: cfg.max_chunk_chars]
            chunks.append(Chunk(i, j, kind, symbol, body))
        if j >= end:
            break
        i = max(i + 1, j - overlap + 1)
    return chunks


# --------------------------------------------------------------------------- #
# Tree-sitter extraction
# --------------------------------------------------------------------------- #

class _Extractor:
    def __init__(self, spec: LangSpec, source: bytes, lines: list[str], cfg: ChunkingConfig):
        self.spec = spec
        self.source = source
        self.lines = lines
        self.cfg = cfg

    # -- helpers ------------------------------------------------------------ #

    def node_text(self, node) -> str:
        return self.source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def span_text(self, start: int, end: int) -> str:
        return "\n".join(self.lines[start - 1:end])

    def inner_definition(self, node):
        """Return the definition node `node` represents, or None."""
        spec = self.spec
        if node.type in spec.definitions:
            return node
        if node.type in spec.wrappers:
            for child in node.named_children:
                inner = self.inner_definition(child)
                if inner is not None:
                    return inner
            return None
        if node.type in spec.var_declarations:
            for child in node.named_children:
                if self._is_function_declarator(child):
                    return child
        return None

    _FUNCTION_VALUES = frozenset({
        "arrow_function", "function", "function_expression", "generator_function", "class",
    })

    def _is_function_declarator(self, node) -> bool:
        """JS/TS `const f = () => ...` style definitions."""
        if node.type != "variable_declarator" or not self.spec.var_declarations:
            return False
        value = node.child_by_field_name("value")
        return value is not None and value.type in self._FUNCTION_VALUES

    def definition_name(self, node) -> str | None:
        """Best-effort name of a definition node."""
        # C/C++ functions: name lives inside the declarator chain
        name_node = node.child_by_field_name("name")
        if name_node is None:
            decl = node.child_by_field_name("declarator")
            depth = 0
            while decl is not None and depth < 8:
                inner = decl.child_by_field_name("declarator")
                if inner is None:
                    name_node = decl
                    break
                decl = inner
                depth += 1
        if name_node is None and node.type == "impl_item":  # rust: impl Trait for Type
            t = node.child_by_field_name("type")
            tr = node.child_by_field_name("trait")
            if t is not None:
                base = self.node_text(t)
                return f"{self.node_text(tr)} for {base}" if tr is not None else base
        if name_node is None and node.type == "type_declaration":  # go
            for child in node.named_children:
                n = child.child_by_field_name("name")
                if n is not None:
                    name_node = n
                    break
        if name_node is None:
            for child in node.named_children:
                if child.type in {"identifier", "type_identifier", "constant", "name",
                                  "field_identifier", "property_identifier"}:
                    name_node = child
                    break
        if name_node is None:
            return None
        name = self.node_text(name_node).strip()
        if node.type in {"method_declaration"} and self.spec.name == "go":
            recv = node.child_by_field_name("receiver")
            if recv is not None:
                ids = _IDENT.findall(_ANGLE.sub("", self.node_text(recv)))
                if ids:
                    return f"{ids[-1]}.{name}"
        # qualified C++ names like Foo::bar -> keep as is; collapse whitespace
        return " ".join(name.split())[:200] or None

    def find_definitions(self, node) -> list:
        """Outermost definitions below `node` (not descending into them).
        Returns (outer_node, definition_node) pairs; outer includes wrappers."""
        found = []
        stack = list(reversed(node.named_children))
        while stack:
            child = stack.pop()
            inner = self.inner_definition(child)
            if inner is not None:
                # outer includes wrappers / the whole `const` statement
                found.append((child, inner))
                continue
            stack.extend(reversed(child.named_children))
        return found

    # -- chunking ----------------------------------------------------------- #

    def chunk_scope(self, node, start: int, end: int, parent: str | None, gap_kind: str) -> list[Chunk]:
        cfg = self.cfg
        chunks: list[Chunk] = []
        covered: list[tuple[int, int]] = []
        for outer, inner in self.find_definitions(node):
            s = outer.start_point[0] + 1
            e = outer.end_point[0] + 1
            if outer.end_point[1] == 0 and e > s:  # node ends at start of next line
                e -= 1
            if s < start or e > end:
                continue
            name = self.definition_name(inner) or "<anonymous>"
            qual = f"{parent}.{name}" if parent else name
            kind = inner.type
            body = self.span_text(s, e)
            covered.append((s, e))
            if len(body) <= cfg.max_chunk_chars:
                if len(body.strip()) >= cfg.min_chunk_chars:
                    chunks.append(Chunk(s, e, kind, qual, body))
            elif self.find_definitions(inner):
                chunks.extend(self.chunk_scope(inner, s, e, qual, "header"))
            else:
                chunks.extend(_window_chunks(self.lines, s, e, kind, qual, cfg))
        chunks.extend(self._gap_chunks(start, end, covered, parent, gap_kind))
        chunks.sort(key=lambda c: (c.start_line, c.end_line))
        return chunks

    def _gap_chunks(self, start, end, covered, parent, kind) -> list[Chunk]:
        covered_lines = set()
        for s, e in covered:
            covered_lines.update(range(s, e + 1))
        chunks: list[Chunk] = []
        run_start = None
        for ln in range(start, end + 2):
            in_gap = ln <= end and ln not in covered_lines
            if in_gap and run_start is None:
                run_start = ln
            elif not in_gap and run_start is not None:
                run_end = ln - 1
                # trim blank lines at both ends
                while run_start <= run_end and not self.lines[run_start - 1].strip():
                    run_start += 1
                while run_end >= run_start and not self.lines[run_end - 1].strip():
                    run_end -= 1
                if run_start <= run_end:
                    body = self.span_text(run_start, run_end)
                    if len(body) <= self.cfg.max_chunk_chars:
                        if len(body.strip()) >= self.cfg.min_chunk_chars:
                            chunks.append(Chunk(run_start, run_end, kind, parent, body))
                    else:
                        chunks.extend(_window_chunks(self.lines, run_start, run_end, kind, parent, self.cfg))
                run_start = None
        return chunks

    # -- dependencies ------------------------------------------------------- #

    def collect_dependencies(self, root, result: ParsedFile) -> None:
        spec = self.spec
        # (node, qualified parent) traversal
        stack = [(root, None)]
        while stack:
            node, parent = stack.pop()
            if node.type in spec.definitions or self._is_function_declarator(node):
                name = self.definition_name(node) or "<anonymous>"
                qual = f"{parent}.{name}" if parent else name
                result.definitions.append(Definition(
                    name=name.split(".")[-1].split("::")[-1],
                    qualified=qual,
                    kind=node.type,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                ))
                parent = qual
            elif node.type in spec.imports:
                for module, name in self.import_modules(node):
                    result.imports.append(Import(module, name, node.start_point[0] + 1))
            elif node.type in spec.calls:
                self.handle_call(node, result)
            for child in reversed(node.named_children):
                stack.append((child, parent))

    def handle_call(self, node, result: ParsedFile) -> None:
        field_name = self.spec.calls[node.type]
        callee = node.child_by_field_name(field_name) if field_name else None
        if callee is None:
            named = node.named_children
            callee = named[0] if named else None
        if callee is None:
            return
        text = self.node_text(callee)
        cleaned = _ANGLE.sub("", text)
        ids = _IDENT.findall(cleaned)
        if not ids:
            return
        name = ids[-1]
        line = node.start_point[0] + 1
        if name in self.spec.import_calls and len(ids) == 1:
            args = node.child_by_field_name("arguments")
            m = _STRING.search(self.node_text(args) if args is not None else self.node_text(node))
            if m:
                module = m.group(1)
                if name == "require_relative" and not module.startswith("."):
                    module = "./" + module
                result.imports.append(Import(module, None, line))
                return
        result.references.append(Reference(name, " ".join(text.split())[:200], line))

    def import_modules(self, node) -> list[tuple[str, str | None]]:
        lang = self.spec.name
        text = " ".join(self.node_text(node).split())
        out: list[tuple[str, str | None]] = []
        if lang == "python":
            m = re.match(r"from\s+([.\w]+)\s+import\s+(.+)", text)
            if m:
                names = m.group(2).strip("() ")
                for part in names.split(","):
                    nm = part.strip().split(" as ")[0].strip()
                    if nm:
                        out.append((m.group(1), nm))
                return out or [(m.group(1), None)]
            m = re.match(r"import\s+(.+)", text)
            if m:
                for part in m.group(1).split(","):
                    mod = part.strip().split(" as ")[0].strip()
                    if mod:
                        out.append((mod, None))
            return out
        if lang in {"javascript", "typescript", "tsx"}:
            m = _STRING.search(text)
            return [(m.group(1), None)] if m else []
        if lang == "go":
            m = _STRING.search(text)
            return [(m.group(1), None)] if m else []
        if lang == "java":
            m = re.match(r"import\s+(?:static\s+)?([\w.*]+)", text)
            return [(m.group(1), None)] if m else []
        if lang in {"c", "cpp"}:
            m = re.search(r"[<\"]([^>\"]+)[>\"]", text)
            return [(m.group(1), None)] if m else []
        if lang == "rust":
            m = re.match(r"(?:pub(?:\([^)]*\))?\s+)?(?:use|extern\s+crate)\s+(.+?);?$", text)
            return [(m.group(1).rstrip(";").strip(), None)] if m else []
        if lang == "csharp":
            m = re.match(r"(?:global\s+)?using\s+(?:static\s+)?(?:\w+\s*=\s*)?([\w.]+)", text)
            return [(m.group(1), None)] if m else []
        if lang == "php":
            m = re.match(r"use\s+(?:function\s+|const\s+)?([\w\\]+)", text)
            return [(m.group(1).lstrip("\\"), None)] if m else []
        return [(text[:200], None)]
