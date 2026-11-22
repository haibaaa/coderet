"""Language registry: file extensions and tree-sitter node types per language.

Languages with a `LangSpec` get syntax-aware chunking and dependency extraction.
Other known extensions are indexed as plain text with line-window chunking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache


@dataclass(frozen=True)
class LangSpec:
    name: str  # tree-sitter-language-pack grammar name
    # Node types that define a named unit (function, class, ...).
    definitions: frozenset[str]
    # Nodes that wrap a definition and should be chunked together with it
    # (decorators, `export`, templates). Name is taken from the wrapped node.
    wrappers: frozenset[str] = frozenset()
    # Variable declarations whose value is a function/class count as definitions
    # (JS/TS `const f = () => ...`).
    var_declarations: frozenset[str] = frozenset()
    imports: frozenset[str] = frozenset()
    # call-like node type -> field name holding the callee ("" = first named child)
    calls: dict[str, str] = field(default_factory=dict)
    # Callee names that are really imports (require, require_relative, ...).
    import_calls: frozenset[str] = frozenset()


_JS_DEFS = frozenset({
    "function_declaration", "generator_function_declaration", "class_declaration",
    "method_definition",
})
_TS_DEFS = _JS_DEFS | frozenset({
    "interface_declaration", "type_alias_declaration", "enum_declaration",
    "abstract_class_declaration", "internal_module", "module",
})
_JS_CALLS = {"call_expression": "function", "new_expression": "constructor"}

SPECS: dict[str, LangSpec] = {
    "python": LangSpec(
        name="python",
        definitions=frozenset({"function_definition", "class_definition"}),
        wrappers=frozenset({"decorated_definition"}),
        imports=frozenset({"import_statement", "import_from_statement"}),
        calls={"call": "function"},
    ),
    "javascript": LangSpec(
        name="javascript",
        definitions=_JS_DEFS,
        wrappers=frozenset({"export_statement"}),
        var_declarations=frozenset({"lexical_declaration", "variable_declaration"}),
        imports=frozenset({"import_statement"}),
        calls=_JS_CALLS,
        import_calls=frozenset({"require", "import"}),
    ),
    "typescript": LangSpec(
        name="typescript",
        definitions=_TS_DEFS,
        wrappers=frozenset({"export_statement"}),
        var_declarations=frozenset({"lexical_declaration", "variable_declaration"}),
        imports=frozenset({"import_statement"}),
        calls=_JS_CALLS,
        import_calls=frozenset({"require", "import"}),
    ),
    "tsx": LangSpec(
        name="tsx",
        definitions=_TS_DEFS,
        wrappers=frozenset({"export_statement"}),
        var_declarations=frozenset({"lexical_declaration", "variable_declaration"}),
        imports=frozenset({"import_statement"}),
        calls=_JS_CALLS,
        import_calls=frozenset({"require", "import"}),
    ),
    "java": LangSpec(
        name="java",
        definitions=frozenset({
            "class_declaration", "interface_declaration", "enum_declaration",
            "record_declaration", "annotation_type_declaration",
            "method_declaration", "constructor_declaration",
        }),
        imports=frozenset({"import_declaration"}),
        calls={"method_invocation": "name", "object_creation_expression": "type"},
    ),
    "go": LangSpec(
        name="go",
        definitions=frozenset({"function_declaration", "method_declaration", "type_declaration"}),
        imports=frozenset({"import_spec"}),
        calls={"call_expression": "function"},
    ),
    "rust": LangSpec(
        name="rust",
        definitions=frozenset({
            "function_item", "struct_item", "enum_item", "trait_item", "impl_item",
            "mod_item", "union_item", "macro_definition", "type_item",
        }),
        imports=frozenset({"use_declaration", "extern_crate_declaration"}),
        calls={"call_expression": "function", "macro_invocation": "macro"},
    ),
    "c": LangSpec(
        name="c",
        definitions=frozenset({"function_definition", "struct_specifier", "enum_specifier", "union_specifier"}),
        imports=frozenset({"preproc_include"}),
        calls={"call_expression": "function"},
    ),
    "cpp": LangSpec(
        name="cpp",
        definitions=frozenset({
            "function_definition", "class_specifier", "struct_specifier",
            "enum_specifier", "union_specifier", "namespace_definition",
        }),
        wrappers=frozenset({"template_declaration"}),
        imports=frozenset({"preproc_include"}),
        calls={"call_expression": "function", "new_expression": "type"},
    ),
    "ruby": LangSpec(
        name="ruby",
        definitions=frozenset({"method", "singleton_method", "class", "module"}),
        calls={"call": "method"},
        import_calls=frozenset({"require", "require_relative", "load"}),
    ),
    "csharp": LangSpec(
        name="csharp",
        definitions=frozenset({
            "class_declaration", "interface_declaration", "struct_declaration",
            "enum_declaration", "record_declaration", "method_declaration",
            "constructor_declaration", "namespace_declaration",
        }),
        imports=frozenset({"using_directive"}),
        calls={"invocation_expression": "function", "object_creation_expression": "type"},
    ),
    "php": LangSpec(
        name="php",
        definitions=frozenset({
            "function_definition", "class_declaration", "interface_declaration",
            "trait_declaration", "method_declaration", "enum_declaration",
        }),
        imports=frozenset({"namespace_use_declaration"}),
        calls={
            "function_call_expression": "function", "member_call_expression": "name",
            "scoped_call_expression": "name", "object_creation_expression": "",
        },
    ),
}

# extension -> language id (a key of SPECS, or a plain-text label)
EXTENSIONS: dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c", ".h": "c",
    ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp", ".hxx": "cpp",
    ".rb": "ruby",
    ".cs": "csharp",
    ".php": "php",
    # Plain-text (line-window chunking, no dependency extraction)
    ".kt": "kotlin", ".kts": "kotlin", ".scala": "scala", ".swift": "swift",
    ".lua": "lua", ".sh": "shell", ".bash": "shell", ".sql": "sql",
    ".md": "markdown", ".rst": "rst", ".txt": "text",
    ".toml": "toml", ".yaml": "yaml", ".yml": "yaml", ".cfg": "ini", ".ini": "ini",
}


def language_for(path: str, allowed: list[str] | None = None) -> str | None:
    dot = path.rfind(".")
    if dot < 0 or "/" in path[dot:]:
        return None
    ext = path[dot:].lower()
    if allowed is not None and ext not in allowed:
        return None
    return EXTENSIONS.get(ext)


@lru_cache(maxsize=None)
def get_parser(lang: str):
    """Return a tree-sitter parser for `lang`, or None if unavailable."""
    if lang not in SPECS:
        return None
    try:
        from tree_sitter_language_pack import get_parser as _gp
        return _gp(SPECS[lang].name)
    except Exception:  # grammar missing from the installed pack
        return None
