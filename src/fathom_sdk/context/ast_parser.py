"""tree-sitter wrapper — parse source code into ASTs.

This module owns all tree-sitter interaction.  The rest of the codebase
should call :func:`parse` or :func:`get_parser` rather than touching
tree-sitter directly.

Uses the modern tree-sitter Python API (v0.23+) where each language is
distributed as its own package.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import tree_sitter


def _python_language() -> object:
    from tree_sitter_python import language as _lang
    return _lang()


def _javascript_language() -> object:
    from tree_sitter_javascript import language as _lang
    return _lang()


def _typescript_language() -> object:
    from tree_sitter_typescript import language_typescript as _lang
    return _lang()


def _java_language() -> object:
    from tree_sitter_java import language as _lang
    return _lang()


def _go_language() -> object:
    from tree_sitter_go import language as _lang
    return _lang()


SUPPORTED_LANGUAGES: dict[str, Callable[[], object]] = {
    "python": _python_language,
    "javascript": _javascript_language,
    "typescript": _typescript_language,
    "java": _java_language,
    "go": _go_language,
}

_ALIASES: dict[str, str] = {
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "golang": "go",
}


def _resolve_language(language: str) -> str:
    """Normalise a language name to its canonical key."""
    key = language.strip().lower()
    return _ALIASES.get(key, key)


@lru_cache(maxsize=16)
def get_parser(language: str) -> tree_sitter.Parser:
    """Return a cached tree-sitter Parser for the given language.

    Raises ValueError if the language is not supported.
    """
    canonical = _resolve_language(language)
    lang_fn = SUPPORTED_LANGUAGES.get(canonical)
    if lang_fn is None:
        supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
        raise ValueError(
            f"Unsupported language: {language!r}. "
            f"Supported languages: {supported}"
        )
    return tree_sitter.Parser(tree_sitter.Language(lang_fn()))


def parse(source_code: str, language: str) -> tree_sitter.Node:
    """Parse source_code and return the root AST node.

    Parameters
    ----------
    source_code:
        The full source text of the file being reviewed.
    language:
        A language identifier (or alias like "py", "js").

    Returns
    -------
    tree_sitter.Node
        The root node of the concrete syntax tree.
    """
    parser = get_parser(language)
    tree = parser.parse(source_code.encode("utf-8"))
    return tree.root_node
