"""Uninitialized Variable Detector — flags variables used before assignment."""

from __future__ import annotations

import re

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_functions, get_node_text
from fathom_sdk.context.ast_parser import parse

# ---------------------------------------------------------------------------
# Language-specific configuration
# ---------------------------------------------------------------------------

# Built-in names / keywords that should never be flagged as uninitialized.
_BUILTINS: dict[str, set[str]] = {
    "python": {
        "True", "False", "None", "self", "cls", "print", "len", "range",
        "int", "str", "float", "list", "dict", "set", "tuple", "bool",
        "type", "super", "object", "Exception", "ValueError", "TypeError",
        "KeyError", "IndexError", "AttributeError", "RuntimeError",
        "StopIteration", "isinstance", "issubclass", "hasattr", "getattr",
        "setattr", "delattr", "property", "staticmethod", "classmethod",
        "enumerate", "zip", "map", "filter", "sorted", "reversed", "min",
        "max", "sum", "abs", "round", "any", "all", "open", "input",
        "format", "repr", "id", "hash", "iter", "next", "callable",
        "breakpoint", "exit", "quit", "help", "dir", "vars", "globals",
        "locals", "exec", "eval", "compile", "chr", "ord", "hex", "oct",
        "bin", "pow", "divmod", "complex", "bytes", "bytearray",
        "memoryview", "frozenset", "slice", "NotImplemented", "Ellipsis",
        "__name__", "__file__", "__doc__", "__all__", "__builtins__",
    },
    "javascript": {
        "undefined", "null", "true", "false", "NaN", "Infinity",
        "console", "window", "document", "global", "globalThis",
        "process", "require", "module", "exports", "__dirname", "__filename",
        "this", "arguments", "Math", "JSON", "Date", "Array", "Object",
        "String", "Number", "Boolean", "RegExp", "Error", "TypeError",
        "RangeError", "Map", "Set", "Promise", "Symbol", "Proxy",
        "parseInt", "parseFloat", "isNaN", "isFinite",
        "setTimeout", "setInterval", "clearTimeout", "clearInterval",
        "fetch", "alert", "confirm", "prompt",
    },
    "typescript": {
        "undefined", "null", "true", "false", "NaN", "Infinity",
        "console", "window", "document", "global", "globalThis",
        "process", "require", "module", "exports",
        "this", "arguments", "Math", "JSON", "Date", "Array", "Object",
        "String", "Number", "Boolean", "RegExp", "Error", "TypeError",
        "RangeError", "Map", "Set", "Promise", "Symbol", "Proxy",
        "parseInt", "parseFloat", "isNaN", "isFinite",
        "setTimeout", "setInterval", "clearTimeout", "clearInterval",
        "fetch",
    },
    "java": {
        "this", "super", "true", "false", "null",
        "System", "String", "Integer", "Long", "Double", "Float",
        "Boolean", "Byte", "Short", "Character", "Object", "Class",
        "Math", "Thread", "Runnable", "Exception", "RuntimeException",
        "Throwable", "Error", "NullPointerException",
        "IllegalArgumentException", "IllegalStateException",
        "Collections", "Arrays", "List", "Map", "Set", "Optional",
    },
    "go": {
        "true", "false", "nil", "iota",
        "append", "cap", "close", "complex", "copy", "delete",
        "imag", "len", "make", "new", "panic", "print", "println",
        "real", "recover", "error", "string", "int", "int8", "int16",
        "int32", "int64", "uint", "uint8", "uint16", "uint32", "uint64",
        "float32", "float64", "complex64", "complex128", "byte", "rune",
        "bool", "fmt", "os", "io", "log", "strings", "strconv",
        "context", "sync", "time", "net", "http", "errors",
    },
}

# Patterns that indicate an identifier on a line is being assigned (LHS).
_ASSIGNMENT_RE: dict[str, re.Pattern[str]] = {
    "python": re.compile(
        r"^(?:\s*)"              # leading whitespace
        r"(\w+)"                 # variable name
        r"\s*(?:[+\-*/|&^%]?=)" # assignment or augmented assignment
        r"(?!=)"                 # not == comparison
    ),
    "javascript": re.compile(
        r"(?:(?:let|var|const)\s+)?(\w+)\s*(?:[+\-*/|&^%]?=)(?!=)"
    ),
    "typescript": re.compile(
        r"(?:(?:let|var|const)\s+)?(\w+)\s*(?::\s*\w[\w<>\[\]|&, ]*\s*)?(?:[+\-*/|&^%]?=)(?!=)"
    ),
    "java": re.compile(
        r"(?:(?:final\s+)?(?:[A-Z]\w*(?:<[^>]*>)?(?:\[\])*)\s+)?(\w+)\s*(?:[+\-*/|&^%]?=)(?!=)"
    ),
    "go": re.compile(
        r"(\w+)\s*(?::=|=[^=])"
    ),
}

# Pattern matching a for-loop variable (the iterator variable).
_FOR_VAR_RE: dict[str, re.Pattern[str]] = {
    "python": re.compile(r"^\s*for\s+(\w+)"),
    "javascript": re.compile(r"(?:for\s*\(\s*(?:let|var|const)\s+)(\w+)"),
    "typescript": re.compile(r"(?:for\s*\(\s*(?:let|var|const)\s+)(\w+)"),
    "java": re.compile(r"for\s*\(\s*(?:\w+(?:<[^>]*>)?)\s+(\w+)"),
    "go": re.compile(r"for\s+(?:\w+\s*,\s*)?(\w+)\s*:=\s*range"),
}

# Import statement patterns (lines to skip entirely).
_IMPORT_RE: dict[str, re.Pattern[str]] = {
    "python": re.compile(r"^\s*(?:from\s+\S+\s+)?import\s+"),
    "javascript": re.compile(r"^\s*import\s+"),
    "typescript": re.compile(r"^\s*import\s+"),
    "java": re.compile(r"^\s*import\s+"),
    "go": re.compile(r'^\s*(?:import\s+|"[^"]+")'),
}

# Pattern to extract identifiers that look like variable reads.
# Negative lookbehind for '.' excludes member/property accesses (e.g. obj.method).
_IDENT_RE = re.compile(r"(?<!\.)\b([a-zA-Z_]\w*)\b")

# Pattern to strip string literals so identifiers inside strings are not matched.
_STRING_LITERAL_RE = re.compile(r'''(?:"[^"]*"|'[^']*'|`[^`]*`)''')

# Words that are keywords, not variable reads.
_KEYWORDS: dict[str, set[str]] = {
    "python": {
        "def", "class", "return", "if", "elif", "else", "for", "while",
        "try", "except", "finally", "with", "as", "import", "from",
        "pass", "break", "continue", "raise", "yield", "lambda",
        "and", "or", "not", "in", "is", "del", "global", "nonlocal",
        "assert", "async", "await",
    },
    "javascript": {
        "function", "return", "if", "else", "for", "while", "do",
        "switch", "case", "break", "continue", "try", "catch", "finally",
        "throw", "new", "delete", "typeof", "instanceof", "void",
        "var", "let", "const", "class", "extends", "import", "export",
        "default", "from", "async", "await", "yield", "in", "of",
    },
    "typescript": {
        "function", "return", "if", "else", "for", "while", "do",
        "switch", "case", "break", "continue", "try", "catch", "finally",
        "throw", "new", "delete", "typeof", "instanceof", "void",
        "var", "let", "const", "class", "extends", "import", "export",
        "default", "from", "async", "await", "yield", "in", "of",
        "type", "interface", "enum", "implements", "namespace", "declare",
    },
    "java": {
        "public", "private", "protected", "static", "final", "abstract",
        "class", "interface", "extends", "implements", "return", "if",
        "else", "for", "while", "do", "switch", "case", "break",
        "continue", "try", "catch", "finally", "throw", "throws",
        "new", "import", "package", "void", "int", "long", "double",
        "float", "boolean", "char", "byte", "short", "instanceof",
        "synchronized", "volatile", "transient", "enum",
    },
    "go": {
        "func", "return", "if", "else", "for", "range", "switch",
        "case", "break", "continue", "select", "defer", "go",
        "package", "import", "type", "struct", "interface", "map",
        "chan", "var", "const", "fallthrough", "default", "goto",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_file_level_names(source: str, language: str) -> set[str]:
    """Extract names defined at file/module scope (imports, top-level assignments).

    These names are available inside functions and should not be flagged
    as uninitialized when referenced.
    """
    lang = language.lower()
    names: set[str] = set()
    import_re = _IMPORT_RE.get(lang)
    assign_re = _ASSIGNMENT_RE.get(lang)

    for line in source.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//", "/*", "*")):
            continue

        # Import lines: extract the imported module / names.
        if import_re and import_re.match(stripped):
            if lang == "python":
                # "import os" -> os
                m = re.match(r"import\s+(\w+)", stripped)
                if m:
                    names.add(m.group(1))
                # "from os import path" -> path (and os)
                m = re.match(r"from\s+(\w+)", stripped)
                if m:
                    names.add(m.group(1))
                # Extract imported names after "import"
                parts = stripped.split("import")
                if len(parts) > 1:
                    for name_part in parts[-1].split(","):
                        name_part = name_part.strip()
                        # Handle "name as alias"
                        tokens = name_part.split(" as ")
                        if tokens:
                            final = tokens[-1].strip()
                            nm = re.match(r"(\w+)", final)
                            if nm:
                                names.add(nm.group(1))
            elif lang in ("javascript", "typescript"):
                # Extract imported names
                idents = _IDENT_RE.findall(stripped)
                for ident in idents:
                    if ident not in ("import", "from", "as", "default", "export"):
                        names.add(ident)
            elif lang == "java":
                # "import com.foo.Bar;" -> last component
                m = re.search(r"\.(\w+)\s*;", stripped)
                if m:
                    names.add(m.group(1))
            elif lang == "go":
                m = re.search(r'"[^"]*?(\w+)"', stripped)
                if m:
                    names.add(m.group(1))
            continue

        # Top-level assignments (indent level 0 or very low).
        if assign_re and _indent_level(line) == 0:
            am = assign_re.match(stripped)
            if am:
                names.add(am.group(1))

    return names


def _extract_function_bodies(source: str, language: str) -> list[tuple[list[str], int, str]]:
    """Return a list of (lines, start_line_1based, func_name) per function.

    Uses tree-sitter to find functions, then slices source lines.
    Falls back to whole-file analysis if parsing fails.
    """
    try:
        root = parse(source, language)
    except (ValueError, Exception):
        # Cannot parse — treat entire file as one scope
        return [(source.splitlines(), 1, "<module>")]

    functions = find_functions(root, language)
    if not functions:
        # No functions found — analyse file-level code
        return [(source.splitlines(), 1, "<module>")]

    all_lines = source.splitlines()
    bodies: list[tuple[list[str], int, str]] = []
    for fn in functions:
        start_idx = fn.start_line - 1  # 0-based
        end_idx = fn.end_line           # exclusive
        fn_lines = all_lines[start_idx:end_idx]
        bodies.append((fn_lines, fn.start_line, fn.name))
    return bodies


def _extract_params(func_lines: list[str], language: str) -> set[str]:
    """Extract parameter names from the first line(s) of a function definition."""
    params: set[str] = set()
    lang = language.lower()

    if not func_lines:
        return params

    # Collect the signature text (may span multiple lines for Python).
    sig_text = func_lines[0]
    if lang == "python":
        # Accumulate lines until we find the closing paren
        i = 0
        while ")" not in sig_text and i + 1 < len(func_lines):
            i += 1
            sig_text += " " + func_lines[i]
        # Extract everything between ( and )
        m = re.search(r"\(([^)]*)\)", sig_text)
        if m:
            raw = m.group(1)
            for part in raw.split(","):
                part = part.strip()
                if not part or part == "self" or part == "cls":
                    params.add(part)
                    continue
                # Handle type annotations: name: type = default
                name = re.match(r"(\w+)", part)
                if name:
                    params.add(name.group(1))
        # Always include self/cls
        params.discard("")
        params.add("self")
        params.add("cls")
    elif lang in ("javascript", "typescript"):
        m = re.search(r"\(([^)]*)\)", sig_text)
        if m:
            for part in m.group(1).split(","):
                part = part.strip()
                name = re.match(r"(\w+)", part)
                if name:
                    params.add(name.group(1))
    elif lang == "java":
        m = re.search(r"\(([^)]*)\)", sig_text)
        if m:
            for part in m.group(1).split(","):
                part = part.strip()
                # Type name pattern
                tokens = part.split()
                if tokens:
                    params.add(tokens[-1])
    elif lang == "go":
        m = re.search(r"\(([^)]*)\)", sig_text)
        if m:
            for part in m.group(1).split(","):
                part = part.strip()
                name = re.match(r"(\w+)", part)
                if name:
                    params.add(name.group(1))

    return params


def _is_comment_line(line: str, language: str) -> bool:
    """Check if a stripped line is a comment."""
    stripped = line.strip()
    if not stripped:
        return True
    lang = language.lower()
    if lang == "python":
        return stripped.startswith("#")
    return stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*")


def _is_conditional_block_start(line: str, language: str) -> bool:
    """Check if a line starts an if/elif/else block."""
    stripped = line.strip()
    lang = language.lower()
    if lang == "python":
        return bool(re.match(r"^(if|elif|else)\b", stripped))
    return bool(re.match(r"^(if|else\s+if|else)\b", stripped))


def _is_try_block_start(line: str, language: str) -> bool:
    """Check if a line starts a try block."""
    stripped = line.strip()
    lang = language.lower()
    if lang == "python":
        return stripped.startswith("try:")
    return stripped.startswith("try") and "{" in stripped


def _indent_level(line: str) -> int:
    """Return the number of leading spaces (tabs count as 4 spaces)."""
    count = 0
    for ch in line:
        if ch == " ":
            count += 1
        elif ch == "\t":
            count += 4
        else:
            break
    return count


def _analyze_function(
    func_lines: list[str],
    start_line: int,
    func_name: str,
    language: str,
    file_path: str,
    file_level_names: set[str] | None = None,
) -> list[Finding]:
    """Analyse a single function body for uninitialized variable usage."""
    lang = language.lower()
    builtins = _BUILTINS.get(lang, set())
    keywords = _KEYWORDS.get(lang, set())
    assign_re = _ASSIGNMENT_RE.get(lang)
    for_var_re = _FOR_VAR_RE.get(lang)
    import_re = _IMPORT_RE.get(lang)

    if assign_re is None:
        return []

    # Variables definitely assigned before use (in all paths).
    assigned: set[str] = set()
    # Variables assigned only in conditional branches (if without else, try blocks).
    conditionally_assigned: set[str] = set()
    # Track the indentation level of the current conditional/try scope.
    # -1 means "not inside a conditional/try block".
    conditional_indent: int = -1
    in_try_block = False

    # Parameters count as assigned.
    params = _extract_params(func_lines, language)
    assigned.update(params)

    # File-level names (imports, module-scope assignments) count as assigned.
    if file_level_names:
        assigned.update(file_level_names)

    findings: list[Finding] = []

    for i, line in enumerate(func_lines):
        lineno = start_line + i
        stripped = line.strip()

        # Skip blank lines and comments.
        if _is_comment_line(line, language):
            continue

        # Skip import lines.
        if import_re and import_re.match(stripped):
            continue

        # Skip function/class definition lines (the signature itself).
        if lang == "python" and re.match(r"^\s*(?:def|class)\s+", stripped):
            continue
        if lang in ("javascript", "typescript") and re.match(
            r"^\s*(?:function|class)\s+", stripped
        ):
            continue
        if lang == "java" and re.match(
            r"^\s*(?:public|private|protected|static|void|class|interface)\s+", stripped
        ):
            continue
        if lang == "go" and re.match(r"^\s*(?:func|type)\s+", stripped):
            continue

        current_indent = _indent_level(line)
        in_conditional = conditional_indent >= 0

        # Detect exit from conditional/try block.
        if lang == "python":
            if (in_conditional or in_try_block) and stripped:
                # If the line is at the same or lesser indentation as the
                # if/try keyword, and is not a continuation (elif/else/except/finally),
                # we have exited the block.
                if current_indent <= conditional_indent and not re.match(
                    r"^(elif|else|except|finally)\b", stripped
                ):
                    conditional_indent = -1
                    in_conditional = False
                    in_try_block = False
        else:
            # For brace-based languages, use a simple heuristic:
            # a closing brace at the same indent as the opening resets scope.
            if (in_conditional or in_try_block) and stripped == "}":
                conditional_indent = -1
                in_conditional = False
                in_try_block = False

        # Detect entry into conditional or try block.
        if _is_conditional_block_start(line, language):
            conditional_indent = current_indent
            in_conditional = True
        if _is_try_block_start(line, language):
            conditional_indent = current_indent
            in_try_block = True

        # Track for-loop variables.
        if for_var_re:
            fm = for_var_re.search(stripped)
            if fm:
                assigned.add(fm.group(1))
                continue

        # Track assignments.
        am = assign_re.search(stripped)
        if am:
            var_name = am.group(1)
            if in_conditional or in_try_block:
                # Only conditionally assigned — might not execute.
                if var_name not in assigned:
                    conditionally_assigned.add(var_name)
            else:
                assigned.add(var_name)
                conditionally_assigned.discard(var_name)
            # Don't check the RHS of assignments on the same line for
            # the variable being assigned (self-assignment is fine).
            continue

        # Check for reads of uninitialized or conditionally-assigned variables.
        # Strip string literals so identifiers inside strings are not matched.
        stripped_no_strings = _STRING_LITERAL_RE.sub("", stripped)
        identifiers = _IDENT_RE.findall(stripped_no_strings)
        for ident in identifiers:
            if ident in assigned:
                continue
            if ident in builtins or ident in keywords:
                continue
            if ident in params:
                continue
            # Skip UPPER_CASE identifiers (likely constants/globals).
            if ident.isupper() and len(ident) > 1:
                continue
            # Skip identifiers that start with uppercase (likely class names/types).
            if ident[0].isupper():
                continue
            # Skip decorator lines in Python.
            if lang == "python" and stripped.startswith("@"):
                break

            if ident in conditionally_assigned:
                findings.append(
                    Finding(
                        agent_name="uninitialized_variable",
                        severity="medium",
                        category="bug",
                        title=(
                            f"Variable '{ident}' may be uninitialized: "
                            f"only assigned in conditional/try block"
                        ),
                        description=(
                            f"'{ident}' is assigned inside a conditional or try block "
                            f"but used at line {lineno} where that assignment may "
                            f"not have executed."
                        ),
                        file_path=file_path,
                        line_start=lineno,
                        line_end=lineno,
                        confidence=0.75,
                        tags=["bug", "uninitialized", "variable"],
                    )
                )
                # Avoid duplicate findings for the same variable.
                conditionally_assigned.discard(ident)
                break
            else:
                # Completely unassigned before use.
                findings.append(
                    Finding(
                        agent_name="uninitialized_variable",
                        severity="medium",
                        category="bug",
                        title=(
                            f"Variable '{ident}' used before assignment"
                        ),
                        description=(
                            f"'{ident}' is used at line {lineno} in function "
                            f"'{func_name}' but was not assigned earlier in this scope."
                        ),
                        file_path=file_path,
                        line_start=lineno,
                        line_end=lineno,
                        confidence=0.75,
                        tags=["bug", "uninitialized", "variable"],
                    )
                )
                # Mark as "seen" so we don't re-flag on every line.
                assigned.add(ident)
                break

    return findings


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------


class UninitializedVariableAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="uninitialized_variable",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java", "go"],
            domains=["web_development", "systems_programming"],
            methodology="bug_detection",
            axis_type="aware",
            tags=["bug", "uninitialized", "variable"],
            model_required=False,
            estimated_cost_cents=0.0,
        )

    def analyze(self, context: CodeContext) -> list[Finding]:
        if not context.source_code.strip():
            return []
        lang = context.language.lower()
        if lang not in _BUILTINS:
            return []

        file_names = _extract_file_level_names(context.source_code, context.language)
        bodies = _extract_function_bodies(context.source_code, context.language)
        findings: list[Finding] = []
        for func_lines, start_line, func_name in bodies:
            findings.extend(
                _analyze_function(
                    func_lines, start_line, func_name, context.language,
                    context.file_path, file_names,
                )
            )
        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Using a variable before it has been assigned a value "
            f"can cause NameError (Python), ReferenceError (JavaScript/TypeScript), "
            f"or undefined behavior. Ensure all variables are initialized before use."
        )
