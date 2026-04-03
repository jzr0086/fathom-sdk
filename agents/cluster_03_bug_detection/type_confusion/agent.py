"""Type Confusion Detector — flags mixed return types, unsafe casts, and loose comparisons."""

from __future__ import annotations

import re

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_functions, find_return_statements, get_node_text
from fathom_sdk.context.ast_parser import parse

# ---------------------------------------------------------------------------
# Python: classify return value types from the return-statement text
# ---------------------------------------------------------------------------

_PY_STRING_RE = re.compile(r"""^(['"]|f['"]|b['"]|r['"])""")
_PY_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")
_PY_NONE_RE = re.compile(r"^None$")
_PY_BOOL_RE = re.compile(r"^(True|False)$")
_PY_LIST_RE = re.compile(r"^\[")
_PY_DICT_RE = re.compile(r"^\{")
_PY_TUPLE_RE = re.compile(r"^\(")

_PY_OPTIONAL_RE = re.compile(r"Optional\[|Union\[.*None|-> .+\| *None|\| *None")


def _python_return_type_label(value_text: str) -> str | None:
    """Return a coarse type label for a Python return value, or None if unknown."""
    v = value_text.strip()
    if not v:
        return "none"  # bare ``return``
    if _PY_NONE_RE.match(v):
        return "none"
    if _PY_BOOL_RE.match(v):
        return "bool"
    if _PY_NUMERIC_RE.match(v):
        return "numeric"
    if _PY_STRING_RE.match(v):
        return "string"
    if _PY_LIST_RE.match(v):
        return "list"
    if _PY_DICT_RE.match(v):
        return "dict"
    if _PY_TUPLE_RE.match(v):
        return "tuple"
    return None  # cannot determine — skip


# ---------------------------------------------------------------------------
# Go: type assertion without comma-ok
# ---------------------------------------------------------------------------

_GO_TYPE_ASSERT_RE = re.compile(r"\.\(\s*[A-Z]\w*")
_GO_COMMA_OK_RE = re.compile(r",\s*(ok|_)\s*[:=]=?\s*\w+\.\(")

# ---------------------------------------------------------------------------
# JavaScript / TypeScript: loose equality and typeof-then-numeric
# ---------------------------------------------------------------------------

_JS_LOOSE_EQ_RE = re.compile(r"[^!=]==[^=]")
_JS_TYPEOF_STRING_RE = re.compile(r'typeof\s+\w+\s*===?\s*["\']string["\']')
_JS_NUMERIC_OP_RE = re.compile(r"[\+\-\*/%]\s*\w+|\w+\s*[\+\-\*/%]")

# ---------------------------------------------------------------------------
# Java: cast without instanceof
# ---------------------------------------------------------------------------

_JAVA_CAST_RE = re.compile(r"\(\s*([A-Z]\w*(?:<[^>]+>)?)\s*\)\s*(\w+)")
_JAVA_INSTANCEOF_RE = re.compile(r"(\w+)\s+instanceof\s+(\w+)")


class TypeConfusionAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="type_confusion",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java", "go"],
            domains=["web_development", "systems_programming"],
            methodology="bug_detection",
            axis_type="critical",
            tags=["bug", "type-confusion", "type-safety"],
            model_required=False,
            estimated_cost_cents=0.0,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze(self, context: CodeContext) -> list[Finding]:
        if not context.source_code.strip():
            return []
        lang = context.language.lower()
        if lang == "python":
            return self._analyze_python(context)
        if lang in ("javascript", "typescript"):
            return self._analyze_js_ts(context)
        if lang == "go":
            return self._analyze_go(context)
        if lang == "java":
            return self._analyze_java(context)
        return []

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Type confusion bugs cause unexpected runtime "
            f"behaviour and are a common source of subtle errors. "
            f"Ensure consistent types or add explicit type guards."
        )

    # ------------------------------------------------------------------
    # Python — mixed return types
    # ------------------------------------------------------------------

    def _analyze_python(self, context: CodeContext) -> list[Finding]:
        root = context.ast
        if root is None:
            try:
                root = parse(context.source_code, context.language)
            except (ValueError, Exception):
                return []

        findings: list[Finding] = []
        functions = find_functions(root, context.language)
        returns = find_return_statements(root, context.language)

        # Check if source has Optional annotation (coarse heuristic)
        has_optional = bool(_PY_OPTIONAL_RE.search(context.source_code))

        for func in functions:
            func_returns = [
                r for r in returns
                if r.enclosing_function == func.name
            ]
            if len(func_returns) < 2:
                continue

            type_labels: set[str] = set()
            for ret in func_returns:
                label = _python_return_type_label(ret.value_text)
                if label is not None:
                    type_labels.add(label)

            # A mix of "none" and one other type is only flagged when there
            # is no Optional annotation anywhere in the source.
            if has_optional and type_labels == {"none"} | (type_labels - {"none"}):
                if len(type_labels - {"none"}) <= 1:
                    continue

            if len(type_labels) >= 2:
                # If the only pair is {none, X} and Optional is declared, skip
                if type_labels - {"none"} and len(type_labels) == 2 and "none" in type_labels and has_optional:
                    continue

                findings.append(
                    Finding(
                        agent_name="type_confusion",
                        severity="medium",
                        category="bug",
                        title=(
                            f"Mixed return types in '{func.name}': "
                            f"{', '.join(sorted(type_labels))}"
                        ),
                        description=(
                            f"Function '{func.name}' returns values of different "
                            f"types ({', '.join(sorted(type_labels))}). "
                            f"This can cause type confusion at call sites."
                        ),
                        file_path=context.file_path,
                        line_start=func.start_line,
                        line_end=func.end_line,
                        confidence=0.78,
                        tags=["bug", "type-confusion", "type-safety"],
                    )
                )
        return findings

    # ------------------------------------------------------------------
    # JavaScript / TypeScript — loose equality & typeof confusion
    # ------------------------------------------------------------------

    def _analyze_js_ts(self, context: CodeContext) -> list[Finding]:
        findings: list[Finding] = []
        lines = context.source_code.splitlines()

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith(("//", "/*", "*")):
                continue

            # Loose equality (== instead of ===)
            if _JS_LOOSE_EQ_RE.search(stripped):
                findings.append(
                    Finding(
                        agent_name="type_confusion",
                        severity="medium",
                        category="bug",
                        title="Loose equality (==) may cause type coercion",
                        description=(
                            f"Line {i}: Using == instead of === allows implicit "
                            f"type coercion which can lead to unexpected matches."
                        ),
                        file_path=context.file_path,
                        line_start=i,
                        line_end=i,
                        confidence=0.78,
                        tags=["bug", "type-confusion", "type-safety"],
                    )
                )
                continue  # one finding per line

            # typeof string check followed by numeric operation on next line
            if _JS_TYPEOF_STRING_RE.search(stripped):
                if i < len(lines):
                    next_line = lines[i].strip()  # lines is 0-indexed, i is 1-indexed
                    if _JS_NUMERIC_OP_RE.search(next_line):
                        findings.append(
                            Finding(
                                agent_name="type_confusion",
                                severity="medium",
                                category="bug",
                                title="Numeric operation after typeof string check",
                                description=(
                                    f"Line {i}: typeof check identifies a string, "
                                    f"but line {i + 1} performs a numeric operation."
                                ),
                                file_path=context.file_path,
                                line_start=i,
                                line_end=i + 1,
                                confidence=0.78,
                                tags=["bug", "type-confusion", "type-safety"],
                            )
                        )
        return findings

    # ------------------------------------------------------------------
    # Go — type assertion without comma-ok
    # ------------------------------------------------------------------

    def _analyze_go(self, context: CodeContext) -> list[Finding]:
        findings: list[Finding] = []
        lines = context.source_code.splitlines()

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("//"):
                continue

            if _GO_TYPE_ASSERT_RE.search(stripped):
                # Check that the assignment uses the comma-ok pattern
                if not _GO_COMMA_OK_RE.search(stripped):
                    findings.append(
                        Finding(
                            agent_name="type_confusion",
                            severity="medium",
                            category="bug",
                            title="Type assertion without comma-ok pattern",
                            description=(
                                f"Line {i}: Type assertion without the comma-ok "
                                f"pattern will panic at runtime if the type does "
                                f"not match. Use `val, ok := x.(Type)` instead."
                            ),
                            file_path=context.file_path,
                            line_start=i,
                            line_end=i,
                            confidence=0.78,
                            tags=["bug", "type-confusion", "type-safety"],
                        )
                    )
        return findings

    # ------------------------------------------------------------------
    # Java — unsafe cast without instanceof
    # ------------------------------------------------------------------

    def _analyze_java(self, context: CodeContext) -> list[Finding]:
        findings: list[Finding] = []
        lines = context.source_code.splitlines()

        # First pass: collect instanceof checks   var -> set of checked types
        instanceof_vars: dict[str, set[str]] = {}
        for line in lines:
            for m in _JAVA_INSTANCEOF_RE.finditer(line):
                var = m.group(1)
                checked_type = m.group(2)
                instanceof_vars.setdefault(var, set()).add(checked_type)

        # Second pass: find casts and check for preceding instanceof
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith(("//", "/*", "*")):
                continue

            for m in _JAVA_CAST_RE.finditer(stripped):
                cast_type = m.group(1)
                cast_var = m.group(2)

                # Skip primitive-like lowercase casts (int, long, etc.)
                if cast_type[0].islower():
                    continue

                # Check if this variable was verified with instanceof for this type
                checked_types = instanceof_vars.get(cast_var, set())
                if cast_type in checked_types:
                    continue

                findings.append(
                    Finding(
                        agent_name="type_confusion",
                        severity="medium",
                        category="bug",
                        title=(
                            f"Unsafe cast to {cast_type} without instanceof check"
                        ),
                        description=(
                            f"Line {i}: Casting '{cast_var}' to {cast_type} "
                            f"without a preceding instanceof check may throw "
                            f"ClassCastException at runtime."
                        ),
                        file_path=context.file_path,
                        line_start=i,
                        line_end=i,
                        confidence=0.78,
                        tags=["bug", "type-confusion", "type-safety"],
                    )
                )
        return findings
